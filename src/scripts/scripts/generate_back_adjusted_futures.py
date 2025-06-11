#!/usr/bin/env python
"""
Generates back-adjusted continuous futures contracts.

Reads individual contract data from the 'market_data' table for specified intervals,
loads roll dates from the 'futures_roll_dates' table, performs back-adjustment
(currently constant price), and saves the adjusted continuous contract data
to the 'continuous_contracts' table.
Supports Nth contract position and specific roll times for sub-daily data.
"""

import logging
import argparse
import sys
from pathlib import Path
from datetime import timedelta, datetime, time
import pandas as pd
import duckdb
import yaml
from typing import Dict, Any, List, Optional, Tuple
import pytz
import subprocess
import re

# Add project root to sys.path BEFORE attempting to import from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: # Avoid adding multiple times if script is reloaded
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "market_symbols.yaml"

# Global constant for Primary Key columns to ensure consistency
CONTINUOUS_CONTRACTS_PK_COLUMNS = [
    "timestamp", 
    "symbol", 
    "interval_value", 
    "interval_unit", 
    "source"
]

# --- Database & Config Functions ---

def connect_db(db_path: Path, read_only: bool = True):
    """Connects to the DuckDB database."""
    try:
        logger.info(f"Connecting to database ({'read-only' if read_only else 'read-write'}): {db_path}")
        con = duckdb.connect(database=str(db_path), read_only=read_only)
        return con
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the market symbols configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def load_roll_dates_from_db(root_symbol: str, roll_type: str, con) -> pd.DataFrame:
    """Loads roll dates for a specific root symbol and roll type from the database."""
    logger.info(f"Loading '{roll_type}' roll dates for {root_symbol} from database...")
    query = """
    SELECT Contract, RollDate
    FROM futures_roll_dates
    WHERE SymbolRoot = ? AND RollType = ?
    ORDER BY RollDate; 
    """
    try:
        roll_dates_df = con.execute(query, [root_symbol, roll_type]).fetchdf()
        if roll_dates_df.empty:
            logger.error(f"No roll dates found for {root_symbol} with type '{roll_type}'. Ensure 'calculate_volume_roll_dates.py' has been run for this RollType.")
            return pd.DataFrame()
        
        roll_dates_df['RollDate'] = pd.to_datetime(roll_dates_df['RollDate']).dt.normalize()
        logger.info(f"Loaded {len(roll_dates_df)} roll dates.")
        return roll_dates_df
    except Exception as e:
        logger.error(f"Error loading roll dates from database: {e}")
        return pd.DataFrame()

def get_all_contracts_by_expiry(root_symbol: str, con) -> pd.DataFrame:
    """
    Retrieves all individual contracts for a root symbol from futures_roll_calendar,
    ordered by their last_trading_day.
    Returns a DataFrame with ['contract_code', 'last_trading_day'].
    """
    logger.info(f"Fetching all contract expiries for {root_symbol} from futures_roll_calendar...")
    query = """
    SELECT contract_code, last_trading_day
    FROM futures_roll_calendar
    WHERE root_symbol = ?
    ORDER BY last_trading_day;
    """
    try:
        contracts_df = con.execute(query, [root_symbol]).fetchdf()
        if contracts_df.empty:
            logger.warning(f"No contracts found in futures_roll_calendar for {root_symbol}.")
            return pd.DataFrame(columns=['contract_code', 'last_trading_day'])
        contracts_df['last_trading_day'] = pd.to_datetime(contracts_df['last_trading_day'])
        logger.info(f"Found {len(contracts_df)} contracts in futures_roll_calendar for {root_symbol}.")
        return contracts_df
    except Exception as e:
        logger.error(f"Error fetching contracts by expiry for {root_symbol}: {e}")
        return pd.DataFrame(columns=['contract_code', 'last_trading_day'])

def get_ordered_contracts(root_symbol: str, interval_value: int, interval_unit: str, con) -> List[str]:
    """Gets an ordered list of individual contract symbols from the market_data table for a specific interval. (Potentially Deprecated if futures_roll_calendar is primary source)"""
    logger.info(f"Fetching ordered contract list for {root_symbol} ({interval_value}-{interval_unit}) from market_data (may be deprecated)...")
    like_pattern = f"{root_symbol}%"
    exclude_pattern1 = f"@{root_symbol}%" 
    exclude_pattern2 = f"{root_symbol}c%"
    
    query = f"""
    SELECT DISTINCT symbol 
    FROM market_data 
    WHERE symbol LIKE ? 
      AND symbol NOT LIKE ? 
      AND symbol NOT LIKE ? 
      AND interval_value = ? 
      AND interval_unit = ?
    ORDER BY 
        CAST(SUBSTRING(symbol, LENGTH(symbol) - 1, 2) AS INTEGER) + 
            CASE WHEN CAST(SUBSTRING(symbol, LENGTH(symbol) - 1, 2) AS INTEGER) < 70 THEN 2000 ELSE 1900 END,
        CASE SUBSTRING(symbol, LENGTH(symbol) - 2, 1) 
                 WHEN 'F' THEN 1 WHEN 'G' THEN 2 WHEN 'H' THEN 3 WHEN 'J' THEN 4
                 WHEN 'K' THEN 5 WHEN 'M' THEN 6 WHEN 'N' THEN 7 WHEN 'Q' THEN 8
                 WHEN 'U' THEN 9 WHEN 'V' THEN 10 WHEN 'X' THEN 11 WHEN 'Z' THEN 12
                 ELSE 99
             END;
    """
    try:
        contracts_df = con.execute(query, [like_pattern, exclude_pattern1, exclude_pattern2, interval_value, interval_unit]).fetchdf()
        contracts = contracts_df['symbol'].tolist()
        if not contracts:
            logger.warning(f"No individual contracts found for {root_symbol} at {interval_value}-{interval_unit} in market_data using get_ordered_contracts.")
        else:
            logger.info(f"Found {len(contracts)} ordered contracts for {root_symbol} ({interval_value}-{interval_unit}) using get_ordered_contracts.")
        return contracts
    except Exception as e:
        logger.error(f"Error fetching ordered contracts for {root_symbol} using get_ordered_contracts: {e}")
        return []

def load_market_data_for_contract(symbol: str, interval_value: int, interval_unit: str, con) -> pd.DataFrame:
    """Fetches OHLCV and OpenInterest data for a specific contract and interval."""
    logger.debug(f"Fetching all data for {symbol} ({interval_value}-{interval_unit})")
    query = """
    SELECT 
        timestamp, open, high, low, close, volume, open_interest
    FROM market_data
    WHERE symbol = ? 
      AND interval_value = ? 
      AND interval_unit = ? 
    ORDER BY timestamp;
    """
    try:
        df = con.execute(query, [symbol, interval_value, interval_unit]).fetchdf()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']) # Should be UTC from DB or assumed UTC
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
            df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0).astype('int64')
            
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill().bfill() 
            df = df.dropna(subset=price_cols)
        logger.debug(f"Loaded {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for contract {symbol}: {e}")
        return pd.DataFrame()

def get_prices_for_adjustment(
    front_contract_symbol: str, 
    back_contract_symbol: str,
    roll_datetime_utc: pd.Timestamp, 
    front_contract_data: pd.DataFrame, # Should be UTC indexed
    back_contract_data: pd.DataFrame,  # Should be UTC indexed
    interval_unit: str, # For logging
    roll_time_specified: bool 
) -> Tuple[Optional[float], Optional[float]]:
    """
    Gets the close price of the front contract and the back contract
    at or immediately before the precise roll_datetime_utc.
    Assumes front_contract_data and back_contract_data have UTC TimestampIndex.
    """
    close_front = None
    close_back = None

    # Ensure roll_datetime_utc is timezone-aware (UTC)
    if roll_datetime_utc.tzinfo is None:
        roll_datetime_utc = roll_datetime_utc.tz_localize('UTC')
    elif roll_datetime_utc.tzinfo != pytz.UTC:
        roll_datetime_utc = roll_datetime_utc.tz_convert('UTC')

    # Front contract: price at or immediately BEFORE roll_datetime_utc
    # Data is already UTC indexed.
    front_bars_le_roll = front_contract_data[front_contract_data.index <= roll_datetime_utc]
    if not front_bars_le_roll.empty:
        close_front = front_bars_le_roll.iloc[-1]['close']
    else:
        logger.warning(f"No data found for front contract {front_contract_symbol} at or before roll datetime {roll_datetime_utc} to get its close.")

    # Back contract: price at or immediately BEFORE roll_datetime_utc for true close-to-close
    back_bars_le_roll = back_contract_data[back_contract_data.index <= roll_datetime_utc]
    if not back_bars_le_roll.empty:
        close_back = back_bars_le_roll.iloc[-1]['close']
    else:
        logger.warning(f"No data found for back contract {back_contract_symbol} at or before roll datetime {roll_datetime_utc} to get its close.")
        # Fallback: if no data at/before, try strictly after (first bar of new period). This might be an issue.
        back_bars_gt_roll = back_contract_data[back_contract_data.index > roll_datetime_utc]
        if not back_bars_gt_roll.empty:
            logger.warning(f"  Fallback: using first bar close of {back_contract_symbol} strictly after {roll_datetime_utc}.")
            close_back = back_bars_gt_roll.iloc[0]['close']

    if pd.isna(close_front):
        logger.error(f"Could not determine front contract {front_contract_symbol} close for roll at {roll_datetime_utc}.")
    if pd.isna(close_back):
        logger.error(f"Could not determine back contract {back_contract_symbol} close for roll at {roll_datetime_utc}.")
        
    logger.info(f"Prices for adjustment at {roll_datetime_utc} (UTC): {front_contract_symbol} close={close_front}, {back_contract_symbol} close={close_back}")
    return close_front, close_back

def generate_adjusted_series(
    root_symbol: str,
    contracts_by_expiry: pd.DataFrame, # DF of ['contract_code', 'last_trading_day'] from futures_roll_calendar
    all_contracts_data: Dict[str, pd.DataFrame], # Dict of individual contract DFs, UTC indexed
    roll_dates_df: pd.DataFrame, # DF of ['Contract', 'RollDate'] for the 1st position front month
    adjustment_type: str, # Changed from adjustment_method to adjustment_type
    interval_value: int,
    interval_unit: str,
    contract_position: int, # The desired contract position (1st, 2nd, ...)
    roll_time_obj: Optional[time] = None, # Specific roll time (naive)
    roll_time_zone_str: Optional[str] = None # Original timezone string for roll_time
) -> pd.DataFrame:
    """
    Generates a continuous future series, applying adjustments if specified.

    Args:
        root_symbol (str): The root symbol (e.g., 'ES').
        contracts_by_expiry (pd.DataFrame): DataFrame of ['contract_code', 'last_trading_day'] from futures_roll_calendar.
        all_contracts_data (Dict[str, pd.DataFrame]): Dict of individual contract DataFrames, UTC indexed.
        roll_dates_df (pd.DataFrame): DataFrame of ['Contract', 'RollDate'] for the 1st position front month.
        adjustment_type (str): 'constant' or 'none'. 
        interval_value (int): Interval value.
        interval_unit (str): Interval unit.
        contract_position (int): The desired contract position (1st, 2nd, ...).
        roll_time_obj (Optional[time]): Specific roll time (naive).
        roll_time_zone_str (Optional[str]): Timezone for the roll_time_obj.

    Returns:
        pd.DataFrame: The generated continuous futures series, UTC indexed.
    """
    logger.info(f"Generating {contract_position}P continuous series for {root_symbol} ({interval_value}-{interval_unit}) using {adjustment_type} adjustment.")
    
    if adjustment_type not in ['constant', 'none']:
        logger.error(f"Unsupported adjustment_type: {adjustment_type}. Supported: 'constant', 'none'.")
        return pd.DataFrame()

    if roll_dates_df.empty:
        logger.error("Roll dates DataFrame is empty. Cannot generate adjusted series.")
        return pd.DataFrame()
    if contracts_by_expiry.empty:
        logger.error("Contracts by expiry DataFrame is empty. Cannot determine contract sequence.")
        return pd.DataFrame()

    # Filter contracts_by_expiry to only those present in all_contracts_data
    available_contract_codes = list(all_contracts_data.keys())
    contracts_sequence_df = contracts_by_expiry[contracts_by_expiry['contract_code'].isin(available_contract_codes)].copy()
    contracts_sequence_df.sort_values(by='last_trading_day', ascending=True, inplace=True) # Oldest to newest

    if len(contracts_sequence_df) < contract_position:
        logger.error(f"Not enough contracts ({len(contracts_sequence_df)}) with market data and expiry to form the {contract_position}P series for {root_symbol}.")
        return pd.DataFrame()

    logger.info(f"Generating {contract_position}P continuous series for {root_symbol} ({interval_value}-{interval_unit}) using {adjustment_type}.")
    if roll_time_obj:
        logger.info(f"Specified roll time: {roll_time_obj} in timezone {roll_time_zone_str}.")
    else:
        logger.info("Using end-of-day roll (default).")

    final_adjusted_series_parts = []
    cumulative_adjustment = 0.0

    # Iterate through the sequence of contracts that will form the P-th continuous series.
    # Example: For ES 1P, this is ESH08, ESM08, ESU08, ...
    # For ES 2P, this is ESM08, ESU08, ESZ08, ...
    # The loop iterates over `current_contract_in_Pth_series` and `next_contract_in_Pth_series`.
    
    # Determine the actual list of contracts forming the Pth series
    # The Pth series uses the Pth contract from the global list, then (P+1)th, (P+2)th etc.
    if len(contracts_sequence_df) < contract_position:
         logger.error(f"Not enough contracts ({len(contracts_sequence_df)}) to form the {contract_position}P series after filtering for market data.")
         return pd.DataFrame()
         
    # `pth_series_contracts` are the actual contracts that will make up this P-th continuous line.
    # It starts from the (P-1)th index of the globally sorted list.
    pth_series_contracts = contracts_sequence_df.iloc[contract_position-1:].copy()
    
    if pth_series_contracts.empty:
        logger.error(f"No contracts identified to form the {contract_position}P series for {root_symbol}.")
        return pd.DataFrame()

    logger.info(f"Identified {len(pth_series_contracts)} contracts for the {contract_position}P series, starting with {pth_series_contracts.iloc[0]['contract_code']}.")

    for i in range(len(pth_series_contracts)):
        current_contract_symbol = pth_series_contracts.iloc[i]['contract_code']
        current_contract_data = all_contracts_data.get(current_contract_symbol)

        if current_contract_data is None or current_contract_data.empty:
            logger.warning(f"No market data for {current_contract_symbol} (part of {contract_position}P series), skipping its segment.")
            continue
        
        # Data should already be UTC indexed from loading step
        segment_to_add = current_contract_data.copy()
        segment_to_add[['open', 'high', 'low', 'close']] += cumulative_adjustment
        segment_to_add['adjustment_factor'] = cumulative_adjustment
        segment_to_add['individual_contract_front'] = current_contract_symbol 
        segment_to_add['individual_contract_next'] = pd.NA

        if i < len(pth_series_contracts) - 1: # If there's a newer contract in our P-th series to roll into
            next_contract_in_Pth_series_symbol = pth_series_contracts.iloc[i+1]['contract_code']
            segment_to_add['individual_contract_next'] = next_contract_in_Pth_series_symbol
            
            # The roll happens when `current_contract_symbol` (as a member of the Pth series) 
            # hands over to `next_contract_in_Pth_series_symbol`.
            # This roll date is determined by when the *underlying front month* that `current_contract_symbol`
            # was tracking rolls. This is complex.
            # Simpler: The P-th series rolls when the main (1st position) front month rolls.
            # We need the RollDate from `roll_dates_df` for the `current_contract_symbol` when IT IS THE FRONT contract.
            
            month_code_current = current_contract_symbol[len(root_symbol)] # e.g. H from ESH24
            year_short_current = current_contract_symbol[len(root_symbol)+1:] # e.g. 24 from ESH24
            roll_dates_key_contract = f"{month_code_current}{year_short_current}" # e.g. H24

            relevant_roll_info = roll_dates_df[roll_dates_df['Contract'] == roll_dates_key_contract]

            if relevant_roll_info.empty:
                logger.warning(f"No roll date found in roll_dates_df for Contract '{roll_dates_key_contract}' (key for {current_contract_symbol}). Cannot determine roll to {next_contract_in_Pth_series_symbol}. Adding full segment.")
                final_adjusted_series_parts.append(segment_to_add)
                continue # Adjustment does not change for next (older) segment
            
            roll_date_naive = relevant_roll_info.iloc[0]['RollDate'] # pd.Timestamp, normalized (date part only)

            # Construct precise roll_datetime_utc
            if roll_time_obj and roll_time_zone_str:
                try:
                    tz = pytz.timezone(roll_time_zone_str)
                    # Combine naive date from RollDate with naive time from roll_time_obj, then localize
                    localized_roll_dt = tz.localize(datetime.combine(roll_date_naive.date(), roll_time_obj))
                    roll_datetime_utc = localized_roll_dt.astimezone(pytz.utc)
                except Exception as e:
                    logger.error(f"Error localizing roll time {roll_time_obj} in {roll_time_zone_str} for date {roll_date_naive.date()}: {e}. Defaulting to EOD UTC for this roll.")
                    roll_datetime_utc = pd.Timestamp(datetime.combine(roll_date_naive.date(), time(23,59,59)), tz='UTC')
            else: # Default to end of day (UTC) for the given RollDate
                roll_datetime_utc = pd.Timestamp(datetime.combine(roll_date_naive.date(), time(23,59,59)), tz='UTC')
            
            logger.info(f"  Segment from {current_contract_symbol}. Rolling to {next_contract_in_Pth_series_symbol} at {roll_datetime_utc} (UTC).")

            # Get prices for adjustment
            # For 'none' adjustment, we still need to identify prices to know the roll point, but no adjustment factor is calculated/applied.
            close_front, close_back = get_prices_for_adjustment(
                current_contract_symbol, next_contract_in_Pth_series_symbol, 
                roll_datetime_utc, 
                current_contract_data, all_contracts_data.get(next_contract_in_Pth_series_symbol),
                interval_unit, roll_time_specified=(roll_time_obj is not None)
            )

            gap = 0.0
            if adjustment_type == 'constant':
                if pd.notna(close_front) and pd.notna(close_back):
                    gap = close_front - close_back
                    logger.info(f"    Adjusting: {current_contract_symbol} ({close_front:.2f}) to {next_contract_in_Pth_series_symbol} ({close_back:.2f}). Gap={gap:.2f}. New Cum.Adj for older segments: {cumulative_adjustment + gap:.2f}")
                    cumulative_adjustment += gap
                else:
                    logger.warning(f"    Cannot calculate gap for roll from {current_contract_symbol} to {next_contract_in_Pth_series_symbol} due to missing close prices. Adjustment not applied for this roll.")
            # If adjustment_type is 'none', gap remains 0.0, cumulative_adjustment remains unchanged (or 0.0 if it's the first effective segment)
            
            # Apply current cumulative_adjustment to prices of current_segment_to_add
            # For 'none' adjustment, current_cumulative_adjustment_for_segment will be 0.0 (or its prior value if we start from newest, which we dont)
            if adjustment_type == 'constant':
                segment_to_add[['open', 'high', 'low', 'close']] -= cumulative_adjustment
                segment_to_add['adjustment_factor'] = cumulative_adjustment
            else: # adjustment_type == 'none'
                segment_to_add['adjustment_factor'] = 0.0 # Explicitly 0 for unadjusted

            # Store metadata for the segment
            segment_to_add['individual_contract_front'] = current_contract_symbol
            segment_to_add['individual_contract_next'] = next_contract_in_Pth_series_symbol

            if not segment_to_add.empty:
                final_adjusted_series_parts.append(segment_to_add)
                # First segment's adjustment_factor is the final cumulative_adjustment from all older rolls.
                # Subsequent segments added will have progressively less adjustment.
                # This is confusing. Let's re-verify. We are building oldest to newest.
                # `cumulative_adjustment` at step `i` is the total adjustment to apply to contract `i` and all OLDER contracts.
                # When we add `segment_to_add` for `current_contract_symbol`, its prices were adjusted by the `cumulative_adjustment`
                # that was calculated based on rolls of contracts *newer* than `current_contract_symbol`.
                # This seems correct for building oldest to newest.
                logger.info(f"    Added {len(segment_to_add)} rows from {current_contract_symbol} (this segment's prices adj by: {segment_to_add['adjustment_factor'].iloc[0]:.2f})")
            else:
                logger.info(f"    No rows to add from {current_contract_symbol} for this segment of {contract_position}P series.")

    if not final_adjusted_series_parts:
        logger.error(f"No data parts to concatenate for {root_symbol} {contract_position}P series.")
        return pd.DataFrame()

    final_series_df = pd.concat(final_adjusted_series_parts)
    final_series_df.sort_index(inplace=True) 

    price_cols = ['open', 'high', 'low', 'close']
    if final_series_df[price_cols].isnull().any().any():
        logger.warning(f"NaNs found in price columns of final {contract_position}P series for {root_symbol}. Forward-filling.")
        final_series_df[price_cols] = final_series_df[price_cols].ffill()
    
    logger.info(f"Successfully generated {contract_position}P continuous series for {root_symbol} with {len(final_series_df)} rows.")
    return final_series_df

def _ensure_continuous_contracts_table(con, cli_args):
    """Ensures the continuous_contracts table exists with all necessary columns and the correct PK."""
    base_table_name = "continuous_contracts"
    
    # Define the target schema including all columns and the PK
    desired_columns_with_types = {
        "timestamp": "TIMESTAMP NOT NULL",
        "symbol": "VARCHAR NOT NULL",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "settle": "DOUBLE",
        "volume": "BIGINT",
        "open_interest": "BIGINT",
        "interval_value": "INTEGER NOT NULL",
        "interval_unit": "VARCHAR NOT NULL",
        "source": "VARCHAR NOT NULL",
        "adjustment_factor": "DOUBLE",
        "individual_contract_front": "VARCHAR",
        "individual_contract_next": "VARCHAR",
        "is_adjusted": "BOOLEAN",
        "roll_date_event": "VARCHAR",
        "created_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
    }
    # Use the global constant for PK definition order
    primary_key_columns_definition_order = CONTINUOUS_CONTRACTS_PK_COLUMNS

    table_exists = False
    existing_column_names = []

    try:
        table_check_df = con.execute(f"SELECT table_name FROM information_schema.tables WHERE table_name = '{base_table_name}' AND table_schema = current_schema()").fetchdf()
        if not table_check_df.empty:
            table_exists = True
            logger.info(f"Table '{base_table_name}' found in schema '{con.execute("SELECT current_schema()").fetchone()[0]}'.")
            pragma_df = con.execute(f"PRAGMA table_info('{base_table_name}');").fetchdf()
            if not pragma_df.empty:
                existing_column_names = pragma_df['name'].tolist()
        else:
            logger.info(f"Table '{base_table_name}' does not exist yet.")
    except Exception as e:
        logger.warning(f"Could not get info for table '{base_table_name}', assuming it does not exist or error: {e}")
        # table_exists remains False, existing_column_names remains empty

    table_recreated_due_to_pk_mismatch = False

    if table_exists:
        # Add missing columns if table exists
        for col_name, col_type_full in desired_columns_with_types.items():
            alter_col_type = col_type_full.split(" ")[0] # Basic type for ALTER
            if col_name not in existing_column_names:
                logger.info(f"Adding missing column '{col_name}' ({alter_col_type}) to existing table '{base_table_name}'.")
                try:
                    con.execute(f"ALTER TABLE {base_table_name} ADD COLUMN {col_name} {alter_col_type};")
                    existing_column_names.append(col_name) # Assume success, add to list
                except Exception as e:
                    logger.error(f"Failed to add column {col_name} to {base_table_name}: {e}")
        
        # Introspect existing table's primary key
        pk_info_df = con.execute(f"""
            SELECT STRING_AGG(name, ', ' ORDER BY pk) AS pk_columns
            FROM pragma_table_info('{base_table_name}')
            WHERE pk > 0
        """).fetchdf()

        current_pk_cols_str_for_display = "<none>"
        current_pk_set = set()
        if not pk_info_df.empty and 'pk_columns' in pk_info_df.columns and pk_info_df['pk_columns'].iloc[0] is not None:
            pk_str_from_db = pk_info_df['pk_columns'].iloc[0]
            current_pk_cols_str_for_display = pk_str_from_db
            current_pk_set = set(col.strip() for col in pk_str_from_db.split(','))
        
        desired_pk_set = set(primary_key_columns_definition_order)
        
        pk_mismatch = False
        if current_pk_set != desired_pk_set:
            pk_mismatch = True
        elif not current_pk_set and desired_pk_set: # No PK in DB, but we want one
            pk_mismatch = True
            current_pk_cols_str_for_display = "<none defined in DB>"


        if pk_mismatch:
            desired_pk_display_canonical = ", ".join(primary_key_columns_definition_order)
            logger.warning(f"Primary key mismatch for '{base_table_name}'.")
            logger.info(f"  PRAGMA table_info('{base_table_name}'):\n{pk_info_df}") # Log PRAGMA output
            logger.warning(f"  Current (from DB, order as in DB): '{current_pk_cols_str_for_display}'")
            logger.warning(f"  Desired (canonical order for script): '{desired_pk_display_canonical}'")

            if cli_args.recreate_table_on_pk_mismatch: # Use cli_args
                logger.warning(f"DEV OPTION: --recreate-table-on-pk-mismatch is set. DROPPING AND RECREATING TABLE '{base_table_name}'.")
                try:
                    con.execute(f"DROP TABLE IF EXISTS {base_table_name}")
                    logger.info(f"Successfully dropped table '{base_table_name}'. It will be recreated.")
                    table_exists = False # Signal for recreation
                    table_recreated_due_to_pk_mismatch = True
                except Exception as e:
                    logger.error(f"Failed to drop table '{base_table_name}' for recreation: {e}")
                    raise
            else:
                logger.error("Primary key mismatch detected and --recreate-table-on-pk-mismatch is NOT set. Upserts may fail.")
        elif current_pk_set: # PKs match
             logger.info(f"Primary key for '{base_table_name}' is correctly defined as: {', '.join(primary_key_columns_definition_order)}")
             logger.info(f"  PRAGMA table_info('{base_table_name}'):\n{pk_info_df}") # Log PRAGMA output even on match
             logger.info(f"  Current PK (from DB, order as in DB): '{current_pk_cols_str_for_display}'")


    # Create table if it doesn't exist or was just dropped for PK recreation
    if not table_exists:
        column_definitions = []
        for col_name in primary_key_columns_definition_order: # Start with PK columns in order
             if col_name in desired_columns_with_types:
                column_definitions.append(f"{col_name} {desired_columns_with_types[col_name]}")
        for col_name, col_type in desired_columns_with_types.items(): # Add remaining columns
            if col_name not in primary_key_columns_definition_order:
                 column_definitions.append(f"{col_name} {col_type}")

        primary_key_clause = f"PRIMARY KEY ({', '.join(primary_key_columns_definition_order)})"
        create_table_sql = f"CREATE TABLE {base_table_name} ({', '.join(column_definitions)}, {primary_key_clause})"
        
        try:
            con.execute(create_table_sql)
            action_word = "Recreated" if table_recreated_due_to_pk_mismatch else "Created"
            logger.info(f"Table '{base_table_name}' {action_word.lower()} successfully with PK: {', '.join(primary_key_columns_definition_order)}.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create table '{base_table_name}': {e}")
        raise

    logger.info(f"Table '{base_table_name}' schema setup complete.")

def save_continuous_futures_to_db(
    df: pd.DataFrame, 
    output_symbol: str, 
    source_id: str,
    interval_val: int,
    interval_u: str,
    is_adjusted: bool, # This flag is now from args.adjustment_type != "none"
    force_delete: bool, 
    con
):
    """Saves the generated continuous futures contract data to the database."""
    if df.empty:
        logger.warning(f"DataFrame for {output_symbol} is empty. Nothing to save.")
        return

    logger.info(f"Preparing to save {len(df)} bars for {output_symbol} to database.")
    
    df_to_save = df.copy()
    df_to_save = df_to_save.reset_index() # Convert timestamp index to column

    # Add / ensure core metadata columns
    df_to_save['symbol'] = output_symbol
    df_to_save['source'] = source_id
    df_to_save['interval_value'] = interval_val
    df_to_save['interval_unit'] = interval_u
    df_to_save['is_adjusted'] = is_adjusted 
    df_to_save['created_at'] = datetime.now(pytz.utc) # Use timezone-aware datetime.now(pytz.utc)
    df_to_save['updated_at'] = datetime.now(pytz.utc) # Use timezone-aware datetime.now(pytz.utc)

    # Columns from generation logic: adjustment_factor, individual_contract_front, individual_contract_next
    # Ensure they exist, default if not (though they should)
    for gen_col in ['adjustment_factor', 'individual_contract_front', 'individual_contract_next']:
        if gen_col not in df_to_save.columns:
            logger.warning(f"Generated column '{gen_col}' missing in df_to_save for {output_symbol}. Defaulting.")
            if gen_col == 'adjustment_factor': df_to_save[gen_col] = 0.0
            else: df_to_save[gen_col] = pd.NA


    # Target columns for the table (should align with _ensure_continuous_contracts_table)
    target_db_cols = [
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest',
        'interval_value', 'interval_unit', 'source', 
        'adjustment_factor', 'individual_contract_front', 'individual_contract_next',
        'is_adjusted', 'created_at', 'updated_at'
        # 'settle', 'underlying_symbol' -- Omitting unless explicitly needed and populated
    ]
    
    # Select and rename columns to match DB
    # 'timestamp' is already the name of the index after reset_index()
    
    # Ensure all target columns are present in df_to_save, adding NaNs if not (except for specific defaults)
    for db_col in target_db_cols:
        if db_col not in df_to_save.columns:
            if db_col == 'open_interest' and 'open_interest' not in df.columns: # df is original from generation
                 df_to_save[db_col] = 0 
            elif db_col == 'volume' and 'volume' not in df.columns:
                 df_to_save[db_col] = 0
            elif db_col in ['settle', 'underlying_symbol']: # Optional, not generated by default
                 df_to_save[db_col] = pd.NA
            else:
                logger.warning(f"Target DB column '{db_col}' missing in DataFrame for {output_symbol}. Will be pd.NA.")
                df_to_save[db_col] = pd.NA # Let DB handle null for other missing optional columns

    df_to_save = df_to_save[target_db_cols] # Select and order columns

    # Timestamp handling: Ensure UTC for DB
    if df_to_save['timestamp'].dt.tz is None:
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.tz_localize('UTC')
    elif df_to_save['timestamp'].dt.tz != pytz.UTC:
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.tz_convert('UTC')

    # PK for deletion and upsert: (timestamp, symbol, interval_value, interval_unit, source)
    pk_cols_for_delete = ['symbol', 'interval_value', 'interval_unit', 'source']

    if force_delete:
        try:
            logger.info(f"Force delete enabled. Removing existing data for PK components: {output_symbol}, {interval_val}-{interval_u}, source: {source_id}...")
            delete_conditions = " AND ".join([f"{col} = ?" for col in pk_cols_for_delete])
            delete_query = f"DELETE FROM continuous_contracts WHERE {delete_conditions}"
            con.execute(delete_query, [output_symbol, interval_val, interval_u, source_id])
            logger.info(f"Deletion complete for primary key set related to {output_symbol}.")
        except Exception as e:
            logger.error(f"Error deleting existing data for {output_symbol}: {e}")
            # Depending on policy, may want to raise or return here

    # Chunking parameters
    chunk_size = 100000
    num_chunks = (len(df_to_save) - 1) // chunk_size + 1
    logger.info(f"Total rows to save: {len(df_to_save)}. Processing in {num_chunks} chunk(s) of up to {chunk_size} rows each.")

    for i in range(num_chunks):
        chunk_df = df_to_save.iloc[i * chunk_size : (i + 1) * chunk_size]
        logger.info(f"Processing chunk {i+1}/{num_chunks} with {len(chunk_df)} rows for {output_symbol}.")

        # Sanitize temp_view_name further for chunking, though unregister should handle reuse
        # Using a slightly different name per chunk might be safer if multiple scripts run concurrently
        # but for sequential execution within one script, unregistering makes reuse fine.
        temp_view_name_base = f"temp_cont_{output_symbol.replace('@','').replace('=','_')}_{interval_val}{interval_u}_{source_id[:5]}"
        temp_view_name = f"{temp_view_name_base}_chunk{i+1}".replace('-', '_') # Sanitize

        try:
            con.register(temp_view_name, chunk_df)
            cols_for_sql = ", ".join([f'"{c}"' for c in chunk_df.columns])
            
            # Simplified to a direct INSERT for diagnostics, as --force-delete handles prior data
            sql = f"""
            INSERT INTO continuous_contracts ({cols_for_sql})
            SELECT {cols_for_sql} FROM {temp_view_name}
            """
            
            con.execute(sql)
            con.unregister(temp_view_name)
            logger.info(f"Successfully inserted chunk {i+1}/{num_chunks} ({len(chunk_df)} rows) for {output_symbol}.")

        except Exception as e:
            logger.error(f"Error saving chunk {i+1}/{num_chunks} for {output_symbol} to database: {e}")
            if temp_view_name in con.execute("SHOW TABLES").df()['name'].tolist(): # Check if view exists before unregistering
                con.unregister(temp_view_name)
            con.rollback() # Rollback on error for this chunk
            import traceback
            traceback.print_exc()
            # Decide if we should break or try next chunk. For now, let's break.
            logger.error(f"Aborting further chunk processing for {output_symbol} due to error in chunk {i+1}.")
            break 
    else: # This else block executes if the loop completed without a 'break'
        logger.info(f"All {num_chunks} chunks for {output_symbol} processed.")

def generate_output_symbol_name(root_symbol: str, contract_position: int, roll_type_cli: str, suffix: str, adjustment_type_arg: str) -> str:
    """
    Generates the continuous contract symbol based on root, position, roll rule, adjustment type, and suffix.
    Example: @ES=101XC_d
    roll_type_cli is the user-provided value (e.g., '01X', 'volume').
    adjustment_type_arg is the normalized value from args ('constant' or 'none').
    """
    adj_char = 'C' # Default to Constant
    if adjustment_type_arg == 'none':
        adj_char = 'N'
    
    # Ensure roll_type_cli is uppercase and remove any leading/trailing whitespace for consistency in the symbol.
    # The actual roll_type logic for DB queries should use the original case if necessary.
    roll_rule_code_for_symbol = roll_type_cli.strip().upper()

    # Format: @{SYMBOL_ROOT}={CONTRACT_POSITION}{ROLL_RULE_CODE}{ADJ_CHAR}{SUFFIX}
    return f"@{root_symbol}={contract_position}{roll_rule_code_for_symbol}{adj_char}{suffix}"

def main():
    parser = argparse.ArgumentParser(description="Generates back-adjusted or non-adjusted continuous futures contracts.")
    parser.add_argument("--root-symbol", type=str, required=True, help="Base symbol of the future (e.g., ES, NQ).")
    parser.add_argument("--roll-type", type=str, required=True, help="Roll type identifier from futures_roll_dates (e.g., 01X, volume). This is the roll rule code for the symbol.")
    parser.add_argument("--contract-position", type=int, default=1, help="The contract position to generate (e.g., 1 for 1st front month, 2 for 2nd). Default: 1.")
    parser.add_argument("--interval-value", type=int, required=True, help="Interval value (e.g., 1, 15).")
    parser.add_argument("--interval-unit", type=str, choices=['minute', 'daily', 'hour'], required=True, help="Interval unit.")
    parser.add_argument("--adjustment-method", type=str, default="constant_price", choices=["constant_price"], help="Back-adjustment method. Default: constant_price.")
    parser.add_argument("--output-symbol-suffix", type=str, default="_d", help="Suffix for the output continuous contract symbol (e.g., _d for derived).")
    parser.add_argument("--source-identifier", type=str, default="inhouse_backadj_const", help="Base source identifier to save in the database (position will be appended).")
    
    parser.add_argument("--roll-time", type=str, help="Specific roll time for sub-daily intervals (HH:MM format, e.g., '15:00'). Optional.")
    parser.add_argument("--roll-time-zone", type=str, default="America/Chicago", help="Timezone for the --roll-time (e.g., 'America/Chicago', 'UTC'). Default: America/Chicago.")

    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to the DuckDB database file.")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the market symbols configuration file.") # Not used currently
    parser.add_argument("--force-delete", action="store_true", help="If set, deletes existing data for the target continuous symbol PK set before inserting new data.")
    parser.add_argument("--recreate-table-on-pk-mismatch", action="store_true", help="DEV ONLY: If a primary key mismatch is detected on an existing continuous_contracts table, drop and recreate it. May cause data loss if other data exists in the table.")
    parser.add_argument(
        "--adjustment-type", 
        type=str, 
        default="constant", 
        choices=["constant", "C", "none", "N"],
        help="Type of adjustment to apply: 'constant' (or 'C') for constant price adjustment, 'none' (or 'N') for no adjustment (raw stitched series). Default is 'constant'."
    )

    args = parser.parse_args()

    if args.contract_position < 1:
        logger.error("--contract-position must be 1 or greater.")
        sys.exit(1)

    # Normalize adjustment type for internal use
    if args.adjustment_type.upper() == 'C':
        args.adjustment_type = 'constant'
    elif args.adjustment_type.upper() == 'N':
        args.adjustment_type = 'none'
        
    # Validate roll_time format if provided
    if args.roll_time:
        try:
            roll_time_obj = datetime.strptime(args.roll_time, "%H:%M").time()
            logger.info(f"Specified roll time: {roll_time_obj} in timezone {args.roll_time_zone}")
            pytz.timezone(args.roll_time_zone) # Validate timezone
        except ValueError:
            logger.error(f"Invalid --roll-time format '{args.roll_time}'. Please use HH:MM.")
            sys.exit(1)
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Unknown --roll-time-zone '{args.roll_time_zone}'.")
            sys.exit(1)

    logger.info(f"Starting continuous future generation for {args.root_symbol}, RollTypeRule: {args.roll_type}, Position: {args.contract_position}P, Interval: {args.interval_value}-{args.interval_unit}")

    db_con = connect_db(Path(args.db_path), read_only=False)

    roll_time_obj: Optional[time] = None
    if args.roll_time:
        roll_time_obj = args.roll_time_obj

    # 1. Load Roll Dates (these are for the 1st position front-month roll)
    # The `args.roll_type` here refers to the rule code like "01X", "02X", "volume"
    roll_dates_df = load_roll_dates_from_db(args.root_symbol, args.roll_type, db_con)
    if roll_dates_df.empty:
        logger.error(f"No roll dates loaded for {args.root_symbol} with rule '{args.roll_type}'. Aborting.")
        db_con.close()
        sys.exit(1)

    # 2. Get all contracts ordered by expiry from futures_roll_calendar
    contracts_by_expiry_df = get_all_contracts_by_expiry(args.root_symbol, db_con)
    if contracts_by_expiry_df.empty:
        logger.error(f"No contracts found in futures_roll_calendar for {args.root_symbol}. Aborting.")
        db_con.close()
        sys.exit(1)
    
    # 3. Load all necessary individual contract data into memory
    all_contracts_to_load = contracts_by_expiry_df['contract_code'].unique().tolist()
    all_contracts_data: Dict[str, pd.DataFrame] = {}
    logger.info(f"Loading market data for {len(all_contracts_to_load)} potential contracts for {args.root_symbol} {args.interval_value}-{args.interval_unit}...")
    for contract_sym in all_contracts_to_load:
        df = load_market_data_for_contract(contract_sym, args.interval_value, args.interval_unit, db_con)
        if not df.empty:
            # Ensure data is UTC localized after loading
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != pytz.UTC:
                df.index = df.index.tz_convert('UTC')
            all_contracts_data[contract_sym] = df
        else:
            logger.warning(f"No market data loaded for {contract_sym}. It will not be usable.")
    
    if not all_contracts_data:
        logger.error(f"No market data loaded for any contract of {args.root_symbol}. Aborting.")
        db_con.close()
        sys.exit(1)
    logger.info(f"Successfully loaded market data for {len(all_contracts_data)} contracts.")

    # 4. Generate continuous series
    continuous_futures_df = generate_adjusted_series(
        root_symbol=args.root_symbol,
        contracts_by_expiry=contracts_by_expiry_df,
        all_contracts_data=all_contracts_data,
        roll_dates_df=roll_dates_df, # These are the primary front-month roll dates for the specified rule
        adjustment_type=args.adjustment_type,
        interval_value=args.interval_value,
        interval_unit=args.interval_unit,
        contract_position=args.contract_position,
        roll_time_obj=roll_time_obj,
        roll_time_zone_str=args.roll_time_zone if roll_time_obj else None
    )

    if continuous_futures_df.empty:
        logger.error(f"Failed to generate continuous futures for {args.root_symbol} (Pos: {args.contract_position}P). No data produced.")
    else:
        logger.info(f"Generated continuous series with {len(continuous_futures_df)} data points.")

        # Prepare metadata for PK and saving
        output_symbol = generate_output_symbol_name(
            args.root_symbol, 
            args.contract_position,
            args.roll_type, 
            args.output_symbol_suffix,
            args.adjustment_type
        )
        logger.info(f"Output symbol will be: {output_symbol}")

        # Update source identifier to include contract position and adjustment type
        adj_type_suffix_for_source = "_C" if args.adjustment_type == 'constant' else "_N"
        source_id_final = f"{args.source_identifier}_p{args.contract_position}{adj_type_suffix_for_source}"
        logger.info(f"Final source identifier for DB: {source_id_final}")

        # Add columns needed for PK to the DataFrame before deduplication if they aren't already there
        # Timestamp is the index, will be reset in save_continuous_futures_to_db
        # Other PK columns are added in save_continuous_futures_to_db, BUT we need them for deduplication here.
        # Let's ensure they are present for deduplication based on how they are defined for the PK.
        # The df passed to save_continuous_futures_to_db expects index to be timestamp.
        temp_df_for_dedup = continuous_futures_df.copy()
        temp_df_for_dedup.reset_index(inplace=True) # 'timestamp' becomes a column
        temp_df_for_dedup['symbol'] = output_symbol
        temp_df_for_dedup['interval_value'] = args.interval_value
        temp_df_for_dedup['interval_unit'] = args.interval_unit
        temp_df_for_dedup['source'] = source_id_final
        
        # Ensure PK columns are exactly as defined in CONTINUOUS_CONTRACTS_PK_COLUMNS
        # 'timestamp' is from reset_index() above.
        # 'symbol', 'interval_value', 'interval_unit', 'source' were just added.

        pk_cols_for_dedup = list(CONTINUOUS_CONTRACTS_PK_COLUMNS) # Make a copy
        
        initial_row_count = len(temp_df_for_dedup)
        logger.info(f"Before deduplication based on PK {pk_cols_for_dedup}, row count: {initial_row_count}")
        temp_df_for_dedup.drop_duplicates(subset=pk_cols_for_dedup, keep='first', inplace=True)
        deduplicated_row_count = len(temp_df_for_dedup)
        logger.info(f"After deduplication, row count: {deduplicated_row_count}. Rows removed: {initial_row_count - deduplicated_row_count}")

        # Re-set the index to timestamp for save_continuous_futures_to_db
        # and pass only the original columns from continuous_futures_df plus new generated ones like adjustment_factor etc.
        # The save function will re-add symbol, source, interval_value, interval_unit.
        continuous_futures_df_deduped = temp_df_for_dedup.set_index('timestamp')
        # Select original columns plus any new ones added during generation, but not the PK helper columns we just added for dedup
        # This is a bit tricky. The `continuous_futures_df` already has 'adjustment_factor', 'individual_contract_front', etc.
        # We need to make sure we pass a df to save_continuous_futures_to_db that it expects.
        # It expects the index to be timestamp and the necessary data columns.
        # The save function adds: symbol, source, interval_value, interval_unit, is_adjusted, created_at, updated_at
        # The save function takes: df (with ohlcv, adjustment_factor etc.), output_symbol, source_id, interval_val, interval_u, is_adjusted

        # Let's reconstruct the DataFrame to be saved from temp_df_for_dedup,
        # ensuring it has the original index and the necessary columns for the save function.
        # The save function expects a DataFrame that is the result of generation, with timestamp as index.
        # The columns 'symbol', 'interval_value', 'interval_unit', 'source' will be re-added by save_continuous_futures_to_db.
        # So, from temp_df_for_dedup, we need to drop these before setting index if they were not original.
        
        cols_to_keep_from_dedup = [col for col in temp_df_for_dedup.columns if col not in ['symbol', 'interval_value', 'interval_unit', 'source'] or col == 'timestamp']
        final_df_to_save = temp_df_for_dedup[cols_to_keep_from_dedup].set_index('timestamp')

        _ensure_continuous_contracts_table(db_con, args)
        save_continuous_futures_to_db(
            df=final_df_to_save, # Use the deduplicated DataFrame
            output_symbol=output_symbol,
            source_id=source_id_final,
            interval_val=args.interval_value,
            interval_u=args.interval_unit,
            is_adjusted=(args.adjustment_type != "none"),
            force_delete=args.force_delete,
            con=db_con
        )
        logger.info(f"Process completed for {output_symbol}.")

    db_con.close()
    logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 