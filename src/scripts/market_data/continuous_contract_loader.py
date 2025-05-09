import duckdb
import os
import sys
import logging
import argparse
import yaml
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import pandas_market_calendars as mcal
from src.scripts.market_data.fetch_market_data import MarketDataFetcher, get_trading_calendar
import re
from typing import Optional

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_continuous_db():
    """Create the continuous contracts table if it doesn't exist"""
    con = duckdb.connect('data/financial_data.duckdb')
    
    # First drop the existing table
    con.execute("""
    DROP TABLE IF EXISTS continuous_contracts;
    """)
    
    # Create the table with proper constraints
    create_table_sql = """
    CREATE TABLE continuous_contracts (
        timestamp TIMESTAMP,      -- The date of the data point
        symbol VARCHAR,           -- The continuous contract symbol (e.g., @ES=202XC)
        underlying_symbol VARCHAR,-- Specific contract used for this row
        open DOUBLE,             -- Opening price
        high DOUBLE,             -- Highest price
        low DOUBLE,             -- Lowest price
        close DOUBLE,            -- Closing price
        volume BIGINT,           -- Trading volume
        open_interest BIGINT,    -- Open Interest
        up_volume BIGINT,        -- Optional up volume
        down_volume BIGINT,      -- Optional down volume
        source VARCHAR,          -- Data source (origin of underlying data)
        built_by VARCHAR,        -- How the continuous row was built ('local_generator', 'tradestation')
        interval_value INTEGER,  -- Interval length (e.g., 1, 15)
        interval_unit VARCHAR,   -- Interval unit (e.g., 'day', 'minute')
        adjusted BOOLEAN,        -- Whether the price is adjusted
        quality INTEGER,         -- Data quality indicator
        settle DOUBLE,           -- Settlement price
        CONSTRAINT continuous_contracts_pkey PRIMARY KEY (symbol, timestamp, interval_value, interval_unit)
    );
    """
    
    con.execute(create_table_sql)
    
    # Create an index on the primary key columns
    con.execute("""
    CREATE INDEX IF NOT EXISTS continuous_contracts_idx 
    ON continuous_contracts(symbol, timestamp, interval_value, interval_unit);
    """)
    
    con.close()

def load_market_symbols_config():
    """Load market symbols configuration from YAML file."""
    config_path = os.path.join('config', 'market_symbols.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_next_roll_date(symbol_root: str, config: dict, fetcher: MarketDataFetcher) -> pd.Timestamp:
    """
    Calculate the next roll date for the given futures symbol based on its configured cycle 
    and expiry rule from the YAML config.
    Roll date is calculated as 2 trading days prior to the calculated expiration date.
    
    Args:
        symbol_root: The root symbol (ES, NQ, VX, etc.)
        config: Market symbols configuration dictionary
        fetcher: An initialized MarketDataFetcher instance (needed for expiry calc)
        
    Returns:
        pd.Timestamp of the next roll date
    """
    if not fetcher:
         raise ValueError("MarketDataFetcher instance is required to calculate expiration dates.")
         
    # Get the futures configuration for this symbol
    futures_config = next(
        (f for f in config.get('futures', []) if f['base_symbol'] == symbol_root),
        None
    )
    if not futures_config:
        # Raise specific error for missing symbol config
        raise ValueError(f"No 'futures' configuration found for '{symbol_root}' in market_symbols.yaml. Please add or verify the entry.")
        
    # --- Determine Next Contract Month/Year --- 
    # Get the contract months patterns specific to this symbol from config
    if 'historical_contracts' not in futures_config or 'patterns' not in futures_config['historical_contracts']:
         # Raise specific error for missing patterns
         raise ValueError(f"No 'historical_contracts.patterns' defined for futures symbol '{symbol_root}' in market_symbols.yaml. Please add the contract month codes (e.g., [H, M, U, Z] or [F, G, ...]).")
    contract_months_codes = futures_config['historical_contracts']['patterns']
    
    full_month_map = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    reverse_month_map = {v: k for k, v in full_month_map.items()}
    
    contract_month_numbers = sorted([full_month_map[code] for code in contract_months_codes if code in full_month_map])
    if not contract_month_numbers:
         # Raise specific error for invalid/missing month codes in patterns
         raise ValueError(f"No valid month codes found in patterns for {symbol_root}: {contract_months_codes}. Check market_symbols.yaml.")

    today = pd.Timestamp.now().normalize()
    current_month = today.month
    current_year = today.year
    
    next_contract_month = None
    next_contract_year = current_year
    
    found_next = False
    temp_idx = 0 # Loop counter for safety break
    # Need mutable state for year and current_month_start within the loop
    check_year = current_year 
    check_month = current_month

    # Find the first contract month >= current month whose roll hasn't passed
    while not found_next:
         # Determine the next month in the symbol's cycle to check
         current_cycle_month_num = None
         for month_num in contract_month_numbers:
             if month_num >= check_month:
                  current_cycle_month_num = month_num
                  break
         # If no month found this year in the cycle, wrap to next year's first cycle month
         if current_cycle_month_num is None:
              current_cycle_month_num = contract_month_numbers[0]
              check_year += 1
              check_month = 1 # Start checking from Jan next year
              
         # Construct the specific contract symbol for expiry calculation
         next_contract_month_code = reverse_month_map.get(current_cycle_month_num)
         if not next_contract_month_code:
              logger.error(f"Logic error: Could not find month code for month number {current_cycle_month_num}")
              # This should not happen if contract_month_numbers is valid
              raise RuntimeError("Internal logic error in roll date calculation.")
              
         next_contract_symbol = f"{symbol_root}{next_contract_month_code}{str(check_year)[-2:]}"
         logger.debug(f"Checking potential next contract: {next_contract_symbol}")
         
         # Calculate expiry for this potential next contract using fetcher's method
         expiry_date = fetcher.calculate_expiration_date(next_contract_symbol)
         
         if expiry_date:
             # Calculate roll date (2 trading days before expiry)
             calendar_name = futures_config.get('calendar', 'NYSE') # Get calendar name from config
             calendar = get_trading_calendar(calendar_name) # Use fetcher's helper
             schedule_around_expiry = calendar.schedule(start_date=expiry_date - pd.Timedelta(days=10), end_date=expiry_date)
             
             if not schedule_around_expiry.empty:
                 # Ensure expiry_date is naive for comparison with naive index
                 naive_expiry_date = expiry_date.tz_localize(None) if expiry_date.tz is not None else expiry_date
                 trading_days_before_expiry = schedule_around_expiry.index[schedule_around_expiry.index < naive_expiry_date] # Strictly before expiry
                 if len(trading_days_before_expiry) >= 2:
                     potential_roll_date = trading_days_before_expiry[-2] # 2nd day back is 2 trading days before expiry
                     
                     # Check if today is strictly before this potential roll date
                     if today < potential_roll_date:
                          # Found the next contract whose roll date hasn't passed
                          next_contract_month = current_cycle_month_num
                          next_contract_year = check_year
                          found_next = True
                          break # Exit while loop
                     else:
                          logger.debug(f"Roll date {potential_roll_date.date()} for {next_contract_symbol} has passed today ({today.date()}). Checking next contract month.")
                 else:
                     # Handle case where expiry is very soon, fewer than 2 trading days before it
                     logger.warning(f"Could not find 2 trading days before expiry {expiry_date.date()} for {next_contract_symbol}. Assuming roll has passed or is today. Checking next contract month.")
             else:
                  # Handle case where calendar schedule might be empty (unlikely for near dates)
                  logger.warning(f"Could not find trading days around expiry {expiry_date.date()} for {next_contract_symbol}. Checking next contract month.")
         else:
             # Expiry calculation failed for this potential symbol
             logger.warning(f"Could not calculate expiry date for potential next contract {next_contract_symbol}. Checking next contract month.")

         # --- Advance to the next month in the cycle for the next iteration --- 
         current_month_index_in_cycle = contract_month_numbers.index(current_cycle_month_num)
         next_month_index_in_cycle = (current_month_index_in_cycle + 1) % len(contract_month_numbers)
         check_month = contract_month_numbers[next_month_index_in_cycle]
         # If we wrapped around the cycle (e.g., from Z back to H), increment the check_year
         if next_month_index_in_cycle == 0:
              check_year += 1 
              
         # Safety break to prevent infinite loops in case of unexpected logic error or config issue
         temp_idx += 1
         if temp_idx > len(contract_month_numbers) * 2: # Allow checking up to two full cycles
             logger.error(f"Potential infinite loop detected while determining next roll date for {symbol_root}. Check config and logic.")
             raise RuntimeError(f"Could not determine next roll date for {symbol_root} after checking multiple cycles.")
             
    # --- End While Loop --- 
    
    if not found_next:
         # This should ideally not be reached if the loop logic is correct
         logger.error(f"Failed to find the next contract roll date for {symbol_root}. Loop completed without success.")
         raise ValueError(f"Unable to determine next roll date for {symbol_root}")

    # --- Now we have the correct next_contract_month/year, calculate its final expiry and roll date --- 
    final_next_month_code = reverse_month_map.get(next_contract_month)
    final_next_symbol = f"{symbol_root}{final_next_month_code}{str(next_contract_year)[-2:]}"
    
    logger.info(f"Determined next contract for roll calculation: {final_next_symbol}")
    
    final_expiry_date = fetcher.calculate_expiration_date(final_next_symbol)
    if not final_expiry_date:
         # This is more serious, calculation failed for the identified next contract
         raise ValueError(f"Failed to calculate final expiration date for identified next contract: {final_next_symbol}")

    # Calculate final roll date (2 trading days prior)
    calendar_name = futures_config.get('calendar', 'NYSE')
    calendar = get_trading_calendar(calendar_name)
    schedule = calendar.schedule(start_date=final_expiry_date - pd.Timedelta(days=10), end_date=final_expiry_date)
    
    # Check if schedule is valid and contains enough days before expiry
    if schedule.empty:
        logger.warning(f"Could not find trading days around expiry {final_expiry_date.date()} for {final_next_symbol}. Returning expiry date minus 2 calendar days as fallback.")
        roll_date = final_expiry_date - pd.Timedelta(days=2) # Fallback
    else:
        # Ensure final_expiry_date is naive for comparison
        naive_final_expiry_date = final_expiry_date.tz_localize(None) if final_expiry_date.tz is not None else final_expiry_date
        trading_days_before_expiry = schedule.index[schedule.index < naive_final_expiry_date] # <-- NEW
        if len(trading_days_before_expiry) < 2:
            logger.warning(f"Could not find 2 trading days before expiry {final_expiry_date.date()} for {final_next_symbol}. Returning expiry date minus 2 calendar days as fallback.")
            roll_date = final_expiry_date - pd.Timedelta(days=2) # Fallback
        else:
             roll_date = trading_days_before_expiry[-2] # 2nd day back = 2 trading days before
    
    logger.info(f"Calculated expiry for {final_next_symbol}: {final_expiry_date.date()}, Roll Date: {roll_date.date()}")
    return roll_date.normalize() # Normalize to remove time component

def is_near_roll(symbol_root: str, config: dict, fetcher: MarketDataFetcher, proximity_days: int = 7) -> bool:
    """
    Check if the current date is near the roll date for the given futures symbol.
    "Near roll" is defined as being within `proximity_days` trading days OF or AFTER 
    the calculated roll date (which is typically 2 days before expiry).
    Effectively, this means we are past the ideal roll point or very close to it.

    Args:
        symbol_root: The root symbol (ES, NQ, VX, etc.)
        config: Market symbols configuration dictionary
        fetcher: An initialized MarketDataFetcher instance
        proximity_days: Number of calendar days to define the "near roll" window. 
                        If today is within roll_date - proximity_days and roll_date, it's near.
                        Actually, it's more like if roll_date is within today to today + proximity_days.
                        Let's redefine: True if today is within `proximity_days` of the next roll date,
                        OR if today is past the next roll date.

    Returns:
        True if near roll date, False otherwise.
    """
    try:
        next_roll_timestamp = get_next_roll_date(symbol_root, config, fetcher)
        if not next_roll_timestamp:
            logger.warning(f"Could not determine next roll date for {symbol_root}. Assuming not near roll.")
            return False

        # Ensure next_roll_timestamp is a timezone-naive pd.Timestamp for comparison
        if isinstance(next_roll_timestamp, datetime):
            next_roll_timestamp = pd.Timestamp(next_roll_timestamp)
        
        if next_roll_timestamp.tzinfo is not None:
            next_roll_timestamp = next_roll_timestamp.tz_localize(None)

        today = pd.Timestamp.now().normalize() # Today, also timezone-naive

        # Check if today is on or after the roll date
        if today >= next_roll_timestamp:
            logger.info(f"Today ({today.date()}) is ON or AFTER the roll date ({next_roll_timestamp.date()}) for {symbol_root}. Considered near roll.")
            return True
        
        # Check if the roll date is within the proximity window from today
        # e.g. proximity_days = 7. If roll is 5 days from now, today + 7 days > roll_date. TRUE.
        if next_roll_timestamp <= today + pd.Timedelta(days=proximity_days):
            logger.info(f"Roll date ({next_roll_timestamp.date()}) for {symbol_root} is within {proximity_days} days from today ({today.date()}). Considered near roll.")
            return True
            
        logger.info(f"Today ({today.date()}) is not near roll date ({next_roll_timestamp.date()}) for {symbol_root} (proximity: {proximity_days} days).")
        return False
        
    except ValueError as e:
        logger.error(f"Error calculating roll date for {symbol_root} in is_near_roll: {e}. Assuming not near roll.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in is_near_roll for {symbol_root}: {e}", exc_info=True)
        return False # Default to False on any other unexpected error

def get_start_date(symbol_root: str, config: dict, force: bool = False, interval_unit: str = 'daily') -> str:
    """
    Determine the start date for data fetching based on force flag, symbol type, and interval.
    
    Args:
        symbol_root: The root symbol (e.g., 'ES' or 'NQ')
        config: Market symbols configuration
        force: Whether to force update the entire series
        interval_unit: The interval unit (e.g., 'daily' or 'minute')
    
    Returns:
        start_date: The date to start fetching data from
    """
    # Get the futures configuration for this symbol
    futures_config = next(
        (f for f in config.get('futures', []) if f['base_symbol'] == symbol_root),
        None
    )
    if not futures_config:
        # Use a default start date if config not found, maybe log warning
        logger.warning(f"No futures config found for {symbol_root} in get_start_date. Using default lookbacks.")
        default_start_daily = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        default_start_intraday = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        return default_start_daily if interval_unit == 'daily' else default_start_intraday

    if force:
        # Use the start_date from config for daily data
        config_start = futures_config.get('start_date')
        if not config_start:
             logger.warning(f"'start_date' not found in config for {symbol_root} during force update. Using default.")
             config_start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d') # Default 5 years
             
        # For intraday data, limit to 30 days even when forced
        if interval_unit == 'daily':
            return config_start
        else:
            return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # For normal updates, use different lookback periods based on interval
    if interval_unit == 'daily':
        return (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        # For intraday data, use a shorter lookback to avoid API limitations
        return (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

def load_continuous_data(fetcher: MarketDataFetcher, symbol_root: str, config: dict, force: bool = False, interval_value: int = 1, interval_unit: str = 'daily', specific_symbol: Optional[str] = None, fetch_mode: str = 'auto', lookback_days: int = 90, roll_proximity_threshold_days: int = 7):
    """
    Load continuous contract data for a given symbol root using the existing MarketDataFetcher.
    Handles different fetch modes and roll proximity for adjusted contracts.

    Args:
        fetcher: Initialized MarketDataFetcher instance.
        symbol_root: The base future root (e.g., ES, NQ, VX).
        config: The market symbols configuration dictionary.
        force: If True, overrides fetch_mode to 'full'.
        interval_value: Interval value (e.g., 1, 15).
        interval_unit: The interval unit (e.g., 'daily' or 'minute').
        specific_symbol: The exact continuous symbol to process (e.g., @ES, @VX=101XN).
        fetch_mode: The mode of fetching data ('auto', 'latest', 'full').
        lookback_days: Days to look back for 'latest' mode or 'auto' (not near roll).
        roll_proximity_threshold_days: Days before expiry to consider 'near roll' in 'auto' mode.
    """
    logger.info(f"Starting load_continuous_data for: {specific_symbol or symbol_root}, Interval: {interval_value}{interval_unit}, Mode: {fetch_mode}, Force: {force}")

    if not specific_symbol:
        specific_symbol = f"@{symbol_root}"
        logger.warning(f"specific_symbol was None, defaulting to {specific_symbol}. This might indicate an issue in the calling logic.")
        logger.error("Cannot load continuous data without a specific symbol.")
        return

    logger.info(f"Requesting processing for: {specific_symbol}, Interval: {interval_value}{interval_unit}, Force Fetch: {force}")

    # --- Determine Fetch Start Date --- 
    fetch_start_date_str = None
    config_start_date_str = None
    try:
        # Get config start date for the base symbol (ES, NQ, VX)
        base_config = fetcher._get_symbol_config(symbol_root) # Use the root symbol
        if base_config:
            config_start_date_str = base_config.get('start_date')
        
        if force:
            fetch_start_date_str = config_start_date_str or '1970-01-01' # Use config date or very old date if forced
            logger.info(f"Force fetch requested. Fetching from: {fetch_start_date_str}")
        else:
            # Query continuous_contracts for the latest timestamp
            latest_ts = None
            try:
                query_latest = "SELECT MAX(timestamp) FROM continuous_contracts WHERE symbol=? AND interval_unit=? AND interval_value=?"
                result_latest = fetcher.conn.execute(query_latest, [specific_symbol, interval_unit, interval_value]).fetchone()
                if result_latest and result_latest[0]:
                    latest_ts = pd.Timestamp(result_latest[0])
            except Exception as db_e:
                logger.warning(f"Could not query latest timestamp for {specific_symbol} from continuous_contracts: {db_e}. Will fetch from config start date.")
            
            if latest_ts:
                 # Fetch from the day after the latest timestamp found
                 fetch_start_date = latest_ts + pd.Timedelta(seconds=1) # Add a second to avoid refetching last bar
                 fetch_start_date_str = fetch_start_date.strftime('%Y-%m-%d')
                 logger.info(f"Found existing data up to {latest_ts}. Fetching new data from {fetch_start_date_str}")
            else:
                 fetch_start_date_str = config_start_date_str or '1970-01-01'
                 logger.info(f"No existing data found for {specific_symbol} in continuous_contracts. Fetching from: {fetch_start_date_str}")
                 
    except Exception as e:
        logger.error(f"Error determining start date for {specific_symbol}: {e}. Aborting fetch.")
        return

    # --- Fetch Data using fetch_data_since --- 
    try:
        logger.info(f"Fetching {interval_value}{interval_unit} data for {specific_symbol} from {fetch_start_date_str}")
        df = fetcher.fetch_data_since(
            symbol=specific_symbol, 
            interval=interval_value, 
            unit=interval_unit, 
            start_date=fetch_start_date_str
        )

        if df is None or df.empty:
            logger.warning(f"No data returned by fetch_data_since for {specific_symbol} from {fetch_start_date_str}. Nothing to save.")
            return # Exit function if no data
            
        logger.info(f"fetch_data_since returned {len(df)} rows for {specific_symbol}.")

        # --- Process Dataframe for continuous_contracts table --- 
        is_adjusted = False # Determine adjustment based on symbol name convention
        if specific_symbol and "=" in specific_symbol and "C" in specific_symbol.split('=')[-1]:
             is_adjusted = True
        elif specific_symbol and specific_symbol.startswith(('@ES', '@NQ')) and '=' not in specific_symbol:
             # Assume generic @ES/@NQ implies adjusted if no specific settings code provided?
             # This might need refinement based on actual usage/config intent.
             is_adjusted = True 
             
        df['symbol'] = specific_symbol
        # fetch_data_since doesn't provide underlying symbol
        df['underlying_symbol'] = None 
        df['source'] = 'tradestation' # Source is TS, even though we build it
        # interval_value, interval_unit are already correct from input
        df['adjusted'] = is_adjusted
        df['quality'] = 100  # Assume good quality 
        df['built_by'] = 'continuous_contract_loader' # This script processed it
        
        # Ensure standard OHLCV columns exist, handle potential missing ones if needed
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing from fetch_data_since output. Setting to NaN/0.")
                df[col] = pd.NA if col != 'volume' else 0
                
        # Add other potentially missing columns expected by the database table
        # Use .get() for safety, defaulting to None or another value
        df['open_interest'] = df.get('open_interest', None)
        df['up_volume'] = df.get('up_volume', None)
        df['down_volume'] = df.get('down_volume', None)
        df['settle'] = df.get('settle', df['close']) # Use settle if returned, else default to close

        # Select only columns that exist in the target table to avoid errors
        target_columns = []
        try:
            table_info = fetcher.conn.execute(f"PRAGMA table_info('continuous_contracts')").fetchall()
            target_columns = [col[1] for col in table_info]
        except Exception as pragma_e:
            logger.error(f"Could not get columns for continuous_contracts table: {pragma_e}. Save might fail.")
            # Define expected columns as fallback
            target_columns = ['timestamp', 'symbol', 'underlying_symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'up_volume', 'down_volume', 'source', 'built_by', 'interval_value', 'interval_unit', 'adjusted', 'quality', 'settle']

        df_to_save = df[[col for col in target_columns if col in df.columns]].copy()
        
        # --- Delete existing data if force=True --- 
        if force:
            try:
                logger.info(f"Force mode: Deleting existing data for {specific_symbol} ({interval_unit}/{interval_value}) from continuous_contracts table.")
                delete_sql = "DELETE FROM continuous_contracts WHERE symbol=? AND interval_unit=? AND interval_value=?"
                delete_params = [specific_symbol, interval_unit, interval_value]
                cur = fetcher.conn.execute(delete_sql, delete_params)
                logger.info(f"Deleted {cur.fetchone()[0]} rows.")
                # No commit needed usually for DELETE in DuckDB unless in explicit transaction
            except Exception as del_e:
                 logger.error(f"Error deleting existing data for {specific_symbol}: {del_e}. Proceeding with upsert.")
        
        # --- Save to continuous_contracts table --- 
        if not hasattr(fetcher, 'conn') or not fetcher.conn:
             logger.error(f"Fetcher database connection not available. Cannot save data for {specific_symbol}.")
             return # Skip saving if connection invalid
             
        try:
            # Get row count before upsert (for logging)
            rows_before_query = "SELECT COUNT(*) FROM continuous_contracts WHERE symbol=? AND interval_unit=? AND interval_value=?"
            rows_before_result = fetcher.conn.execute(rows_before_query, [specific_symbol, interval_unit, interval_value]).fetchone()
            rows_before = rows_before_result[0] if rows_before_result else 0
        except Exception as count_e:
            logger.error(f"Error getting row count before save for {specific_symbol}: {count_e}")
            rows_before = -1 # Indicate error
        
        # Use DuckDB's efficient UPSERT capability via temp view registration
        safe_symbol_name = re.sub(r'[^a-zA-Z0-9_]', '', specific_symbol)
        temp_view_name = f"temp_cont_{safe_symbol_name}_{interval_value}{interval_unit}_view"
        try:
             fetcher.conn.register(temp_view_name, df_to_save) # Register the filtered dataframe
             
             # Define columns to insert/update dynamically based on df_to_save
             insert_columns = list(df_to_save.columns) 
             insert_columns_str = ", ".join([f'\"{col}\"' for col in insert_columns]) # Quote column names
             pk_columns = ['symbol', 'timestamp', 'interval_value', 'interval_unit']
             update_columns = [col for col in insert_columns if col not in pk_columns]
             update_setters_str = ", ".join([f'\"{col}\" = EXCLUDED.\"{col}\"' for col in update_columns])

             if not update_setters_str:
                  # Handle case where only PK columns are present
                  logger.warning(f"No columns to update for {specific_symbol}. Only inserting.")
                  upsert_sql = f"""
                  INSERT INTO continuous_contracts ({insert_columns_str})
                  SELECT {insert_columns_str}
                  FROM {temp_view_name}
                  ON CONFLICT DO NOTHING
                  """
             else:
                  upsert_sql = f"""
                  INSERT INTO continuous_contracts ({insert_columns_str})
                  SELECT {insert_columns_str}
                  FROM {temp_view_name}
                  ON CONFLICT (symbol, timestamp, interval_value, interval_unit) DO UPDATE SET 
                      {update_setters_str}
                  """
             
             logger.debug(f"Executing UPSERT SQL for {specific_symbol}:")#\n{upsert_sql}") 
             fetcher.conn.execute(upsert_sql)
             # fetcher.conn.commit() # Let context manager handle commit
             
             try:
                 rows_after_query = "SELECT COUNT(*) FROM continuous_contracts WHERE symbol=? AND interval_unit=? AND interval_value=?"
                 rows_after_result = fetcher.conn.execute(rows_after_query, [specific_symbol, interval_unit, interval_value]).fetchone()
                 rows_after = rows_after_result[0] if rows_after_result else 0
                 rows_added = rows_after - rows_before if rows_before != -1 else -1
             except Exception as count_e:
                 logger.error(f"Error getting row count after save for {specific_symbol}: {count_e}")
                 rows_after = -1
                 rows_added = -1
                 
             logger.info(f"Successfully saved continuous contract data for {specific_symbol} to continuous_contracts table")
             if rows_added != -1:
                  logger.info(f"Upserted {len(df_to_save)} rows (approx {rows_added} new) to the database")
             else:
                  logger.info(f"Upserted {len(df_to_save)} rows to the database (count change unavailable)")
                  
        except Exception as save_e:
             logger.error(f"Error saving data to database for {specific_symbol}: {save_e}", exc_info=True)
             try: fetcher.conn.rollback() 
             except Exception as rb_e: logger.error(f"Rollback attempt failed: {rb_e}")
             return # Stop processing this symbol on save error
        finally:
             try:
                  fetcher.conn.unregister(temp_view_name) # Clean up the view
             except Exception as unreg_e:
                  logger.warning(f"Could not unregister temp view {temp_view_name}: {unreg_e}")
                  
    except Exception as e:
        logger.error(f"Error processing {specific_symbol}: {str(e)}", exc_info=True)
        return

def main():
    DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")
    DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")

    parser = argparse.ArgumentParser(description="Load and build continuous futures contracts from TradeStation data.")
    parser.add_argument(
        "symbol",
        help="The continuous contract symbol to process (e.g., @ES, @VX=101XN, ES for generic @ES)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch of all historical data for underlying contracts."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to the DuckDB database file (default: {DEFAULT_DB_PATH})."
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the market symbols YAML configuration file (default: {DEFAULT_CONFIG_PATH})."
    )
    # Add interval arguments, defaulting to daily
    parser.add_argument("--interval-value", type=int, default=1, help="Interval value (default: 1).")
    parser.add_argument("--interval-unit", type=str, default='daily', choices=['daily', 'minute', 'hour'], help="Interval unit (default: 'daily').")

    # New arguments for fetch control
    parser.add_argument(
        "--fetch-mode",
        type=str,
        default='auto',
        choices=['auto', 'latest', 'full'],
        help="Fetch mode: 'auto' (default, intelligent roll handling), 'latest' (fetch recent N days), 'full' (fetch all history)."
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Number of days to look back when fetch_mode is 'latest' or 'auto' (and not near roll for adjusted). Default: 90."
    )
    parser.add_argument(
        "--roll-proximity-threshold-days",
        type=int,
        default=7,
        help="Days before expiry to trigger full series rebuild for adjusted contracts in 'auto' mode (default: 7)"
    )

    args = parser.parse_args()

    logger.info(f"Continuous Contract Loader started for: {args.symbol}")
    logger.info(f"Using DB Path: {args.db_path}, Config Path: {args.config_path}") # DEBUG DB PATH

    # Setup database table
    # setup_continuous_db() # Removed setup call - should be handled by main orchestrator
    
    # Load market symbols configuration
    try:
        config = load_market_symbols_config()
    except Exception as e:
        logger.error(f"Failed to load market symbols configuration: {e}")
        sys.exit(1)
    
    # Initialize the fetcher and authenticate with TradeStation
    config_file_path = args.config_path # Define path used by load_config
    
    fetcher = None
    try:
        # Pass the config_path, not the loaded config dictionary
        fetcher = MarketDataFetcher(
            config_path=config_file_path, 
            db_path=args.db_path # This db_path is from args
        )
        logger.info(f"DEBUG_LOADER: MarketDataFetcher initialized with db_path: {fetcher.db_path}") # Log the path fetcher is using
        # Authenticate the fetcher's agent
        if not fetcher.ts_agent or not fetcher.ts_agent.authenticate():
            logger.error("Failed to initialize or authenticate TradeStation fetcher")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize or authenticate TradeStation fetcher: {e}", exc_info=True)
        sys.exit(1) 

    # Determine the actual symbol to process and the root for config lookup
    user_input_symbol = args.symbol
    actual_symbol_to_process = ""
    root_symbol_for_config = ""

    if '=' in user_input_symbol: # Parameterized like @VX=101XN
        actual_symbol_to_process = user_input_symbol
        match = re.match(r"^(@[A-Z]{1,3})", user_input_symbol)
        if match:
            root_symbol_for_config = match.group(1).lstrip('@')
        else:
            logger.error(f"Invalid parameterized continuous symbol format: {user_input_symbol}")
            sys.exit(1)
    elif user_input_symbol.startswith('@'): # Generic like @ES
        actual_symbol_to_process = user_input_symbol
        root_symbol_for_config = user_input_symbol.lstrip('@')
    else: # Base like ES, implies @ES
        actual_symbol_to_process = "@" + user_input_symbol
        root_symbol_for_config = user_input_symbol

    logger.info(f"Processing continuous contract: {actual_symbol_to_process} (config root: {root_symbol_for_config})")

    # Metadata Check (using actual_symbol_to_process for query)
    metadata_source = ''
    metadata_found = False
    try:
        logger.info(f"Querying metadata for continuous contract group based on: '{actual_symbol_to_process}' (derived root: '{root_symbol_for_config}') using existing fetcher connection.")
        
        # Attempt to find a specific entry first (e.g. for @VX=101XN where base_symbol IS @VX=101XN)
        query_specific = """ 
            SELECT data_source, base_symbol, data_table, asset_type
            FROM symbol_metadata 
            WHERE base_symbol = ? AND asset_type = 'continuous_future' AND data_table = ?
            LIMIT 1
        """
        params_specific = [actual_symbol_to_process, actual_symbol_to_process]
        logger.info(f"DEBUG_LOADER: specific_symbol_arg for metadata query_specific: '{actual_symbol_to_process}'") # DEBUG LINE 1
        metadata_rows = fetcher.conn.execute(query_specific, params_specific).fetchall()
        logger.info(f"DEBUG_LOADER: query_specific returned {len(metadata_rows)} rows.") # DEBUG LINE 2
        if metadata_rows:
            logger.info(f"DEBUG_LOADER: metadata_rows[0] from query_specific: {metadata_rows[0]}") # DEBUG LINE 3

        if metadata_rows:
            # We expect one row per interval. For now, just take the first to get common properties.
            metadata_record = metadata_rows[0]
            metadata_found = True # Set flag to True
            metadata_source = metadata_record['data_source'] # Set variable
            logger.info(f"Found base metadata for {actual_symbol_to_process} (Source: {metadata_source}): {metadata_record}")
            logger.info(f"Will attempt to process for requested interval: {args.interval_unit}/{args.interval_value}.")

            if metadata_source.lower() != 'tradestation':
                logger.error(f"Symbol {actual_symbol_to_process} (or its group {metadata_record['base_symbol']}) is configured with data_source='{metadata_source}' in symbol_metadata, not 'tradestation'. Cannot proceed with this loader.")
                sys.exit(1)
            # No need to check base_symbol_from_meta vs root_symbol_for_config here as the query logic handles it.
        else:
            # This 'else' means neither the specific nor the generic query yielded results
            logger.error(f"No 'continuous_future' metadata record found for symbol '{actual_symbol_to_process}' or its generic group ('{f'@{root_symbol_for_config}'}') in symbol_metadata. Cannot determine data_source.")
            sys.exit(1) 
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR querying metadata for {actual_symbol_to_process}: {e}", exc_info=True)
        sys.exit(1) # Exit if metadata query fails critically

    logger.info(f"Metadata source verified. Proceeding to load data for {actual_symbol_to_process}...")

    load_continuous_data(
        fetcher=fetcher,
        symbol_root=root_symbol_for_config, 
        config=config,
        force=args.force,
        interval_value=args.interval_value,
        interval_unit=args.interval_unit,
        specific_symbol=actual_symbol_to_process,
        fetch_mode=args.fetch_mode,
        lookback_days=args.lookback_days,
        roll_proximity_threshold_days=args.roll_proximity_threshold_days
    )

    # Close fetcher connection if opened
    if fetcher and hasattr(fetcher, 'conn') and fetcher.conn: # Check if fetcher manages its own conn
         try:
             fetcher.conn.close()
             logger.info("Fetcher database connection closed.")
         except Exception as e:
             logger.warning(f"Error closing fetcher connection: {e}")

    logger.info(f"Continuous contract loading finished for {args.symbol}.")

if __name__ == '__main__':
    main() 