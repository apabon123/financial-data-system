import sys
import os
from pathlib import Path

# Add project root to the path for imports
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(project_root)

import yaml
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from exchange_calendars import get_calendar # Import calendar library
import pandas_market_calendars as mcal # Add pandas_market_calendars for fallback
from typing import Dict, List, Optional, Tuple # Import for type hints

# Add import for continuous contract mapping utilities
from src.utils.continuous_contracts import get_active_contract, get_all_active_contracts

# REMOVED: Basic logging setup here
logger = logging.getLogger(__name__) # Get logger instance

DEFAULT_END_DATE = datetime.today().strftime('%Y-%m-%d')
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"
DEFAULT_DB_PATH = "data/financial_data.duckdb" # Updated default

# Configuration
DEFAULT_START_DATE = '2004-01-01' # Example default, adjust as needed

# --- Logging Setup --- #
def setup_logging(log_file_path):
    """Configures logging to output to both console and a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger() # Get root logger
    root_logger.setLevel(logging.INFO) # Set level on root logger

    # Remove existing handlers if any (e.g., from basicConfig)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging to {log_file_path}: {e}", file=sys.stderr)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout) # Log INFO and above to stdout
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging configured. Outputting to console and file: {log_file_path}")

# --- Configuration Loading ---
def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config {config_path}: {e}")
        return None

# --- Database Operations ---
def connect_db(db_path, read_only=False):
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to database: {db_path} (Read-Only: {read_only})")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def load_market_data(conn, symbol_config, root_symbol, config, specific_symbol: Optional[str] = None):
    """Loads relevant market data based on source, optionally filtering for a specific symbol."""
    
    # Determine the correct table and source based on interval (assuming daily for this script)
    interval_unit = 'daily' # Continuous generation is based on daily data
    interval_value = 1
    
    # Find the daily frequency config override, if it exists
    daily_freq_config = None
    frequencies = symbol_config.get('frequencies', [])
    if isinstance(frequencies, list):
        for freq in frequencies:
            if isinstance(freq, dict) and freq.get('name') == interval_unit:
                daily_freq_config = freq
                break
                
    # Get table and source, prioritizing the daily override
    if daily_freq_config:
        table_name = daily_freq_config.get('raw_table', symbol_config.get('default_raw_table'))
        source = daily_freq_config.get('source', symbol_config.get('default_source', '')).lower()
    else:
        # Fallback to defaults if no specific daily config
        table_name = symbol_config.get('default_raw_table')
        source = symbol_config.get('default_source', '').lower()

    # Original interval unit/value logic (might not be needed if we assume daily)
    # interval_unit = symbol_config.get('interval_unit', 'day')
    # interval_value = symbol_config.get('interval_value', 1)

    if not table_name:
        logger.error(f"Raw table name not configured for {root_symbol} (daily)")
        return pd.DataFrame()

    # Define base columns required
    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest', 'source']
    select_cols_str = ", ".join([f'"{c}"' for c in required_cols])

    # Build WHERE clause and parameters
    where_clauses = ["interval_unit = ?", "interval_value = ?"]
    params = [interval_unit, interval_value]

    if specific_symbol:
        where_clauses.append("symbol = ?")
        params.append(specific_symbol)
        log_target = specific_symbol
    else:
        where_clauses.append("symbol LIKE ?")
        params.append(f"{root_symbol}%")
        log_target = f"pattern {root_symbol}%"
        
    where_clause_str = " AND ".join(where_clauses)

    query = f"""
        SELECT {select_cols_str}
        FROM {table_name}
        WHERE {where_clause_str}
        ORDER BY timestamp ASC
    """
    
    logger.info(f"Loading data for {log_target} from {table_name} with columns: {required_cols}")

    try:
        df = conn.execute(query, params).fetchdf()
        logger.info(f"Loaded {len(df)} rows from {table_name} for {log_target}")
        
        if df.empty:
            logger.warning(f"No data found in {table_name} for {log_target}")
            return pd.DataFrame()

        # Ensure correct data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 logger.warning(f"Column '{col}' missing in loaded data for {log_target}. Will be filled with NaN.")
                 df[col] = pd.NA 
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure source is string if present
        if 'source' in df.columns:
            df['source'] = df['source'].astype(str)
        else:
            logger.warning(f"Column 'source' missing in loaded data for {log_target}.") # Should not happen if selected
            df['source'] = None # Add as None if somehow missing
        
        df.dropna(subset=['timestamp'], inplace=True)
        
        return df

    except duckdb.CatalogException:
        logger.error(f"Error loading data: Table '{table_name}' not found.")
        return pd.DataFrame()
    except duckdb.BinderException as e:
         logger.error(f"Binder Error loading data from {table_name} for {log_target}. Check columns. Query: {query}, Params: {params}, Error: {e}")
         # Attempt to list columns if BinderError occurs
         try:
             cols_in_table = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
             logger.error(f"Columns found in {table_name}: {cols_in_table}")
         except Exception as ie:
             logger.error(f"Could not retrieve columns for {table_name}: {ie}")
         return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading market data for {log_target} from {table_name}: {e}", exc_info=True)
        return pd.DataFrame()

def load_roll_calendar(conn, root_symbol):
    """Loads the roll calendar for a root symbol as a DataFrame."""
    query = f"""
        SELECT contract_code, last_trading_day
        FROM futures_roll_calendar
        WHERE root_symbol = ?
        ORDER BY last_trading_day ASC
    """
    try:
        df_calendar = conn.execute(query, [root_symbol]).fetchdf()
        if df_calendar.empty:
            logger.warning(f"No roll calendar entries found for {root_symbol}.")
            return pd.DataFrame() # Return empty DataFrame
        df_calendar['last_trading_day'] = pd.to_datetime(df_calendar['last_trading_day'])
        # Add sequence column based on order of last_trading_day
        df_calendar['sequence'] = range(len(df_calendar))
        logger.info(f"Loaded {len(df_calendar)} roll calendar entries for {root_symbol} into DataFrame.")
        # Return the DataFrame directly
        return df_calendar
    except duckdb.Error as e:
        logger.error(f"Error loading roll calendar for {root_symbol}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        logger.error(f"Unexpected error loading roll calendar for {root_symbol}: {e}")
        return pd.DataFrame()

def delete_continuous_data(conn, continuous_symbol):
    """Deletes existing data for a continuous contract symbol."""
    try:
        # Target the correct table
        table_name = 'continuous_contracts' # Changed from market_data
        query = f"DELETE FROM {table_name} WHERE symbol = ?"
        conn.execute(query, [continuous_symbol])
        logger.info(f"Executed delete for existing entries of {continuous_symbol} from {table_name}.")
    except Exception as e:
        logger.error(f"Error deleting data for {continuous_symbol} from {table_name}: {e}")

# --- Contract Logic ---
def generate_symbol_map(root_symbol, year, patterns):
    """Generates a list of contract symbols for a given year."""
    return [f"{root_symbol}{p}{str(year)[-2:]}" for p in patterns]

def _get_expiry_date(symbol):
    """DEPRECATED/Placeholder: Extracts year/month - expiry logic now uses roll calendar."""
    # This function's logic is no longer the primary driver for rolls.
    # It might still be useful for basic validation or ordering if needed.
    # We keep it simple for now.
    # DEPRECATED - Not reliable without calendar
    # if len(symbol) < 4:
    #     raise ValueError(f"Invalid symbol format: {symbol}")
    # month_code = symbol[-3]
    # year_short = symbol[-2:]
    # month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    #              'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    # if month_code not in month_map:
    #     raise ValueError(f"Invalid month code in symbol: {symbol}")
    # month = month_map[month_code]
    # year = int(f"20{year_short}") # Assumes 2000s, might need adjustment for other centuries
    # # Return the *first* day of the expiry month as a placeholder date
    # # The actual roll decision uses the calendar's last_trading_day.
    # return datetime(year, month, 1)
    return None # Explicitly return None as it's deprecated

def _get_active_contract_for_date(current_date, roll_calendar_df):
    """Finds the active contract for the current date based on the roll calendar DataFrame."""
    if roll_calendar_df is None or roll_calendar_df.empty:
        logger.error("Roll calendar DataFrame is missing or empty. Cannot determine active contract.")
        return None

    # Ensure current_date is timezone-naive Timestamp for comparison
    current_date_ts = pd.Timestamp(current_date).normalize().tz_localize(None)

    # Filter calendar for contracts whose LTD is on or after the current date
    active_potential = roll_calendar_df[roll_calendar_df['last_trading_day'] >= current_date_ts]

    if active_potential.empty:
        # If no contract LTD is >= current date, maybe return the last known contract?
        # Or log an error/warning. For now, log and return None.
        logger.debug(f"No active contract found for {current_date_ts.date()} using roll calendar.")
        # Fallback: consider the contract that expired *most recently* before current_date?
        expired_potential = roll_calendar_df[roll_calendar_df['last_trading_day'] < current_date_ts]
        if not expired_potential.empty:
            last_expired_contract = expired_potential.sort_values('last_trading_day', ascending=False).iloc[0]['contract_code']
            logger.warning(f"No contract active on {current_date_ts.date()}. Using last expired as potential fallback: {last_expired_contract}")
            # Decide if returning this fallback is appropriate. For now, still return None from main logic.
            # return last_expired_contract # Uncomment if this fallback is desired behavior
        return None

    # The first contract in the sorted list (by LTD) is the active one
    active_contract = active_potential.sort_values('last_trading_day', ascending=True).iloc[0]['contract_code']
    return active_contract

# --- Get active contracts based on roll dates ---
def get_active_contracts_for_date(conn, date, root_symbol: str, roll_type: str, num_contracts: int) -> Dict[str, Optional[str]]:
    """
    Gets the active contracts (1st, 2nd, ..., nth month) for a date
    using the futures_roll_dates table.

    Args:
        conn: Existing database connection
        date: The date to look up (string, date, or datetime)
        root_symbol: The root symbol (e.g., VX, ES)
        roll_type: The roll type identifier (e.g., 'volume')
        num_contracts: The number of continuous contracts to retrieve (e.g., 9 for VX)

    Returns:
        Dictionary mapping standard continuous symbol (@VX=101XN) to underlying contract symbol.
    """
    # Convert input date to date object
    if isinstance(date, str):
        current_date = pd.to_datetime(date).date()
    elif isinstance(date, (datetime, pd.Timestamp)):
        current_date = date.date()
    else: # Assuming it's already a date object
        current_date = date

    mapping = {}

    try:
        # Fetch the ordered roll dates for the root symbol and roll type
        # Filter for roll dates *after* the current date
        query = """
            SELECT Contract, RollDate
            FROM futures_roll_dates
            WHERE SymbolRoot = ?
              AND RollType = ?
              AND RollDate > ?
            ORDER BY RollDate ASC;
        """
        params = [root_symbol, roll_type, current_date]
        roll_dates_df = conn.execute(query, params).fetchdf()

        if roll_dates_df.empty:
            logger.debug(f"No future roll dates found for {root_symbol} (type: {roll_type}) after {current_date}. Cannot determine active contracts.")
            for i in range(1, num_contracts + 1):
                 continuous_symbol = f"@{root_symbol}={i}01XN" # Standard key format
                 mapping[continuous_symbol] = None
            return mapping

        # The results are already sorted by RollDate, representing the 1st, 2nd, ... contracts
        underlying_contracts = roll_dates_df['Contract'].tolist()

    except Exception as e:
        logger.error(f"Error querying futures_roll_dates for {root_symbol} (type: {roll_type}) on {current_date}: {e}", exc_info=True)
        for i in range(1, num_contracts + 1):
             continuous_symbol = f"@{root_symbol}={i}01XN"
             mapping[continuous_symbol] = None
        return mapping

    # Build the mapping dictionary using the standard key format
    for i in range(1, num_contracts + 1):
        continuous_symbol = f"@{root_symbol}={i}01XN" # Standard key for lookup
        if i <= len(underlying_contracts):
            underlying_symbol = underlying_contracts[i-1]
            mapping[continuous_symbol] = underlying_symbol
        else:
            mapping[continuous_symbol] = None
            logger.debug(f"Could not determine {i}th contract for {root_symbol} on {current_date} (only found {len(underlying_contracts)} future roll dates).")

    return mapping

# --- Main Generation Function ---
def generate_continuous(conn, root_symbol, config, start_date, end_date, roll_type, force):
    """Generates continuous futures contracts (c1, c2, ...). Assumes conn is a valid DuckDB connection."""
    # Find the specific contract config within the main config
    contract_config = next((item for item in config.get('futures', []) if item['base_symbol'] == root_symbol), None)
    if not contract_config:
        logger.error(f"Configuration for root symbol '{root_symbol}' not found in provided config object.")
        return

    num_contracts = contract_config.get('num_active_contracts', 1)
    calendar_name = contract_config.get('calendar') # Get calendar name from config

    # --- Define Roll Type to Symbol Suffix Mapping ---
    # For VX, we specifically want the '01X' suffix for the '@VX=N01XN' format
    # when using the 'volume' roll type data from the database.
    if root_symbol == 'VX' and roll_type == 'volume':
        roll_suffix = '01X' # Hardcode for VX volume roll -> 01X suffix
    else:
        # Fallback mapping for other types or symbols
        roll_suffix_map = {
            'volume': 'V',
            # Add other mappings if needed
        }
        roll_suffix = roll_suffix_map.get(roll_type, roll_type) # Default to roll_type if not found

    logger.info(f"Beginning continuous generation for {root_symbol} (Roll Type: {roll_type}, Suffix: {roll_suffix}N) from {start_date} to {end_date}")

    # --- Delete existing data first if forced (runs ONCE for all years) ---
    if force:
        logger.info(f"Force flag specified. Deleting all existing data for {root_symbol} continuous contracts matching the generated suffix...")
        for c_num in range(1, num_contracts + 1):
            continuous_symbol = f"@{root_symbol}={c_num}{roll_suffix}N" # Use determined suffix
            logger.info(f"Deleting existing data for {continuous_symbol} due to --force flag.")
            delete_continuous_data(conn, continuous_symbol)
        logger.info(f"Finished deleting existing data for {root_symbol} with suffix {roll_suffix}N.")

    # --- Determine Year Range ---
    try:
        start_dt_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        start_year = start_dt_obj.year
        end_year = end_dt_obj.year
    except ValueError:
        logger.error("Invalid start or end date format. Use YYYY-MM-DD.")
        return

    # --- Loop Through Each Year ---
    missing_data_tracker_overall = {}
    date_missing_contracts_overall = {}

    for current_year in range(start_year, end_year + 1):
        logger.info(f"--- Processing Year: {current_year} ---")

        # Determine start/end dates for this specific year, respecting overall bounds
        year_start_date = f"{current_year}-01-01"
        year_end_date = f"{current_year}-12-31"

        current_year_start_dt_obj = datetime.strptime(year_start_date, '%Y-%m-%d').date()
        current_year_end_dt_obj = datetime.strptime(year_end_date, '%Y-%m-%d').date()

        actual_year_start_date = max(start_dt_obj, current_year_start_dt_obj).strftime('%Y-%m-%d')
        actual_year_end_date = min(end_dt_obj, current_year_end_dt_obj).strftime('%Y-%m-%d')

        logger.info(f"Processing date range for year {current_year}: {actual_year_start_date} to {actual_year_end_date}")

        # --- Generate Date Range for the Current Year --- #
        calendar_dates = []
        if calendar_name:
            try:
                logger.info(f"Using trading calendar: {calendar_name}")
                try:
                    trading_calendar = get_calendar(calendar_name)
                    schedule = trading_calendar.schedule(start_date=actual_year_start_date, end_date=actual_year_end_date)
                    calendar_dates = schedule.index.normalize().tolist()
                    logger.info(f"Generated yearly date range with {len(calendar_dates)} trading days using {calendar_name} calendar.")
                except Exception as exc_cal_err:
                    logger.warning(f"Error with exchange_calendars for '{calendar_name}': {exc_cal_err}. Trying pandas_market_calendars.")
                    try:
                        trading_calendar = mcal.get_calendar(calendar_name)
                        schedule = trading_calendar.schedule(start_date=actual_year_start_date, end_date=actual_year_end_date)
                        calendar_dates = schedule.index.tolist()
                        logger.info(f"Generated yearly date range with {len(calendar_dates)} trading days using pandas_market_calendars {calendar_name}.")
                    except Exception as mcal_err:
                        logger.error(f"Both calendar libraries failed for '{calendar_name}' for year {current_year}. Falling back to business days (freq='B'). Error: {mcal_err}")
                        date_range_pd = pd.date_range(start=actual_year_start_date, end=actual_year_end_date, freq='B')
                        calendar_dates = date_range_pd.tolist()
            except Exception as cal_err:
                logger.error(f"Error loading or using calendar '{calendar_name}' for year {current_year}: {cal_err}. Falling back to business days (freq='B').")
                date_range_pd = pd.date_range(start=actual_year_start_date, end=actual_year_end_date, freq='B')
                calendar_dates = date_range_pd.tolist()
        else:
            logger.warning(f"No calendar specified for {root_symbol} in config. Using business days (freq='B').")
            date_range_pd = pd.date_range(start=actual_year_start_date, end=actual_year_end_date, freq='B')
            calendar_dates = date_range_pd.tolist()

        # Supplement with market data dates (e.g., VIX for VX) to ensure all trading days are covered
        market_data_dates = []
        supplement_symbol = '$VIX.X' if root_symbol == 'VX' else None # Add other cases if needed
        if supplement_symbol:
            supplement_table = 'market_data_cboe' if root_symbol == 'VX' else 'market_data' # Adjust table if needed
            try:
                market_query = f"""
                    SELECT DISTINCT timestamp::DATE
                    FROM {supplement_table}
                    WHERE symbol = ?
                    AND interval_unit = 'daily' -- Assuming daily interval
                    AND timestamp::DATE >= ?
                    AND timestamp::DATE <= ?
                    ORDER BY timestamp::DATE
                """
                market_dates_df = conn.execute(market_query, [supplement_symbol, actual_year_start_date, actual_year_end_date]).fetchdf()
                if not market_dates_df.empty:
                    market_data_dates = pd.to_datetime(market_dates_df['timestamp']).tolist() # Keep as Timestamp
                    logger.info(f"Found {len(market_data_dates)} {supplement_symbol} trading days for {current_year}.")
            except Exception as e:
                logger.error(f"Error retrieving {supplement_symbol} dates: {e}")

        # Combine calendar dates with market data dates
        # Convert calendar_dates (which might be Timestamps or other datetime objects) to pd.Timestamp
        calendar_dates_ts = [pd.Timestamp(d) for d in calendar_dates]
        all_dates = sorted(list(set(calendar_dates_ts + market_data_dates)))
        # Ensure all are date objects at the end for consistency in loops
        # date_range = pd.DatetimeIndex(all_dates).date # Convert to array of date objects
        date_range = pd.DatetimeIndex(all_dates) # Keep as DatetimeIndex for easier processing

        logger.info(f"Using {len(date_range)} unique trading days for {current_year} after combining calendar and market data.")

        if date_range.empty:
            logger.warning(f"No trading days found for year {current_year} in the specified range. Skipping.")
            continue

        # --- Determine and Load contracts needed for the Current Year ---
        contracts_to_load = set()
        logger.info(f"Determining contracts needed for year {current_year} using roll type '{roll_type}'...")
        # Sample dates within the current year's range to find relevant contracts
        sample_dates_year = []
        total_days_year = len(date_range)
        if total_days_year == 0:
             logger.warning(f"Date range for year {current_year} is empty. No contracts to load.")
        elif total_days_year <= 10:
             sample_dates_year = date_range.tolist() # Use the list of Timestamps
        else:
             step_year = max(1, total_days_year // 10)
             sample_dates_year = [date_range[i] for i in range(0, total_days_year, step_year)]
             if date_range[0] not in sample_dates_year: sample_dates_year.insert(0, date_range[0])
             if date_range[-1] not in sample_dates_year: sample_dates_year.append(date_range[-1])

        # Get unique contracts based on roll dates for this year's sample dates
        for sample_date in sample_dates_year:
            contracts_for_date = get_active_contracts_for_date(conn, sample_date, root_symbol, roll_type, num_contracts)
            for continuous_symbol, underlying_symbol in contracts_for_date.items():
                if underlying_symbol:
                     contracts_to_load.add(underlying_symbol)
        logger.info(f"Found {len(contracts_to_load)} unique underlying contracts to load for year {current_year}")

        # --- Load Underlying Data for the Current Year ---
        underlying_data = {}
        if not contracts_to_load:
             logger.warning(f"No underlying contracts identified to load for year {current_year}.")
        else:
            logger.info(f"Loading data for {len(contracts_to_load)} underlying contracts for year {current_year}...")
            loaded_count = 0
            for symbol in contracts_to_load:
                # Load data ONLY for the relevant year to save memory
                # TODO: Optimize load_market_data to accept date ranges
                df_sym = load_market_data(conn, contract_config, root_symbol, config, symbol)

                if not df_sym.empty:
                    # Filter data for the current year's date range
                    df_sym_filtered = df_sym[(df_sym['timestamp'] >= pd.to_datetime(actual_year_start_date)) &
                                             (df_sym['timestamp'] <= pd.to_datetime(actual_year_end_date))].copy()

                    if not df_sym_filtered.empty:
                         underlying_data[symbol] = df_sym_filtered.set_index('timestamp')[['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest', 'source']]
                         loaded_count += 1
                    else:
                         logger.debug(f"No data found for symbol {symbol} within the date range {actual_year_start_date} to {actual_year_end_date} after loading.")
            logger.info(f"Successfully loaded data for {loaded_count} underlying contracts for year {current_year}.")

        # --- Process Each Date within the Current Year --- #
        results_for_year = []
        missing_data_tracker_year = {}
        date_missing_contracts_year = {}

        for current_dt in date_range: # Iterate through combined & sorted Timestamps
            # Use the Timestamp object directly for lookups
            # current_date_str = current_dt.strftime('%Y-%m-%d') # Not needed if using Timestamp

            # Get contracts for this date using the new function
            # Pass the original roll_type identifier used for DB lookup
            contracts_map = get_active_contracts_for_date(conn, current_dt, root_symbol, roll_type, num_contracts)

            date_missing = []

            for c_num in range(1, num_contracts + 1):
                continuous_symbol = f"@{root_symbol}={c_num}{roll_suffix}N" # Use dynamic suffix

                # The get_active_contracts_for_date function returns keys like @VX=101XN, @VX=201XN
                # We need to look up the underlying contract using the key format it expects
                # OR modify get_active_contracts_for_date to return keys with the dynamic suffix (more complex)
                # Let's keep get_active_contracts_for_date returning the standard format for now
                # and extract the underlying symbol from its output if needed.
                # HOWEVER, the current get_active_contracts_for_date uses the standard format for the *keys*
                # Let's adjust get_active_contracts_for_date to use the correct key format.

                # Re-thinking: Let get_active_contracts_for_date return the standard keys (@VX=101XN)
                # We look up the corresponding underlying symbol using the standard key.
                standard_lookup_key = f"@{root_symbol}={c_num}01XN" # Standard key format used by get_active_contracts_for_date
                underlying_symbol = contracts_map.get(standard_lookup_key)

                if not underlying_symbol:
                    logger.debug(f"No underlying contract determined for position {c_num} ({standard_lookup_key}) on {current_dt.date()}")
                    date_missing.append(f"{continuous_symbol}(no mapping)") # Log with the target symbol format

                # Initialize data points
                open_price, high_price, low_price, close_price, settle_price, volume_value, open_interest_value, source_value = [None] * 8
                valid_price_found = False
                actual_underlying_for_record = None

                if underlying_symbol:
                    # Look up data in the pre-loaded dictionary
                    if underlying_symbol in underlying_data:
                        try:
                            # Use current_dt (Timestamp) for indexing
                            row_data_series = underlying_data[underlying_symbol].loc[current_dt]

                            if row_data_series is not None and not row_data_series.empty:
                                open_price = row_data_series.get('open')
                                high_price = row_data_series.get('high')
                                low_price = row_data_series.get('low')
                                close_price = row_data_series.get('close') # Store original close
                                volume_value = row_data_series.get('volume')
                                open_interest_value = row_data_series.get('open_interest')
                                source_value = row_data_series.get('source')
                                actual_underlying_for_record = underlying_symbol

                                # Try settle first, then fallback to close
                                temp_settle = row_data_series.get('settle')
                                temp_close = row_data_series.get('close')

                                if pd.notna(temp_settle) and temp_settle > 0:
                                    settle_price = temp_settle
                                    valid_price_found = True
                                elif pd.notna(temp_close) and temp_close > 0:
                                    settle_price = temp_close # Use close as fallback
                                    valid_price_found = True
                                    logger.debug(f"Using CLOSE price ({settle_price}) for {continuous_symbol} on {current_dt.date()} as SETTLE is missing/invalid.")
                                else:
                                    settle_price = None

                        except KeyError:
                            # Data for this specific date not found in the loaded DataFrame
                            logger.debug(f"Data for date {current_dt.date()} not found in loaded data for {underlying_symbol}")
                        except Exception as e:
                            logger.error(f"Error processing row for {underlying_symbol} on {current_dt.date()}: {e}")
                    else:
                        logger.debug(f"No pre-loaded data available for {underlying_symbol} (required for {current_dt.date()})")

                # Append result row
                results_for_year.append({
                    'timestamp': current_dt, # Store the Timestamp
                    'symbol': continuous_symbol,
                    'interval_value': 1,
                    'interval_unit': 'daily',
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price, # Store original close if available
                    'settle': settle_price, # Store settle (or close fallback)
                    'volume': volume_value,
                    'open_interest': open_interest_value,
                    'underlying_symbol': actual_underlying_for_record,
                    'source': source_value if source_value else ('cboe' if root_symbol == 'VX' else 'generated')
                })

                # Track Missing Data
                if not valid_price_found:
                    missing_reason = "no mapping" if not underlying_symbol else f"no data for {underlying_symbol}"
                    date_missing.append(f"{continuous_symbol}({missing_reason})") # Log with target format
                    # Use the *target* continuous symbol format for tracking
                    missing_key = f"{continuous_symbol}_{underlying_symbol or 'NoMapping'}"
                    missing_data_tracker_overall[missing_key] = missing_data_tracker_overall.get(missing_key, 0) + 1

            if date_missing:
                logger.debug(f"Missing on {current_dt.date()}: {', '.join(date_missing)}")
                date_missing_contracts_overall[current_dt.date()] = date_missing

        # --- Insert/Update Data for the Current Year ---
        if results_for_year:
            logger.info(f"Preparing to insert/update {len(results_for_year)} data points for year {current_year}...")
            try:
                df_insert = pd.DataFrame(results_for_year)

                # Data Type Conversions
                df_insert['timestamp'] = pd.to_datetime(df_insert['timestamp']).dt.tz_localize(None) # Ensure timezone naive
                numeric_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']
                for col in numeric_cols:
                    df_insert[col] = pd.to_numeric(df_insert[col], errors='coerce')
                string_cols = ['symbol', 'interval_unit', 'underlying_symbol', 'source']
                for col in string_cols:
                     df_insert[col] = df_insert[col].astype(str).fillna('')
                df_insert['interval_value'] = pd.to_numeric(df_insert['interval_value'], errors='coerce').fillna(1).astype(int)
                # Set specific types for DB insert
                df_insert['volume'] = df_insert['volume'].fillna(0).astype('Int64') # Use nullable Int64
                df_insert['open_interest'] = df_insert['open_interest'].fillna(0).astype('Int64') # Use nullable Int64

                # Define insert columns based on DataFrame
                insert_columns = df_insert.columns.tolist()
                insert_columns_str = ", ".join([f'"{c}"' for c in insert_columns])
                
                # Define primary key columns for conflict resolution
                pk_cols = ['timestamp', 'symbol', 'interval_value', 'interval_unit'] # Assuming this PK
                update_cols = [c for c in insert_columns if c not in pk_cols]
                update_setters_str = ", ".join([f'"{c}" = excluded."{c}"' for c in update_cols])

                # Register DataFrame as a temporary view
                temp_view_name = f"temp_insert_{root_symbol}_{current_year}"
                conn.register(temp_view_name, df_insert[insert_columns])

                # Upsert SQL
                upsert_sql = f"""
                    INSERT INTO continuous_contracts ({insert_columns_str})
                    SELECT {insert_columns_str} FROM {temp_view_name}
                    ON CONFLICT (timestamp, symbol, interval_value, interval_unit) DO UPDATE SET
                        {update_setters_str}
                """

                conn.execute(upsert_sql)
                conn.unregister(temp_view_name)
                logger.info(f"Successfully upserted {len(df_insert)} rows for year {current_year}.")

            except Exception as e:
                logger.error(f"Error during database upsert for year {current_year}: {e}", exc_info=True)
                # Potentially add rollback logic here if using transactions

    # --- Final Summary Logging ---
    logger.info("--- Continuous Generation Summary ---")
    if missing_data_tracker_overall:
        logger.warning("Missing Data Summary (Contract_Underlying: Count):")
        for key, count in sorted(missing_data_tracker_overall.items()):
            logger.warning(f"  {key}: {count}")
    else:
        logger.info("No missing data points tracked during generation.")
        
    if date_missing_contracts_overall:
        logger.warning(f"Dates with at least one missing contract mapping/data: {len(date_missing_contracts_overall)}")
        # Optionally log the first few dates:
        # for i, (date, missing) in enumerate(date_missing_contracts_overall.items()):
        #     if i >= 10: break
        #     logger.warning(f"  {date}: {', '.join(missing)}")
    else:
         logger.info("All requested contract positions had underlying data found for all processed dates.")

# --- Constants ---
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'market_symbols.yaml')
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'financial_data.duckdb')
DEFAULT_START_DATE = '1980-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
DEFAULT_ROLL_TYPE = 'volume' # Default roll type changed to volume

# --- Main Execution Logic ---
def main(args_dict=None, existing_conn=None):
    """Main function to run generation from CLI or direct call."""
    if args_dict:
        args = argparse.Namespace(**args_dict)
        logger.info("Running generate_continuous_futures from direct call.")
    else:
        parser = argparse.ArgumentParser(description='Generate continuous futures contracts.')
        parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to DuckDB database')
        parser.add_argument('--config-path', default=DEFAULT_CONFIG_PATH, help='Path to YAML configuration file')
        parser.add_argument('--root-symbol', required=True, help='Root symbol (e.g., VX, ES, NQ)')
        parser.add_argument('--start-date', default=DEFAULT_START_DATE, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', default=DEFAULT_END_DATE, help='End date (YYYY-MM-DD)')
        parser.add_argument('--roll-type', default=DEFAULT_ROLL_TYPE, help=f'Roll type from futures_roll_dates (default: {DEFAULT_ROLL_TYPE})')
        parser.add_argument('--force', action='store_true', help='Delete existing data before generating')
        args = parser.parse_args()
        logger.info("Running generate_continuous_futures from command line.")

    # --- Setup Logging --- #
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"generate_continuous_{args.root_symbol}_{timestamp_str}.log")
    setup_logging(log_file)
    # --- End Logging Setup ---

    logger.info(f"Effective args: {vars(args)}")

    config = load_config(args.config_path)
    if config is None:
        sys.exit(1)

    conn = None
    close_conn_locally = False

    try:
        if existing_conn:
            conn = existing_conn
            logger.info("Using existing database connection.")
        else:
            conn = connect_db(args.db_path, read_only=False)
            close_conn_locally = True

        if conn:
            # Pass roll_type to generate_continuous
            generate_continuous(conn, args.root_symbol, config, args.start_date, args.end_date, args.roll_type, args.force)
            logger.info("Continuous futures generation completed.")
        else:
             logger.error("Failed to establish database connection.")
             sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred during continuous generation: {e}", exc_info=True)
        # Consider adding conn.rollback() here if using explicit transactions
    finally:
        if conn and close_conn_locally:
            try:
                conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing database connection: {e_close}")

if __name__ == "__main__":
    main() 