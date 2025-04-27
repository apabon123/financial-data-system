import sys
import yaml
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_END_DATE = datetime.today().strftime('%Y-%m-%d')
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"
DEFAULT_DB_PATH = "data/financial_data.duckdb" # Updated default

# Configuration
DEFAULT_START_DATE = '2004-01-01' # Example default, adjust as needed

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

def load_market_data(conn, symbol, start_date, end_date):
    """Loads market data for a specific symbol and date range."""
    try:
        query = f"""
            SELECT timestamp, open, high, low, settle, source
            FROM market_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        """
        df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
        logger.info(f"Loaded {len(df)} market data entries for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def load_roll_calendar(conn, root_symbol):
    """Loads the roll calendar (last trading days) for a root symbol."""
    query = f"""
        SELECT contract_code, last_trading_day
        FROM futures_roll_calendar
        WHERE root_symbol = ?
        ORDER BY last_trading_day ASC
    """
    try:
        df_calendar = conn.execute(query, [root_symbol]).fetchdf()
        df_calendar['last_trading_day'] = pd.to_datetime(df_calendar['last_trading_day'])
        calendar_dict = pd.Series(df_calendar.last_trading_day.values, index=df_calendar.contract_code).to_dict()
        logger.info(f"Loaded {len(calendar_dict)} roll calendar entries for {root_symbol}")
        return calendar_dict
    except duckdb.Error as e:
        logger.error(f"Error loading roll calendar for {root_symbol}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading roll calendar for {root_symbol}: {e}")
        return {}

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
    if len(symbol) < 4:
        raise ValueError(f"Invalid symbol format: {symbol}")
    month_code = symbol[-3]
    year_short = symbol[-2:]
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    if month_code not in month_map:
        raise ValueError(f"Invalid month code in symbol: {symbol}")
    month = month_map[month_code]
    year = int(f"20{year_short}") # Assumes 2000s, might need adjustment for other centuries
    # Return the *first* day of the expiry month as a placeholder date
    # The actual roll decision uses the calendar's last_trading_day.
    return datetime(year, month, 1)

def _get_active_contract_for_date(current_date, contracts, roll_calendar):
    """Finds the active contract for the current date based on the roll calendar's last_trading_day."""
    if not roll_calendar:
        logger.error("Roll calendar data is missing. Cannot determine active contract.")
        return None

    # Ensure current_date is timezone-naive for comparison
    current_date_naive = current_date.replace(tzinfo=None) if current_date.tzinfo else current_date

    active_contract = None
    # Sort contracts by their last trading day from the calendar
    # Use datetime.max for any contract not found in the calendar (shouldn't happen ideally)
    sorted_contracts = sorted(contracts, key=lambda c: roll_calendar.get(c, datetime.max))

    for contract in sorted_contracts:
        last_trading_day = roll_calendar.get(contract)
        if last_trading_day is None:
            # This case means the contract from generate_symbol_map wasn't in the calendar table
            logger.warning(f"Contract {contract} not found in loaded roll calendar. Skipping.")
            continue

        last_trading_day_naive = last_trading_day.replace(tzinfo=None) if last_trading_day.tzinfo else last_trading_day

        # The contract is active if the current_date is less than or equal to its last trading day
        if current_date_naive <= last_trading_day_naive:
            active_contract = contract
            break # Found the first contract that is still active

    # If no active contract found (current_date is past all known last trading days)
    if active_contract is None:
         logger.debug(f"No active contract found for {current_date_naive} among provided contracts.")
         # Depending on need, could fallback to last contract: active_contract = sorted_contracts[-1] if sorted_contracts else None

    return active_contract

# --- Main Generation Function ---
def generate_continuous(conn, root_symbol, contract_config, start_date, end_date, force):
    """Generates continuous futures contracts (c1, c2, ...). Assumes conn is a valid DuckDB connection."""
    num_contracts = contract_config.get('num_active_contracts', 1)
    patterns = contract_config.get('historical_contracts', {}).get('patterns', [])
    hist_start_year = contract_config.get('historical_contracts', {}).get('start_year', datetime.strptime(start_date, '%Y-%m-%d').year)

    # Load Roll Calendar for this root symbol using the main connection
    roll_calendar = load_roll_calendar(conn, root_symbol)
    if not roll_calendar:
        logger.error(f"Failed to load roll calendar for {root_symbol}. Aborting generation.")
        return

    # Generate list of all potential historical contracts (needed for _get_active_contract_for_date)
    all_contracts = []
    current_year = datetime.strptime(end_date, '%Y-%m-%d').year
    for year in range(hist_start_year, current_year + 2): # Extend year range slightly for future contracts
        all_contracts.extend(generate_symbol_map(root_symbol, year, patterns))

    # --- Delete existing data if forced ---
    if force:
        for c_num in range(1, num_contracts + 1):
            continuous_symbol = f"@{root_symbol}={c_num}01XN"
            logger.info(f"Deleting existing data for {continuous_symbol} due to --force flag.")
            delete_continuous_data(conn, continuous_symbol)

    # --- Generate Date Range ---
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Use business days
    logger.info(f"Generating continuous data for {root_symbol} from {start_date} to {end_date}...")

    # --- Load all relevant underlying contract data into memory ---
    # This avoids querying the database repeatedly inside the loop
    underlying_data = {}
    
    # --- Optimized contract selection --- #
    contracts_to_load = set()
    gen_start_dt = pd.to_datetime(start_date)
    gen_end_dt = pd.to_datetime(end_date)
    
    # Find the contract active just before the generation period starts
    # We need this to correctly identify the c1 contract at the start of the range
    contract_before_start = None
    latest_ltd_before_start = pd.Timestamp.min
    for contract_code, last_trading_day in roll_calendar.items():
        if last_trading_day < gen_start_dt and last_trading_day > latest_ltd_before_start:
            latest_ltd_before_start = last_trading_day
            contract_before_start = contract_code
    if contract_before_start:
        contracts_to_load.add(contract_before_start)
        logger.debug(f"Identified contract active before start date: {contract_before_start}")
        
    # Find all contracts whose last trading day is within or slightly after the generation period
    # Add a buffer (e.g., 3 months) to ensure we have data for c2, c3 etc. near the end date
    buffer_end_date = gen_end_dt + pd.DateOffset(months=3) 
    for contract_code, last_trading_day in roll_calendar.items():
        # Load if the contract expires after our generation period started 
        # (or includes the contract active just before the start)
        # AND it expires before a reasonable buffer after our period ends.
        if last_trading_day >= latest_ltd_before_start and last_trading_day <= buffer_end_date:
            contracts_to_load.add(contract_code)
            
    # Ensure the contract active at the very start date is included if missed
    start_contract_seq = _get_active_contract_for_date(gen_start_dt, list(roll_calendar.keys()), roll_calendar)
    if start_contract_seq:
         contracts_to_load.add(start_contract_seq) # Add the first contract active on start date

    # --- Fallback if filtering is too aggressive (should be rare) ---
    if not contracts_to_load:
        logger.warning("Optimized contract pre-load filtering yielded no results. Falling back to broader load (LTD >= start_date - 3 months). This might happen for very short generation periods.")
        fallback_start_date = gen_start_dt - pd.DateOffset(months=3) 
        for contract_code, last_trading_day in roll_calendar.items():
            if last_trading_day >= fallback_start_date:
                 contracts_to_load.add(contract_code)
    # --- End of optimized contract selection ---

    logger.info(f"Attempting to load data for {len(contracts_to_load)} underlying contracts relevant to the period {start_date} - {end_date}...")
    loaded_count = 0
    for symbol in contracts_to_load:
        # Load data for the entire potential period for each contract
        df_sym = load_market_data(conn, symbol, start_date, end_date) # Adjust range if needed
        if not df_sym.empty:
            # Ensure timestamp is datetime and set it as index
            df_sym['timestamp'] = pd.to_datetime(df_sym['timestamp'])
            # Store Series including settle and source, indexed by timestamp
            underlying_data[symbol] = df_sym.set_index('timestamp')[['settle', 'source']]
            loaded_count += 1
    logger.info(f"Successfully loaded data for {loaded_count} underlying contracts.")

    # --- Process Each Date --- #
    results = []
    for current_dt in date_range:
        # Find the sequence of active contracts for this date
        active_contracts_sequence = []
        # Start search from contracts potentially active around current_dt
        potential_contracts = sorted([c for c, ltd in roll_calendar.items() if ltd >= current_dt], key=lambda c: roll_calendar[c])

        temp_contract = None
        contracts_pool = list(all_contracts) # Use the full list generated earlier

        for i in range(num_contracts):
             # Find the next active contract based on the calendar
             active_contract = _get_active_contract_for_date(current_dt, contracts_pool, roll_calendar)
             if active_contract:
                 active_contracts_sequence.append(active_contract)
                 # Remove this contract and any before it (based on roll calendar order) from the pool for the next iteration
                 contracts_pool = sorted([c for c in contracts_pool if roll_calendar.get(c, datetime.min) > roll_calendar.get(active_contract, datetime.min)], key=lambda c: roll_calendar.get(c, datetime.max))
             else:
                 # Not enough future contracts found for this date
                 break

        # For each continuous contract (c1, c2, ...)
        for c_num, underlying_symbol in enumerate(active_contracts_sequence, 1):
            # continuous_symbol = f"{root_symbol}c{c_num}" # Old format
            # Generate symbol like @VX=101XN, @VX=201XN etc. Using 01XN as placeholder suffix for now.
            continuous_symbol = f"@{root_symbol}={c_num}01XN"

            # Get the settlement price and source from the pre-loaded data
            settle_price = None
            source_value = None # Default source to None
            underlying_df = underlying_data.get(underlying_symbol)
            if underlying_df is not None and not underlying_df.empty:
                try:
                    # Use .loc for potential index lookup
                    row_data = underlying_df.loc[current_dt]

                    # Ensure row_data is a Series (take the first row if it's a DataFrame)
                    if isinstance(row_data, pd.DataFrame):
                        if not row_data.empty:
                            row_data = row_data.iloc[0] # Take the first row for the date
                        else:
                            # Handle case where loc returns an empty DataFrame (should be caught by KeyError ideally)
                            raise KeyError(f"No data found for {current_dt} despite non-empty DataFrame check.")

                    # Now row_data is guaranteed to be a Series (or raised an error)
                    # Check if settle exists and is not NaN before assigning
                    if 'settle' in row_data and pd.notna(row_data['settle']): # Check column exists first
                        settle_price = row_data['settle']
                        source_value = row_data.get('source') # Use .get for robustness
                except KeyError:
                    # Date not found in this specific underlying contract's data
                    logger.debug(f"Data for {current_dt.strftime('%Y-%m-%d')} not found in pre-loaded data for {underlying_symbol}")
                except Exception as e_lookup:
                    logger.warning(f"Error looking up data for {underlying_symbol} on {current_dt}: {e_lookup}")

            # Always append a row, use None for settle if price is missing/NaN
            results.append({
                'timestamp': current_dt,
                'Symbol': continuous_symbol,
                'settle': settle_price if pd.notna(settle_price) else None,
                'UnderlyingSymbol': underlying_symbol,
                'interval_value': 1,
                'interval_unit': 'daily',
                'source': source_value,
                'BuiltBy': 'local_generator'
            })

    # --- Save Results ---
    if results:
        df_results = pd.DataFrame(results)
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        
        # Use the passed connection `conn`
        try:
            # Target the correct table: continuous_contracts
            conn.register('results_df', df_results)
            # Rename columns in DataFrame to match target table
            df_results = df_results.rename(columns={
                'Symbol': 'symbol',
                'UnderlyingSymbol': 'underlying_symbol',
                'settle': 'settle',
                'BuiltBy': 'built_by'
            })
            
            # Define columns for insertion
            # Match these to the columns in continuous_contracts table schema
            insert_columns = ['timestamp', 'symbol', 'underlying_symbol', 'settle', 'interval_value', 'interval_unit', 'source'] 
            # Add 'built_by' and other columns if they exist
            if 'built_by' in df_results.columns and 'built_by' not in insert_columns: insert_columns.append('built_by')
            # Ensure 'adjusted' is set correctly (likely FALSE if we only have settle)
            df_results['adjusted'] = False # Assuming not adjusted for now
            if 'adjusted' not in insert_columns: insert_columns.append('adjusted')

            # Filter DataFrame to only include insert_columns
            df_insert = df_results[insert_columns]

            conn.register('insert_view', df_insert)

            insert_columns_str = ", ".join(insert_columns)
            update_setters_str = ", ".join([f"{col} = excluded.{col}" for col in insert_columns if col not in ['timestamp', 'symbol', 'interval_value', 'interval_unit']])
            
            sql = f"""
                INSERT INTO continuous_contracts ({insert_columns_str})
                SELECT {insert_columns_str}
                FROM insert_view
                ON CONFLICT (timestamp, symbol, interval_value, interval_unit) DO UPDATE SET
                    {update_setters_str}
            """
            conn.execute(sql)
            conn.unregister('insert_view') # Unregister the final view used
            conn.unregister('results_df') # Also unregister the original df
            logger.info(f"Successfully inserted/updated {len(df_insert)} continuous data points into continuous_contracts table.")
        except Exception as e:
            logger.error(f"Error saving continuous data to database: {e}")
            # Ensure unregistration even on error if views were created
            try:
                conn.unregister('insert_view')
            except: pass
            try:
                 conn.unregister('results_df')
            except: pass
    else:
        logger.warning("No continuous contract data generated.")

# --- Main Execution ---
def main(args_dict=None, existing_conn=None):
    """Main function to run generation from CLI or direct call."""
    if args_dict:
        # Called directly with arguments
        args = argparse.Namespace(**args_dict)
        logger.info("Running generate_continuous_futures from direct call.")
    else:
        # Called from command line
        parser = argparse.ArgumentParser(description='Generate continuous futures contracts.')
        parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to DuckDB database')
        parser.add_argument('--config-path', default=DEFAULT_CONFIG_PATH, help='Path to YAML configuration file')
        parser.add_argument('--root-symbol', required=True, help='Root symbol (e.g., VX, ES, NQ)')
        parser.add_argument('--start-date', default=DEFAULT_START_DATE, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', default=DEFAULT_END_DATE, help='End date (YYYY-MM-DD)')
        parser.add_argument('--force', action='store_true', help='Delete existing data before generating')
        args = parser.parse_args()
        logger.info("Running generate_continuous_futures from command line.")
    
    logger.info(f"Effective args: {vars(args)}")

    config = load_config(args.config_path)
    if config is None:
        sys.exit(1)

    contract_config = next((item for item in config.get('futures', []) if item['base_symbol'] == args.root_symbol), None)
    if not contract_config:
        logger.error(f"Configuration for root symbol '{args.root_symbol}' not found in {args.config_path}")
        sys.exit(1)
        
    conn = None # Initialize conn to None
    close_conn_locally = False # Flag to track if we need to close the connection here
    
    try:
        if existing_conn:
            conn = existing_conn
            logger.info("Using existing database connection.")
        else:
            # Need write access for delete and insert/update
            conn = connect_db(args.db_path, read_only=False) 
            close_conn_locally = True # We created it, so we should close it
            
        if conn:
            generate_continuous(conn, args.root_symbol, contract_config, args.start_date, args.end_date, args.force)
            logger.info("Continuous futures generation completed.")
        else:
             logger.error("Failed to establish database connection.")
             sys.exit(1)
             
    except Exception as e:
        logger.error(f"An error occurred during continuous generation: {e}")
    finally:
        if conn and close_conn_locally:
            conn.close()
            logger.info("Database connection closed.")

# Guard execution for when script is run directly
if __name__ == "__main__":
    # Call main without args_dict when run directly
    main() 