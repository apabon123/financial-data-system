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
def connect_db(db_path):
    """Connects to the DuckDB database."""
    try:
        # Connect read-only unless force deleting
        # However, the delete function needs write access later.
        # Let's connect read-write for simplicity, assuming it's needed.
        conn = duckdb.connect(database=db_path, read_only=False)
        logger.info(f"Connected to database: {db_path}")
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
        # Using 'market_data' table based on previous context
        table_name = 'market_data'
        query = f"DELETE FROM {table_name} WHERE symbol = ?"
        # DuckDB execute often doesn't return rowcount directly
        conn.execute(query, [continuous_symbol])
        # Assuming auto-commit or handled elsewhere if transaction needed
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
def generate_continuous(conn, db_path, root_symbol, contract_config, start_date, end_date, force):
    """Generates continuous futures contracts (c1, c2, ...)."""
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
            continuous_symbol = f"{root_symbol}c{c_num}"
            logger.info(f"Deleting existing data for {continuous_symbol} due to --force flag.")
            delete_continuous_data(conn, continuous_symbol)

    # --- Generate Date Range ---
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Use business days
    logger.info(f"Generating continuous data for {root_symbol} from {start_date} to {end_date}...")

    # --- Load all relevant underlying contract data into memory ---
    # This avoids querying the database repeatedly inside the loop
    underlying_data = {}
    # Determine which contracts might be needed based on date range and calendar
    contracts_to_load = set()
    for contract_code, last_trading_day in roll_calendar.items():
        # Load if the contract's last trading day is within or after our range start
        # And potentially started before our range end (rough filter)
        contract_year = int(f"20{contract_code[-2:]}")
        if last_trading_day.year >= hist_start_year and contract_year <= current_year + 1:
             contracts_to_load.add(contract_code)

    logger.info(f"Attempting to load data for {len(contracts_to_load)} potential underlying contracts...")
    for symbol in contracts_to_load:
        # Load data for the entire potential period for each contract
        df_sym = load_market_data(conn, symbol, start_date, end_date) # Adjust range if needed
        if not df_sym.empty:
            # Ensure timestamp is datetime and set it as index
            df_sym['timestamp'] = pd.to_datetime(df_sym['timestamp'])
            # Store Series including settle and source, indexed by timestamp
            underlying_data[symbol] = df_sym.set_index('timestamp')[['settle', 'source']]
    logger.info(f"Successfully loaded data for {len(underlying_data)} underlying contracts.")

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
            continuous_symbol = f"{root_symbol}c{c_num}"

            # Get the settlement price and source from the pre-loaded data
            settle_price = None
            source_value = None # Default source to None
            underlying_df = underlying_data.get(underlying_symbol)
            if underlying_df is not None and not underlying_df.empty:
                try:
                    # Use .loc for potential index lookup
                    row_data = underlying_df.loc[current_dt]
                    # Check if settle exists and is not NaN before assigning
                    if pd.notna(row_data['settle']):
                        settle_price = row_data['settle']
                        source_value = row_data['source'] # Assign source only if settle is valid
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
                'interval_unit': 'day',
                'source': source_value # Add the retrieved source
            })

    # --- Write Results to Database --- #
    if results:
        results_df = pd.DataFrame(results)
        try:
            # --- Ensure UnderlyingSymbol column exists in market_data ---
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('market_data')
                WHERE name = 'UnderlyingSymbol'
            """)
            column_exists = cursor.fetchone()[0] > 0
            cursor.close()

            if not column_exists:
                logger.info("Column 'UnderlyingSymbol' not found in 'market_data'. Adding column...")
                conn.execute("ALTER TABLE market_data ADD COLUMN UnderlyingSymbol VARCHAR") # Use VARCHAR or TEXT
                logger.info("Column 'UnderlyingSymbol' added successfully.")
            else:
                logger.info("Column 'UnderlyingSymbol' already exists in 'market_data'.")

            # --- Write data ---
            # Assuming 'market_data' table with columns: timestamp, Symbol, settle, UnderlyingSymbol
            logger.info(f"Writing {len(results_df)} continuous contract entries to database...")
            conn.register('results_df_view', results_df)
            conn.execute(f"""
                INSERT OR REPLACE INTO market_data (timestamp, Symbol, settle, UnderlyingSymbol, interval_value, interval_unit, source)
                SELECT timestamp, Symbol, settle, UnderlyingSymbol, interval_value, interval_unit, source FROM results_df_view
            """)
            conn.commit() # Necessary for DuckDB when not using autocommit context
            logger.info("Successfully wrote continuous contract data.")
        except Exception as e:
            logger.error(f"Error writing continuous data to database: {e}")
    else:
        logger.warning("No continuous contract data generated.")

# --- Main Execution ---
def main(args_dict=None):
    effective_args = {}
    if args_dict is None:
        # --- Argument Parsing (if run directly) ---
        parser = argparse.ArgumentParser(description='Generate Continuous Futures Contracts.')
        parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
        parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the market symbols YAML config file.')
        parser.add_argument('--root-symbol', type=str, required=True, help='Root symbol to generate continuous contracts for (e.g., ES, VX).')
        parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD). Defaults to config or earliest data.')
        parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE, help='End date (YYYY-MM-DD). Defaults to today.')
        parser.add_argument('--force', action='store_true', help='Force delete existing data before generating new contracts.')
        parsed_args = parser.parse_args()
        # Populate effective_args from parsed command-line args
        effective_args['db_path'] = parsed_args.db_path
        effective_args['config_path'] = parsed_args.config_path
        effective_args['root_symbol'] = parsed_args.root_symbol
        effective_args['start_date'] = parsed_args.start_date
        effective_args['end_date'] = parsed_args.end_date
        effective_args['force'] = parsed_args.force
        logger.info("Running generate_continuous_futures from command line.")
    else:
        # --- Use Provided Args (if called programmatically) ---
        logger.info("Running generate_continuous_futures programmatically.")
        effective_args['db_path'] = args_dict.get('db_path', DEFAULT_DB_PATH)
        effective_args['config_path'] = args_dict.get('config_path', DEFAULT_CONFIG_PATH)
        effective_args['root_symbol'] = args_dict.get('root_symbol') # Required, should be in args_dict
        effective_args['start_date'] = args_dict.get('start_date') # Optional, defaults to None
        effective_args['end_date'] = args_dict.get('end_date', DEFAULT_END_DATE)
        effective_args['force'] = args_dict.get('force', False) # Default force to False

        # Basic validation for required arg
        if not effective_args['root_symbol']:
             logger.error("Root symbol must be provided in args_dict.")
             sys.exit(1)


    # --- Common Logic using effective_args ---
    logger.info(f"Effective args: {effective_args}")

    # Load configuration
    config = load_config(effective_args['config_path'])

    # Find the correct contract config within the 'futures' list
    contract_config = None
    if config and 'futures' in config:
        for future_config in config['futures']:
            if future_config.get('base_symbol') == effective_args['root_symbol']:
                contract_config = future_config
                break

    if contract_config is None:
        logger.error(f"Config for root symbol {effective_args['root_symbol']} not found under 'futures' list in {effective_args['config_path']}.")
        sys.exit(1)

    # Determine effective start/end dates (handle None defaults)
    config_start_date = contract_config.get('start_date', DEFAULT_START_DATE)
    # Use start_date from effective_args if provided, else use config start_date
    gen_start_date = effective_args['start_date'] if effective_args['start_date'] else config_start_date
    gen_end_date = effective_args['end_date'] # Already defaulted

    # Validate dates if needed (basic check)
    try:
        datetime.strptime(gen_start_date, '%Y-%m-%d')
        datetime.strptime(gen_end_date, '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format provided. Please use YYYY-MM-DD.")
        sys.exit(1)
    except TypeError:
        logger.error("Date validation failed - check start/end date values.")
        sys.exit(1)


    conn = None # Initialize conn
    try:
        conn = connect_db(effective_args['db_path'])
        logger.info(f"Generating continuous contracts for {effective_args['root_symbol']} from {gen_start_date} to {gen_end_date}")
        generate_continuous(conn,
                            effective_args['db_path'],
                            effective_args['root_symbol'],
                            contract_config,
                            gen_start_date,
                            gen_end_date,
                            effective_args['force'])
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1) # Exit after logging the error
    finally:
        if conn:
            try:
                 if not conn.closed:
                     conn.close()
                     logger.info("Database connection closed.")
            except Exception as close_e:
                 logger.error(f"Error closing database connection: {close_e}")


# Guard execution for when script is run directly
if __name__ == "__main__":
    # Call main without args_dict when run directly
    main() 