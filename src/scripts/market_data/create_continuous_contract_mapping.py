import sys
import duckdb
import pandas as pd
import argparse
import logging
import os
from datetime import datetime, timedelta
from exchange_calendars import get_calendar # Import calendar library
import pandas_market_calendars as mcal # Add pandas_market_calendars for fallback
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db(db_path, read_only=False):
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to database: {db_path} (Read-Only: {read_only})")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

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
            return pd.DataFrame()
        df_calendar['last_trading_day'] = pd.to_datetime(df_calendar['last_trading_day'])
        logger.info(f"Loaded {len(df_calendar)} roll calendar entries for {root_symbol}.")
        return df_calendar
    except Exception as e:
        logger.error(f"Error loading roll calendar for {root_symbol}: {e}")
        return pd.DataFrame()

def get_active_contract_for_date(current_date, roll_calendar_df):
    """Finds the active contract for the current date based on the roll calendar."""
    if roll_calendar_df is None or roll_calendar_df.empty:
        logger.error("Roll calendar DataFrame is missing or empty.")
        return None

    # Ensure current_date is timezone-naive Timestamp for comparison
    current_date_ts = pd.Timestamp(current_date).normalize().tz_localize(None)

    # Filter calendar for contracts whose LTD is on or after the current date
    active_potential = roll_calendar_df[roll_calendar_df['last_trading_day'] >= current_date_ts]

    if active_potential.empty:
        # If no contracts available, get the most recently expired one
        expired = roll_calendar_df[roll_calendar_df['last_trading_day'] < current_date_ts]
        if not expired.empty:
            last_expired = expired.sort_values('last_trading_day', ascending=False).iloc[0]['contract_code']
            logger.debug(f"No active contract found for {current_date_ts.date()}. Using last expired: {last_expired}")
            return last_expired
        return None

    # The first contract in the sorted list (by LTD) is the active one
    active_contract = active_potential.sort_values('last_trading_day', ascending=True).iloc[0]['contract_code']
    return active_contract

def generate_continuous_mapping(conn, root_symbol, num_contracts, start_date, end_date):
    """
    Generates a mapping between dates and continuous contracts efficiently.
    Creates a table showing which underlying contract corresponds to each continuous contract on each date.
    """
    # 1. Load and Prepare Roll Calendar
    roll_calendar_df = load_roll_calendar(conn, root_symbol)
    if roll_calendar_df.empty:
        logger.error(f"Failed to load roll calendar for {root_symbol}. Aborting.")
        return
    # Ensure sorted by LTD
    roll_calendar_df = roll_calendar_df.sort_values('last_trading_day').reset_index(drop=True)
    logger.info(f"Using {len(roll_calendar_df)} roll calendar entries.")

    # 2. Generate Date Range
    calendar_name = 'CFE' # Explicitly use CFE for VX mapping
    date_range = None
    try:
        logger.info(f"Using trading calendar: {calendar_name} for date range generation.")
        # (Calendar loading logic - keeping the existing robust version)
        try:
            trading_calendar = get_calendar(calendar_name)
            schedule = trading_calendar.schedule(start_date=start_date, end_date=end_date)
            date_range = schedule.index.normalize()
            logger.info(f"Generated date range with {len(date_range)} trading days using exchange_calendars {calendar_name}.")
        except Exception as exc_cal_err:
            logger.warning(f"exchange_calendars failed for '{calendar_name}': {exc_cal_err}. Trying pandas_market_calendars.")
            try:
                trading_calendar = mcal.get_calendar(calendar_name)
                schedule = trading_calendar.schedule(start_date=start_date, end_date=end_date)
                date_range = schedule.index
                logger.info(f"Generated date range with {len(date_range)} trading days using pandas_market_calendars {calendar_name}.")
            except Exception as mcal_err:
                logger.error(f"Both calendar libraries failed for '{calendar_name}'. Error: {mcal_err}")
                raise # Re-raise if both fail
    except Exception as cal_err:
        logger.error(f"Error generating trading days using calendar '{calendar_name}': {cal_err}. Using business days as fallback.")
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        logger.warning(f"Falling back to {len(date_range)} business days.")

    if date_range is None or date_range.empty:
         logger.error("Failed to generate a valid date range. Aborting mapping.")
         return

    # 3. Initialize Active Contracts Queue
    # Use a deque for efficient popping from the left
    active_contracts_queue = deque(maxlen=num_contracts)
    next_contract_index = 0

    # Pre-fill the queue with the first num_contracts
    while len(active_contracts_queue) < num_contracts and next_contract_index < len(roll_calendar_df):
        active_contracts_queue.append({
            'code': roll_calendar_df.loc[next_contract_index, 'contract_code'],
            'ltd': roll_calendar_df.loc[next_contract_index, 'last_trading_day']
        })
        next_contract_index += 1
        
    if not active_contracts_queue:
        logger.error("Roll calendar has fewer than 1 contract. Cannot initialize mapping.")
        return

    # 4. Process Dates and Generate Mapping
    mapping_data = []
    logger.info("Processing dates to generate mapping...")

    for current_dt in date_range:
        current_date_ts = pd.Timestamp(current_dt).normalize() # Ensure normalized timestamp

        # Roll contracts if the current date is past the first contract's LTD
        # Continue rolling until the front contract is the active one for the current date
        while active_contracts_queue and current_date_ts > active_contracts_queue[0]['ltd']:
            # Roll occurred: remove the expired front contract
            expired_contract = active_contracts_queue.popleft()
            logger.debug(f"Rolled off {expired_contract['code']} (LTD: {expired_contract['ltd'].date()}) on date {current_date_ts.date()}")
            
            # Add the next available contract from the calendar to the end
            if next_contract_index < len(roll_calendar_df):
                next_contract = {
                     'code': roll_calendar_df.loc[next_contract_index, 'contract_code'],
                     'ltd': roll_calendar_df.loc[next_contract_index, 'last_trading_day']
                }
                active_contracts_queue.append(next_contract)
                logger.debug(f"Rolled on {next_contract['code']} (LTD: {next_contract['ltd'].date()})")
                next_contract_index += 1
            else:
                 logger.warning(f"Reached end of roll calendar while looking for next contract after {expired_contract['code']}.")
                 # The queue will now have less than num_contracts items

        # Record the mapping for the current date
        row = {'date': current_date_ts.date()} # Store only the date part
        active_codes = [c['code'] for c in active_contracts_queue] # Get current codes in order
        for i, contract_code in enumerate(active_codes):
            continuous_symbol = f"@{root_symbol}={i+1}01XN" # e.g., @VX=101XN, @VX=201XN
            row[continuous_symbol] = contract_code
            
        # If queue has fewer than num_contracts, remaining mappings are None (implicitly)
        mapping_data.append(row)

    logger.info("Finished processing dates. Converting mapping to DataFrame.")
    # 5. Convert to DataFrame
    df_mapping = pd.DataFrame(mapping_data)
    if df_mapping.empty:
         logger.warning("Generated mapping DataFrame is empty.")
         return
         
    # Set date as index temporarily for easier processing if needed, or keep as column
    # df_mapping = df_mapping.set_index('date')

    logger.info(f"Generated mapping DataFrame shape: {df_mapping.shape}")

    # 6. Save to Database (using the existing long-format transformation)
    try:
        conn.register('mapping_df_view', df_mapping) # Use a different view name
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS continuous_contract_mapping (
                date DATE,
                continuous_symbol VARCHAR,
                underlying_symbol VARCHAR,
                PRIMARY KEY (date, continuous_symbol)
            )
        """)
        
        continuous_cols = [col for col in df_mapping.columns if col.startswith('@')]
        logger.info(f"Found continuous columns: {continuous_cols}")
        
        # Clear old mappings for this root symbol first
        logger.info(f"Deleting existing mappings for symbols like '@{root_symbol}=%'")
        delete_pattern = f'@{root_symbol}=%' # Define the pattern
        conn.execute(f"DELETE FROM continuous_contract_mapping WHERE continuous_symbol LIKE ?", parameters=[delete_pattern])
        logger.info("Existing mappings deleted.")

        # Insert new mappings in long format
        # Use INSERT OR IGNORE to handle potential duplicate dates from fallback calendar
        sql_parts = []
        for col in continuous_cols:
             # Extract non-null mappings for this specific continuous symbol
             sql_parts.append(f"""
                 SELECT date, '{col}' as continuous_symbol, "{col}" as underlying_symbol
                 FROM mapping_df_view
                 WHERE "{col}" IS NOT NULL
             """)
        
        if not sql_parts:
             logger.warning("No valid mapping data found in DataFrame to insert.")
             return
             
        full_insert_query = " UNION ALL ".join(sql_parts)
        
        final_sql = f"""
            INSERT OR IGNORE INTO continuous_contract_mapping (date, continuous_symbol, underlying_symbol)
            {full_insert_query}
        """
        
        logger.info("Executing final batch insert into continuous_contract_mapping...")
        cursor = conn.execute(final_sql)
        # Safely check fetchall result before accessing
        fetchall_result = cursor.fetchall()
        # Check if list and first element exist before accessing [0][0]
        inserted_rows = fetchall_result[0][0] if fetchall_result and fetchall_result[0] else 0 
        logger.info(f"Finished inserting mappings. Rows affected/inserted (approx): {inserted_rows}")
        
        conn.unregister('mapping_df_view')
        logger.info(f"Successfully created/updated continuous contract mapping for {root_symbol}.")
        
    except Exception as e:
        logger.error(f"Error saving mapping table: {e}", exc_info=True)
        try:
            conn.unregister('mapping_df_view')
        except Exception as unreg_e:
            logger.error(f"Error unregistering view during error handling: {unreg_e}")

def main():
    parser = argparse.ArgumentParser(description='Create continuous contract mapping table.')
    parser.add_argument('--db-path', default='data/financial_data.duckdb', help='Path to DuckDB database')
    parser.add_argument('--root-symbol', default='VX', help='Root symbol (e.g., VX)')
    parser.add_argument('--num-contracts', type=int, default=9, help='Number of continuous contracts')
    parser.add_argument('--start-date', default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2050-12-31', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Set up logging file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"continuous_mapping_{args.root_symbol}_{timestamp_str}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting continuous contract mapping creation for {args.root_symbol}")
    logger.info(f"Parameters: start_date={args.start_date}, end_date={args.end_date}, num_contracts={args.num_contracts}")
    
    # Connect to database
    conn = connect_db(args.db_path, read_only=False)
    
    try:
        # Generate the mapping
        generate_continuous_mapping(
            conn, 
            args.root_symbol, 
            args.num_contracts, 
            args.start_date, 
            args.end_date
        )
        logger.info("Mapping creation completed successfully.")
    except Exception as e:
        logger.error(f"Error in mapping creation: {e}")
    finally:
        # Close the connection
        conn.close()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 