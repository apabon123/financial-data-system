import yaml
import pandas as pd
from pathlib import Path
import argparse
from datetime import timedelta, date, datetime
import duckdb
import pandas_market_calendars as mcal

# --- Configuration Loading ---

def load_config(config_path='config/market_symbols.yaml'):
    """Loads the market symbols configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Database Connection ---

# Determine the database path relative to the script location or workspace root
# Assumes the script is run from the workspace root or the path is adjusted accordingly
DEFAULT_DB_PATH = Path("data/financial_data.duckdb")
DEFAULT_WRITE_DB_PATH = Path("data/financial_data.duckdb") # For explicit write connection path

def get_db_connection(db_path=DEFAULT_DB_PATH, read_only=True):
    """Establishes a connection to the DuckDB database."""
    try:
        print(f"Attempting to connect to DB: {db_path}, read_only={read_only}")
        con = duckdb.connect(database=str(db_path), read_only=read_only)
        print(f"Successfully connected to DB: {db_path}")
        return con
    except Exception as e:
        print(f"Error connecting to database at {db_path}: {e}")
        raise

# --- Volume Data Retrieval ---

def get_historical_volume(con, contract_symbol, start_date, end_date):
    """
    Retrieves historical daily volume data for a specific contract from the database.

    Args:
        con: Active DuckDB database connection.
        contract_symbol (str): The specific futures contract symbol (e.g., 'ESU23').
        start_date (pd.Timestamp): Start date for volume data (inclusive).
        end_date (pd.Timestamp): End date for volume data (inclusive).

    Returns:
        pd.DataFrame: DataFrame with 'date' (index, pd.Timestamp) and 'volume' columns.
                      Returns an empty DataFrame with correct columns if no data or error.
    """
    query = f"""
    SELECT
        DATE_TRUNC('day', timestamp) AS date,
        MAX(volume) AS volume -- Assuming volume is consistent for the day, or take max
    FROM market_data
    WHERE symbol = ?
      AND timestamp >= ?::TIMESTAMP
      AND timestamp <= ?::TIMESTAMP
      AND (
            (interval_unit = 'day' AND interval_value = 1)
            OR
            (interval_unit = 'daily') -- Handle potential variations in naming
          )
    GROUP BY date
    ORDER BY date;
    """
    try:
        # Ensure dates are passed correctly to DuckDB
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d 23:59:59') # Include the whole end day

        df = con.execute(query, [contract_symbol, start_str, end_str]).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=['volume'], index=pd.DatetimeIndex([], name='date'))

        # Convert date column to DatetimeIndex
        # Ensure conversion handles potential timezone if DB returns aware timestamps
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None) # Ensure naive
        df = df.set_index('date')

        # Ensure volume is numeric, handle potential DBNull or None
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna(subset=['volume'])
        df['volume'] = df['volume'].astype(int)

        return df

    except Exception as e:
        print(f"Error querying volume for {contract_symbol} ({start_str} to {end_str}): {e}")
        return pd.DataFrame(columns=['volume'], index=pd.DatetimeIndex([], name='date'))


# --- Calendar and Expiry Logic ---

# Cache for market calendars to avoid reloading
_calendar_cache = {}

def get_contract_month_code(month_num: int) -> str | None:
    """Convert month number to futures contract month code."""
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    return month_codes.get(month_num)

def get_trading_calendar(calendar_name='CME'):
    """Gets a pandas_market_calendars object for the given exchange name."""
    if calendar_name not in _calendar_cache:
        try:
            print(f"Loading trading calendar: {calendar_name}")
            _calendar_cache[calendar_name] = mcal.get_calendar(calendar_name)
            print(f"Calendar {calendar_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading calendar '{calendar_name}': {e}. Falling back to NYSE.")
            # Fallback or raise error - choosing NYSE as a common default
            if 'NYSE' not in _calendar_cache:
                 _calendar_cache['NYSE'] = mcal.get_calendar('NYSE')
            _calendar_cache[calendar_name] = _calendar_cache['NYSE'] # Use NYSE as fallback
    return _calendar_cache[calendar_name]

def get_expiry_date(calendar, symbol_config, contract_year, contract_month_code):
    """
    Calculates the expiry date using pandas_market_calendars and rules from config.
    """
    rule = symbol_config.get('expiry_rule', {})
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    contract_month = month_map.get(contract_month_code)
    if not contract_month:
        raise ValueError(f"Invalid contract month code: {contract_month_code}")

    # Define start and end of the contract month for calendar searching
    month_start = pd.Timestamp(f'{contract_year}-{contract_month:02d}-01')
    # Get a sensible range around the month to find dates
    search_start = month_start - timedelta(days=5)
    search_end = month_start + timedelta(days=40) # Ensure we cover next month start potentially

    # Get valid trading days within the potential range
    valid_days = calendar.valid_days(start_date=search_start.strftime('%Y-%m-%d'),
                                     end_date=search_end.strftime('%Y-%m-%d'))

    # --- Apply Specific Expiry Rules ---

    # Rule: Nth specific weekday of the month (e.g., 3rd Friday)
    if rule.get('day_type') in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] and rule.get('day_number'):
        day_name_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}
        target_weekday = day_name_map[rule['day_type']]
        occurrence = int(rule['day_number'])

        # Find all instances of that weekday within the contract month
        month_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == contract_year)]
        target_days = month_days[month_days.weekday == target_weekday]

        if len(target_days) >= occurrence:
            return target_days[occurrence - 1] # 0-indexed
        else:
            print(f"Warning: Could not find {occurrence} occurrences of {rule['day_type']} in {contract_year}-{contract_month} for {symbol_config['base_symbol']}. Rule: {rule}")
            # Fallback or raise error? Returning dummy date for now.
            return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()

    # Rule: N business days before a specific day of the month (e.g., 3 days before 25th for CL)
    elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_day'):
        days_before = int(rule['days_before'])
        reference_day_num = int(rule['reference_day'])
        reference_date = pd.Timestamp(f'{contract_year}-{contract_month:02d}-{reference_day_num:02d}').normalize()

        # Find trading days before or on the reference date
        days_in_month_leading_to_ref = valid_days[valid_days <= reference_date]

        if len(days_in_month_leading_to_ref) >= days_before:
            # The expiry is the day that is `days_before` trading days prior to the reference date.
            # Example: 3 days before 25th. If 25th is a trading day, count back 3 incl 25th -> 23rd (if 23,24,25 are trading days)
            # Find the index of the last valid day <= reference_date
            # This logic might need refinement depending on exact definition (strictly before or can be the ref day?)
            # Let's assume 'before' means the Nth trading day counting backwards *from the day before* the reference date.

            # Find trading days strictly *before* the reference date
            days_strictly_before_ref = valid_days[valid_days < reference_date]
            if len(days_strictly_before_ref) >= days_before:
                 return days_strictly_before_ref[-days_before]
            else:
                 # Handle cases where reference day is very early or many holidays
                 print(f"Warning: Not enough trading days found before {reference_date.date()} to satisfy {days_before} days_before rule for {symbol_config['base_symbol']}. Rule: {rule}")
                 return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()
        else:
             print(f"Warning: Not enough trading days found leading up to {reference_date.date()} for {symbol_config['base_symbol']}. Rule: {rule}")
             return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()

    # Rule: N business days before the last business day of the month (e.g., GC)
    elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_point') == 'last_business_day':
        days_before = int(rule['days_before'])
        # Find all trading days in the contract month
        month_trading_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == contract_year)]

        if len(month_trading_days) >= days_before + 1: # Need at least N+1 days to count back N from last
            # The expiry day is the Nth day before the last business day.
            # E.g., days_before=3. Last day is index -1. Day before is -2. Day before that -3. Target is index -4.
            return month_trading_days[-(days_before + 1)]
        else:
            print(f"Warning: Not enough trading days in {contract_year}-{contract_month} to find {days_before} days before last business day for {symbol_config['base_symbol']}. Rule: {rule}")
            return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()

    # Rule: Special VX expiry (Wednesday 30 days prior to 3rd Friday of *following* month)
    elif rule.get('special_rule') == 'VX_expiry':
        # 1. Find the 3rd Friday of the *following* month (SPX options expiry)
        next_month = contract_month + 1
        next_year = contract_year
        if next_month > 12:
            next_month = 1
            next_year += 1

        next_month_start = pd.Timestamp(f'{next_year}-{next_month:02d}-01')
        search_end_spx = next_month_start + timedelta(days=35)
        valid_days_spx = calendar.valid_days(start_date=next_month_start.strftime('%Y-%m-%d'),
                                             end_date=search_end_spx.strftime('%Y-%m-%d'))

        next_month_days = valid_days_spx[(valid_days_spx.month == next_month) & (valid_days_spx.year == next_year)]
        fridays_next_month = next_month_days[next_month_days.weekday == 4] # 4 is Friday

        if len(fridays_next_month) >= 3:
            spx_expiry_friday = fridays_next_month[2] # 3rd Friday (0-indexed)
            # 2. Calculate target date: 30 calendar days before spx_expiry_friday
            target_date = spx_expiry_friday - timedelta(days=30)
            # 3. Find the Wednesday on or before the target date
            # Search backwards from target_date
            potential_expiry = target_date
            while potential_expiry.weekday() != 2: # 2 is Wednesday
                potential_expiry -= timedelta(days=1)
            # Ensure it's a trading day
            if calendar.is_session(potential_expiry.strftime('%Y-%m-%d')):
                 return potential_expiry.normalize()
            else:
                 # If that Wednesday is a holiday, go back to the previous trading day (usually Tuesday)
                 # Find the schedule for that day and get previous open
                 schedule = calendar.schedule(start_date=(potential_expiry - timedelta(days=5)).strftime('%Y-%m-%d'),
                                            end_date=potential_expiry.strftime('%Y-%m-%d'))
                 if not schedule.empty:
                     # Return the last valid trading day before the non-session Wednesday
                     return schedule.index[-1].normalize()
                 else:
                     # Should not happen if calendar is correct, but fallback
                      print(f"Warning: Could not find previous trading day for VX calculated expiry {potential_expiry.date()} for {symbol_config['base_symbol']}.")
                      return potential_expiry.normalize() # Return calculated Wed even if holiday?

        else:
            print(f"Warning: Could not find 3rd Friday in {next_year}-{next_month} for VX expiry calculation for {symbol_config['base_symbol']}. Rule: {rule}")
            return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()

    # --- Fallback for unhandled rules ---
    else:
        print(f"Warning: Expiry rule calculation not implemented for rule: {rule}. Returning dummy date.")
        return pd.Timestamp(f'{contract_year}-{contract_month:02d}-15').normalize()


def generate_contract_symbols(symbol_config):
    """Generates a list of historical contract symbols."""
    base = symbol_config['base_symbol']
    hist_contracts = symbol_config['historical_contracts']
    patterns = hist_contracts['patterns']
    start_year = hist_contracts['start_year']
    current_year = pd.Timestamp.now().year
    symbols = []
    month_map_inv = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                     7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

    for year in range(start_year, current_year + 2): # Generate a bit into the future
        yr_code = str(year)[-2:]
        for month_code in patterns:
             symbols.append(f"{base}{month_code}{yr_code}")
    return symbols

def _extract_year_month_code(contract_symbol_with_root):
    # This function is not used in the new version of the script
    # It's kept here for potential future use
    # The logic for extracting year and month code from a contract symbol is not provided in the new version
    # It's assumed to be unchanged from the original version
    # If you need to implement this function, you'll need to add the appropriate logic here
    # This is a placeholder and should be replaced with the actual implementation
    return None

# --- Database Table Handling ---

def _ensure_roll_dates_table(con):
    """Ensures the futures_roll_dates table exists in the database."""
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS futures_roll_dates (
                SymbolRoot VARCHAR NOT NULL,
                Contract VARCHAR NOT NULL, -- e.g., U23, Z23
                RollDate DATE NOT NULL,
                RollType VARCHAR NOT NULL, -- e.g., 'volume', 'open_interest', 'calendar'
                CalculationTimestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (SymbolRoot, Contract, RollType)
            );
        """)
        print("Table 'futures_roll_dates' verified/created successfully.")
    except Exception as e:
        print(f"Error creating/verifying futures_roll_dates table: {e}")
        raise # Re-raise the exception to stop execution if table can't be created

def save_roll_dates_to_db(df: pd.DataFrame, roll_type: str, db_con):
    """Saves the calculated roll dates DataFrame to the database.

    Args:
        df (pd.DataFrame): DataFrame with columns ['SymbolRoot', 'Contract', 'RollDate'].
        roll_type (str): The type of roll calculation (e.g., 'volume').
        db_con: Active DuckDB database connection.
    """
    if df.empty:
        print(f"No roll dates to save for RollType '{roll_type}'.")
        return

    df_to_save = df.copy()
    # Ensure RollType column is set from the argument for this specific save operation
    df_to_save['RollType'] = roll_type
    df_to_save['RollDate'] = pd.to_datetime(df_to_save['RollDate']).dt.date

    required_cols = ['SymbolRoot', 'Contract', 'RollDate', 'RollType']
    
    # Ensure all required columns exist, add if not (though they should be from calculation)
    for col in required_cols:
        if col not in df_to_save.columns:
            if col == 'RollType': # Should be set above
                 print(f"Warning: 'RollType' column was missing before explicit set in save_roll_dates_to_db for {roll_type}.")
            else:
                 print(f"Warning: Column '{col}' was missing in DataFrame for RollType '{roll_type}'. This might indicate an issue.")
                 # df_to_save[col] = None # Or handle more gracefully

    df_to_save = df_to_save[required_cols]


    table_name = "futures_roll_dates"
    temp_view_name = f"temp_{table_name}_upsert_view"

    try:
        db_con.register(temp_view_name, df_to_save)
        
        sql = f"""
        INSERT INTO {table_name} (SymbolRoot, Contract, RollDate, RollType)
        SELECT SymbolRoot, Contract, RollDate, RollType FROM {temp_view_name}
        ON CONFLICT (SymbolRoot, Contract, RollType) DO UPDATE SET
            RollDate = EXCLUDED.RollDate
        """
        
        db_con.execute(sql)
        db_con.unregister(temp_view_name)
        db_con.commit()
        print(f"Successfully upserted {len(df_to_save)} rows into {table_name} with RollType '{roll_type}'.")

    except Exception as e:
        print(f"Error saving roll dates to database: {e}")
        db_con.rollback()
        import traceback
        traceback.print_exc()

# --- Main Calculation Logic ---

def calculate_volume_roll(symbol_root, config, db_con, num_days_before_expiry=5):
    """
    Calculates the volume-based roll dates for a given futures symbol root.

    Args:
        symbol_root (str): The base symbol (e.g., 'ES', 'NQ').
        config (dict): The loaded market symbols configuration.
        db_con: Active DuckDB database connection.
        num_days_before_expiry (int): Number of trading days before expiry to check volume.

    Returns:
        pd.DataFrame: DataFrame with columns ['SymbolRoot', 'Contract', 'RollDate'].
    """
    futures_config = next((item for item in config.get('futures', []) if item['base_symbol'] == symbol_root), None)
    if not futures_config:
        raise ValueError(f"Symbol root '{symbol_root}' not found in futures configuration.")

    calendar_name = futures_config.get('calendar', 'CME') # Get calendar name from config
    trading_calendar = get_trading_calendar(calendar_name)
    contract_symbols = generate_contract_symbols(futures_config)
    roll_dates = []

    for i in range(len(contract_symbols) - 1):
        first_contract = contract_symbols[i]
        second_contract = contract_symbols[i+1]

        print(f"Processing roll from {first_contract} to {second_contract}...")

        try:
            first_month_code = first_contract[-3]
            first_year_short = first_contract[-2:]
            current_century = (pd.Timestamp.now().year // 100) * 100
            short_year_int = int(first_year_short)
            first_year = (current_century - 100 + short_year_int) if short_year_int > (pd.Timestamp.now().year % 100 + 10) else (current_century + short_year_int)

            # Get expiry date using the new function and the specific calendar
            expiry_date = get_expiry_date(trading_calendar, futures_config, first_year, first_month_code)
            print(f"  Expiry Date for {first_contract}: {expiry_date.date()}")

            # Define the check window using the trading calendar
            # Get schedule around expiry to find trading days
            schedule_start = expiry_date - timedelta(days=num_days_before_expiry + 15) # Look back enough days
            schedule_end = expiry_date - timedelta(days=1) # End day before expiry

            if schedule_start > schedule_end:
                 print(f"  Skipping {first_contract}: Calculation range invalid ({schedule_start.date()} > {schedule_end.date()}).")
                 continue

            schedule = trading_calendar.schedule(start_date=schedule_start.strftime('%Y-%m-%d'),
                                               end_date=schedule_end.strftime('%Y-%m-%d'))

            if len(schedule) < num_days_before_expiry:
                 print(f"  Skipping {first_contract}: Not enough trading days ({len(schedule)}) found before expiry {expiry_date.date()} in schedule range {schedule_start.date()} to {schedule_end.date()}.")
                 continue

            # The check dates are the last `num_days_before_expiry` from the schedule
            check_dates = schedule.index[-num_days_before_expiry:]
            check_start_date = check_dates.min()
            check_end_date = check_dates.max() # Use the actual last day in the window

            # --- Added Check: Stop if check window is in the future ---
            today = pd.Timestamp.now().normalize() # Get today's date, ignore time
            if check_start_date > today:
                print(f"  Check window start date {check_start_date.date()} is after today {today.date()}. Stopping further processing.")
                break # Stop the loop for this symbol_root
            # -----------------------------------------------------------

            print(f"  Volume Check Window: {check_start_date.date()} to {check_end_date.date()}")

            # Fetch volume data using the database connection
            vol1 = get_historical_volume(db_con, first_contract, check_start_date, check_end_date)
            vol2 = get_historical_volume(db_con, second_contract, check_start_date, check_end_date)

            if vol1.empty and vol2.empty:
                 print(f"  Skipping {first_contract}: No volume data found for either contract in the check window.")
                 continue
            # Handle cases where one might be empty
            if vol1.empty:
                 print(f"  Note: {first_contract} has no volume data in check window.")
            if vol2.empty:
                 print(f"  Note: {second_contract} has no volume data in check window.")

            # Combine volumes, reindex to ensure all check_dates are present
            combined_vol = pd.DataFrame(index=check_dates) # Use exact check dates from calendar
            combined_vol['vol1'] = vol1['volume']
            combined_vol['vol2'] = vol2['volume']
            combined_vol = combined_vol.fillna(0) # Fill missing volume with 0

            # Find the roll date: Iterate backwards through the check_dates
            roll_date_found = None
            for date in sorted(combined_vol.index, reverse=True):
                vol_f = combined_vol.loc[date, 'vol1']
                vol_s = combined_vol.loc[date, 'vol2']

                # Roll if second contract volume is greater than first contract volume
                if vol_s > vol_f:
                    roll_date_found = date
                    print(f"  Volume Crossover: {second_contract}({vol_s}) > {first_contract}({vol_f}) on {date.date()}")
                    break

            if roll_date_found:
                 contract_month_year = f"{first_month_code}{str(first_year)[-2:]}"
                 roll_dates.append({
                     'SymbolRoot': symbol_root,
                     'Contract': contract_month_year,
                     'RollDate': roll_date_found.strftime('%Y-%m-%d')
                 })
                 print(f"  >>> Roll Date Determined: {roll_date_found.date()} (rolling out of {first_contract})")
            else:
                 print(f"  No volume crossover found in the window for {first_contract} -> {second_contract}.")

        except Exception as e:
            print(f"Error processing {first_contract} -> {second_contract}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    return pd.DataFrame(roll_dates)

def calculate_calendar_offset_rolls(symbol_root: str, config: dict, db_con, offsets: list[int]) -> pd.DataFrame:
    """
    Calculates roll dates based on N business days before the last trading day from futures_roll_calendar.

    Args:
        symbol_root (str): The root symbol (e.g., 'ES').
        config (dict): The market symbols configuration dictionary.
        db_con: Active DuckDB database connection.
        offsets (list[int]): List of N days to offset (e.g., [1, 2, 3] for 1, 2, 3 days before).

    Returns:
        pd.DataFrame: DataFrame with columns [SymbolRoot, Contract, RollDate, RollType].
                      RollType will be like '01X', '02X', etc.
    """
    print(f"Calculating calendar offset rolls for {symbol_root} with offsets: {offsets}")
    all_calendar_rolls = []

    # Get the specific configuration for the futures of the given root symbol
    # The config['futures'] is a LIST of configurations.
    symbol_config = None
    if 'futures' in config and isinstance(config['futures'], list):
        for future_conf in config['futures']:
            if future_conf.get('base_symbol') == symbol_root:
                symbol_config = future_conf
                break
    
    if not symbol_config:
        print(f"Warning: No futures configuration found for base_symbol '{symbol_root}' in the list under 'futures' in market_symbols.yaml. Skipping calendar rolls.")
        return pd.DataFrame()

    calendar_name = symbol_config.get('calendar', 'CME')
    trading_calendar = get_trading_calendar(calendar_name)

    query = f"""
    SELECT contract_code, last_trading_day, year, month
    FROM futures_roll_calendar
    WHERE root_symbol = ?
    ORDER BY last_trading_day;
    """
    try:
        calendar_entries_df = db_con.execute(query, [symbol_root]).fetchdf()
        if calendar_entries_df.empty:
            print(f"No entries found in futures_roll_calendar for {symbol_root}. Cannot calculate calendar offset rolls.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error querying futures_roll_calendar for {symbol_root}: {e}")
        return pd.DataFrame()

    for _, row in calendar_entries_df.iterrows():
        try:
            last_trading_day_dt = pd.to_datetime(row['last_trading_day']).normalize()
            # Expect year and month to come from the DB query
            contract_year = int(row['year'])
            contract_month_num = int(row['month'])

            month_code = get_contract_month_code(contract_month_num)
            if not month_code:
                print(f"Warning: Could not get month code for month number {contract_month_num}. Skipping contract {row['contract_code']}.")
                continue
            
            # Construct Contract field like "H24"
            contract_identifier = f"{month_code}{str(contract_year)[-2:]}"

            # Fetch a schedule around the last_trading_day to find its index for safe offsetting
            # Ensure the schedule covers enough range for offsets
            schedule_start = last_trading_day_dt - timedelta(days=max(offsets) + 15) # Extra buffer
            schedule_end = last_trading_day_dt + timedelta(days=5) # Small buffer past
            
            schedule = trading_calendar.schedule(start_date=schedule_start.strftime('%Y-%m-%d'), 
                                                 end_date=schedule_end.strftime('%Y-%m-%d'))
            
            if last_trading_day_dt not in schedule.index:
                # This can happen if last_trading_day from DB isn't an actual trading day on the calendar
                # Try to find the closest preceding trading day
                print(f"Warning: last_trading_day {last_trading_day_dt.date()} for {row['contract_code']} is not on the {calendar_name} trading calendar. Attempting to find prior trading day.")
                # Get valid days up to last_trading_day_dt
                valid_days_up_to = trading_calendar.valid_days(start_date=schedule_start.strftime('%Y-%m-%d'), end_date=last_trading_day_dt.strftime('%Y-%m-%d'))
                if not valid_days_up_to.empty:
                    actual_ref_date = valid_days_up_to[-1] # Closest actual trading day
                    print(f"Using {actual_ref_date.date()} as reference for {row['contract_code']} instead of {last_trading_day_dt.date()}.")
                else: # Should not happen if schedule_start is reasonable
                    print(f"Error: Could not find any valid trading day before or on {last_trading_day_dt.date()} for {row['contract_code']}. Skipping.")
                    continue
            else:
                actual_ref_date = last_trading_day_dt

            # Find the index of actual_ref_date in the full schedule (not just the small one we fetched for safety)
            # For simplicity, we'll use the small schedule, assuming it's sufficient
            # Or, more robustly, use date_offset if available and reliable.
            # pandas_market_calendars schedule is a DatetimeIndex.
            
            try:
                # Get the series of trading sessions (dates)
                session_dates = schedule.index.normalize()
                loc = session_dates.get_loc(actual_ref_date)
            except KeyError:
                print(f"Error: {actual_ref_date.date()} (derived from {row['last_trading_day']}) not found in fetched {calendar_name} schedule for {row['contract_code']}. Dates available: {session_dates[:3]}...{session_dates[-3:]}. Skipping contract.")
                continue


            for offset in offsets:
                if loc >= offset:
                    roll_date = session_dates[loc - offset]
                    roll_type_str = f"{offset:02d}X"
                    all_calendar_rolls.append({
                        'SymbolRoot': symbol_root,
                        'Contract': contract_identifier,
                        'RollDate': roll_date.to_pydatetime().date(), # Store as python date
                        'RollType': roll_type_str
                    })
                else:
                    print(f"Warning: Not enough trading days before {actual_ref_date.date()} to offset by {offset} days for {row['contract_code']}. Skipping this offset.")
        
        except Exception as e:
            print(f"Error processing contract {row.get('contract_code', 'N/A')} for calendar offset rolls: {e}")
            import traceback
            traceback.print_exc()


    return pd.DataFrame(all_calendar_rolls)

def main():
    parser = argparse.ArgumentParser(description="Calculate and store futures roll dates based on volume or calendar rules.")
    parser.add_argument("--symbol-roots", type=str, help="Comma-separated list of symbol roots to process for volume rolls (e.g., ES,NQ).")
    parser.add_argument("--config-path", type=str, default="config/market_symbols.yaml", help="Path to the market symbols configuration file.")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to the DuckDB database file (for read operations).")
    parser.add_argument("--db-write-path", type=str, default=str(DEFAULT_WRITE_DB_PATH), help="Path to the DuckDB database file (for write operations).")
    parser.add_argument("--write-to-db", action="store_true", help="If set, writes calculated roll dates to the database.")
    parser.add_argument("--days-before-expiry-volume", type=int, default=5, help="Number of trading days before theoretical expiry to scan for volume roll.")
    
    # Arguments for calendar-based rolls
    parser.add_argument("--calculate-calendar-rolls", action="store_true", help="If set, calculates calendar-offset based roll dates.")
    parser.add_argument("--calendar-roll-offsets", type=str, default="1,2,3", help="Comma-separated list of N days before expiry to calculate (e.g., '1,2,3' for 01X, 02X, 03X).")
    parser.add_argument("--symbol-roots-calendar", type=str, help="Comma-separated list of symbol roots for calendar rolls (e.g., ES,NQ). Default is same as --symbol-roots if not provided and calendar rolls are active.")

    args = parser.parse_args()

    config = load_config(args.config_path)
    db_connection = None # Single connection object

    try:
        # Determine connection mode based on whether writing is needed
        read_only_mode = not args.write_to_db
        db_path_to_use = args.db_write_path if args.write_to_db else args.db_path
        
        print(f"Connecting to database: {db_path_to_use}, Read-only: {read_only_mode}")
        db_connection = get_db_connection(db_path_to_use, read_only=read_only_mode)

        if args.write_to_db:
            _ensure_roll_dates_table(db_connection) # Ensure table exists before writing

        # Volume Roll Calculation (existing logic)
        if args.symbol_roots:
            symbol_roots_volume = [s.strip().upper() for s in args.symbol_roots.split(',')]
            print(f"Processing volume rolls for: {symbol_roots_volume}")
            for symbol_root in symbol_roots_volume:
                print(f"--- Calculating Volume Roll for {symbol_root} ---")
                volume_roll_df = calculate_volume_roll(symbol_root, config, db_connection, args.days_before_expiry_volume)
                if not volume_roll_df.empty:
                    if args.write_to_db:
                        save_roll_dates_to_db(volume_roll_df, 'volume', db_connection)
                    else:
                        print(f"Volume rolls for {symbol_root} (not writing to DB):\n{volume_roll_df}")
                else:
                    print(f"No volume roll dates calculated for {symbol_root}.")
        
        # Calendar Roll Calculation (new logic)
        if args.calculate_calendar_rolls:
            symbol_roots_cal = [] # Initialize
            if args.symbol_roots_calendar:
                symbol_roots_cal = [s.strip().upper() for s in args.symbol_roots_calendar.split(',')]
            elif args.symbol_roots: # Fallback to general symbols if specific calendar symbols not given
                print("Using --symbol-roots for calendar roll calculation as --symbol-roots-calendar not specified.")
                symbol_roots_cal = [s.strip().upper() for s in args.symbol_roots.split(',')]
            
            if not symbol_roots_cal:
                print("No symbol roots specified for calendar roll calculation. Use --symbol-roots-calendar or --symbol-roots when --calculate-calendar-rolls is active.")
            else: # symbol_roots_cal is populated, proceed with offset parsing and calculation
                offsets = [] # Initialize offsets
                try:
                    offsets = [int(offset.strip()) for offset in args.calendar_roll_offsets.split(',')]
                    if not all(o > 0 for o in offsets):
                        raise ValueError("Calendar roll offsets must be positive integers.")
                except ValueError as e:
                    print(f"Error: Invalid format for --calendar-roll-offsets. Expected comma-separated positive integers (e.g., '1,2,3'). {e}")
                    # offsets will remain empty, so the next 'if offsets:' block won't run

                if offsets: # Proceed only if offsets were successfully parsed and are not empty
                    print(f"Processing calendar offset rolls for: {symbol_roots_cal} with offsets {offsets}")
                    for symbol_root in symbol_roots_cal:
                        print(f"--- Calculating Calendar Offset Rolls for {symbol_root} ---")
                        # Pass the single connection for querying futures_roll_calendar
                        calendar_df = calculate_calendar_offset_rolls(symbol_root, config, db_connection, offsets)
                        
                        if not calendar_df.empty:
                            if args.write_to_db:
                                # Iterate through each RollType in the df and save separately
                                # because save_roll_dates_to_db expects a single roll_type argument
                                # to set for the entire batch it saves.
                                for roll_type_val in calendar_df['RollType'].unique():
                                    df_subset = calendar_df[calendar_df['RollType'] == roll_type_val]
                                    # The 'RollType' column is already correctly set in df_subset
                                    # The `roll_type_val` argument to save_roll_dates_to_db will ensure
                                    # the print messages and any internal logic dependent on it are correct.
                                    print(f"Saving {len(df_subset)} rows for {symbol_root} with RollType '{roll_type_val}'")
                                    save_roll_dates_to_db(df_subset, roll_type_val, db_connection)
                            else:
                                print(f"Calendar offset rolls for {symbol_root} (not writing to DB):\n{calendar_df}")
                        else:
                            print(f"No calendar offset roll dates calculated for {symbol_root}.")
        # This implicit else (for `if args.calculate_calendar_rolls:`) means no calendar rolls are processed if flag is false.
        
        if not args.symbol_roots and not args.calculate_calendar_rolls:
            print("No action specified. Use --symbol-roots for volume rolls or --calculate-calendar-rolls for calendar rolls.")

    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_connection:
            db_connection.close()
            print("Database connection closed.") 

if __name__ == "__main__":
    main() 