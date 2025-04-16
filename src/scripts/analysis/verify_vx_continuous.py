#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verifies the generated VX continuous futures contracts for data quality issues.

Checks performed:
1.  Data points on Sundays.
2.  Large day-over-day price gaps (close-to-close).
3.  Missing trading days (date gaps, excluding weekends).
"""

import os
import sys
import logging
import argparse
import duckdb
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"
PRICE_GAP_THRESHOLD = 0.20 # Percentage (e.g., 0.20 for 20%)
CONFIG_PATH = PROJECT_ROOT / "config" / "market_symbols.yaml"

def connect_db(db_file: Path = DB_PATH, read_only: bool = True) -> Optional[duckdb.DuckDBPyConnection]:
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=str(db_file), read_only=read_only)
        logger.info(f"Successfully connected to database: {db_file} (Read-Only)")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database {db_file}: {e}")
        return None

def get_continuous_contracts(conn: duckdb.DuckDBPyConnection, symbol_prefix: str = 'VXc') -> List[str]:
    """Gets a list of continuous contract symbols from the database."""
    query = "SELECT DISTINCT symbol FROM continuous_contracts WHERE symbol LIKE ? ORDER BY symbol"
    try:
        symbols = conn.execute(query, [f"{symbol_prefix}%"]).df()['symbol'].tolist()
        logger.info(f"Found continuous contracts: {symbols}")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching continuous contract symbols: {e}")
        return []

def load_config(config_path: Path = CONFIG_PATH) -> Optional[Dict]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config file: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return None

def get_holidays(config: Dict, calendar_name: str = "NYSE") -> Set[date]:
    """Extracts holidays for a given calendar from the config."""
    holidays = set()
    try:
        holiday_config = config.get('settings', {}).get('holidays', {}).get(calendar_name, {})
        fixed_dates = holiday_config.get('fixed_dates', [])
        relative_dates = holiday_config.get('relative_dates', [])
        current_year = date.today().year # Use a reasonable range if needed

        # Process fixed date holidays (assuming they apply every year within a reasonable range)
        # A more robust solution would consider the year range of the data
        for year in range(current_year - 20, current_year + 5): # Example range
            for fixed_date in fixed_dates:
                month, day = map(int, fixed_date.split('-'))
                try:
                    holidays.add(date(year, month, day))
                except ValueError: # Handle cases like Feb 29
                    pass

        # Process relative date holidays (complex - requires dateutil or similar logic)
        # Basic implementation for common cases
        weekday_map = {"monday": MO, "tuesday": TU, "wednesday": WE, "thursday": TH, "friday": FR, "saturday": SA, "sunday": SU}
        for year in range(current_year - 20, current_year + 5): # Example range
            for rel_date in relative_dates:
                month = rel_date['month']
                day_type = rel_date['day_type'].lower()
                occurrence = rel_date['occurrence']
                weekday = weekday_map.get(day_type)

                if not weekday:
                    # Handle 'Good Friday' or other complex relative dates if needed
                    if rel_date['name'] == 'Good Friday':
                         # Requires Easter calculation - skipping for simplicity for now
                        pass
                    continue

                # Use relativedelta to find the Nth weekday
                if occurrence > 0: # Nth weekday from start
                    dt = date(year, month, 1) + relativedelta(weekday=weekday(occurrence))
                else: # Nth weekday from end (e.g., last Monday)
                    # Find first day of *next* month, subtract 1 day to get last day of current month
                    next_month_start = date(year, month, 1) + relativedelta(months=1)
                    last_day_of_month = next_month_start - timedelta(days=1)
                    # Find the Nth last weekday
                    dt = last_day_of_month + relativedelta(weekday=weekday(occurrence)) # occurrence is negative

                # Ensure the calculated date is still in the correct month
                if dt.month == month:
                    holidays.add(dt)

        logger.info(f"Loaded {len(holidays)} holidays for calendar '{calendar_name}'")
        return holidays

    except Exception as e:
        logger.error(f"Error processing holidays from config: {e}")
        return set()

def get_nth_weekday_of_month(year: int, month: int, weekday_int: int, n: int) -> date:
    """
    Gets the date of the nth occurrence of a specific weekday in a given month and year.
    weekday_int: 0=Mon, 1=Tue, ..., 6=Sun
    n: 1 for 1st, 2 for 2nd, etc.
    """
    first_day_of_month = date(year, month, 1)
    day_of_week_first = first_day_of_month.weekday() # Monday is 0, Sunday is 6

    # Calculate days to add to reach the first target weekday
    days_to_add = (weekday_int - day_of_week_first + 7) % 7
    first_occurrence_date = first_day_of_month + timedelta(days=days_to_add)

    # Add (n-1) weeks to get the nth occurrence
    nth_occurrence_date = first_occurrence_date + timedelta(weeks=n - 1)

    # Check if the calculated date is still in the same month
    if nth_occurrence_date.month != month:
        raise ValueError(f"The {n}th weekday {weekday_int} does not exist in {year}-{month:02d}")

    return nth_occurrence_date

def get_previous_weekday(target_date: date, weekday_int: int) -> date:
    """Gets the date of the most recent specified weekday before or on the target_date."""
    days_to_subtract = (target_date.weekday() - weekday_int + 7) % 7
    return target_date - timedelta(days=days_to_subtract)

def adjust_for_holiday(expiry_date: date, holidays: Set[date], direction: int = -1) -> date:
    """Adjusts a date backward (default) or forward if it falls on a holiday."""
    adjusted_date = expiry_date
    while adjusted_date in holidays or adjusted_date.weekday() >= 5: # Skip weekends too
        adjusted_date += timedelta(days=direction)
    return adjusted_date

def calculate_expected_vx_expiry_dates(start_date: date, end_date: date, holidays: Set[date]) -> Dict[date, str]:
    """
    Calculates the expected VIX futures expiry dates within a range.
    Rule: Wednesday preceding the 3rd Friday of the month, adjusted for holidays.
    Returns a dictionary mapping expiry_date -> contract_month_str (e.g., '2010-01')
    """
    expected_expiry_dates = {}
    current_date = date(start_date.year, start_date.month, 1)

    logger.info(f"Calculating expected VIX expiry dates (Wed before 3rd Fri + holidays) from {start_date} to {end_date}...")

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        try:
            # Find the 3rd Friday of the month (Friday is 4 in weekday())
            third_friday = get_nth_weekday_of_month(year, month, 4, 3)

            # Find the Wednesday immediately preceding the 3rd Friday (Wednesday is 2 in weekday())
            expected_expiry = get_previous_weekday(third_friday - timedelta(days=1), 2) # Look back from Thursday before 3rd Fri

            # Adjust if the calculated Wednesday is a holiday or weekend (roll back)
            # Use the *same* holiday adjustment logic as the generation script
            final_expiry = adjust_for_holiday(expected_expiry, holidays, direction=-1)

            if final_expiry >= start_date: # Only include if within our overall range
                 contract_month_str = f"{year}-{month:02d}"
                 expected_expiry_dates[final_expiry] = contract_month_str
                 logger.debug(f"[{contract_month_str}] 3rd Fri={third_friday}, Exp Wed={expected_expiry}, Final Exp={final_expiry}")

        except ValueError as e:
             logger.warning(f"Could not calculate expiry for {year}-{month:02d}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calculating expiry for {year}-{month:02d}: {e}")

        # Move to the next month
        current_date += relativedelta(months=1)

    logger.info(f"Calculated {len(expected_expiry_dates)} expected expiry dates (using robust rule).")
    return expected_expiry_dates

def get_data_range(conn: duckdb.DuckDBPyConnection, symbol: str) -> Tuple[Optional[date], Optional[date]]:
    """Gets the min and max date for a symbol in continuous_contracts."""
    query = "SELECT MIN(timestamp::DATE), MAX(timestamp::DATE) FROM continuous_contracts WHERE symbol = ?"
    try:
        result = conn.execute(query, [symbol]).fetchone()
        if result and result[0] and result[1]:
            return result[0], result[1]
        else:
            logger.warning(f"[{symbol}] Could not determine data range.")
            return None, None
    except Exception as e:
        logger.error(f"[{symbol}] Error fetching data range: {e}")
        return None, None

def check_sunday_data(conn: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    """Checks for data points falling on a Sunday."""
    logger.info(f"[{symbol}] Checking for data on Sundays...")
    query = f"""
    SELECT timestamp, symbol, close
    FROM continuous_contracts
    WHERE symbol = ?
      AND EXTRACT(DOW FROM timestamp) = 0 -- 0 = Sunday in DuckDB's EXTRACT(DOW)
    ORDER BY timestamp;
    """
    try:
        sunday_data = conn.execute(query, [symbol]).fetchdf()
        if not sunday_data.empty:
            logger.warning(f"[{symbol}] Found {len(sunday_data)} data points on Sundays:")
            # Log details of the first few issues
            for _, row in sunday_data.head().iterrows():
                logger.warning(f"  - {row['timestamp'].date()}")
        else:
            logger.info(f"[{symbol}] No data found on Sundays.")
        return sunday_data
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for Sunday data: {e}")
        return pd.DataFrame()

def check_price_gaps(conn: duckdb.DuckDBPyConnection, symbol: str, threshold: float) -> pd.DataFrame:
    """Checks for large day-over-day percentage changes in the closing price."""
    logger.info(f"[{symbol}] Checking for price gaps larger than {threshold:.0%}...")
    query = f"""
    WITH lagged_data AS (
        SELECT
            timestamp,
            close,
            LAG(close, 1) OVER (ORDER BY timestamp) as prev_close
        FROM continuous_contracts
        WHERE symbol = ?
    )
    SELECT
        timestamp,
        close,
        prev_close,
        (close - prev_close) / prev_close as pct_change
    FROM lagged_data
    WHERE prev_close IS NOT NULL
      AND ABS((close - prev_close) / prev_close) > ?
    ORDER BY timestamp;
    """
    try:
        gaps_df = conn.execute(query, [symbol, threshold]).fetchdf()
        if not gaps_df.empty:
            logger.warning(f"[{symbol}] Found {len(gaps_df)} large price gaps (threshold > {threshold:.0%}):")
            # Log details of the first few issues
            for _, row in gaps_df.head().iterrows():
                 logger.warning(f"  - {row['timestamp'].date()}: Close={row['close']:.2f}, PrevClose={row['prev_close']:.2f}, Change={row['pct_change']:.1%}")
        else:
            logger.info(f"[{symbol}] No large price gaps found.")
        return gaps_df
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for price gaps: {e}")
        return pd.DataFrame()

def check_date_gaps(conn: duckdb.DuckDBPyConnection, symbol: str) -> List[pd.Timestamp]:
    """Checks for missing trading days (excluding weekends)."""
    logger.info(f"[{symbol}] Checking for missing trading days (date gaps)...")
    query = f"""
    SELECT DISTINCT timestamp::DATE as date
    FROM continuous_contracts
    WHERE symbol = ?
    ORDER BY 1; -- Order by the first selected column (timestamp::DATE)
    """
    try:
        dates_df = conn.execute(query, [symbol]).fetchdf()
        if dates_df.empty:
            logger.warning(f"[{symbol}] No data found to check for date gaps.")
            return []

        dates_df['date'] = pd.to_datetime(dates_df['date'])
        # Create a complete date range from min to max date
        min_date = dates_df['date'].min()
        max_date = dates_df['date'].max()
        all_days = pd.date_range(start=min_date, end=max_date, freq='B') # 'B' is business day frequency

        # Find missing dates
        missing_dates = all_days.difference(dates_df['date']).tolist()

        if missing_dates:
            logger.warning(f"[{symbol}] Found {len(missing_dates)} missing trading days (date gaps):")
            # Log the first few missing dates
            for missing_date in missing_dates[:5]:
                logger.warning(f"  - {missing_date.date()}")
            if len(missing_dates) > 5:
                logger.warning(f"  - ... and {len(missing_dates) - 5} more")
        else:
            logger.info(f"[{symbol}] No missing trading days found.")
        return missing_dates
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for date gaps: {e}")
        return []

def check_rollover_consistency(conn: duckdb.DuckDBPyConnection, symbol: str, expected_expiry_dates: Dict[date, str]) -> Tuple[int, int]:
    """
    Checks if the actual rollovers in the continuous contract match expected expiry dates.
    Rollover is defined as the day the underlying_symbol changes.
    Expected rollover day = The calculated expiry day itself.
    Returns (number of matches, number of mismatches).
    """
    logger.info(f"[{symbol}] Checking rollover consistency against expected expiry dates...")
    query = f"""
    WITH lagged_data AS (
        SELECT
            timestamp::DATE as date,
            underlying_symbol,
            LAG(underlying_symbol, 1) OVER (ORDER BY timestamp) as prev_underlying_symbol
        FROM continuous_contracts
        WHERE symbol = ?
    )
    SELECT date, underlying_symbol, prev_underlying_symbol
    FROM lagged_data
    WHERE underlying_symbol != prev_underlying_symbol
      AND prev_underlying_symbol IS NOT NULL -- Exclude the very first record
    ORDER BY date;
    """
    matches = 0
    mismatches = 0
    # Expected rollover happens ON the expiry date according to generation script's logic
    expected_rollover_dates_on_expiry = {exp_date: month_str for exp_date, month_str in expected_expiry_dates.items()}

    try:
        actual_rollovers = conn.execute(query, [symbol]).fetchdf()
        # Ensure we are comparing date objects
        actual_rollover_dates = set(pd.to_datetime(actual_rollovers['date']).dt.date)

        if actual_rollovers.empty:
            logger.warning(f"[{symbol}] No actual rollovers found based on underlying_symbol changes.")
            return 0, len(expected_rollover_dates_on_expiry)

        # Ensure keys are date objects
        expected_dates_set = set(expected_rollover_dates_on_expiry.keys())
        # Verify keys are date objects (optional debug)
        # if not all(isinstance(d, date) for d in expected_dates_set):
        #     logger.error(f"[{symbol}] ERROR: Expected expiry dates are not all date objects!")

        # Find matches
        matched_dates = actual_rollover_dates.intersection(expected_dates_set)
        matches = len(matched_dates)

        # Find mismatches
        unexpected_actual = actual_rollover_dates - expected_dates_set
        missing_expected = expected_dates_set - actual_rollover_dates
        mismatches = len(unexpected_actual) + len(missing_expected)

        logger.info(f"[{symbol}] Rollover comparison (expecting roll ON expiry): {matches} matches, {mismatches} mismatches.")

        if matched_dates:
            logger.info(f"[{symbol}] Matched rollover dates (Actual vs Expected Expiry):")
            # Convert matched_dates back to list of datetime.date for sorting/lookup if needed
            sorted_matched_dates = sorted(list(matched_dates))
            for dt in sorted_matched_dates:
                 exp_month = expected_rollover_dates_on_expiry.get(dt, "N/A")
                 # Find the original row in actual_rollovers DataFrame
                 row_matches = actual_rollovers[pd.to_datetime(actual_rollovers['date']).dt.date == dt]
                 if not row_matches.empty:
                     row = row_matches.iloc[0]
                     logger.info(f"  - {dt}: Matched expiry. Changed from {row['prev_underlying_symbol']} to {row['underlying_symbol']} (Contract Month: {exp_month})")
                 else:
                      logger.warning(f"  - {dt}: Matched expiry, but could not find original row details? (Contract Month: {exp_month})")

        if unexpected_actual:
            logger.warning(f"[{symbol}] Found {len(unexpected_actual)} actual rollovers NOT on an expected expiry date:")
            sorted_unexpected_actual = sorted(list(unexpected_actual))
            for dt in sorted_unexpected_actual:
                 row_matches = actual_rollovers[pd.to_datetime(actual_rollovers['date']).dt.date == dt]
                 if not row_matches.empty:
                     row = row_matches.iloc[0]
                     logger.warning(f"  - {dt}: Changed from {row['prev_underlying_symbol']} to {row['underlying_symbol']}")
                 else:
                     logger.warning(f"  - {dt}: Unexpected rollover, but could not find original row details?")

        if missing_expected:
            logger.warning(f"[{symbol}] Did not find rollovers on {len(missing_expected)} expected expiry dates:")
            sorted_missing_expected = sorted(list(missing_expected))
            for dt in sorted_missing_expected:
                 exp_month = expected_rollover_dates_on_expiry.get(dt, "N/A")
                 logger.warning(f"  - Expected rollover ON {dt} (Expiry for contract month {exp_month})")

    except Exception as e:
        logger.error(f"[{symbol}] Error checking rollover consistency: {e}")
        import traceback
        traceback.print_exc() # Add traceback for detailed errors

    return matches, mismatches

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Verify continuous futures contract data.")
    parser.add_argument("--symbol-prefix", type=str, default="VXc", help="Prefix for continuous contract symbols (e.g., 'VXc')")
    parser.add_argument("--gap-threshold", type=float, default=PRICE_GAP_THRESHOLD, help="Percentage threshold for price gap detection (e.g., 0.2 for 20%)")
    args = parser.parse_args()

    logger.info(f"Starting continuous contract verification for symbols starting with '{args.symbol_prefix}'")

    # Load Config and Holidays
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)

    # Assuming VIX uses NYSE holidays as specified in config structure
    holidays = get_holidays(config, calendar_name="NYSE")
    # Debug: Print first few holidays
    # if holidays:
    #     logger.debug(f"First 5 loaded holidays: {sorted(list(holidays))[:5]}")


    conn = connect_db()
    if not conn:
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)

    symbols_to_check = get_continuous_contracts(conn, args.symbol_prefix)
    if not symbols_to_check:
        logger.warning(f"No continuous contracts found with prefix '{args.symbol_prefix}'. Exiting.")
        conn.close()
        sys.exit(0)

    total_issues = 0
    expected_rollover_issues = 0
    try:
        # Calculate expected rollovers once for the likely range
        # Need min/max dates across all symbols, or calculate per symbol?
        # Let's calculate per symbol for now, might be slightly inefficient if ranges overlap heavily
        all_expected_expiry_dates = {}


        for symbol in symbols_to_check:
            logger.info(f"--- Verifying {symbol} ---")

            # Get data range for this specific symbol
            min_date, max_date = get_data_range(conn, symbol)
            if not min_date or not max_date:
                logger.warning(f"[{symbol}] Skipping rollover check as data range couldn't be determined.")
                symbol_expected_expiry = {}
            else:
                 # Calculate expected expiry dates for this symbol's range
                 symbol_expected_expiry = calculate_expected_vx_expiry_dates(min_date, max_date, holidays)


            sunday_issues = check_sunday_data(conn, symbol)
            price_gap_issues = check_price_gaps(conn, symbol, args.gap_threshold)
            date_gap_issues = check_date_gaps(conn, symbol)
            # Pass the calculated expected dates for *this symbol's range*
            rollover_matches, rollover_mismatches = check_rollover_consistency(conn, symbol, symbol_expected_expiry)


            symbol_issues = len(sunday_issues) + len(price_gap_issues) + len(date_gap_issues) + rollover_mismatches
            total_issues += symbol_issues
            expected_rollover_issues += rollover_mismatches # Track separately for summary

            if symbol_issues == 0:
                logger.info(f"[{symbol}] Verification complete. No issues found.")
            else:
                 logger.warning(f"[{symbol}] Verification complete. Found {symbol_issues} potential issues ({rollover_mismatches} rollover mismatches)." )
            logger.info("------------------------")

    except Exception as e:
        logger.error(f"An unexpected error occurred during verification: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    logger.info(f"=== Verification Summary ===")
    if total_issues == 0:
        logger.info("All checked contracts passed verification.")
    else:
        logger.warning(f"Verification finished. Found a total of {total_issues} potential issues across all checked contracts.")
        if expected_rollover_issues > 0:
             logger.warning(f"  - Includes {expected_rollover_issues} rollover date mismatches.")
    logger.info("==========================")

if __name__ == "__main__":
    main() 