#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fills historical gaps in VXc1 and VXc2 data (specifically 2004-2005)
by deriving prices from $VIX.X using a calculated historical ratio.
Marks the inserted data with changed=TRUE and source='DERIVED_VIX_RATIO'.
"""

import sys
import duckdb
import pandas as pd
from datetime import datetime
import logging
import argparse

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"
TARGET_SYMBOLS = ['VXc1', 'VXc2']
REFERENCE_SYMBOL = '$VIX.X'
GAP_START_DATE = '2004-01-01'
GAP_END_DATE = '2005-12-31'
RATIO_CALC_DAYS = 30 # Number of trading days to use for ratio calculation
DERIVED_SOURCE_TAG = 'DERIVED_VIX_RATIO'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Operations ---
def connect_db(db_path):
    """Connects to the DuckDB database (read-write)."""
    try:
        conn = duckdb.connect(database=db_path, read_only=False)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def load_data_for_period(conn, symbol, start_date, end_date):
    """Loads settle prices for a symbol within a date range."""
    logger.debug(f"Loading data for {symbol} from {start_date} to {end_date}")
    try:
        # Select OHLC and Settle columns
        query = f"""
            SELECT timestamp, open, high, low, settle
            FROM market_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
        if df.empty:
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
        else:
             # Ensure timestamp is datetime object
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             logger.info(f"Loaded {len(df)} entries for {symbol} from {start_date} to {end_date}")
        return df.set_index('timestamp')
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

# --- Calculation Logic ---
def calculate_average_ratio(target_series, reference_series):
    """Calculates the average ratio between two Pandas Series (target/reference)."""
    # Align series by timestamp (index) and drop rows where either value is missing
    df_merged = pd.merge(target_series, reference_series, left_index=True, right_index=True, how='inner', suffixes=('_target', '_ref'))
    df_merged.dropna(inplace=True)

    if df_merged.empty:
        logger.warning("No overlapping data found to calculate ratio.")
        return None

    # Avoid division by zero or near-zero
    df_merged = df_merged[df_merged['settle_ref'].abs() > 1e-6] # Check reference denom

    if df_merged.empty:
        logger.warning("Reference data is zero or near-zero, cannot calculate stable ratio.")
        return None

    df_merged['ratio'] = df_merged['settle_target'] / df_merged['settle_ref']
    average_ratio = df_merged['ratio'].mean()
    logger.info(f"Calculated average ratio: {average_ratio:.4f} from {len(df_merged)} data points")
    return average_ratio

# --- Main Execution ---
def main(args_dict=None):
    effective_args = {}
    if args_dict is None:
        # --- Argument Parsing (if run directly) ---
        parser = argparse.ArgumentParser(description='Fill 2004-2005 gaps in VXc1/VXc2 using $VIX.X ratio.')
        parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
        parsed_args = parser.parse_args()
        effective_args['db_path'] = parsed_args.db_path
        logger.info("Running fill_vx_continuous_gaps from command line.")
    else:
        # --- Use Provided Args (if called programmatically) ---
        logger.info("Running fill_vx_continuous_gaps programmatically.")
        effective_args['db_path'] = args_dict.get('db_path', DEFAULT_DB_PATH)

    # --- Common Logic using effective_args ---
    logger.info(f"Effective args: {effective_args}")

    conn = None
    all_derived_data = []

    try:
        # Use db_path from effective_args
        conn = connect_db(effective_args['db_path'])

        # Load reference data ($VIX.X) covering gap period
        logger.info(f"Loading reference data: {REFERENCE_SYMBOL}")
        df_vix = load_data_for_period(conn, REFERENCE_SYMBOL, GAP_START_DATE, GAP_END_DATE)
        if df_vix.empty:
            logger.error(f"Cannot proceed without reference data ({REFERENCE_SYMBOL}). Exiting.")
            # Ensure connection is closed before exiting
            if conn: conn.close()
            sys.exit(1)

        # Process each target symbol (VXc1, VXc2)
        for target_symbol in TARGET_SYMBOLS:
            logger.info(f"--- Processing target symbol: {target_symbol} ---")

            # Load target data for the entire gap period
            df_target = load_data_for_period(conn, target_symbol, GAP_START_DATE, GAP_END_DATE)
            # Note: df_target might be empty or partially empty, especially in the gap period

            # --- Calculate Ratio ---
            logger.info(f"Calculating historical ratio for {target_symbol}/{REFERENCE_SYMBOL}")

            # Find the first date within the gap period where the target symbol has data
            first_valid_date = df_target.first_valid_index()

            if first_valid_date is None or first_valid_date > pd.Timestamp(GAP_END_DATE):
                logger.warning(f"No data found for target symbol {target_symbol} within the period {GAP_START_DATE}-{GAP_END_DATE}. Cannot calculate ratio.")
                continue

            # Determine the date range for ratio calculation (first N days from first_valid_date)
            ratio_calc_start = first_valid_date
            # Calculate end date using Business Day frequency
            # Get the index of dates where target *settle* is valid, starting from ratio_calc_start
            valid_target_dates = df_target.loc[ratio_calc_start:].dropna(subset=['settle']).index
            if len(valid_target_dates) < RATIO_CALC_DAYS:
                logger.warning(f"Fewer than {RATIO_CALC_DAYS} valid data points found for {target_symbol} after {ratio_calc_start.strftime('%Y-%m-%d')}. Using all available points ({len(valid_target_dates)})." )
                ratio_calc_end = valid_target_dates[-1] if not valid_target_dates.empty else ratio_calc_start
            else:
                ratio_calc_end = valid_target_dates[RATIO_CALC_DAYS - 1]

            logger.info(f"Using data from {ratio_calc_start.strftime('%Y-%m-%d')} to {ratio_calc_end.strftime('%Y-%m-%d')} for ratio calculation.")

            # Filter dataframes for the dynamically determined ratio period
            df_target_ratio_period = df_target.loc[ratio_calc_start:ratio_calc_end].copy()
            df_vix_ratio_period = df_vix.loc[ratio_calc_start:ratio_calc_end].copy()

            # Ensure dataframes are not empty before slicing
            if df_target_ratio_period.empty or df_vix_ratio_period.empty:
                logger.warning(f"No overlapping data found in the dynamic ratio period ({ratio_calc_start.strftime('%Y-%m-%d')} to {ratio_calc_end.strftime('%Y-%m-%d')}) for {target_symbol} or {REFERENCE_SYMBOL}.")
                continue # Skip to next target symbol

            # Ratio calculation still based on settle prices only
            avg_ratio = calculate_average_ratio(df_target_ratio_period['settle'], df_vix_ratio_period['settle'])

            if avg_ratio is None:
                logger.warning(f"Could not calculate average ratio for {target_symbol}. Skipping gap filling for this symbol.")
                continue

            # --- Identify Gaps and Generate Derived Data ---
            logger.info(f"Identifying gaps for {target_symbol} between {GAP_START_DATE} and {GAP_END_DATE}")
            df_vix_gap_period = df_vix.loc[GAP_START_DATE:GAP_END_DATE]
            # Use .copy() to avoid SettingWithCopyWarning later if df_target is empty/sliced
            # Target data for gap identification still uses the full loaded period (2004-2005)
            df_target_gap_period = df_target.copy()

            if df_vix_gap_period.empty:
                logger.warning(f"No reference data ({REFERENCE_SYMBOL}) found in the gap period {GAP_START_DATE}-{GAP_END_DATE}. Cannot fill gaps for {target_symbol}.")
                continue

            # Find dates where VIX exists but target doesn't
            df_merged_gaps = pd.merge(
                df_vix_gap_period[['open', 'high', 'low', 'settle']], # Include VIX OHLC
                df_target_gap_period[['settle']], # Only need target settle to identify gap
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('_vix', '_target')
            )

            # Filter for the actual gaps (where target settle is NaN, but VIX settle is not)
            df_gaps = df_merged_gaps[df_merged_gaps['settle_target'].isna() & df_merged_gaps['settle_vix'].notna()]

            if df_gaps.empty:
                logger.info(f"No gaps found for {target_symbol} in the specified period where {REFERENCE_SYMBOL} data exists.")
                continue

            logger.info(f"Found {len(df_gaps)} gap dates for {target_symbol} to fill.")

            # Calculate derived OHLC and Settle prices
            # Apply ratio to VIX OHLC/Settle. Use original column names for OHLC from VIX df,
            # as suffixes are only added by merge on conflicting columns (like 'settle').
            derived_open = df_gaps['open'] * avg_ratio
            derived_high = df_gaps['high'] * avg_ratio
            derived_low = df_gaps['low'] * avg_ratio
            derived_settle = df_gaps['settle_vix'] * avg_ratio # Settle used the suffix

            # Prepare DataFrame for insertion
            df_derived = pd.DataFrame({
                'timestamp': derived_settle.index,
                'symbol': target_symbol,
                'open': derived_open.values,
                'high': derived_high.values,
                'low': derived_low.values,
                'close': None, # 'close' column might not be used if 'settle' is primary
                'settle': derived_settle.values,
                'volume': None,
                'open_interest': None,
                'interval_value': 1,
                'interval_unit': 'day',
                'source': DERIVED_SOURCE_TAG,
                'changed': True,
                'adjusted': False, # Keep adjusted as False unless specifically calculated
                'quality': 80 # Assign lower quality score for derived data
                # Add 'UnderlyingSymbol': None if that column exists and should be null
            })
            # Add UnderlyingSymbol if it exists in the table schema
            try:
                 # Quick check if column exists - assumes it does if no error
                 conn.execute("SELECT UnderlyingSymbol FROM market_data LIMIT 1")
                 df_derived['UnderlyingSymbol'] = None
            except duckdb.CatalogException:
                 logger.debug("UnderlyingSymbol column not found, not adding to derived data.")
            except Exception as e:
                 logger.warning(f"Could not check for UnderlyingSymbol column: {e}")


            all_derived_data.append(df_derived)

        # --- Combine and Insert All Derived Data ---
        if not all_derived_data:
            logger.info("No derived data generated across all target symbols. Nothing to insert.")
            # No return here, let finally handle closing
        else:
            df_to_insert = pd.concat(all_derived_data, ignore_index=True)
            logger.info(f"Attempting to insert {len(df_to_insert)} derived rows into market_data...")
            try:
                conn.register('df_derived_view', df_to_insert)
                cols = df_to_insert.columns
                col_names_db = ", ".join([f'"{c}"' for c in cols])
                col_names_view = ", ".join([f'v."{c}"' for c in cols])
                sql = f"""
                    INSERT OR REPLACE INTO market_data ({col_names_db})
                    SELECT {col_names_view}
                    FROM df_derived_view v
                """
                logger.debug(f"Executing INSERT OR REPLACE SQL: {sql}")
                conn.execute(sql)
                conn.commit()
                logger.info(f"Successfully inserted {len(df_to_insert)} derived rows.")

            except duckdb.Error as e:
                logger.error(f"Database error inserting derived data: {e}")
                conn.rollback()
            except Exception as e:
                logger.error(f"Unexpected error inserting derived data: {e}")
                conn.rollback()

    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        # Let finally block handle connection closing

    finally:
        if conn:
            try:
                 if not conn.closed:
                     conn.close()
                     logger.info("Database connection closed.")
            except Exception as close_e:
                 logger.error(f"Error closing database connection: {close_e}")

if __name__ == "__main__":
    # Call main without args_dict when run directly
    main() 