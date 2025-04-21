#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Downloads and updates historical daily data for active VIX futures contracts
from the CBOE website into the market_data table.
"""

import os
import sys
import duckdb
import pandas as pd
import argparse
import logging
from datetime import date, datetime, timedelta
import requests
import io
import yaml

# Import necessary functions from the generator script
# Assuming scripts are runnable from the root directory or PYTHONPATH is set
try:
    # Import main functions to call scripts sequentially
    from src.scripts.market_data.generate_continuous_futures import main as generate_continuous_main
    from src.scripts.market_data.fill_vx_continuous_gaps import main as fill_gaps_main
    from src.scripts.market_data.fill_vx_zero_prices import main as fill_zero_prices_main
except ImportError as e:
    print(f"Error importing continuous generator or filler functions: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is configured.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"
CBOE_BASE_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{settlement_date}.csv"
ROOT_SYMBOL = "VX"
VIX_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_INDEX_SYMBOL = "$VIX.X"
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"

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

def get_active_contracts(conn):
    """Gets VIX contracts from the calendar whose settlement date is today or later."""
    today_str = date.today().strftime('%Y-%m-%d')
    query = f"""
        SELECT contract_code, final_settlement_date
        FROM futures_roll_calendar
        WHERE root_symbol = ? AND final_settlement_date >= ?
        ORDER BY final_settlement_date ASC
        LIMIT 8 -- Limit to the first 8 contracts
    """
    try:
        df_active = conn.execute(query, [ROOT_SYMBOL, today_str]).fetchdf()
        logger.info(f"Found {len(df_active)} active/future VIX contracts in calendar (limit 8).")
        # Convert date column to actual date objects if needed
        df_active['final_settlement_date'] = pd.to_datetime(df_active['final_settlement_date']).dt.date
        return df_active
    except Exception as e:
        logger.error(f"Error querying active contracts: {e}")
        return pd.DataFrame()

# --- CBOE Data Download ---
def download_cboe_data(contract_code: str, settlement_date: date):
    """Downloads historical data CSV for a given contract from CBOE."""
    url = CBOE_BASE_URL.format(settlement_date=settlement_date.strftime('%Y-%m-%d'))
    logger.info(f"Attempting to download data for {contract_code} from {url}")
    try:
        response = requests.get(url, timeout=30) # Add timeout
        if response.status_code == 200:
            logger.info(f"Successfully downloaded data for {contract_code}.")
            # Use io.StringIO to read the CSV content directly into pandas
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df
        elif response.status_code == 404:
            logger.warning(f"Data file not found (404) for {contract_code} at {url}. It might not be generated yet.")
            return None
        else:
            logger.error(f"Failed to download data for {contract_code}. Status code: {response.status_code}, URL: {url}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading data for {contract_code} from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing download for {contract_code}: {e}")
        return None

def download_vix_index_data():
    """Downloads historical data CSV for the VIX index from CBOE."""
    url = VIX_INDEX_URL
    logger.info(f"Attempting to download VIX index data from {url}")
    try:
        response = requests.get(url, timeout=60) # Longer timeout for potentially larger file
        if response.status_code == 200:
            logger.info(f"Successfully downloaded VIX index data.")
            csv_data = io.StringIO(response.text)
            # Skip header row if it exists and isn't standard
            # Check first few lines if necessary, but usually pandas handles it.
            df = pd.read_csv(csv_data)
            return df
        else:
            logger.error(f"Failed to download VIX index data. Status code: {response.status_code}, URL: {url}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading VIX index data from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing VIX index download: {e}")
        return None

# --- Data Preparation & DB Update ---
def prepare_data_for_db(df: pd.DataFrame, contract_code: str):
    """Prepares the downloaded DataFrame for insertion into market_data."""
    required_cols_in = ['Trade Date', 'Open', 'High', 'Low', 'Settle']
    if not all(col in df.columns for col in required_cols_in):
        # Log a warning but continue if possible, maybe only settle is needed sometimes?
        # For now, let's require all. If files genuinely lack OHLC, we might adjust.
        logger.error(f"Futures data for {contract_code} missing one or more required columns: {required_cols_in}")
        return pd.DataFrame()

    try:
        # Select and rename columns (including OHLC)
        df_prep = df[required_cols_in].copy()
        df_prep.rename(columns={
            'Trade Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Settle': 'settle'
        }, inplace=True)

        # Assign metadata
        df_prep['symbol'] = contract_code
        df_prep['interval_value'] = 1
        df_prep['interval_unit'] = 'day'
        df_prep['source'] = 'CBOE' # Add data source

        # Convert timestamp
        df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'])

        # Convert OHLC and settle to numeric, coercing errors
        num_cols = ['open', 'high', 'low', 'settle']
        for col in num_cols:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')

        # Drop rows where settle became NaN after coercion, or where timestamp is NaT
        rows_before = len(df_prep)
        df_prep.dropna(subset=['timestamp', 'open', 'high', 'low', 'settle'], inplace=True)
        rows_after = len(df_prep)
        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows from {contract_code} due to missing/invalid date or OHLC/settle price.")

        # Ensure column order matches typical primary key for INSERT OR REPLACE
        # PK: timestamp, symbol, interval_value, interval_unit
        final_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'settle', 'interval_value', 'interval_unit', 'source']
        return df_prep[final_cols]

    except KeyError as e:
        logger.error(f"Missing expected column in downloaded data for {contract_code}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error preparing data for {contract_code}: {e}")
        return pd.DataFrame()

def prepare_vix_index_data_for_db(df: pd.DataFrame):
    """Prepares the downloaded VIX index DataFrame for insertion into market_data."""
    required_cols_in = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
    if not all(col in df.columns for col in required_cols_in):
        logger.error(f"VIX index data missing one or more required columns: {required_cols_in}")
        return pd.DataFrame()

    try:
        # Select and rename columns
        df_prep = df[required_cols_in].copy()
        df_prep.rename(columns={
            'DATE': 'timestamp',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'settle' # Use CLOSE for settle column
        }, inplace=True)

        # Assign metadata
        df_prep['symbol'] = VIX_INDEX_SYMBOL
        df_prep['interval_value'] = 1
        df_prep['interval_unit'] = 'day'
        df_prep['source'] = 'CBOE' # Add data source

        # Convert timestamp
        df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'])

        # Convert OHLC and settle to numeric, coercing errors
        num_cols = ['open', 'high', 'low', 'settle']
        for col in num_cols:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')

        # Drop rows where essential columns became NaN/NaT
        rows_before = len(df_prep)
        df_prep.dropna(subset=['timestamp', 'open', 'high', 'low', 'settle'], inplace=True)
        rows_after = len(df_prep)
        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows from VIX index data due to missing/invalid values.")

        # Define final columns for the database table
        final_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'settle', 'interval_value', 'interval_unit', 'source']
        return df_prep[final_cols]

    except Exception as e:
        logger.error(f"Error preparing VIX index data: {e}")
        return pd.DataFrame()

def update_market_data(conn, df_insert: pd.DataFrame):
    """Updates the market_data table using INSERT OR REPLACE."""
    if df_insert.empty:
        logger.info("No valid data prepared for database update.")
        return

    contract_code = df_insert['symbol'].iloc[0] # Get symbol for logging
    logger.info(f"Attempting to INSERT OR REPLACE {len(df_insert)} rows for {contract_code}...")
    try:
        conn.register('df_insert_view', df_insert)
        # Construct column list dynamically from the DataFrame
        cols = df_insert.columns
        col_names_db = ", ".join([f'\"{c}\"'.lower() for c in cols]) # Assuming DB cols are lowercase, quote if needed
        col_names_df = ", ".join([f'\"{c}\"'.lower() for c in cols]) # Match case for selection

        # PK: timestamp, symbol, interval_value, interval_unit
        # Use INSERT OR REPLACE INTO ... SELECT ... FROM view
        sql = f"""
            INSERT OR REPLACE INTO market_data ({col_names_db})
            SELECT {col_names_df} FROM df_insert_view
        """
        # print(f"Executing SQL: {sql}") # Debug SQL if needed
        conn.execute(sql)
        conn.commit() # Commit changes
        logger.info(f"Successfully updated database for {contract_code}.")
    except duckdb.Error as e:
        logger.error(f"Database error updating data for {contract_code}: {e}")
        conn.rollback() # Rollback on error
    except Exception as e:
        logger.error(f"Unexpected error updating database for {contract_code}: {e}")
        conn.rollback()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Download and update active VIX futures data from CBOE.')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
    parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the market symbols config YAML file.')
    parser.add_argument('--full-regen', action='store_true', help='Perform a full regeneration of continuous contracts from history and fill 2004-2005 gaps.')

    args = parser.parse_args()

    conn = None
    try:
        conn = connect_db(args.db_path)

        # ======= RESTORE ORIGINAL LOGIC =======
        active_contracts = get_active_contracts(conn)

        if active_contracts.empty:
            logger.info("No active VIX contracts found in the roll calendar. Exiting.")
            sys.exit(0)

        for _, contract_row in active_contracts.iterrows():
            contract_code = contract_row['contract_code']
            settlement_date = contract_row['final_settlement_date']

            df_downloaded = download_cboe_data(contract_code, settlement_date)

            if df_downloaded is not None and not df_downloaded.empty:
                df_prepared_futures = prepare_data_for_db(df_downloaded, contract_code)
                update_market_data(conn, df_prepared_futures)
            else:
                logger.info(f"Skipping database update for {contract_code} due to download/empty data.")
            logger.info("---") # Separator between contracts
        # ======================================

        # --- VIX Index Data Update REMOVED - Handled by update_vix_index.py ---
        # logger.info("*** Processing VIX Index ($VIX.X) ***")
        # df_vix_downloaded = download_vix_index_data()
        # if df_vix_downloaded is not None and not df_vix_downloaded.empty:
        #     df_vix_prepared = prepare_vix_index_data_for_db(df_vix_downloaded)
        #     update_market_data(conn, df_vix_prepared)
        # else:
        #     logger.warning("Skipping database update for VIX index due to download/empty data.")
        # logger.info("*** Finished VIX Index Processing ***")
        # ---------------------------------------------------------------------

        # ======= RESTORE ORIGINAL LOGIC =======
        # --- Call Continuous Contract Generation Script ---
        # Generate only the last ~90 days
        logger.info(f"*** Calling Continuous Contracts Generation for {ROOT_SYMBOL} ***")
        try:
            # Determine date range and force flag based on --full-regen
            gen_end_date = date.today().strftime('%Y-%m-%d')
            force_flag = False
            gen_start_date = None # Initialize
            if args.full_regen:
                logger.info("Full regeneration requested.")
                # Set start_date to None; generate_continuous_main will use config default
                gen_start_date = None
                force_flag = True # Force delete for a clean full regeneration
            else:
                # Default: ~95 day lookback
                gen_start_date = (date.today() - timedelta(days=95)).strftime('%Y-%m-%d')

            # Log appropriately based on start date
            log_start = "config default" if gen_start_date is None else gen_start_date
            logger.info(f"Generating continuous contracts from {log_start} to {gen_end_date} (force={force_flag})")

            # Prepare args dictionary
            gen_args = {
                'db_path': args.db_path,
                'config_path': args.config_path,
                'root_symbol': ROOT_SYMBOL,
                'start_date': gen_start_date,
                'end_date': gen_end_date,
                'force': force_flag
            }
            generate_continuous_main(args_dict=gen_args)
            logger.info(f"*** Finished Continuous Contracts Generation for {ROOT_SYMBOL} ***")
        except Exception as gen_e:
            logger.error(f"Error running generate_continuous_main: {gen_e}", exc_info=True)

        # --- Conditionally Call Gap Filling Script ---
        if args.full_regen:
            logger.info(f"*** Full regen requested, running Gap Filling Script... ***")
            try:
                # Prepare args dictionary
                fill_args = {
                    'db_path': args.db_path
                }
                fill_gaps_main(args_dict=fill_args)
                logger.info(f"*** Finished Gap Filling Script ***")
            except Exception as fill_e:
                logger.error(f"Error running fill_gaps_main: {fill_e}", exc_info=True)
                
            # --- Call Zero Price Filling Script ---
            logger.info(f"*** Full regen requested, running Zero Price Filling Script... ***")
            try:
                # Prepare args dictionary
                zero_fill_args = {
                    'db_path': args.db_path
                }
                fill_zero_prices_main(args_dict=zero_fill_args)
                logger.info(f"*** Finished Zero Price Filling Script ***")
            except Exception as zero_fill_e:
                logger.error(f"Error running fill_zero_prices_main: {zero_fill_e}", exc_info=True)
        else:
            logger.info("Skipping historical gap and zero filling (run with --full-regen to execute).")
        # ======================================

    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        sys.exit(1) # Exit on main execution error
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 