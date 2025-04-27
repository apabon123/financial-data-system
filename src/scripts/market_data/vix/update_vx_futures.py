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
def main(args_dict=None, existing_conn=None):
    """Main execution function."""
    if args_dict:
        # Called directly
        args = argparse.Namespace()
        args.db_path = args_dict.get('db_path', DEFAULT_DB_PATH)
        args.config_path = args_dict.get('config_path', DEFAULT_CONFIG_PATH)
        args.full_regen = args_dict.get('full_regen', False)
        if 'db_conn' in args_dict:
            existing_conn = args_dict['db_conn']
        logger.info("Running update_vx_futures from direct call.")
    else:
        # Called from command line
        parser = argparse.ArgumentParser(description='Download and update active VIX futures data from CBOE.')
        parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
        parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the market symbols YAML config file.')
        parser.add_argument('--full-regen', action='store_true', help='Force generation of continuous contracts after update.')
        args = parser.parse_args()
        logger.info("Running update_vx_futures from command line.")

    conn = None
    close_conn_locally = False
    try:
        if existing_conn:
            conn = existing_conn
            logger.info("Using existing database connection for VX futures update.")
        else:
            conn = connect_db(args.db_path)
            close_conn_locally = True

        if not conn:
            logger.error("Failed to establish database connection.")
            sys.exit(1)

        # --- Get active contracts ---
        df_active_contracts = get_active_contracts(conn)
        if df_active_contracts.empty:
            logger.warning("No active VIX futures contracts found. Skipping download.")
        else:
            # --- Download and update each active contract ---
            for _, contract in df_active_contracts.iterrows():
                contract_code = contract['contract_code']
                settlement_date = contract['final_settlement_date']
                logger.info(f"*** Processing {contract_code} (Settles: {settlement_date}) ***")
                df_downloaded = download_cboe_data(contract_code, settlement_date)
                if df_downloaded is not None and not df_downloaded.empty:
                    logger.info(f"Raw DF shape: {df_downloaded.shape}")
                    df_prepared = prepare_data_for_db(df_downloaded, contract_code)
                    logger.info(f"Prepared DF shape: {df_prepared.shape}")
                    update_market_data(conn, df_prepared) # Use the established connection
                else:
                    logger.warning(f"Skipping database update for {contract_code} due to download/empty data.")
                logger.info(f"*** Finished Processing {contract_code} ***")

        # --- Regenerate Continuous Contracts (if requested or by default) ---
        # Note: The original script called other main functions here.
        # This logic should now reside in update_all_market_data.py
        # We keep this script focused ONLY on updating the underlying VX futures.
        # if args.full_regen:
        #     logger.info("Full regeneration requested. Running continuous generation...")
        #     gen_args = { 'db_path': args.db_path, 'config_path': args.config_path, 'root_symbol': ROOT_SYMBOL, 'force': True }
        #     generate_continuous_main(args_dict=gen_args)
        #     fill_args = { 'db_path': args.db_path }
        #     fill_gaps_main(args_dict=fill_args)
        #     fill_zero_prices_main(args_dict=fill_args)
        # else:
        #     logger.info("Skipping full regeneration. Standard update only.")
        logger.info("VX Futures update complete. Continuous generation handled by orchestrator script.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        if not args_dict:
            sys.exit(1)
        else:
            raise # Re-raise exception if called programmatically
    finally:
        if conn and close_conn_locally:
            conn.close()
            logger.info("Database connection closed (local VX futures).")

if __name__ == "__main__":
    main() 