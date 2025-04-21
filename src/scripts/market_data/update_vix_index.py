#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Downloads and updates historical daily data for the VIX index ($VIX.X)
from the CBOE website into the market_data table.
"""

import sys
import duckdb
import pandas as pd
import argparse
import logging
import requests
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"
VIX_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_INDEX_SYMBOL = "$VIX.X"

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

# --- CBOE Data Download ---
def download_vix_index_data():
    """Downloads historical data CSV for the VIX index from CBOE."""
    url = VIX_INDEX_URL
    logger.info(f"Attempting to download VIX index data from {url}")
    try:
        response = requests.get(url, timeout=60) # Longer timeout for potentially larger file
        if response.status_code == 200:
            logger.info(f"Successfully downloaded VIX index data.")
            csv_data = io.StringIO(response.text)
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

    symbol_code = df_insert['symbol'].iloc[0] # Get symbol for logging
    logger.info(f"Attempting to INSERT OR REPLACE {len(df_insert)} rows for {symbol_code}...")
    try:
        conn.register('df_insert_view', df_insert)
        # Construct column list dynamically from the DataFrame
        cols = df_insert.columns
        col_names_db = ", ".join([f'\"{c}\"'.lower() for c in cols]) # Assuming DB cols are lowercase
        col_names_df = ", ".join([f'\"{c}\"'.lower() for c in cols]) # Match case for selection

        # Use INSERT OR REPLACE INTO ... SELECT ... FROM view
        sql = f"""
            INSERT OR REPLACE INTO market_data ({col_names_db})
            SELECT {col_names_df} FROM df_insert_view
        """
        conn.execute(sql)
        conn.commit() # Commit changes
        logger.info(f"Successfully updated database for {symbol_code}.")
    except duckdb.Error as e:
        logger.error(f"Database error updating data for {symbol_code}: {e}")
        conn.rollback() # Rollback on error
    except Exception as e:
        logger.error(f"Unexpected error updating database for {symbol_code}: {e}")
        conn.rollback()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Download and update VIX index data from CBOE.')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
    args = parser.parse_args()

    conn = None
    try:
        conn = connect_db(args.db_path)

        # --- Update VIX Index Data ---
        logger.info(f"*** Processing VIX Index ({VIX_INDEX_SYMBOL}) ***")
        df_vix_downloaded = download_vix_index_data()
        if df_vix_downloaded is not None and not df_vix_downloaded.empty:
            logger.info(f"Raw VIX DF shape: {df_vix_downloaded.shape}")
            df_vix_prepared = prepare_vix_index_data_for_db(df_vix_downloaded)
            logger.info(f"Prepared VIX DF shape: {df_vix_prepared.shape}")
            update_market_data(conn, df_vix_prepared)
        else:
            logger.warning(f"Skipping database update for VIX index due to download/empty data.")
        logger.info(f"*** Finished VIX Index Processing ({VIX_INDEX_SYMBOL}) ***")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        sys.exit(1) # Exit on main execution error
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 