#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loads historical CBOE VIX Futures data from local files into the DuckDB database.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import duckdb
from pathlib import Path
import glob
from io import StringIO
from typing import List, Tuple, Optional

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to stdout
        # Optionally add logging.FileHandler('load_cboe_data.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Use a relative path within the project for the database
DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"
# Absolute path provided by the user for the CBOE data files
CBOE_DATA_DIR = r"C:/Users/alexp/OneDrive/Gdrive/Trading/Data Downloads/VX Futures"
# Source identifier for this data
DATA_SOURCE = "CBOE"

def connect_db(db_file: Path = DB_PATH, read_only: bool = False) -> Optional[duckdb.DuckDBPyConnection]:
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=str(db_file), read_only=read_only)
        logger.info(f"Successfully connected to database: {db_file} {'(Read-Only)' if read_only else ''}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database {db_file}: {e}")
        return None

def find_data_files(data_dir: str) -> List[str]:
    """Finds all data files (e.g., CSV) in the specified directory."""
    # Assuming CSV files for now, adjust pattern if needed (e.g., '*.csv', '*.xlsx')
    file_pattern = os.path.join(data_dir, '*.csv')
    files = glob.glob(file_pattern)
    logger.info(f"Found {len(files)} potential data files in {data_dir} matching pattern '{os.path.basename(file_pattern)}'.")
    if not files:
        logger.warning(f"No files found matching pattern in {data_dir}. Please check the directory and pattern.")
    return files

def _parse_futures_symbol(futures_str: str) -> Optional[str]:
    """Parses the 'Futures' column string (e.g., 'F (Jan 2024)') into a standard symbol (e.g., 'VXF24')."""
    try:
        # Basic split, might need refinement if format varies significantly
        parts = futures_str.split('(')
        if len(parts) < 2:
            logger.warning(f"Could not parse month/year from Futures string: {futures_str}")
            return None

        month_year_part = parts[1].split(')')[0]
        month_abbr, year_str = month_year_part.split()

        year = int(year_str)
        year_short = str(year)[-2:] # Get last two digits

        month_map = {
            'Jan': 'F', 'Feb': 'G', 'Mar': 'H', 'Apr': 'J', 'May': 'K', 'Jun': 'M',
            'Jul': 'N', 'Aug': 'Q', 'Sep': 'U', 'Oct': 'V', 'Nov': 'X', 'Dec': 'Z'
        }

        month_code = month_map.get(month_abbr)
        if not month_code:
            logger.warning(f"Unknown month abbreviation '{month_abbr}' in Futures string: {futures_str}")
            return None

        # Assume root symbol is always VX for these files
        return f"VX{month_code}{year_short}"

    except Exception as e:
        logger.error(f"Error parsing Futures string '{futures_str}': {e}")
        return None

def parse_cboe_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parses a single CBOE data file based on the observed format.
    Handles column renaming, type conversion, and symbol generation.
    """
    logger.info(f"Parsing file: {file_path}...")
    try:
        # Read the CSV file. Adjust 'skiprows' if needed based on other files.
        # We assume the first row is the header based on the sample.
        df = pd.read_csv(file_path, sep=',') # Explicitly set separator

        # --- Data Cleaning and Transformation ---

        # Check for required columns
        required_headers = ['Trade Date', 'Futures', 'Open', 'High', 'Low', 'Settle', 'Total Volume', 'Open Interest']
        if not all(col in df.columns for col in required_headers):
            logger.error(f"File {file_path} is missing one or more required columns: {required_headers}. Found: {df.columns.tolist()}")
            return None

        # 1. Parse Symbol first - needed for potential early exit or grouping
        df['symbol'] = df['Futures'].apply(_parse_futures_symbol)
        # Handle cases where symbol parsing failed
        if df['symbol'].isnull().any():
            logger.warning(f"Symbol parsing failed for some rows in {file_path}. Dropping those rows.")
            df = df.dropna(subset=['symbol'])
            if df.empty:
                logger.error(f"No valid symbols found after parsing {file_path}. Skipping file.")
                return None

        # Check if multiple symbols were generated (shouldn't happen based on assumption)
        unique_symbols_in_file = df['symbol'].unique()
        if len(unique_symbols_in_file) > 1:
            logger.warning(f"File {file_path} resulted in multiple symbols after parsing ({unique_symbols_in_file}). This is unexpected. Processing only the first symbol found: {unique_symbols_in_file[0]}")
            # Filter to keep only the first symbol encountered if this happens
            df = df[df['symbol'] == unique_symbols_in_file[0]].copy()


        # 2. Rename columns to match market_data schema
        rename_map = {
            'Trade Date': 'timestamp',
            # 'Futures' column is processed above into 'symbol'
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Settle': 'close', # Using 'Settle' as the closing price
            'Total Volume': 'volume',
            'Open Interest': 'open_interest'
        }
        df = df.rename(columns=rename_map)

        # 3. Ensure correct data types
        # Date conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y', errors='coerce').dt.date # Keep as date object for daily data

        # Numeric conversion (handle potential non-numeric like commas if necessary later)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        # 4. Add standard columns
        df['interval_value'] = 1
        df['interval_unit'] = 'daily'
        df['source'] = DATA_SOURCE
        df['up_volume'] = None  # CBOE data doesn't provide this
        df['down_volume'] = None  # CBOE data doesn't provide this
        df['adjusted'] = False  # Raw data, not adjusted
        df['quality'] = 100  # Raw data quality score (100 = raw data from source)

        # 5. Select final columns in the correct order for the table
        final_cols = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
            'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
            'adjusted', 'quality', 'open_interest'
        ]
        df = df[final_cols]

        # 6. Data Quality Checks (Optional but recommended)
        # Drop rows where essential data is missing after conversion
        initial_rows = len(df)
        df = df.dropna(subset=['timestamp', 'symbol', 'close']) # Require timestamp, symbol, and close price
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows from {file_path} due to missing timestamp, symbol, or close price after conversion.")

        if df.empty:
             logger.warning(f"No valid data remaining in {file_path} after cleaning. Skipping.")
             return None

        logger.info(f"Successfully parsed and cleaned {len(df)} rows from {file_path} for symbol {unique_symbols_in_file[0]}.")
        return df

    except Exception as e:
        logger.error(f"Failed to parse file {file_path}: {e}")
        # Raise exception or return None based on desired error handling
        # raise e # Option 1: Fail fast
        return None # Option 2: Log error and skip file

def clean_existing_data(conn: duckdb.DuckDBPyConnection, symbol: str, min_date, max_date):
    """Deletes existing daily data from the CBOE source for the given symbol and date range."""
    if not symbol or min_date is None or max_date is None:
        logger.error("Cannot clean data: Missing symbol or date range.")
        return

    delete_query = """
    DELETE FROM market_data
    WHERE symbol = ?
      AND source = ?
      AND interval_value = 1
      AND interval_unit = 'daily'
      AND timestamp BETWEEN ? AND ?
    """
    try:
        logger.info(f"Deleting existing CBOE daily data for {symbol} between {min_date} and {max_date}...")
        with conn.cursor() as cur:
            cur.execute(delete_query, [symbol, DATA_SOURCE, min_date, max_date])
            deleted_count = cur.rowcount
            logger.info(f"Deleted {deleted_count} existing rows for {symbol}.")
            # No explicit commit needed in DuckDB Python API unless in manual transaction mode
    except Exception as e:
        logger.error(f"Error cleaning existing data for {symbol}: {e}")
        # Consider adding rollback if using explicit transactions

def insert_data(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Inserts the DataFrame into the market_data table."""
    if df.empty:
        logger.info("No data to insert.")
        return

    table_name = "market_data"
    try:
        logger.info(f"Inserting {len(df)} rows into {table_name}...")
        # Use DuckDB's efficient DataFrame insertion
        conn.register('df_to_insert', df)
        # Explicitly list columns to ensure order matches table schema
        sql = f"""
            INSERT INTO {table_name} (
                timestamp, symbol, open, high, low, close, volume, 
                open_interest, up_volume, down_volume, source, 
                interval_value, interval_unit, adjusted, quality
            )
            SELECT 
                timestamp, symbol, open, high, low, close, volume, 
                open_interest, up_volume, down_volume, source, 
                interval_value, interval_unit, adjusted, quality
            FROM df_to_insert
        """
        conn.execute(sql)
        conn.unregister('df_to_insert') # Clean up temporary registration
        logger.info(f"Successfully inserted {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error inserting data into {table_name}: {e}")

def main():
    """Main script execution function."""
    logger.info("Starting CBOE VIX Futures data loading process...")

    # Argument parsing (optional, could add flags for date ranges, specific files, etc.)
    parser = argparse.ArgumentParser(description='Load CBOE VIX Futures data from local files.')
    # Example: parser.add_argument('--force-reload', action='store_true', help='Force reload all data')
    args = parser.parse_args()

    data_files = find_data_files(CBOE_DATA_DIR)
    if not data_files:
        logger.error("No data files found. Exiting.")
        sys.exit(1)

    conn = connect_db()
    if not conn:
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)

    processed_files = 0
    total_rows_added = 0

    try:
        for file_path in data_files:
            df = parse_cboe_file(file_path)

            if df is not None and not df.empty:
                # Assume all rows in a file are for the same symbol for cleaning
                # If a file could contain multiple symbols, logic needs adjustment
                unique_symbols = df['symbol'].unique()
                if len(unique_symbols) == 1:
                    symbol = unique_symbols[0]
                    min_date = df['timestamp'].min()
                    max_date = df['timestamp'].max()

                    clean_existing_data(conn, symbol, min_date, max_date)
                    insert_data(conn, df)
                    total_rows_added += len(df)
                    processed_files += 1
                else:
                    logger.warning(f"File {file_path} contains multiple symbols ({unique_symbols}). Skipping cleaning/insertion for this file. Manual review needed.")
            else:
                 logger.warning(f"Skipping file due to parsing error or empty content: {file_path}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    logger.info(f"--- Load Summary ---")
    logger.info(f"Processed {processed_files} out of {len(data_files)} files.")
    logger.info(f"Added {total_rows_added} rows to the market_data table.")
    logger.info("--------------------")
    logger.info("CBOE VIX Futures data loading process finished.")

if __name__ == "__main__":
    main() 