#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loads futures roll calendar data from a CSV file into the database.

Reads a CSV file containing contract month, final settlement date, and
last trading day, and inserts/replaces the data into the 'futures_roll_calendar' table.
"""

import os
import sys
import duckdb
import pandas as pd
import argparse
import logging
from contextlib import closing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Consider making this configurable or deriving from a central config if needed
DEFAULT_DB_PATH = "data/financial_data.db"

# --- Database Operations ---

def create_table(conn):
    """Creates the futures_roll_calendar table if it doesn't exist, adding the contract_code column if needed."""
    try:
        with closing(conn.cursor()) as cursor:
            # Create table if it doesn't exist (original 4 or new 5 columns)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS futures_roll_calendar (
                root_symbol TEXT NOT NULL,
                contract_month TEXT NOT NULL, -- Format YYYY-MM
                contract_code TEXT NOT NULL, -- Format SYMBOL+MCODE+YY (e.g., VXF05)
                final_settlement_date DATE NOT NULL,
                last_trading_day DATE NOT NULL,
                PRIMARY KEY (root_symbol, contract_month)
            )
            """)
            logger.info("Table 'futures_roll_calendar' checked/created.")

            # Check if the contract_code column exists
            cursor.execute("""
                SELECT COUNT(*) FROM pragma_table_info('futures_roll_calendar')
                WHERE name = 'contract_code'
            """)
            column_exists = cursor.fetchone()[0] > 0

            if not column_exists:
                logger.info("Column 'contract_code' not found in 'futures_roll_calendar'. Adding column...")
                cursor.execute("""
                    ALTER TABLE futures_roll_calendar
                    ADD COLUMN contract_code TEXT -- Allow NULL temporarily for existing rows, or set a default?
                                                 -- Let's add NOT NULL and rely on INSERT OR REPLACE to fix old rows.
                                                 -- Better: Add without NOT NULL, then update, then add constraint. Safest: Add without NOT NULL.
                                                 -- Simplest for now: Add column, rely on INSERT OR REPLACE
                """)
                # Update existing rows potentially? Safer just to let INSERT OR REPLACE handle it.
                logger.info("Column 'contract_code' added successfully.")
            else:
                logger.info("Column 'contract_code' already exists.")

    except duckdb.Error as e:
        logger.error(f"Error checking/modifying table 'futures_roll_calendar': {e}")
        raise

def load_calendar_data(conn, csv_path: str, root_symbol: str):
    """Loads data from the CSV file into the database table using DuckDB."""
    try:
        df = pd.read_csv(csv_path, parse_dates=['FinalSettlementDate', 'LastTradingDay'])
        logger.info(f"Read {len(df)} records from {csv_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        sys.exit(1)

    added_count = 0
    replaced_count = 0

    # Prepare DataFrame for DuckDB insertion
    df['root_symbol'] = root_symbol
    # Rename columns to match the table schema exactly
    df.rename(columns={
        'ContractMonth': 'contract_month',
        'ContractCode': 'contract_code',
        'FinalSettlementDate': 'final_settlement_date_dt',
        'LastTradingDay': 'last_trading_day_dt'
        }, inplace=True)

    # Ensure date columns are in YYYY-MM-DD string format for DuckDB DATE type
    df['final_settlement_date'] = df['final_settlement_date_dt'].dt.strftime('%Y-%m-%d')
    df['last_trading_day'] = df['last_trading_day_dt'].dt.strftime('%Y-%m-%d')

    # Select and order columns for insertion
    df_to_insert = df[['root_symbol', 'contract_month', 'contract_code', 'final_settlement_date', 'last_trading_day']]

    try:
        # Use DuckDB's ability to query pandas DataFrames directly
        # The df_to_insert variable holds the prepared DataFrame
        conn.sql(f"""
            INSERT OR REPLACE INTO futures_roll_calendar
            (root_symbol, contract_month, contract_code, final_settlement_date, last_trading_day)
            SELECT root_symbol, contract_month, contract_code, final_settlement_date, last_trading_day
            FROM df_to_insert
        """)
        # DuckDB commits automatically by default unless explicitly in a transaction block
        logger.info(f"Successfully processed {len(df_to_insert)} records for symbol '{root_symbol}' into 'futures_roll_calendar'.")

    except duckdb.Error as e:
        logger.error(f"Database error during data insertion for {root_symbol}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data insertion: {e}")
        raise


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Load Futures Roll Calendar CSV to Database.')
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the roll calendar CSV file.')
    parser.add_argument('--root-symbol', type=str, required=True, help='The root symbol for this calendar (e.g., VX, ES).')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the SQLite database file.')

    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found at {args.db_path}. Please ensure the path is correct or the database exists.")
        # Depending on desired behavior, we could create the DB dir/file, but safer to require it exists.
        sys.exit(1)

    try:
        # Connect to the database
        conn = duckdb.connect(database=args.db_path, read_only=False)
        logger.info(f"Connected to database: {args.db_path}")

        # Ensure the table exists
        create_table(conn)

        # Load the data
        load_calendar_data(conn, args.csv_path, args.root_symbol.upper()) # Standardize symbol case

    except duckdb.Error as e:
        logger.error(f"A database error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 