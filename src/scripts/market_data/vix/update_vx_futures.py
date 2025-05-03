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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Default paths
# DEFAULT_DB_PATH = "C:/temp/data/financial_data.duckdb" # Point back to original DB in temp location
DEFAULT_DB_PATH = "data/financial_data.duckdb" # Use workspace-relative path
DEFAULT_CONFIG_PATH = "configs/market_symbols.yaml" # Use simple relative path
CBOE_BASE_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{settlement_date}.csv"
ROOT_SYMBOL = "VX"
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
TARGET_TABLE_CBOE = "market_data_cboe"

def prepare_data_for_db(df: pd.DataFrame, contract_code: str, settlement_date: date, conn):
    """Prepares the downloaded DataFrame for insertion into market_data_cboe, handling potential column variations."""
    # --- Flexible Column Name Mapping ---
    actual_cols_lower = {col.lower().strip(): col for col in df.columns}
    date_cols = ['trade date', 'date']
    open_cols = ['open']
    high_cols = ['high']
    low_cols = ['low']
    close_cols = ['close', 'last']
    settle_cols = ['settle'] # Define settle column names
    volume_cols = ['total volume', 'volume', 'vol']
    open_interest_cols = ['open interest', 'oi'] # Add potential names for Open Interest

    def find_col(potential_names):
        for name in potential_names:
            if name in actual_cols_lower:
                return actual_cols_lower[name]
        return None

    date_col = find_col(date_cols)
    open_col = find_col(open_cols)
    high_col = find_col(high_cols)
    low_col = find_col(low_cols)
    close_col = find_col(close_cols)
    settle_col = find_col(settle_cols) # Find the settle column
    volume_col = find_col(volume_cols)
    open_interest_col = find_col(open_interest_cols) # Find the open interest column

    # --- Check for essential columns ---
    if not date_col:
        logger.error(f"Essential 'Date' column not found in CSV for {contract_code}. Found columns: {list(df.columns)}")
        return pd.DataFrame()

    # Determine primary price: Prefer settle if available, otherwise use close.
    primary_price_col = settle_col or close_col 
    primary_price_db_name = 'settle' if settle_col else 'close' # DB column name for primary price

    if not primary_price_col:
         # Adjust error message if settle was preferred but not found
         preferred_price = 'Settle' if settle_cols else 'Close' 
         logger.error(f"Essential '{preferred_price}' or 'Close/Last' column not found in CSV for {contract_code}. Found columns: {list(df.columns)}")
         return pd.DataFrame()

    # --- Prepare DataFrame ---
    try:
        # Identify columns present in the source file
        present_cols_map = {}
        if date_col: present_cols_map['timestamp'] = date_col
        if primary_price_col: present_cols_map[primary_price_db_name] = primary_price_col # Map primary (settle/close) to its DB name
        # Map close specifically if it exists AND is different from the primary price (i.e., primary was settle)
        if close_col and close_col != primary_price_col: 
             present_cols_map['close'] = close_col
        # Map settle specifically if it exists AND is different from the primary price (i.e., primary was close)
        elif settle_col and settle_col != primary_price_col:
             present_cols_map['settle'] = settle_col
             
        if open_col: present_cols_map['open'] = open_col
        if high_col: present_cols_map['high'] = high_col
        if low_col: present_cols_map['low'] = low_col
        if volume_col: present_cols_map['volume'] = volume_col
        if open_interest_col: present_cols_map['open_interest'] = open_interest_col # Add open interest

        # Select only the columns that were actually found
        df_prep = df[list(present_cols_map.values())].copy()
        # Rename found columns to standard DB names
        df_prep.rename(columns={v: k for k, v in present_cols_map.items()}, inplace=True)

        # --- Ensure All Target Data Columns Exist (Add if missing) ---
        # Define the core data columns expected in the final table (excluding metadata)
        # Now includes 'settle', 'volume', 'open_interest'
        expected_data_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest'] 

        for col in expected_data_cols:
            if col not in df_prep.columns:
                logger.warning(f"Column '{col}' not found in source for {contract_code}. Adding column with NA.")
                df_prep[col] = pd.NA # Add missing column with NA

        # --- Handle Data Types & Missing Values --- 
        df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'], errors='coerce')

        # Convert OHLC, Close, and Settle to numeric, coercing errors to NA
        for col in ['open', 'high', 'low', 'close', 'settle']:
            if col in df_prep.columns:
                 # Use pd.NA for missing numeric values if using pandas >= 1.0
                 df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce') 

        # Convert Volume and Open Interest to numeric, coercing errors, then fill NA with 0 and convert to Int64
        for col in ['volume', 'open_interest']:
            if col in df_prep.columns:
                 df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').fillna(0).astype(pd.Int64Dtype()) # Use nullable integer type

        # Assign metadata
        df_prep['symbol'] = contract_code
        df_prep['interval_value'] = 1
        df_prep['interval_unit'] = 'daily'
        df_prep['source'] = 'CBOE'

        # Drop rows where timestamp OR the primary price (settle if available, else close) are invalid/NaT/NA
        # The primary price column name in df_prep is stored in primary_price_db_name
        rows_before = len(df_prep)
        df_prep.dropna(subset=['timestamp', primary_price_db_name], inplace=True)
        rows_after = len(df_prep)
        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows from {contract_code} due to missing/invalid timestamp or primary price ('{primary_price_db_name}').")

        # Return empty if no valid data remains after dropping NAs
        if df_prep.empty:
            logger.warning(f"No valid data rows remaining for {contract_code} after NA drop.")
            return pd.DataFrame()

        # Filter out any data strictly *after* the settlement date
        settlement_date_ts = pd.Timestamp(settlement_date)
        original_rows = len(df_prep)
        df_prep = df_prep[df_prep['timestamp'] <= settlement_date_ts]
        rows_filtered = original_rows - len(df_prep)
        if rows_filtered > 0:
            logger.info(f"Filtered {rows_filtered} rows with timestamp > settlement date ({settlement_date_ts.date()}) for {contract_code}")
        else:
             logger.info(f"No rows filtered based on settlement date {settlement_date_ts.date()} for {contract_code}")

        # Log the date range after processing
        if not df_prep.empty:
             logger.info(f"Processed data date range for {contract_code}: {df_prep['timestamp'].min().date()} to {df_prep['timestamp'].max().date()}")
        else:
             logger.info(f"No data remains for {contract_code} after processing.")
             return pd.DataFrame() # Return empty if no data left

        # --- Check for duplicate PKs in the prepared data ---
        pk_cols = ['timestamp', 'symbol', 'interval_value', 'interval_unit']
        duplicates = df_prep[df_prep.duplicated(subset=pk_cols, keep=False)]
        if not duplicates.empty:
            logger.error(f"Found {len(duplicates)} duplicate primary key rows in prepared data for {contract_code}:")
            logger.error(duplicates.head().to_string())
            return pd.DataFrame() 

        # Select and order final columns for the DB
        # Target columns for market_data_cboe - NOW INCLUDES SETTLE, VOLUME, OPEN_INTEREST
        target_db_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest', 'interval_value', 'interval_unit', 'source']
        df_final = df_prep[target_db_cols].copy()
        return df_final

    except Exception as e:
        logger.error(f"Error preparing data for {contract_code}: {e}", exc_info=True)
        return pd.DataFrame()

def prepare_vix_index_data_for_db(df: pd.DataFrame):
    """Prepares the downloaded VIX index DataFrame for insertion into market_data_cboe."""
    # Use lowercase for robust matching
    actual_cols_lower = {col.lower().strip(): col for col in df.columns}
    date_col = actual_cols_lower.get('date')
    open_col = actual_cols_lower.get('open')
    high_col = actual_cols_lower.get('high')
    low_col = actual_cols_lower.get('low')
    # Treat the VIX index CSV 'CLOSE' as the definitive 'settle' price
    settle_col_source = actual_cols_lower.get('close') 

    required_cols = [date_col, open_col, high_col, low_col, settle_col_source]
    if not all(required_cols):
        logger.error(f"VIX index data missing one or more required columns (Date, Open, High, Low, Close). Found: {list(df.columns)}")
        return pd.DataFrame()

    try:
        # Select and rename columns
        # Map source 'close' to target 'settle'
        df_prep = df[[date_col, open_col, high_col, low_col, settle_col_source]].copy()
        df_prep.rename(columns={
            date_col: 'timestamp',
            open_col: 'open',
            high_col: 'high',
            low_col: 'low',
            settle_col_source: 'settle' # Map CSV CLOSE to DB settle
        }, inplace=True)

        # Create the 'close' column and fill it with the 'settle' value
        df_prep['close'] = df_prep['settle']

        # Assign metadata
        df_prep['symbol'] = VIX_INDEX_SYMBOL
        df_prep['interval_value'] = 1
        df_prep['interval_unit'] = 'daily'
        df_prep['source'] = 'CBOE' 
        df_prep['volume'] = 0 # VIX index has no volume

        # Convert timestamp
        df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'])

        # Convert OHLC, Close, and Settle to numeric, coercing errors
        num_cols = ['open', 'high', 'low', 'close', 'settle']
        for col in num_cols:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')

        # Drop rows where essential columns became NaN/NaT (prioritize settle)
        rows_before = len(df_prep)
        df_prep.dropna(subset=['timestamp', 'settle'], inplace=True)
        rows_after = len(df_prep)
        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows from VIX index data due to missing/invalid timestamp or settle value.")

        # Define final columns for the database table (matching market_data_cboe, including settle)
        final_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'interval_value', 'interval_unit', 'source']
        # Select and order final columns
        df_final = df_prep[final_cols].copy()
        return df_final

    except Exception as e:
        logger.error(f"Error preparing VIX index data: {e}", exc_info=True)
        return pd.DataFrame()

def update_market_data(conn, df_insert: pd.DataFrame):
    """Updates the CBOE-specific market data table using DELETE then INSERT."""
    if df_insert.empty:
        logger.info("No valid data prepared for database update.")
        return

    target_table = TARGET_TABLE_CBOE # Use the new table name

    # Get symbol and interval details from the DataFrame
    contract_code = df_insert['symbol'].iloc[0]
    interval_val = df_insert['interval_value'].iloc[0]
    interval_u = df_insert['interval_unit'].iloc[0]
    
    logger.info(f"Attempting DELETE/INSERT for {len(df_insert)} rows into {target_table} for {contract_code} ({interval_val}{interval_u})...")
    logger.info(f"Data date range for {contract_code}: {df_insert['timestamp'].min()} to {df_insert['timestamp'].max()}")
    
    try:
        # --- Ensure Table Schema --- 
        # Define desired types explicitly, especially for settle, volume, open_interest
        type_map = {
            'timestamp': duckdb.typing.TIMESTAMP,
            'open': duckdb.typing.DOUBLE,
            'high': duckdb.typing.DOUBLE,
            'low': duckdb.typing.DOUBLE,
            'close': duckdb.typing.DOUBLE,
            'settle': duckdb.typing.DOUBLE, # Explicitly define settle type
            'volume': duckdb.typing.BIGINT,   # Explicitly define volume type
            'open_interest': duckdb.typing.BIGINT, # Explicitly define open_interest type
            'interval_value': duckdb.typing.BIGINT,
            'symbol': duckdb.typing.VARCHAR,
            'interval_unit': duckdb.typing.VARCHAR,
            'source': duckdb.typing.VARCHAR
        }
        
        # 1. Create table if it doesn't exist (defines initial PK and essential cols)
        # Minimal create statement focusing on PK and base columns present initially
        pk_cols_str = ", ".join([f'"{c}" {type_map[c]} NOT NULL' for c in ['timestamp', 'symbol', 'interval_value', 'interval_unit']])
        base_cols_str = ", ".join([f'"{c}" {type_map[c]}' for c in ['open', 'high', 'low', 'settle', 'source'] if c in type_map]) # Initial known columns
        pk_constraint = f"PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)"
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {target_table} ({pk_cols_str}, {base_cols_str}, {pk_constraint});"
        conn.execute(create_table_sql)
        logger.debug(f"Ensured table {target_table} exists with base schema and PK.")
        
        # 2. Add missing columns if they don't exist using ALTER TABLE
        # These are the columns we added later: close, volume, open_interest
        columns_to_ensure = {
             'close': type_map['close'], 
             'volume': type_map['volume'], 
             'open_interest': type_map['open_interest']
        }
        for col, col_type in columns_to_ensure.items():
             try:
                 alter_sql = f'ALTER TABLE {target_table} ADD COLUMN IF NOT EXISTS "{col}" {col_type};'
                 conn.execute(alter_sql)
                 logger.debug(f"Ensured column '{col}' exists in {target_table}.")
             except Exception as alter_e:
                 logger.error(f"Error trying to ALTER TABLE {target_table} ADD COLUMN {col}: {alter_e}")
                 raise # Re-raise the error if altering fails

        # --- Proceed with Data Update --- 
        # Use parameterized query for checks and delete
        params = [contract_code, int(interval_val), interval_u]
        
        # Check existing data (optional logging)
        check_query = f"SELECT COUNT(*) as count FROM {target_table} WHERE symbol = ? AND interval_value = ? AND interval_unit = ?"
        existing_data = conn.execute(check_query, params).fetchdf()
        existing_count = 0
        if not existing_data.empty:
             existing_count = existing_data.iloc[0]['count']
             
        if existing_count > 0:
             logger.info(f"Existing data count for {contract_code} in {target_table}: {existing_count} rows. Will be deleted.")
        else:
            logger.info(f"No existing data found for {contract_code} in {target_table}. Table might be new or contract not present.")

        # *** DELETE existing data first ***
        # Only run delete if there was existing data
        if existing_count > 0:
            delete_query = f"DELETE FROM {target_table} WHERE symbol = ? AND interval_value = ? AND interval_unit = ? "
            delete_result = conn.execute(delete_query, params)
            # Safely check affected rows
            affected_rows = delete_result.fetchone() if delete_result else None
            logger.info(f"Executed DELETE for {contract_code} from {target_table}. Rows affected: {affected_rows[0] if affected_rows else 'N/A'}")
        
        # Register view and perform INSERT (using only columns present in df_insert)
        conn.register('df_insert_view', df_insert)
        cols = df_insert.columns
        col_names_db = ", ".join([f'"{c}"' for c in cols])
        col_names_df = ", ".join([f'"{c}"' for c in cols])
        sql = f""" 
            INSERT INTO {target_table} ({col_names_db})
            SELECT {col_names_df} FROM df_insert_view
        """
        conn.execute(sql)
        logger.info(f"Executed INSERT for {contract_code} into {target_table}. Rows inserted: {len(df_insert)}.")
        
    except duckdb.Error as e:
        logger.error(f"Database error updating data for {contract_code} in {target_table}: {e}", exc_info=True)
        # Rollback is likely not needed/possible here if auto-commit or single statement context
        # conn.rollback() 
    except Exception as e:
        logger.error(f"Unexpected error updating database for {contract_code} in {target_table}: {e}", exc_info=True)
        # conn.rollback()

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
                    df_prepared = prepare_data_for_db(df_downloaded, contract_code, settlement_date, conn)
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
            try:
                logger.info("Forcing database checkpoint before closing...")
                conn.sql('PRAGMA force_checkpoint;') # Explicitly force checkpoint
                logger.info("Checkpoint forced.")
            except Exception as e:
                logger.error(f"Error forcing checkpoint: {e}")
            finally:
                conn.close()
                logger.info("Database connection closed (local VX futures).")

        # *** Add Post-Update Verification Step ***
        logger.info("Performing post-update verification with a new connection...")
        verification_conn = None
        try:
            verification_conn = connect_db(args.db_path) # Reconnect read-write to ensure visibility?
            if verification_conn:
                # Get the expected max date from the last processed contract (if any)
                # Find the latest settlement date among processed contracts
                latest_settlement_date = None
                last_processed_symbol = None
                if 'df_active_contracts' in locals() and not df_active_contracts.empty:
                    latest_settlement_date = df_active_contracts['final_settlement_date'].max()
                    # Find the symbol corresponding to the latest settlement date
                    last_processed_symbol = df_active_contracts.loc[df_active_contracts['final_settlement_date'] == latest_settlement_date, 'contract_code'].iloc[0]
                    
                if last_processed_symbol:
                    # Use the timestamp from the last successfully prepared DataFrame if available
                    expected_max_date = None
                    if 'df_prepared' in locals() and not df_prepared.empty and df_prepared['symbol'].iloc[0] == last_processed_symbol:
                        expected_max_date = df_prepared['timestamp'].max().date()
                    else: # Fallback: Get today's date as a rough check if last df not available
                         expected_max_date = date.today() 

                    if expected_max_date:
                        logger.info(f"Verifying max date for {last_processed_symbol} in {TARGET_TABLE_CBOE}. Expecting >= {expected_max_date}")
                        verify_query = f"SELECT MAX(timestamp)::DATE FROM {TARGET_TABLE_CBOE} WHERE symbol = ? AND interval_value = 1 AND interval_unit = 'daily'"
                        actual_max_date_result = verification_conn.execute(verify_query, [last_processed_symbol]).fetchone()
                        
                        if actual_max_date_result and actual_max_date_result[0]:
                            actual_max_date = actual_max_date_result[0]
                            logger.info(f"Verification query returned max date: {actual_max_date} for {last_processed_symbol} from {TARGET_TABLE_CBOE}")
                            # Allow for slight variations, check if actual date is within a day of expected
                            if actual_max_date >= expected_max_date or (expected_max_date - actual_max_date).days <= 1:
                                logger.info(f"[SUCCESS] Post-update verification passed for {last_processed_symbol}. Max date {actual_max_date} is as expected (>= {expected_max_date}).")
                            else:
                                logger.critical(f"[FAILURE] Post-update verification FAILED for {last_processed_symbol}. Expected max date >= {expected_max_date}, but found {actual_max_date}.")
                        else:
                            logger.error(f"[FAILURE] Post-update verification FAILED for {last_processed_symbol}. Could not retrieve max date from {TARGET_TABLE_CBOE}.")
                    else:
                         logger.warning("Could not determine expected max date for verification.")
                else:
                    logger.warning("Skipping post-update verification: No contracts were processed.")
            else:
                 logger.error("Skipping post-update verification: Failed to establish verification connection.")
        except Exception as e:
            logger.error(f"Error during post-update verification: {e}", exc_info=True)
        finally:
            if verification_conn:
                verification_conn.close()
                logger.info("Verification database connection closed.")

if __name__ == "__main__":
    main() 