#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to diagnose and fix VIX index data loading issues
"""

import sys
import duckdb
import pandas as pd
import logging
import requests
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DB_PATH = "data/financial_data.duckdb"
VIX_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_INDEX_SYMBOL = "$VIX.X"

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
    except Exception as e:
        logger.error(f"Error downloading VIX index data: {e}")
        return None

def prepare_vix_index_data_for_db(df: pd.DataFrame):
    """Prepares the downloaded VIX index DataFrame for insertion into market_data."""
    try:
        # Select and rename columns
        df_prep = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']].copy()
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
        df_prep['source'] = 'CBOE'
        
        # Convert timestamp
        df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'])
        
        # Add any missing columns that might be required by the schema
        df_prep['close'] = df_prep['settle']  # Add close column matching settle
        df_prep['volume'] = None
        df_prep['open_interest'] = None
        df_prep['up_volume'] = None
        df_prep['down_volume'] = None
        df_prep['adjusted'] = False
        df_prep['quality'] = 100
        df_prep['changed'] = False
        df_prep['UnderlyingSymbol'] = None

        # Convert numeric columns
        num_cols = ['open', 'high', 'low', 'settle', 'close']
        for col in num_cols:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')

        # Drop rows with NaN values in critical columns
        df_prep.dropna(subset=['timestamp', 'open', 'high', 'low', 'settle'], inplace=True)
        
        logger.info(f"Prepared data shape: {df_prep.shape}")
        return df_prep
    except Exception as e:
        logger.error(f"Error preparing VIX index data: {e}")
        return pd.DataFrame()

def debug_market_data_table(conn):
    """Print information about the market_data table structure and contents"""
    try:
        logger.info("Checking market_data table structure...")
        columns = conn.execute("PRAGMA table_info(market_data)").fetchdf()
        logger.info(f"Table columns: {columns}")
        
        try:
            pk_info = conn.execute("SELECT name FROM pragma_table_info('market_data') WHERE pk > 0").fetchdf()
            logger.info(f"Primary key columns: {pk_info['name'].tolist() if not pk_info.empty else 'None'}")
        except Exception as pk_e:
            logger.error(f"Error getting primary key info: {pk_e}")
        
        row_count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        logger.info(f"Total market_data rows: {row_count}")
        
        symbol_counts = conn.execute("SELECT symbol, COUNT(*) as count FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 10").fetchdf()
        logger.info(f"Top 10 symbols by count: \n{symbol_counts}")
        
        vix_count = conn.execute(f"SELECT COUNT(*) FROM market_data WHERE symbol = '{VIX_INDEX_SYMBOL}'").fetchone()[0]
        logger.info(f"VIX index row count: {vix_count}")
        
        # Check for any rows with VIX in the symbol name
        vix_like = conn.execute("SELECT symbol, COUNT(*) FROM market_data WHERE symbol LIKE '%VIX%' GROUP BY symbol").fetchdf()
        logger.info(f"Symbols containing 'VIX': \n{vix_like}")
        
    except Exception as e:
        logger.error(f"Error debugging market_data table: {e}")

def attempt_direct_insert(conn, df):
    """Attempt to directly insert VIX data row by row"""
    try:
        # First, delete any existing VIX data to avoid conflicts
        logger.info(f"Deleting existing VIX index data...")
        conn.execute(f"DELETE FROM market_data WHERE symbol = '{VIX_INDEX_SYMBOL}'")
        
        # Use a transaction to ensure all or nothing is inserted
        logger.info(f"Beginning transaction for direct insertion...")
        conn.begin()
        
        # Get a list of column names in the correct order from the database table
        table_info = conn.execute("PRAGMA table_info(market_data)").fetchdf()
        column_names = table_info['name'].tolist()
        logger.info(f"Market data table columns (in order): {column_names}")
        
        sample_size = min(5, len(df))
        logger.info(f"First {sample_size} rows to insert: \n{df.head(sample_size)}")
        
        # Manually prepare data for direct insertion
        # Use a simpler approach with prepared statements
        logger.info("Using raw SQL with parameters for insertion...")
        
        # Create a minimal DataFrame with just the essential columns
        insert_df = pd.DataFrame()
        insert_df['timestamp'] = df['timestamp']
        insert_df['symbol'] = df['symbol']
        insert_df['open'] = df['open']
        insert_df['high'] = df['high']
        insert_df['low'] = df['low']
        insert_df['close'] = df['close']
        insert_df['settle'] = df['settle']
        insert_df['interval_value'] = df['interval_value']
        insert_df['interval_unit'] = df['interval_unit']
        insert_df['source'] = df['source']
        
        # Register the DataFrame as a view
        conn.register('insert_df_view', insert_df)
        
        # Use a simple INSERT with only the essential columns
        insert_sql = """
        INSERT INTO market_data (
            timestamp, symbol, open, high, low, close, settle, 
            interval_value, interval_unit, source
        ) 
        SELECT 
            timestamp, symbol, open, high, low, close, settle, 
            interval_value, interval_unit, source
        FROM insert_df_view
        """
        
        logger.info("Executing insert SQL...")
        conn.execute(insert_sql)
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully inserted {len(df)} VIX index rows")
        
        # Verify insertion
        final_count = conn.execute(f"SELECT COUNT(*) FROM market_data WHERE symbol = '{VIX_INDEX_SYMBOL}'").fetchone()[0]
        logger.info(f"Final VIX index row count: {final_count}")
        
        if final_count > 0:
            # Get a sample of the inserted data
            sample = conn.execute(f"SELECT timestamp::DATE, open, high, low, settle FROM market_data WHERE symbol = '{VIX_INDEX_SYMBOL}' ORDER BY timestamp LIMIT 5").fetchdf()
            logger.info(f"Sample of inserted VIX data: \n{sample}")
        
    except Exception as e:
        logger.error(f"Error during direct insert: {e}")
        try:
            conn.rollback()
            logger.info("Transaction rolled back due to error")
        except:
            pass

def main():
    try:
        # Connect to the database
        logger.info(f"Connecting to database: {DB_PATH}")
        conn = duckdb.connect(DB_PATH, read_only=False)
        
        # Debug the market_data table
        debug_market_data_table(conn)
        
        # Download VIX index data
        df_vix = download_vix_index_data()
        if df_vix is None or df_vix.empty:
            logger.error("Failed to download VIX index data. Exiting.")
            return
        
        logger.info(f"Downloaded VIX data shape: {df_vix.shape}")
        logger.info(f"First 5 rows: \n{df_vix.head()}")
        
        # Prepare data for insertion
        df_prepared = prepare_vix_index_data_for_db(df_vix)
        if df_prepared.empty:
            logger.error("Failed to prepare VIX index data. Exiting.")
            return
        
        # Attempt direct insertion
        attempt_direct_insert(conn, df_prepared)
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        # Close connection
        try:
            if 'conn' in locals() and conn is not None:
                conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

if __name__ == "__main__":
    main() 