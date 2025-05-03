#!/usr/bin/env python
import logging
import sys
from datetime import date
import pandas as pd
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Adjust path if necessary, assuming runnable from root
    from src.scripts.market_data.vix.update_vx_futures import download_cboe_data, prepare_data_for_db
except ImportError:
    logger.error("Could not import functions. Ensure script is run from project root or PYTHONPATH is set.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

CONTRACT_CODE = 'VXU25'
# Settlement date found in previous logs
SETTLEMENT_DATE = date(2025, 9, 17) 
DB_PATH = "data/financial_data.duckdb"
TEMP_TABLE_NAME = "temp_vxu25_insert_test"

if __name__ == "__main__":
    logger.info(f"Attempting to fetch data for {CONTRACT_CODE} (settles: {SETTLEMENT_DATE})")
    conn = None # Initialize conn outside try block
    df_prepared = None # Initialize df_prepared

    try:
        # 1. Download
        df_downloaded = download_cboe_data(CONTRACT_CODE, SETTLEMENT_DATE)
        
        if df_downloaded is not None and not df_downloaded.empty:
            logger.info(f"Download successful. Shape: {df_downloaded.shape}")
            
            # 2. Prepare (using a temporary connection as prepare_data might need it)
            try:
                prep_conn = duckdb.connect(DB_PATH, read_only=True) # Read-only needed?
                df_prepared = prepare_data_for_db(df_downloaded, CONTRACT_CODE, SETTLEMENT_DATE, prep_conn)
                prep_conn.close()
            except Exception as prep_e:
                 logger.error(f"Error during data preparation: {prep_e}", exc_info=True)
                 df_prepared = None # Ensure it's None if prep fails

            if df_prepared is not None and not df_prepared.empty:
                logger.info(f"Preparation successful. Prepared shape: {df_prepared.shape}")
                logger.info(f"Prepared data range: {df_prepared['timestamp'].min()} to {df_prepared['timestamp'].max()}")

                # 3. Connect to DB (Read-Write for temp table creation)
                conn = duckdb.connect(DB_PATH, read_only=False)
                logger.info("Connected to DB for temp table operations.")

                # 4. Create temp table and insert data
                conn.register('prepared_data_view', df_prepared)
                # Use CREATE OR REPLACE in case script run multiple times in session
                create_sql = f"CREATE OR REPLACE TEMP TABLE {TEMP_TABLE_NAME} AS SELECT * FROM prepared_data_view;"
                conn.execute(create_sql)
                conn.commit() # Commit creation/insertion
                logger.info(f"Created/Replaced temp table '{TEMP_TABLE_NAME}' and inserted data.")

                # 5. Query the temp table
                query_sql = f"SELECT COUNT(*) as count, MIN(timestamp)::DATE as min_date, MAX(timestamp)::DATE as max_date FROM {TEMP_TABLE_NAME}"
                result = conn.execute(query_sql).fetchone()
                
                if result:
                    logger.info(f"--- Query Result from TEMP Table '{TEMP_TABLE_NAME}' ---")
                    logger.info(f"Row Count: {result[0]}")
                    logger.info(f"Min Date:  {result[1]}")
                    logger.info(f"Max Date:  {result[2]}")
                    logger.info(f"-----------------------------------------------------")
                    # Explicitly check if max date is as expected
                    if result[2] == date(2025, 4, 25):
                        logger.info("SUCCESS: Max date in temp table is 2025-04-25!")
                    else:
                        logger.error(f"FAILURE: Max date in temp table is {result[2]}, expected 2025-04-25.")
                else:
                    logger.error(f"Could not query results from temp table '{TEMP_TABLE_NAME}'.")

            elif df_prepared is None:
                 logger.error("Data preparation failed. Cannot proceed with database insertion.")
            else: # df_prepared is empty
                logger.warning("Prepared DataFrame was empty. Skipping database insertion.")

        elif df_downloaded is None:
            logger.error(f"Download failed for {CONTRACT_CODE}. Function returned None.")
        else: # DataFrame is empty
             logger.warning(f"Download resulted in an empty DataFrame for {CONTRACT_CODE}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if conn:
            # Optional: Drop temp table if needed, but it's session-scoped anyway
            # try:
            #     conn.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE_NAME};")
            #     logger.info(f"Dropped temp table {TEMP_TABLE_NAME}.")
            # except Exception as drop_e:
            #      logger.warning(f"Could not drop temp table {TEMP_TABLE_NAME}: {drop_e}")
            conn.close()
            logger.info("Database connection closed.") 