#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to delete data from continuous_contracts for generic symbols @ES and @NQ.
"""

import os
import sys
import duckdb
import logging
from pathlib import Path

# Add project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Database Path ---
# Use environment variable or default relative path
DATA_DIR = os.getenv('DATA_DIR', os.path.join(project_root, 'data'))
DEFAULT_DB_PATH = os.path.join(DATA_DIR, 'financial_data.duckdb')
DB_PATH = os.getenv('DUCKDB_PATH', DEFAULT_DB_PATH)
# ---------------------

def cleanup_generic_at_symbols(db_path: str):
    """Connects to the DB and deletes @ES and @NQ entries from continuous_contracts."""
    conn = None
    try:
        logger.info(f"Connecting to database: {db_path}")
        conn = duckdb.connect(database=db_path, read_only=False)
        conn.begin() # Start a transaction explicitly
        
        symbols_to_delete = ('@ES', '@NQ')
        tables_to_clean = ['continuous_contracts', 'market_data']
        
        for table_name in tables_to_clean:
            logger.info(f"--- Processing table: {table_name} ---")
            # Query existing data count first for diagnostics
            logger.info(f"Querying count for symbols {symbols_to_delete} in {table_name}...")
            
            # Check each symbol individually for clarity in logging
            for sym_to_check in symbols_to_delete:
                try:
                    query_count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
                    count_result = conn.execute(query_count_sql, [sym_to_check]).fetchone()
                    count_found = count_result[0] if count_result else 0
                    logger.info(f"Found {count_found} rows matching '{sym_to_check}' in {table_name} before delete.")
                except duckdb.Error as e:
                    logger.warning(f"Could not query count for {sym_to_check} in {table_name} (table might not exist or schema mismatch): {e}")
                    continue # Skip to next symbol or table if count fails

            # Delete from the current table using parameters
            logger.info(f"Attempting to delete data for symbols {symbols_to_delete} from {table_name}...")
            
            for sym_to_del in symbols_to_delete:
                try:
                    delete_data_sql = f"DELETE FROM {table_name} WHERE symbol = ?"
                    conn.execute(delete_data_sql, [sym_to_del])
                    logger.info(f"Executed DELETE for symbol '{sym_to_del}' from {table_name}. (Actual count reflected after commit)")
                except duckdb.Error as e:
                    logger.warning(f"Could not delete {sym_to_del} from {table_name}: {e}")
            
        conn.commit()
        logger.info(f"Successfully executed DELETE for {symbols_to_delete} from tables {tables_to_clean} and committed changes.")
        
        # Verify deletion
        logger.info("Verifying deletion...")
        for table_name in tables_to_clean:
            logger.info(f"--- Verifying table: {table_name} ---")
            for sym_to_check in symbols_to_delete:
                try:
                    query_count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
                    count_result = conn.execute(query_count_sql, [sym_to_check]).fetchone()
                    count_found = count_result[0] if count_result else 0
                    logger.info(f"Found {count_found} rows matching '{sym_to_check}' in {table_name} AFTER delete.")
                except duckdb.Error as e:
                    logger.warning(f"Could not verify count for {sym_to_check} in {table_name}: {e}")

    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                logger.info("Changes rolled back due to error.")
            except Exception as rb_e:
                logger.error(f"Rollback attempt failed: {rb_e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("Starting cleanup of generic @ES, @NQ symbols from continuous_contracts...")
    cleanup_generic_at_symbols(DB_PATH)
    logger.info("Cleanup script finished.") 