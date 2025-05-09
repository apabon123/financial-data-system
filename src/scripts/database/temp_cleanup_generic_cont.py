#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to delete data and metadata for generic continuous symbols @ES and @NQ.
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

def cleanup_generic_symbols(db_path: str):
    """Connects to the DB and deletes @ES and @NQ entries."""
    conn = None
    try:
        logger.info(f"Connecting to database: {db_path}")
        conn = duckdb.connect(database=db_path, read_only=False)
        
        symbols_to_delete = ('@ES', '@NQ')
        # Parameterized queries use placeholders like ?

        # 0. Query existing data count first for diagnostics
        logger.info(f"Querying count for symbols {symbols_to_delete} in continuous_contracts...")
        query_count_sql = "SELECT COUNT(*) FROM continuous_contracts WHERE symbol IN (?, ?)"
        count_result = conn.execute(query_count_sql, symbols_to_delete).fetchone()
        count_found = count_result[0] if count_result else 0
        logger.info(f"Found {count_found} rows matching {symbols_to_delete} in continuous_contracts before delete.")

        # 1. Delete from continuous_contracts using parameters
        logger.info(f"Attempting to delete data for symbols {symbols_to_delete} from continuous_contracts...")
        delete_data_sql = "DELETE FROM continuous_contracts WHERE symbol IN (?, ?)"
        deleted_data_count = conn.execute(delete_data_sql, symbols_to_delete).fetchone()[0]
        logger.info(f"Deleted {deleted_data_count} rows from continuous_contracts for symbols {symbols_to_delete}.")

        # 2. Delete from symbol_metadata (using base_symbol column) using parameters
        logger.info(f"Attempting to delete metadata for base_symbols {symbols_to_delete} from symbol_metadata...")
        delete_meta_sql = "DELETE FROM symbol_metadata WHERE base_symbol IN (?, ?)"
        deleted_meta_count = conn.execute(delete_meta_sql, symbols_to_delete).fetchone()[0]
        logger.info(f"Deleted {deleted_meta_count} rows from symbol_metadata for base_symbols {symbols_to_delete}.")

        conn.commit()
        logger.info("Changes committed.")

    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}", exc_info=True)
        if conn:
            conn.rollback()
            logger.info("Changes rolled back due to error.")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("Starting cleanup script for @ES and @NQ...")
    cleanup_generic_symbols(DB_PATH)
    logger.info("Cleanup script finished.") 