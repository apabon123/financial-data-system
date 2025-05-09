#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to directly verify row counts for specific symbols in specified tables.
"""

import duckdb
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def verify_counts(db_access_path: str):
    """Connects to the DB and queries counts for @ES and @NQ in specified tables."""
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found at resolved path: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path} (Read-Only)")
        conn = duckdb.connect(database=str(actual_db_path), read_only=True)
        
        symbols_to_check = ('@ES', '@NQ')
        # Explicitly include market_data_cboe if it's a potential source for these symbols
        tables_to_check = ['market_data', 'continuous_contracts', 'market_data_cboe'] 

        for table in tables_to_check:
            logger.info(f"--- Checking table: {table} ---")
            for symbol in symbols_to_check:
                try:
                    query = f"SELECT COUNT(*) FROM {table} WHERE symbol = ?"
                    result = conn.execute(query, [symbol]).fetchone()
                    count = result[0] if result else 0
                    logger.info(f"Symbol '{symbol}' in table '{table}': {count} rows.")
                except duckdb.CatalogException:
                    logger.warning(f"Table '{table}' not found or schema mismatch. Skipping checks for this table.")
                    break # No point checking other symbols in a non-existent/problematic table
                except Exception as e:
                    logger.error(f"Error querying '{symbol}' in '{table}': {e}")
        
    except Exception as e:
        logger.error(f"An error occurred during verification: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("Starting direct count verification for @ES, @NQ...")
    
    # Determine DB path: Assumes script is in project_root/src/scripts/database/
    # and DB is in project_root/data/financial_data.duckdb
    try:
        script_dir = Path(__file__).resolve().parent
        db_path_to_use = script_dir.parent.parent.parent / "data" / "financial_data.duckdb"
        
        # Allow override by DUCKDB_PATH environment variable if set
        db_path_to_use_str = os.getenv('DUCKDB_PATH', str(db_path_to_use.resolve()))

        if not Path(db_path_to_use_str).exists():
             logger.error(f"Database path derived or from DUCKDB_PATH does not exist: {db_path_to_use_str}")
             # Fallback for common case: CWD is project root
             fallback_db_path = Path("data/financial_data.duckdb").resolve()
             if fallback_db_path.exists():
                 logger.info(f"Using fallback DB path: {fallback_db_path}")
                 db_path_to_use_str = str(fallback_db_path)
             else:
                 logger.error(f"Fallback DB path also not found: {fallback_db_path}. Please check DB location or DUCKDB_PATH.")
                 exit(1)
        
        verify_counts(db_path_to_use_str)

    except Exception as e:
        logger.critical(f"Failed to initialize and run verification: {e}", exc_info=True)
    
    logger.info("Verification script finished.") 