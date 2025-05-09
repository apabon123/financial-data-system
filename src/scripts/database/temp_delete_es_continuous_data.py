#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to delete all data from continuous_contracts for symbols starting with '@ES='.
"""

import duckdb
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def delete_at_es_data(db_access_path: str):
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path}")
        conn = duckdb.connect(database=str(actual_db_path), read_only=False)

        symbol_pattern_to_delete = '@ES=%'
        table_name = 'continuous_contracts'

        check_sql = f"SELECT COUNT(*) FROM {table_name} WHERE symbol LIKE ?"
        delete_sql = f"DELETE FROM {table_name} WHERE symbol LIKE ? RETURNING symbol;"

        logger.info(f"Checking for rows to delete from '{table_name}' where symbol LIKE '{symbol_pattern_to_delete}'")
        
        conn.begin()

        count_to_delete_res = conn.execute(check_sql, [symbol_pattern_to_delete]).fetchone()
        count_to_delete = count_to_delete_res[0] if count_to_delete_res else 0

        if count_to_delete > 0:
            logger.info(f"Found {count_to_delete} records in '{table_name}' matching pattern '{symbol_pattern_to_delete}' to be deleted.")
            
            result = conn.execute(delete_sql, [symbol_pattern_to_delete])
            affected_rows_list = result.fetchall()
            num_actually_deleted = len(affected_rows_list)

            conn.commit()
            logger.info(f"Successfully deleted {num_actually_deleted} records from '{table_name}' for symbols LIKE '{symbol_pattern_to_delete}'. Commit successful.")
            
            logger.info("Verifying deletion:")
            remaining_count_res = conn.execute(check_sql, [symbol_pattern_to_delete]).fetchone()
            logger.info(f"Rows LIKE '{symbol_pattern_to_delete}' in '{table_name}' after delete: {remaining_count_res[0] if remaining_count_res else 0}")
        else:
            logger.info(f"No records found in '{table_name}' matching pattern '{symbol_pattern_to_delete}'. No deletion performed.")
            conn.rollback() # Rollback if no action taken

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to error.")
            except Exception as rb_err:
                logger.error(f"Failed to rollback: {rb_err}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info(f"Starting deletion of all '@ES=%' data from continuous_contracts...")
    try:
        script_dir = Path(__file__).resolve().parent
        db_path_to_use = script_dir.parent.parent.parent / "data" / "financial_data.duckdb"
        db_path_to_use_str = os.getenv('DUCKDB_PATH', str(db_path_to_use.resolve()))

        if not Path(db_path_to_use_str).exists():
            logger.error(f"Database path not found: {db_path_to_use_str}")
            fallback_db_path = Path("data/financial_data.duckdb").resolve()
            if fallback_db_path.exists():
                logger.info(f"Using fallback DB path: {fallback_db_path}")
                db_path_to_use_str = str(fallback_db_path)
            else:
                logger.error(f"Fallback DB path also not found: {fallback_db_path}.")
                exit(1)
        delete_at_es_data(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run deletion script: {e}", exc_info=True)
    logger.info("Deletion script finished.") 