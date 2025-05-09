#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to delete specific @ES= variants from the market_data table.
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

def delete_specific_es_from_market_data(db_access_path: str):
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path}")
        conn = duckdb.connect(database=str(actual_db_path), read_only=False)

        symbols_to_delete = ['@ES=101XN', '@ES=102XC'] # Include both just in case
        table_name = 'market_data'
        
        conn.begin()
        total_deleted_count = 0

        for symbol_val in symbols_to_delete:
            logger.info(f"Checking for symbol '{symbol_val}' in table '{table_name}'...")
            check_sql = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
            delete_sql = f"DELETE FROM {table_name} WHERE symbol = ? RETURNING symbol;"

            count_res = conn.execute(check_sql, [symbol_val]).fetchone()
            count = count_res[0] if count_res else 0

            if count > 0:
                logger.info(f"Found {count} records for '{symbol_val}' in '{table_name}'. Deleting...")
                result = conn.execute(delete_sql, [symbol_val])
                affected_rows_list = result.fetchall()
                num_actually_deleted = len(affected_rows_list)
                total_deleted_count += num_actually_deleted
                logger.info(f"Successfully deleted {num_actually_deleted} records for '{symbol_val}'.")
            else:
                logger.info(f"No records found for '{symbol_val}' in '{table_name}'.")
        
        if total_deleted_count > 0:
            conn.commit()
            logger.info(f"Total {total_deleted_count} records deleted from '{table_name}'. Commit successful.")
        else:
            logger.info(f"No records matching specified symbols were found in '{table_name}'. No changes made.")
            conn.rollback()

        logger.info("Verifying deletion:")
        for symbol_val in symbols_to_delete:
            remaining_count_res = conn.execute(check_sql, [symbol_val]).fetchone()
            logger.info(f"Rows for '{symbol_val}' in '{table_name}' after operation: {remaining_count_res[0] if remaining_count_res else 0}")

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
    logger.info(f"Starting deletion of specific @ES variants from { 'market_data' }...")
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
        delete_specific_es_from_market_data(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run deletion script: {e}", exc_info=True)
    logger.info("Deletion script for specific ES variants from market_data finished.") 