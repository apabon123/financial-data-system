#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to correct mislabeled @ES=102XN (unadjusted) to @ES=101XN
in the continuous_contracts table.
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

def correct_es_unadjusted_symbol(db_access_path: str):
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path}")
        conn = duckdb.connect(database=str(actual_db_path), read_only=False)

        target_symbol_incorrect = '@ES=102XN'
        target_symbol_correct = '@ES=101XN'
        
        # Condition: symbol is the incorrect one AND it's marked as unadjusted (adjusted = False)
        # or if adjusted column is not reliably populated, one might use other criteria
        # For now, we rely on the symbol name and assume any '@ES=102XN' that ISN'T adjusted=True should be corrected.
        # The safest is to assume that if we find '@ES=102XN' and adjusted is False, it's the one.

        check_sql = f"""SELECT COUNT(*) FROM continuous_contracts 
                       WHERE symbol = ? AND (adjusted = False OR adjusted IS NULL)"""
        
        update_sql = f"""UPDATE continuous_contracts 
                        SET symbol = ? 
                        WHERE symbol = ? AND (adjusted = False OR adjusted IS NULL)
                        RETURNING symbol;""" # Corrected RETURNING clause

        logger.info(f"Checking for rows to update: Incorrect='{target_symbol_incorrect}' (unadjusted) to Correct='{target_symbol_correct}'")
        
        # Using a transaction
        conn.begin()

        count_to_update_res = conn.execute(check_sql, [target_symbol_incorrect]).fetchone()
        count_to_update = count_to_update_res[0] if count_to_update_res else 0

        if count_to_update > 0:
            logger.info(f"Found {count_to_update} groups of records (by interval) for '{target_symbol_incorrect}' (unadjusted) to be updated to '{target_symbol_correct}'.")
            
            # The RETURNING clause in DuckDB for UPDATE returns one row for each *updated* row.
            # To get a count of effective changes (how many rows were actually modified):
            result = conn.execute(update_sql, [target_symbol_correct, target_symbol_incorrect])
            affected_rows_count = result.fetchall() # This will be a list of tuples, len() gives num rows
            num_actually_updated = len(affected_rows_count)

            conn.commit()
            logger.info(f"Successfully updated {num_actually_updated} records from '{target_symbol_incorrect}' (unadjusted) to '{target_symbol_correct}'. Commit successful.")
            
            logger.info("Verifying correction:")
            corrected_count_res = conn.execute("SELECT COUNT(*) FROM continuous_contracts WHERE symbol = ?", [target_symbol_correct]).fetchone()
            remaining_incorrect_count_res = conn.execute("SELECT COUNT(*) FROM continuous_contracts WHERE symbol = ? AND (adjusted = False OR adjusted IS NULL)", [target_symbol_incorrect]).fetchone()
            
            logger.info(f"Rows for '{target_symbol_correct}' after update: {corrected_count_res[0] if corrected_count_res else 0}")
            logger.info(f"Rows for '{target_symbol_incorrect}' (unadjusted) after update: {remaining_incorrect_count_res[0] if remaining_incorrect_count_res else 0}")
        else:
            logger.info(f"No records found for '{target_symbol_incorrect}' (unadjusted). No update performed.")
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
    logger.info("Starting correction for mislabeled unadjusted @ES symbol...")
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
        correct_es_unadjusted_symbol(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run correction script: {e}", exc_info=True)
    logger.info("Correction script finished.") 