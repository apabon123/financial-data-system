#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to find specific @ES=101XN and @ES=102XC symbols across all relevant data tables.
"""

import duckdb
import os
from pathlib import Path
import logging
import pandas as pd
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def find_specific_es_variants_across_tables(db_access_path: str):
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path} (Read-Only)")
        conn = duckdb.connect(database=str(actual_db_path), read_only=True)

        symbols_to_find = ['@ES=101XN', '@ES=102XC']
        tables_to_search = ['market_data', 'market_data_cboe', 'continuous_contracts']
        results_data = []

        for table_name in tables_to_search:
            for symbol_val in symbols_to_find:
                logger.info(f"--- Querying for symbol: '{symbol_val}' in table: '{table_name}' ---")
                query = f"""
                SELECT 
                    '{symbol_val}' as symbol,
                    '{table_name}' as table_name,
                    COUNT(*) as record_count,
                    MIN(timestamp) as min_timestamp,
                    MAX(timestamp) as max_timestamp
                FROM {table_name}
                WHERE symbol = ?;
                """
                # Using .fetchone() because we expect a single row (count summary)
                result = conn.execute(query, [symbol_val]).fetchone()
                
                if result and result[2] > 0: # result[2] is record_count
                    results_data.append([
                        result[0], # symbol
                        result[1], # table_name
                        result[2], # record_count
                        result[3], # min_timestamp
                        result[4]  # max_timestamp
                    ])
                else:
                    results_data.append([symbol_val, table_name, 0, None, None])
        
        if not results_data:
            logger.info("No data found for the specified ES variants in any searched table.")
            return

        console = Console()
        table_display = Table(title="Location of @ES=101XN and @ES=102XC Variants")
        table_display.add_column("Symbol", style="cyan")
        table_display.add_column("Found In Table", style="yellow")
        table_display.add_column("Record Count", style="magenta")
        table_display.add_column("Min Timestamp", style="green")
        table_display.add_column("Max Timestamp", style="blue")

        for row_data in results_data:
            table_display.add_row(
                str(row_data[0]), 
                str(row_data[1]), 
                str(row_data[2]), 
                str(row_data[3]) if row_data[3] else 'N/A', 
                str(row_data[4]) if row_data[4] else 'N/A'
            )
        console.print(table_display)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info(f"Starting search for specific @ES variants across tables...")
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
        find_specific_es_variants_across_tables(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run script: {e}", exc_info=True)
    logger.info("Specific ES variants search finished.") 