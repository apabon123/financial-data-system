#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to check for @ES=101XN and @ES=102XC in continuous_contracts table.
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

def check_es_variants(db_access_path: str):
    conn = None
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path} (Read-Only)")
        conn = duckdb.connect(database=str(actual_db_path), read_only=True)

        symbols_to_check = ['@ES=101XN', '@ES=102XC']
        results_data = []

        for symbol_val in symbols_to_check:
            logger.info(f"--- Querying for symbol: {symbol_val} in continuous_contracts ---")
            query = f"""
            SELECT 
                '{symbol_val}' as symbol,
                COUNT(*) as record_count,
                MIN(timestamp) as min_timestamp,
                MAX(timestamp) as max_timestamp,
                interval_unit,
                interval_value
            FROM continuous_contracts
            WHERE symbol = ?
            GROUP BY interval_unit, interval_value
            ORDER BY interval_unit, interval_value;
            """
            df = conn.execute(query, [symbol_val]).fetchdf()
            if not df.empty:
                for _index, row in df.iterrows():
                    results_data.append([
                        row['symbol'], 
                        row['record_count'], 
                        row['min_timestamp'], 
                        row['max_timestamp'],
                        f"{row['interval_value']}{row['interval_unit']}"
                    ])
            else:
                results_data.append([symbol_val, 0, None, None, 'N/A'])
        
        if not results_data:
            logger.info("No data found for the specified ES variants in continuous_contracts.")
            return

        console = Console()
        table = Table(title="@ES Variants in continuous_contracts")
        table.add_column("Symbol", style="cyan")
        table.add_column("Record Count", style="magenta")
        table.add_column("Min Timestamp", style="green")
        table.add_column("Max Timestamp", style="blue")
        table.add_column("Interval", style="yellow")

        for row_data in results_data:
            table.add_row(
                str(row_data[0]), 
                str(row_data[1]), 
                str(row_data[2]) if row_data[2] else 'N/A', 
                str(row_data[3]) if row_data[3] else 'N/A',
                str(row_data[4])
            )
        console.print(table)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info(f"Starting check for @ES variants in continuous_contracts...")
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
        check_es_variants(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run script: {e}", exc_info=True)
    logger.info("ES variants check finished.") 