#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary script to debug the O1 summary counts for base symbol '@ES'.
Lists the source symbols and their counts that contribute to the O1 '@ES' total.
"""

import duckdb
import os
from pathlib import Path
import logging
import pandas as pd
from rich.console import Console
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# get_base function copied from summarize_symbol_inventory.py
def get_base_for_debug(symbol_str: str) -> str:
    if not isinstance(symbol_str, str):
        return str(symbol_str) # Or handle error appropriately

    if symbol_str.startswith('@') and '=' in symbol_str:
        return symbol_str.split('=')[0]
    elif symbol_str.startswith('$VIX'):
        return "$VIX.X"
    elif symbol_str.startswith('VX'):
        if len(symbol_str) >= 4 and symbol_str[2].isalpha() and symbol_str[3:].isdigit():
            return "VX"
        return symbol_str
    elif symbol_str in ['SPY', 'QQQ', 'AAPL', 'GS']:
        return symbol_str
    elif len(symbol_str) >= 4 and symbol_str[-3].isalpha() and symbol_str[-2:].isdigit():
        if symbol_str[-3] in 'FGHJKMNQUVXZ':
            return symbol_str[:-3]
    return symbol_str

def debug_o1_es_counts(db_access_path: str):
    """Connects to the DB and investigates @ES counts for O1 report."""
    conn = None
    console = Console()
    try:
        actual_db_path = Path(db_access_path).resolve()
        if not actual_db_path.exists():
            logger.error(f"Database file not found at resolved path: {actual_db_path}")
            return

        logger.info(f"Connecting to database: {actual_db_path} (Read-Only)")
        conn = duckdb.connect(database=str(actual_db_path), read_only=True)
        
        query = """
        WITH combined_data_for_debug AS (
            SELECT symbol, interval_unit, interval_value, timestamp FROM market_data
            UNION ALL
            SELECT symbol, 'day' as interval_unit, 1 as interval_value, timestamp FROM market_data_cboe
            UNION ALL
            SELECT symbol, interval_unit, interval_value, timestamp FROM continuous_contracts
        )
        SELECT 
            symbol, 
            interval_unit, 
            interval_value, 
            COUNT(*) as record_count,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date
        FROM combined_data_for_debug
        GROUP BY symbol, interval_unit, interval_value
        ORDER BY symbol, interval_unit, interval_value;
        """
        
        logger.info("Fetching all symbol data from combined sources...")
        all_symbols_df = conn.execute(query).fetchdf()

        if all_symbols_df.empty:
            logger.info("No data found in combined sources.")
            return

        logger.info(f"Applying get_base_for_debug to {len(all_symbols_df)} aggregated symbol/interval groups...")
        all_symbols_df['derived_base_symbol'] = all_symbols_df['symbol'].apply(get_base_for_debug)
        
        es_base_contributors_df = all_symbols_df[all_symbols_df['derived_base_symbol'] == '@ES'].copy()
        
        if es_base_contributors_df.empty:
            logger.info("No contributors found for base symbol '@ES' after applying get_base logic.")
        else:
            logger.info(f"Found {len(es_base_contributors_df)} symbol/interval groups contributing to '@ES' base symbol in O1:")
            
            # Sort for display
            es_base_contributors_df.sort_values(by=['symbol', 'interval_unit', 'interval_value'], inplace=True)
            es_base_contributors_df['first_date'] = pd.to_datetime(es_base_contributors_df['first_date']).dt.strftime('%Y-%m-%d')
            es_base_contributors_df['last_date'] = pd.to_datetime(es_base_contributors_df['last_date']).dt.strftime('%Y-%m-%d')

            display_table = Table(title="[bold blue]Source Contracts Contributing to O1 '@ES' Base Symbol[/bold blue]")
            display_table.add_column("Original Symbol", style="cyan")
            display_table.add_column("Interval Unit", style="white")
            display_table.add_column("Interval Value", style="white")
            display_table.add_column("Record Count", style="green", justify="right")
            display_table.add_column("First Date", style="yellow")
            display_table.add_column("Last Date", style="yellow")

            total_records_for_at_es = 0
            for _, row in es_base_contributors_df.iterrows():
                display_table.add_row(
                    str(row['symbol']),
                    str(row['interval_unit']),
                    str(row['interval_value']),
                    str(row['record_count']),
                    str(row['first_date']),
                    str(row['last_date'])
                )
                total_records_for_at_es += row['record_count']
            
            console.print(display_table)
            logger.info(f"Total records contributing to O1 '@ES' base symbol: {total_records_for_at_es}")
            if total_records_for_at_es != 238200: # O1 reported 238.2K
                 logger.warning(f"Calculated total {total_records_for_at_es} does NOT match O1 summary of 238.2K. Further investigation of O1 script grouping needed or data changed.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("Starting O1 @ES count debugger...")
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
        debug_o1_es_counts(db_path_to_use_str)
    except Exception as e:
        logger.critical(f"Failed to initialize/run: {e}", exc_info=True)
    logger.info("O1 @ES count debugger finished.") 