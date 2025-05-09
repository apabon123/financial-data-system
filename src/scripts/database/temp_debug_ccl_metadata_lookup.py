#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temp script to debug the metadata lookup performed by continuous_contract_loader.py
for a specific continuous symbol.
"""
import os
import sys
from pathlib import Path
import logging
import duckdb
import argparse
from rich.console import Console
from rich.table import Table

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")
METADATA_TABLE_NAME = "symbol_metadata"

def debug_metadata_lookup(db_path: str, specific_symbol_arg: str):
    logger.info(f"--- Debugging metadata lookup for: '{specific_symbol_arg}' in DB: {db_path} ---")
    conn = None
    console = Console()
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        logger.info(f"Successfully connected to database.")

        # --- 1. Test the specific query --- 
        logger.info(f"\n--- Testing specific query for: '{specific_symbol_arg}' ---")
        query_specific = f"""
            SELECT base_symbol, data_table, data_source, asset_type, interval_unit, interval_value 
            FROM {METADATA_TABLE_NAME} 
            WHERE base_symbol = ? AND asset_type = 'continuous_future' AND data_source IS NOT NULL AND data_source != '';
        """
        logger.info(f"Executing query_specific: {query_specific.strip()} with param: '{specific_symbol_arg}'")
        metadata_rows = conn.execute(query_specific, [specific_symbol_arg]).fetchall()
        
        logger.info(f"query_specific returned {len(metadata_rows)} rows.")
        if metadata_rows:
            rich_table_specific = Table(title=f"Specific Metadata Results for {specific_symbol_arg}")
            # Assuming columns are base_symbol, data_table, data_source, asset_type, interval_unit, interval_value
            columns = ["base_symbol", "data_table", "data_source", "asset_type", "interval_unit", "interval_value"]
            for col in columns:
                rich_table_specific.add_column(col)
            for row in metadata_rows:
                rich_table_specific.add_row(*[str(item) for item in row])
            console.print(rich_table_specific)
        else:
            logger.info("No rows found by query_specific.")

        # --- 2. Test the generic query --- 
        config_lookup_root = specific_symbol_arg.split('=')[0].replace('@', '') # E.g., VX, ES, NQ
        generic_base_symbol_query_val = f"@{config_lookup_root}" # E.g., @VX, @ES
        logger.info(f"\n--- Testing generic query for: '{generic_base_symbol_query_val}' (derived from {specific_symbol_arg}) ---")
        
        query_generic = f"""
            SELECT base_symbol, data_table, data_source, asset_type, interval_unit, interval_value 
            FROM {METADATA_TABLE_NAME} 
            WHERE base_symbol = ? AND asset_type = 'continuous_future' AND data_source IS NOT NULL AND data_source != ''
            LIMIT 1; 
        """
        # Note: CCL uses LIMIT 1 for its generic fallback, but for debugging let's see all matches for generic
        query_generic_debug = f"""
            SELECT base_symbol, data_table, data_source, asset_type, interval_unit, interval_value 
            FROM {METADATA_TABLE_NAME} 
            WHERE base_symbol = ? AND asset_type = 'continuous_future' AND data_source IS NOT NULL AND data_source != ''; 
        """
        logger.info(f"Executing query_generic_debug: {query_generic_debug.strip()} with param: '{generic_base_symbol_query_val}'")
        generic_metadata_rows = conn.execute(query_generic_debug, [generic_base_symbol_query_val]).fetchall()

        logger.info(f"query_generic_debug returned {len(generic_metadata_rows)} rows.")
        if generic_metadata_rows:
            rich_table_generic = Table(title=f"Generic Metadata Results for {generic_base_symbol_query_val}")
            for col in columns: # Using same columns as above
                rich_table_generic.add_column(col)
            for row in generic_metadata_rows:
                rich_table_generic.add_row(*[str(item) for item in row])
            console.print(rich_table_generic)
        else:
            logger.info(f"No rows found by query_generic_debug for '{generic_base_symbol_query_val}'.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug metadata lookup for continuous_contract_loader.")
    parser.add_argument("symbol", type=str, help="The specific continuous symbol to check (e.g., @VX=701XN, @ES=102XC).")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"Database path (Default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()

    logger.info("Starting CCL metadata lookup debug script...")
    debug_metadata_lookup(args.db_path, args.symbol)
    logger.info("CCL metadata lookup debug script finished.") 