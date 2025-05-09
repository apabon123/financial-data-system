#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temp script to run populate_symbol_metadata and then check its output for @ES=102XC.
"""
import os
import sys
from pathlib import Path
import logging
import duckdb
import subprocess
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
DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")
METADATA_TABLE_NAME = "symbol_metadata"

def run_and_check_metadata(db_path: str, config_path: str):
    # Step 1: Run populate_symbol_metadata.py
    logger.info(f"--- Running populate_symbol_metadata.py ---")
    metadata_module_path = "src.scripts.database.populate_symbol_metadata"
    metadata_args = ["--db-path", db_path, "--config-path", config_path]
    cmd = [sys.executable, "-m", metadata_module_path] + metadata_args
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"populate_symbol_metadata.py ran successfully. STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR from populate_symbol_metadata.py:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"populate_symbol_metadata.py failed. Error:\n{e}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        return
    except FileNotFoundError:
        logger.error(f"Script module not found or python executable issue: {metadata_module_path}")
        return

    # Step 2: Connect to DB and check for @ES=102XC
    logger.info(f"--- Checking {METADATA_TABLE_NAME} for '@ES=102XC' entries ---")
    conn = None
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        query = f"SELECT * FROM {METADATA_TABLE_NAME} WHERE base_symbol = '@ES=102XC'"
        df = conn.execute(query).fetchdf()

        console = Console()
        if not df.empty:
            logger.info(f"Found {len(df)} entries for '@ES=102XC':")
            rich_table = Table(title="Metadata for @ES=102XC")
            for col in df.columns:
                rich_table.add_column(col)
            for _index, row in df.iterrows():
                rich_table.add_row(*[str(item) for item in row])
            console.print(rich_table)
        else:
            logger.info("No entries found for '@ES=102XC' in symbol_metadata.")
            # Let's check for @ES to see if a generic entry was made instead by mistake
            logger.info("Checking for generic '@ES' with asset_type='continuous_future'...")
            query_generic_es = f"SELECT * FROM {METADATA_TABLE_NAME} WHERE base_symbol = '@ES' AND asset_type = 'continuous_future'"
            df_generic = conn.execute(query_generic_es).fetchdf()
            if not df_generic.empty:
                logger.info(f"Found {len(df_generic)} GENERIC entries for '@ES' (continuous_future):")
                rich_table_generic = Table(title="Generic Metadata for @ES (continuous_future)")
                for col in df_generic.columns:
                    rich_table_generic.add_column(col)
                for _index, row in df_generic.iterrows():
                    rich_table_generic.add_row(*[str(item) for item in row])
                console.print(rich_table_generic)
            else:
                logger.info("No generic '@ES' (continuous_future) entries found either.")

    except Exception as e:
        logger.error(f"Error during database check: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Starting metadata population and check script...")
    db_path_to_use = os.getenv('DUCKDB_PATH', DEFAULT_DB_PATH)
    config_path_to_use = DEFAULT_CONFIG_PATH # Assuming this is fixed for now
    
    run_and_check_metadata(db_path_to_use, config_path_to_use)
    logger.info("Metadata population and check script finished.") 