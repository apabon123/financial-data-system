#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Market Data Update Script (Refactored)

Orchestrates the full update process for financial market data:
1. Ensures symbol metadata is up-to-date.
2. Fetches new data for active individual futures, indices, and equities from TradeStation.
3. Updates raw CBOE VX futures data.
4. Updates TradeStation-sourced continuous contracts with intelligent roll handling.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import subprocess
from datetime import datetime, timedelta
import yaml # For loading config directly if needed, though fetcher handles it
import duckdb

# Add project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.scripts.market_data.fetch_market_data import MarketDataFetcher
# Import main function from populate_symbol_metadata
try:
    from src.scripts.database.populate_symbol_metadata import main as populate_metadata_main
except ImportError as e:
    # Log this early before logger is fully configured if this script is run directly
    print(f"Initial Import Error: Failed to import populate_symbol_metadata: {e}. Ensure it's in the correct path.")
    populate_metadata_main = None 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths - ensure consistency
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")
DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_ROLL_PROXIMITY_DAYS = 7

def run_python_script(script_module_path: str, args: list, cwd: str):
    """Helper to run a python script module (e.g., src.scripts.xyz) via subprocess."""
    cmd = [sys.executable, "-m", script_module_path] + args
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        # It is crucial that the script being called can be found via python -m
        # This means its parent directory needs to be in PYTHONPATH or be a package recognized by Python.
        # The `cwd` argument to subprocess.run does not add that directory to Python's module search path directly.
        # Ensuring project_root is in sys.path (done above) should help Python find the modules.
        env = os.environ.copy()
        # env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "") # Ensure project root is in PYTHONPATH for subprocess

        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        logger.info(f"Successfully ran {script_module_path}. Output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr from {script_module_path}:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_module_path}: {e}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        # This error usually means sys.executable was not found, or python -m could not resolve the module.
        logger.error(f"Script module not found or python executable issue: {script_module_path}. Check path and Python environment.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive market data update utility.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"Database path (Default: {DEFAULT_DB_PATH})")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help=f"Market symbols config path (Default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help=f"Default lookback days for 'latest' fetch mode in continuous contracts (Default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--roll-proximity-days", type=int, default=DEFAULT_ROLL_PROXIMITY_DAYS, help=f"Days before expiry to trigger full rebuild for adjusted continuous contracts in 'auto' mode (Default: {DEFAULT_ROLL_PROXIMITY_DAYS})")
    parser.add_argument("--fetch-all-individual-history", action="store_true", help="Force fetch of all available history for individual contracts (TradeStation). Default is to fetch new data.")
    # Add placeholder for the legacy --verify argument from the batch script
    parser.add_argument("--verify", action="store_true", help="Legacy argument, currently ignored by this script.")

    args = parser.parse_args()

    logger.info("========= Starting Full Market Data Update Orchestrator =========")
    if args.verify:
        logger.info("Received --verify flag; it is currently ignored by the main Python orchestrator.")

    # 0. Update Symbol Metadata
    logger.info("--- Step 0: Updating symbol metadata ---")
    if populate_metadata_main: # Check import worked
        # Call populate_symbol_metadata.py as a module via subprocess
        metadata_module_path = "src.scripts.database.populate_symbol_metadata"
        metadata_args = ["--db-path", args.db_path, "--config-path", args.config_path]
        if run_python_script(metadata_module_path, metadata_args, cwd=project_root):
            logger.info("Symbol metadata update script executed successfully.")
        else:
            logger.error("Symbol metadata update script execution failed. Check logs above.")
            # Decide if critical: For now, log error and continue
    else:
        logger.error("populate_symbol_metadata.main could not be imported. Skipping metadata update.")

    # Initialize MarketDataFetcher
    logger.info("--- Initializing MarketDataFetcher ---")
    fetcher = None # Define fetcher outside try block
    try:
        fetcher = MarketDataFetcher(config_path=args.config_path, db_path=args.db_path)
        logger.info("MarketDataFetcher initialized.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize MarketDataFetcher: {e}", exc_info=True)
        logger.info("========= Full Market Data Update Failed (Fetcher Init) =========")
        return

    # 1. Fetch Active Individual Futures, Indices, Equities from TradeStation
    logger.info("--- Step 1: Updating individual contracts, indices, and equities from TradeStation ---")
    if not fetcher.config:
        logger.error("Fetcher config not loaded. Skipping update of individual TS symbols.")
    else:
        for asset_type_key in ['futures', 'indices', 'equities']:
            if asset_type_key not in fetcher.config:
                continue
            for item_config in fetcher.config.get(asset_type_key, []):
                if 'continuous_group' in item_config: 
                    continue

                # ADDED CHECK: Skip items that are definitions for continuous futures, as they are handled by Step 3.
                if item_config.get('type') == 'continuous_future':
                    logger.debug(f"Skipping {item_config.get('symbol', 'unknown symbol')} in Step 1; it's a continuous_future type meant for Step 3.")
                    continue

                base_symbol = item_config.get('base_symbol') or item_config.get('symbol')
                item_source = item_config.get('source', item_config.get('default_source', 'tradestation')).lower()

                if item_source != 'tradestation':
                    logger.debug(f"Skipping {base_symbol} for TradeStation individual update (source: {item_source}).")
                    continue

                logger.info(f"Processing TradeStation {asset_type_key}: {base_symbol}")
                
                symbols_to_fetch_for_item = []
                if asset_type_key == 'futures':
                    try:
                        active_contracts = fetcher.get_active_futures_symbols(base_symbol, item_config)
                        symbols_to_fetch_for_item.extend(active_contracts)
                        logger.info(f"Active contracts for {base_symbol}: {active_contracts}")
                    except Exception as e:
                        logger.error(f"Could not get active contracts for {base_symbol}: {e}", exc_info=True)
                else: 
                    symbols_to_fetch_for_item.append(base_symbol)

                raw_frequencies = item_config.get('frequencies', [])
                parsed_frequencies = []
                for freq_entry in raw_frequencies:
                    unit, val = None, None
                    if isinstance(freq_entry, str):
                        if freq_entry == 'daily': unit, val = 'daily', 1
                        elif 'min' in freq_entry: 
                            try: val = int(freq_entry.replace('min','')); unit = 'minute'
                            except ValueError: logger.warning(f"Invalid minute string: {freq_entry}")
                    elif isinstance(freq_entry, dict):
                        unit, val = freq_entry.get('unit'), freq_entry.get('interval')
                    
                    if unit and val is not None:
                        parsed_frequencies.append({'unit': unit, 'value': val})
                    else:
                        logger.warning(f"Skipping unparsable frequency entry '{freq_entry}' for {base_symbol}")
                
                for symbol_to_fetch in symbols_to_fetch_for_item:
                    for freq_info in parsed_frequencies:
                        logger.info(f"Fetching {symbol_to_fetch} ({freq_info['value']} {freq_info['unit']}) from TS.")
                        try:
                            fetcher.process_symbol(
                                symbol=symbol_to_fetch,
                                interval_unit=freq_info['unit'],
                                interval_value=freq_info['value'],
                                force=args.fetch_all_individual_history,
                                update_history=args.fetch_all_individual_history
                            )
                            logger.info(f"Successfully processed {symbol_to_fetch} ({freq_info['value']} {freq_info['unit']}).")
                        except Exception as e:
                            logger.error(f"Failed to process {symbol_to_fetch} ({freq_info['value']} {freq_info['unit']}): {e}", exc_info=True)

    # Close the main fetcher connection BEFORE running subprocesses
    logger.info("--- Closing main fetcher DB connection before running subprocesses ---")
    if fetcher and hasattr(fetcher, 'close_connection') and callable(fetcher.close_connection):
        try:
            fetcher.close_connection()
            logger.info("Main MarketDataFetcher database connection closed.")
            fetcher.conn = None # Ensure connection object is cleared
        except Exception as e:
            logger.error(f"Error closing main MarketDataFetcher connection: {e}", exc_info=True)
    else:
        # If fetcher doesn't have close_connection, manually close if possible
        if fetcher and hasattr(fetcher, 'conn') and fetcher.conn:
            try:
                 fetcher.conn.close()
                 logger.info("Main fetcher DuckDB connection closed manually.")
                 fetcher.conn = None # Ensure connection object is cleared
            except Exception as e:
                 logger.error(f"Error closing fetcher.conn manually: {e}")

    # 2. Update Raw CBOE Data (VX Futures)
    logger.info("--- Step 2: Updating raw CBOE VX Futures data (to market_data_cboe) ---")
    # Module path for python -m execution
    cboe_vx_module_path = "src.scripts.market_data.vix.update_vx_futures"
    cboe_vx_args = ["--config-path", args.config_path, "--db-path", args.db_path]
    if not run_python_script(cboe_vx_module_path, cboe_vx_args, cwd=project_root):
        logger.error("Failed to update raw CBOE VX futures data. Check logs for details.")

    # 3. Update TradeStation-sourced Continuous Contracts
    logger.info("--- Step 3: Updating TradeStation-sourced Continuous Contracts ---")
    ts_continuous_symbols = []
    conn_meta = None
    try:
        conn_meta = duckdb.connect(args.db_path, read_only=True)
        # Query base_symbol directly, as this should hold the specific continuous symbol identifier
        # (e.g., @ES=102XC) after changes to populate_symbol_metadata.py
        query = """
            SELECT DISTINCT base_symbol
            FROM symbol_metadata
            WHERE asset_type = 'continuous_future'
              AND data_source = 'tradestation'
              AND base_symbol LIKE '@%'
              AND base_symbol IS NOT NULL AND base_symbol != '';
        """
        rows = conn_meta.execute(query).fetchall()
        ts_continuous_symbols = [row[0] for row in rows if row[0]] # Ensure not None
        logger.info(f"Found TradeStation continuous symbols from metadata: {ts_continuous_symbols}")
    except Exception as e:
        logger.error(f"Error querying continuous symbols from metadata: {e}. Updates might be incomplete.", exc_info=True)
    finally:
        if conn_meta:
             conn_meta.close()

    if not ts_continuous_symbols:
        logger.info("No TradeStation-sourced continuous contracts found in metadata to update.")
    
    ccl_module_path = "src.scripts.market_data.continuous_contract_loader"
    for cont_symbol in ts_continuous_symbols:
        logger.info(f"Updating continuous contract: {cont_symbol}")
        ccl_args = [
            cont_symbol,
            "--config-path", args.config_path,
            "--db-path", args.db_path,
            "--fetch-mode", "auto",
            "--lookback-days", str(args.lookback_days),
            "--roll-proximity-threshold-days", str(args.roll_proximity_days)
        ]
        if not run_python_script(ccl_module_path, ccl_args, cwd=project_root):
            logger.error(f"Failed to update continuous contract {cont_symbol}. Check logs.")

    logger.info("========= Full Market Data Update Orchestrator Finished =========")

if __name__ == "__main__":
    # This initial log might not have full formatting if logger is reconfigured by imported modules
    logger.info("update_all_market_data.py script execution started.")
    # Basic check for critical imports before main logic
    if not populate_metadata_main:
        logger.critical("Halting script: populate_symbol_metadata.main could not be imported.")
        sys.exit(1)
    main() 