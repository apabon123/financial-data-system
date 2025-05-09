#!/usr/bin/env python
"""
Dispatch Fetcher Script

This script acts as a dispatcher for fetching data for a specific symbol and interval.
It reads the symbol_metadata table to determine the correct data source and update script,
then executes that script, passing along necessary arguments.
"""

import os
import sys
import duckdb
from pathlib import Path
import logging
import argparse
import subprocess
from typing import Optional
import re # Import regex

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")
METADATA_TABLE_NAME = "symbol_metadata"
DEFAULT_FETCHER_SCRIPT = "src/scripts/market_data/fetch_market_data.py"

def get_base_symbol(symbol: str) -> str:
    """Determine the base symbol from a specific contract symbol or return the symbol if it's already a base."""
    # Check for known index/equity patterns first
    if symbol.startswith('$') or symbol in ['SPY', 'QQQ', 'AAPL', 'GS']: # Add other known non-futures bases if needed
        return symbol
    # Check for futures pattern (e.g., ESH24, VXU25)
    # Updated regex to handle 1 or 2 digit year and 1-3 char base
    match = re.match(r'^([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$', symbol)
    if match:
        return match.group(1) # Return the base (e.g., ES, VX, ZN)
    # Assume it's already a base symbol if no pattern matches
    return symbol

def get_fetch_metadata(conn: duckdb.DuckDBPyConnection, symbol_to_query: str) -> Optional[dict]:
    """Query metadata for the specific symbol (which could be a base or a direct symbol)."""
    logging.info(f"Querying metadata in get_fetch_metadata for symbol: '{symbol_to_query}' (Type: {type(symbol_to_query)}))")
    try:
        # WARNING: Using f-string for direct embedding - only for diagnostics
        # Ensure the symbol is treated as a string literal in SQL by adding quotes
        # Basic escaping of single quotes within the symbol itself (though unlikely for $VIX.X)
        sql_safe_symbol = str(symbol_to_query).replace("'", "''") 
        
        query = f"""
            SELECT historical_script_path, update_script_path, asset_type 
            FROM {METADATA_TABLE_NAME} 
            WHERE base_symbol = '{sql_safe_symbol}'  -- Symbol directly embedded
            LIMIT 1
        """
        logging.info(f"Executing direct query: {query}") # Log the exact query
        # params = [str(symbol_to_query)] 
        result = conn.execute(query).fetchone() # No params needed now
        
        if result:
            logging.info(f"Metadata FOUND for symbol '{symbol_to_query}' in get_fetch_metadata.")
            return {
                'historical_script_path': result[0],
                'update_script_path': result[1],
                'asset_type': result[2]
            }
        else:
            logging.warning(f"Metadata NOT FOUND for symbol '{symbol_to_query}' in get_fetch_metadata.")
            return None
    except Exception as e:
        logging.error(f"Error querying {METADATA_TABLE_NAME} for symbol '{symbol_to_query}' in get_fetch_metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Dispatch data fetching based on symbol metadata.')
    parser.add_argument("--symbol", required=True, help="Symbol to fetch (e.g., SPY, VIX_INDEX, ES, ESH25).")
    parser.add_argument("--interval-unit", required=False, default='daily', help="Interval unit (e.g., daily, minute). Default: daily")
    parser.add_argument("--interval-value", type=int, required=False, default=1, help="Interval value (e.g., 1, 15). Default: 1")
    parser.add_argument("--force", action='store_true', help="Force fetch (overwrite existing data).")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the DuckDB database file.")
    parser.add_argument("--operation-type", choices=['fetch', 'update'], default='fetch', help="Specify operation: 'fetch' for historical/new data, 'update' for raw source data.")
    args = parser.parse_args()

    # --- ADDED: Handle symbol alias for $VIX.X ---
    symbol_to_process = args.symbol
    if args.symbol == "VIX_INDEX":
        symbol_to_process = "$VIX.X"
        logging.info(f"Received symbol alias 'VIX_INDEX', processing as '{symbol_to_process}'")
    # --- END ADDED ---

    db_file = Path(args.db_path).resolve()
    if not db_file.exists():
        logging.error(f"Error: Database file not found at {db_file}")
        sys.exit(1)

    metadata = None
    conn = None
    # Use symbol_to_process for metadata lookup logic
    symbol_for_metadata_lookup = symbol_to_process 
    
    try:
        conn = duckdb.connect(database=str(db_file), read_only=True)
        logging.info(f"Connected to database to read metadata: {db_file}")
        
        logging.info(f"Attempting metadata lookup with exact symbol: '{symbol_to_process}'")
        metadata = get_fetch_metadata(conn, symbol_to_process)
        
        if not metadata:
            logging.info(f"Exact symbol '{symbol_to_process}' not found in metadata. Trying derived base symbol.")
            # Pass symbol_to_process to get_base_symbol
            derived_base = get_base_symbol(symbol_to_process) 
            if derived_base != symbol_to_process: 
                logging.info(f"Derived base symbol: '{derived_base}'. Attempting metadata lookup with derived base.")
                metadata = get_fetch_metadata(conn, derived_base)
                symbol_for_metadata_lookup = derived_base 
            else:
                logging.info(f"Derived base symbol '{derived_base}' is the same as input '{symbol_to_process}'. No second lookup needed.")

    except duckdb.Error as e:
        logging.error(f"Database connection error while reading metadata: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Closed metadata database connection.")
            
    if not metadata:
        # Use symbol_for_metadata_lookup for the error message
        # The input symbol in the message should be the original args.symbol
        logging.error(f"Could not find fetch metadata using '{symbol_for_metadata_lookup}' (derived from input '{args.symbol}'). Aborting.")
        sys.exit(1)
            
    # --- Select script path based on operation type --- #
    target_script = None
    if args.operation_type == 'fetch':
        target_script = metadata.get('historical_script_path')
        if not target_script:
            logging.warning(f"No historical_script_path found for {symbol_for_metadata_lookup}, falling back to default: {DEFAULT_FETCHER_SCRIPT}")
            target_script = DEFAULT_FETCHER_SCRIPT
        logging.info(f"Operation type 'fetch': using historical script: {target_script}")
    elif args.operation_type == 'update':
        target_script = metadata.get('update_script_path')
        if not target_script:
            logging.error(f"Operation type 'update' requested, but no update_script_path found for {symbol_for_metadata_lookup}. Aborting.")
            sys.exit(1)
        logging.info(f"Operation type 'update': using update script: {target_script}")
    # ------------------------------------------------- #

    target_script_path = os.path.join(project_root, target_script)
        
    if not os.path.exists(target_script_path):
        # Attempt fallback only if the historical script was not found initially and target_script is still default
        if args.operation_type == 'fetch' and target_script == DEFAULT_FETCHER_SCRIPT:
            logging.error(f"Target historical script not found: {target_script_path}")
            logging.error(f"Default fetcher script also not found: {target_script_path}. Aborting.")
            sys.exit(1)
        else:
             logging.error(f"Target script not found: {target_script_path}. Aborting.")
             sys.exit(1)

    # --- Construct command arguments --- #
    cmd = [sys.executable, target_script_path] 
    
    if args.operation_type == 'fetch':
        # Pass the potentially translated symbol_to_process to the downstream script
        cmd.extend(["--symbol", symbol_to_process]) 
        cmd.extend(["--interval-value", str(args.interval_value)]) 
        cmd.extend(["--interval-unit", args.interval_unit])       
        if args.force:
            cmd.append("--force")
        # Use symbol_for_metadata_lookup for the VX CBOE note condition
        if target_script == DEFAULT_FETCHER_SCRIPT and symbol_for_metadata_lookup == 'VX' and args.interval_unit == 'daily':
             logging.info(f"Note: Fetching daily VX data for {symbol_to_process} using '{target_script}', which will query the CBOE source table.")
             
    elif args.operation_type == 'update':
        # Update scripts might have different arg needs
        if target_script == "src/scripts/market_data/vix/update_vx_futures.py":
            logging.info(f"Running {target_script}. This script updates ALL active daily VX futures from CBOE.")
            # This script doesn't take symbol/interval/force args in its current form
            if args.force:
                logging.warning("Force flag provided but may not apply to update_vx_futures.py.")
        elif target_script == "src/scripts/market_data/vix/update_vix_index.py":
             logging.info(f"Running {target_script} (ignores specific symbol/interval args).")
             if args.force:
                 logging.warning("Force flag provided but not supported by update_vix_index.py.")
        else:
            # Unknown update script, pass basic args?
            logging.warning(f"Dispatching to unknown update script {target_script}. Passing --symbol and --force only.")
            # Pass the potentially translated symbol_to_process
            cmd.extend(["--symbol", symbol_to_process])
            if args.force:
                cmd.append("--force")

    # --- Execute the command --- #
    logging.info(f"Executing command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, cwd=project_root) 
        logging.info(f"Script completed successfully (Return Code: {process.returncode}).")
    except subprocess.CalledProcessError as e:
        logging.error(f"Script execution failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
         logging.error(f"Could not execute command. Ensure Python executable and script path are correct: {' '.join(cmd)}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during script execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 