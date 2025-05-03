#!/usr/bin/env python
"""
Populate Symbol Metadata Table (Interval-Aware)

This script reads the market symbols configuration file and populates (or updates)
the symbol_metadata table in the database. This table acts as a central registry
mapping base symbols *and intervals* to their data tables, sources, and asset types.
"""

import os
import sys
import yaml
import duckdb
from pathlib import Path
import logging
from datetime import datetime
import argparse
from typing import Optional

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
CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")
METADATA_TABLE_NAME = "symbol_metadata"

def parse_frequency(freq_str: str) -> tuple[Optional[str], Optional[int]]:
    """Parse frequency string like '1min', '15min', 'daily' into unit and value."""
    if freq_str == 'daily':
        return 'daily', 1
    elif 'min' in freq_str:
        try:
            value = int(freq_str.replace('min', ''))
            return 'minute', value
        except ValueError:
            return None, None
    # Add other parsers if needed (e.g., 'hourly')
    return None, None

def determine_metadata_for_interval(symbol_config: dict, interval_unit: str, interval_value: int, freq_config: Optional[dict] = None) -> dict:
    """Determine the table, source, and script paths for a specific symbol and interval.
    Prioritizes source/table info from freq_config if provided.
    """
    base_symbol = symbol_config.get('base_symbol') or symbol_config.get('symbol')
    # Prioritize source from specific frequency config, then symbol config, then default
    source = (freq_config.get('source') if freq_config else None) or symbol_config.get('source') or symbol_config.get('default_source')
    asset_type = symbol_config.get('type', 'future').lower()
    
    # Defaults
    data_table = 'market_data'
    hist_script = 'src/scripts/market_data/fetch_market_data.py'
    upd_script = None
    data_source = source or 'tradestation' # Default to tradestation if source is ultimately None

    # --- Source/Table Specific Logic (using the determined source) --- #
    if source == 'cboe':
        data_source = 'cboe' # Ensure data_source reflects cboe
        # Check for specific raw_table from freq_config first, then symbol_config
        raw_table = (freq_config.get('raw_table') if freq_config else None) or symbol_config.get('raw_table')
        data_table = raw_table or 'market_data_cboe' # Default CBOE table
        
        # Script assignments based on asset type and interval
        if asset_type == 'index' and base_symbol == '$VIX.X' and interval_unit == 'daily':
            hist_script = 'src/scripts/market_data/vix/update_vix_index.py' 
            upd_script = 'src/scripts/market_data/vix/update_vix_index.py'
        elif asset_type == 'future' and base_symbol == 'VX' and interval_unit == 'daily':
            # Daily VX futures: update from CBOE script, historical fetch from fetch_market_data (which reads CBOE table)
            hist_script = 'src/scripts/market_data/fetch_market_data.py' 
            upd_script = 'src/scripts/market_data/vix/update_vx_futures.py'
        else:
            # Fallback for other potential CBOE sources/intervals - needs specific handling if added
            logging.warning(f"Unhandled CBOE source case for {base_symbol} {interval_unit} {interval_value}. Using default scripts.")
            hist_script = None 
            upd_script = None 
    
    # Handle TradeStation source (explicitly or by default)
    elif source == 'tradestation':
        data_source = 'tradestation'
        data_table = 'market_data' # Default table for TS data
        hist_script = 'src/scripts/market_data/fetch_market_data.py'
        upd_script = None # No separate update script for TS data

    # --- Continuous Contracts Handling (AFTER specific source logic) --- #
    if asset_type == 'continuous_future':
        data_table = 'continuous_contracts'
        # Determine continuous source based on base symbol convention
        if base_symbol == '@VX':
             data_source = 'generated'
        elif base_symbol in ('@ES', '@NQ'):
             data_source = 'tradestation' 
        else:
             # Use symbol-level source if available for other continuous types
             data_source = symbol_config.get('source', 'unknown') 
             
        # Continuous contracts use the loader script for both operations
        hist_script = 'src/scripts/market_data/continuous_contract_loader.py' 
        upd_script = 'src/scripts/market_data/continuous_contract_loader.py' 
        asset_type = 'continuous_future'
        
    # --- Final Overrides --- #
    # NOTE: These overrides apply AFTER all other logic. Use with caution.
    if 'target_table' in symbol_config: data_table = symbol_config['target_table']
    if 'historical_script_path' in symbol_config: hist_script = symbol_config['historical_script_path']
    if 'update_script_path' in symbol_config: upd_script = symbol_config['update_script_path']

    return {
        'data_table': data_table,
        'data_source': data_source,
        'historical_script_path': hist_script,
        'update_script_path': upd_script,
        'asset_type': asset_type
    }

def create_metadata_table(conn: duckdb.DuckDBPyConnection):
    """Create the symbol_metadata table with composite PK if it doesn't exist.
    Drops the table first if it exists with an old schema (missing interval_unit).
    """
    table_exists = False
    old_schema = False
    try:
        # Check if table exists and if it has the new columns
        table_info = conn.execute(f"PRAGMA table_info('{METADATA_TABLE_NAME}')").fetchall()
        if table_info:
            table_exists = True
            column_names = [col[1] for col in table_info]
            if 'interval_unit' not in column_names:
                old_schema = True
                logging.warning(f"Table '{METADATA_TABLE_NAME}' exists with old schema. Dropping it.")
                conn.execute(f"DROP TABLE {METADATA_TABLE_NAME};")
                table_exists = False # Treat as if it doesn't exist now
            else:
                logging.info(f"Table '{METADATA_TABLE_NAME}' already exists with the correct schema.")
        
    except duckdb.CatalogException:
        # Table doesn't exist, which is fine
        table_exists = False
        logging.info(f"Table '{METADATA_TABLE_NAME}' does not exist yet.")
    except duckdb.Error as e:
        logging.error(f"Error checking schema for {METADATA_TABLE_NAME}: {e}")
        # Decide whether to raise or try to proceed
        raise # Re-raise error if schema check fails unexpectedly

    # Proceed with creation only if it doesn't exist or was just dropped
    if not table_exists:
        try:
            conn.execute(f"""
                CREATE TABLE {METADATA_TABLE_NAME} (
                    base_symbol VARCHAR NOT NULL,
                    interval_unit VARCHAR NOT NULL,
                    interval_value INTEGER NOT NULL,
                    data_table VARCHAR NOT NULL,
                    data_source VARCHAR,
                    historical_script_path VARCHAR,
                    update_script_path VARCHAR,
                    asset_type VARCHAR NOT NULL,
                    config_path VARCHAR,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (base_symbol, interval_unit, interval_value)
                );
            """)
            logging.info(f"Created table '{METADATA_TABLE_NAME}' with composite PK.")
        except duckdb.Error as e:
            logging.error(f"Error creating table {METADATA_TABLE_NAME}: {e}")
            raise

def populate_metadata(conn: duckdb.DuckDBPyConnection, config_file: str):
    """Read the config and populate the metadata table for each symbol/interval."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_file}: {e}")
        return

    # --- Clean up old 'day' and 'continuous' entries --- #
    try:
        conn.execute(f"DELETE FROM {METADATA_TABLE_NAME} WHERE interval_unit IN ('day', 'continuous')")
        logging.info(f"Attempted to delete any old entries with interval_unit='day' or 'continuous'.") 
    except duckdb.Error as e:
        logging.error(f"Error trying to delete old 'day'/'continuous' entries: {e}")
    # --- End clean up --- #

    entries_to_insert = []
    now = datetime.now()

    asset_configs = [
        (config.get('futures', []), 'future'), 
        (config.get('indices', []), 'index'), 
        (config.get('equities', []), 'equity')
    ]

    for config_list, default_asset_type in asset_configs:
        for item_config in config_list:
            base_symbol = item_config.get('base_symbol') or item_config.get('symbol')
            if not base_symbol: continue

            frequencies_config = item_config.get('frequencies', [])
            parsed_frequencies_with_config = [] # List of tuples (unit, value, freq_dict or None)

            if isinstance(frequencies_config, list):
                 if all(isinstance(f, dict) for f in frequencies_config):
                     # New structure: list of dicts
                     for freq_dict in frequencies_config:
                          unit = freq_dict.get('unit')
                          value = freq_dict.get('interval')
                          if unit and value is not None:
                               parsed_frequencies_with_config.append((unit, value, freq_dict))
                          else:
                               logging.warning(f"Skipping frequency dict due to missing unit/interval: {freq_dict} for {base_symbol}")
                 elif all(isinstance(f, str) for f in frequencies_config):
                     # Old structure: list of strings
                     for freq_str in frequencies_config:
                          unit, value = parse_frequency(freq_str)
                          if unit and value is not None:
                               parsed_frequencies_with_config.append((unit, value, None)) # No freq_dict for old format
                          else:
                               logging.warning(f"Could not parse frequency string '{freq_str}' for {base_symbol}. Skipping.")
                 else:
                     logging.warning(f"Mixed or invalid frequency types for {base_symbol}. Skipping frequencies.")
            else:
                logging.warning(f"Invalid 'frequencies' format for {base_symbol} (should be a list). Skipping.")
                continue # Skip this symbol if frequencies are invalid
                
            if not parsed_frequencies_with_config:
                 logging.warning(f"No valid frequencies determined for {base_symbol}. Skipping metadata entry.")
                 continue

            asset_type = item_config.get('type', default_asset_type).lower()
            item_config['type'] = asset_type 
            
            # Process Individual Symbol/Intervals (Non-Continuous)
            for unit, value, freq_dict in parsed_frequencies_with_config: # Pass freq_dict now
                # Pass freq_dict to the determination function
                interval_metadata = determine_metadata_for_interval(item_config, unit, value, freq_dict)
                entries_to_insert.append({
                    'base_symbol': base_symbol,
                    'interval_unit': unit,
                    'interval_value': value,
                    'data_table': interval_metadata['data_table'],
                    'data_source': interval_metadata['data_source'],
                    'historical_script_path': interval_metadata['historical_script_path'],
                    'update_script_path': interval_metadata['update_script_path'],
                    'asset_type': interval_metadata['asset_type'], 
                    'config_path': config_file,
                    'last_updated': now
                })
            
            # Add entries for continuous version if applicable
            if default_asset_type == 'future' and item_config.get('is_continuous', False):
                 continuous_base = f"@{base_symbol}"
                 logging.info(f"Processing continuous symbol {continuous_base} based on {base_symbol} frequencies...")
                 for unit, value, freq_dict in parsed_frequencies_with_config: 
                     # Determine metadata for continuous, passing original base symbol info and interval
                     cont_config_for_determine = {
                         'type': 'continuous_future', 
                         'base_symbol': continuous_base, 
                         'source': item_config.get('source') # Pass base source hint
                     }
                     # Pass the specific freq_dict if available (though continuous loader might ignore it)
                     cont_metadata = determine_metadata_for_interval(cont_config_for_determine, unit, value, freq_dict)
                     entries_to_insert.append({
                         'base_symbol': continuous_base,
                         'interval_unit': unit,
                         'interval_value': value,
                         'data_table': cont_metadata['data_table'],
                         'data_source': cont_metadata['data_source'],
                         'historical_script_path': cont_metadata['historical_script_path'],
                         'update_script_path': cont_metadata['update_script_path'],
                         'asset_type': cont_metadata['asset_type'],
                         'config_path': config_file,
                         'last_updated': now
                     })

    # Insert/Replace entries into the database
    if not entries_to_insert:
        logging.warning("No metadata entries generated from config.")
        return

    try:
        conn.begin()
        for entry in entries_to_insert:
            # Use INSERT OR REPLACE to handle updates based on composite PK
            # Ensure new script path columns exist if we are inserting into them
            # Check if new columns exist in the target table
            cols_exist = conn.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{METADATA_TABLE_NAME}'").fetchall()
            col_names = [col[0] for col in cols_exist]
            
            sql_columns = "base_symbol, interval_unit, interval_value, data_table, data_source, asset_type, config_path, last_updated"
            sql_placeholders = "?, ?, ?, ?, ?, ?, ?, ?"
            sql_params = [
                entry['base_symbol'], entry['interval_unit'], entry['interval_value'],
                entry['data_table'], entry['data_source'], entry['asset_type'], 
                entry['config_path'], entry['last_updated']
            ]
            
            # Conditionally add new script path columns
            if 'historical_script_path' in col_names:
                sql_columns += ", historical_script_path"
                sql_placeholders += ", ?"
                sql_params.append(entry.get('historical_script_path')) # Use .get for safety
            if 'update_script_path' in col_names:
                 sql_columns += ", update_script_path"
                 sql_placeholders += ", ?"
                 sql_params.append(entry.get('update_script_path')) # Use .get for safety
            # Handle old column if necessary (only if new ones don't exist)
            elif 'update_script' in col_names: 
                 sql_columns += ", update_script"
                 sql_placeholders += ", ?"
                 sql_params.append(entry.get('update_script')) # Fallback to old key

            sql = f"""
                INSERT INTO {METADATA_TABLE_NAME} ({sql_columns})
                VALUES ({sql_placeholders})
                ON CONFLICT (base_symbol, interval_unit, interval_value) DO UPDATE SET
                    data_table = EXCLUDED.data_table,
                    data_source = EXCLUDED.data_source,
                    asset_type = EXCLUDED.asset_type,
                    config_path = EXCLUDED.config_path,
                    last_updated = EXCLUDED.last_updated{', historical_script_path = EXCLUDED.historical_script_path' if 'historical_script_path' in col_names else ''}{', update_script_path = EXCLUDED.update_script_path' if 'update_script_path' in col_names else (', update_script = EXCLUDED.update_script' if 'update_script' in col_names else '')};
            """
            conn.execute(sql, sql_params)

        conn.commit()
        logging.info(f"Successfully populated/updated {len(entries_to_insert)} entries in {METADATA_TABLE_NAME}.")
    except duckdb.Error as e:
        conn.rollback()
        logging.error(f"Database error populating metadata: {e}")
    except Exception as e:
        conn.rollback()
        logging.error(f"Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Populate the symbol_metadata table from configuration.')
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the DuckDB database file.")
    parser.add_argument("--config-path", default=CONFIG_PATH, help="Path to the market symbols YAML configuration file.")
    args = parser.parse_args()

    db_file = Path(args.db_path).resolve()
    config_file = Path(args.config_path).resolve()

    if not db_file.exists():
        logging.error(f"Error: Database file not found at {db_file}")
        sys.exit(1)
    if not config_file.exists():
        logging.error(f"Error: Configuration file not found at {config_file}")
        sys.exit(1)

    conn = None
    try:
        conn = duckdb.connect(database=str(db_file), read_only=False)
        logging.info(f"Connected to database: {db_file}")

        # Ensure table exists
        create_metadata_table(conn)

        # Populate the table
        populate_metadata(conn, str(config_file))

    except duckdb.Error as e:
        logging.error(f"Failed to connect to or operate on database {db_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main() 