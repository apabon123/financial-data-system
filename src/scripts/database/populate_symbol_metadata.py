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

def get_ordinal_suffix(number: int) -> str:
    """Return the ordinal suffix for a number (e.g., 1st, 2nd, 3rd, 4th)."""
    if 10 <= number % 100 <= 20:
        return str(number) + 'th'
    else:
        return str(number) + {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')

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

def _parse_symbol_frequencies(yaml_freq_list, base_symbol_for_logging: str = "symbol") -> list:
    """Parse a list of frequency configurations (strings or dicts) into a standardized list of tuples."""
    parsed_list = []
    if not isinstance(yaml_freq_list, list):
        logging.warning(f"Invalid 'frequencies' format for {base_symbol_for_logging} (should be a list, got {type(yaml_freq_list)}). Skipping frequencies.")
        return parsed_list

    if all(isinstance(f, dict) for f in yaml_freq_list):
        for freq_dict in yaml_freq_list:
            unit = freq_dict.get('unit')
            value = freq_dict.get('interval')
            # name = freq_dict.get('name') # Also available if needed
            if unit and value is not None:
                parsed_list.append((unit, value, freq_dict))
            else:
                logging.warning(f"Skipping frequency dict due to missing unit/interval: {freq_dict} for {base_symbol_for_logging}")
    elif all(isinstance(f, str) for f in yaml_freq_list):
        for freq_str in yaml_freq_list:
            unit, value = parse_frequency(freq_str) # Uses the existing global parse_frequency helper
            if unit and value is not None:
                # For string frequencies, freq_dict is None, which is handled by determine_metadata_for_interval
                parsed_list.append((unit, value, None))
            else:
                logging.warning(f"Could not parse frequency string '{freq_str}' for {base_symbol_for_logging}. Skipping.")
    elif yaml_freq_list: # If list is not empty but mixed or invalid types
        logging.warning(f"Mixed or invalid frequency types in list for {base_symbol_for_logging}. Skipping frequencies: {yaml_freq_list}")
    
    return parsed_list

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
        data_table = 'continuous_contracts' # Table is specific to continuous futures
        
        # The 'data_source' variable should have been correctly set by the preceding
        # 'if source == 'cboe': ... elif source == 'tradestation': ...' blocks
        # based on the initial 'source' resolution (which considers default_source).
        # We only need to override 'data_source' here for truly special continuous future types
        # that have a source different from what their original symbol_config might imply.
        if base_symbol == '@VX': # The generic @VX symbol, typically locally-generated.
             data_source = 'generated' # Override source to 'generated' for this specific case.
        # For other continuous futures like @ES, @NQ, or our @VX=...XN symbols,
        # the 'data_source' (e.g., 'tradestation') is assumed to be correctly set
        # by the logic that processed the 'source' variable just before this block.
        # No 'elif' or 'else' is needed to re-set it if that prior logic is sound.
             
        # Continuous contracts use the loader script for both operations
        hist_script = 'src/scripts/market_data/continuous_contract_loader.py' 
        upd_script = 'src/scripts/market_data/continuous_contract_loader.py' 
        # asset_type is already 'continuous_future' due to the surrounding if condition.
        # Ensure it's explicitly part of the return if this block is entered.
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

    # --- Clean up old 'day' and 'continuous' interval_unit entries --- #
    try:
        conn.execute(f"DELETE FROM {METADATA_TABLE_NAME} WHERE interval_unit IN ('day', 'continuous')")
        logging.info(f"Attempted to delete any old entries with interval_unit='day' or 'continuous'.")
    except duckdb.Error as e:
        logging.error(f"Error trying to delete old 'day'/'continuous' interval_unit entries: {e}")

    # --- ADDED: Explicitly delete old generic continuous_future metadata for @ES, @NQ, @VX ---
    specific_symbols_to_delete = ('@ES', '@NQ', '@VX')
    try:
        for sym_to_del in specific_symbols_to_delete:
            # Delete any metadata entries for these specific base_symbols, regardless of other attributes,
            # as they represent the old generic continuous entries we want to remove.
            # The new, correct entries (e.g. for @ES=102XC) will be re-added later.
            deleted_rows = conn.execute(f"DELETE FROM {METADATA_TABLE_NAME} WHERE base_symbol = ? RETURNING 1", [sym_to_del]).fetchall()
            if deleted_rows and len(deleted_rows) > 0:
                logging.info(f"Deleted {len(deleted_rows)} old metadata entries for generic symbol '{sym_to_del}'.")
            else:
                logging.info(f"No old metadata entries found/deleted for generic symbol '{sym_to_del}'.")
    except duckdb.Error as e:
        logging.error(f"Error deleting specific old generic symbol metadata (@ES, @NQ, @VX): {e}")
    # --- End explicit clean up ---

    entries_to_insert = []
    now = datetime.now()

    asset_configs = [
        (config.get('futures', []), 'future'), 
        (config.get('indices', []), 'index'), 
        (config.get('equities', []), 'equity')
    ]

    for config_list, default_asset_type in asset_configs:
        for item_config in config_list:
            # --- ADDED: Handle continuous_group ---
            if 'continuous_group' in item_config:
                group_details = item_config['continuous_group']
                identifier_base = group_details.get('identifier_base')
                month_codes = group_details.get('month_codes', [])
                settings_code = group_details.get('settings_code', "") # Ensure it's a string
                description_template = group_details.get('description_template', "Continuous Future {symbol}")

                if not identifier_base or not month_codes:
                    logging.warning(f"Skipping continuous_group due to missing 'identifier_base' or 'month_codes': {group_details}")
                    continue

                # Prepare a base item_config for the generated symbols
                generated_item_config_base = {
                    'exchange': group_details.get('exchange'),
                    'type': group_details.get('type', 'continuous_future'), # Default to continuous_future
                    'default_source': group_details.get('default_source'),
                    'default_raw_table': group_details.get('default_raw_table'),
                    'start_date': group_details.get('start_date'),
                    'calendar': group_details.get('calendar')
                }

                # Parse frequencies ONCE from the group config
                group_yaml_frequencies = group_details.get('frequencies', [])
                group_parsed_frequencies = _parse_symbol_frequencies(group_yaml_frequencies, identifier_base)

                if not group_parsed_frequencies:
                    logging.warning(f"No valid frequencies defined for continuous_group '{identifier_base}'. Skipping this group.")
                    continue
                
                for idx, month_code in enumerate(month_codes):
                    actual_symbol_val = f"{identifier_base}={str(month_code)}{settings_code}" # Ensure month_code is string
                    
                    current_generated_item_config = generated_item_config_base.copy()
                    current_generated_item_config['symbol'] = actual_symbol_val
                    current_generated_item_config['description'] = description_template.format(
                        nth_month=get_ordinal_suffix(idx + 1), symbol=actual_symbol_val
                    )
                    # 'type' is already in current_generated_item_config from generated_item_config_base

                    for unit, value, freq_dict_for_interval in group_parsed_frequencies:
                        interval_metadata = determine_metadata_for_interval(
                            current_generated_item_config,
                            unit,
                            value,
                            freq_dict_for_interval
                        )
                        entries_to_insert.append({
                            'base_symbol': actual_symbol_val, # Use the fully generated symbol
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
                continue # Move to the next item_config in the outer loop
            # --- END ADDED: Handle continuous_group ---

            base_symbol = item_config.get('base_symbol') or item_config.get('symbol')
            if not base_symbol: continue

            # MODIFIED: Use the new _parse_symbol_frequencies helper
            frequencies_config_list = item_config.get('frequencies', [])
            parsed_frequencies_with_config = _parse_symbol_frequencies(frequencies_config_list, base_symbol)
            
            if not parsed_frequencies_with_config:
                logging.warning(f"No valid frequencies determined for {base_symbol}. Skipping metadata entry.")
                continue

            asset_type = item_config.get('type', default_asset_type).lower()
            item_config['type'] = asset_type 
            
            # Process Individual Symbol/Intervals (Non-Continuous)
            for unit, value, freq_config_for_interval in parsed_frequencies_with_config:
                interval_specific_metadata = determine_metadata_for_interval(item_config, unit, value, freq_config_for_interval)
                
                # Debug logging for individual symbol items (which includes explicitly defined continuous_future types)
                if base_symbol in ['@ES=102XC', '@VX=101XN', '@ES=101XN', '@NQ=101XN', '@NQ=102XC']:
                    logging.info(f"DEBUG_POPULATOR (individual_symbol): Preparing to insert for {base_symbol} | {unit} | {value} with metadata: {interval_specific_metadata}")

                try:
                    conn.execute(f"""
                        INSERT INTO {METADATA_TABLE_NAME} (base_symbol, interval_unit, interval_value, data_table, data_source, asset_type, config_path, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (base_symbol, interval_unit, interval_value) DO UPDATE SET
                            data_table = EXCLUDED.data_table,
                            data_source = EXCLUDED.data_source,
                            asset_type = EXCLUDED.asset_type,
                            config_path = EXCLUDED.config_path,
                            last_updated = EXCLUDED.last_updated;
                    """, (
                        base_symbol,
                        unit,
                        value,
                        interval_specific_metadata['data_table'],
                        interval_specific_metadata['data_source'],
                        interval_specific_metadata['asset_type'],
                        config_file,
                        now
                    ))
                except duckdb.Error as e:
                    logging.error(f"Database error populating metadata: {e}")
                    raise

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