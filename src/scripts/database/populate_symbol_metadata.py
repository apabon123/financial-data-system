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
from typing import Optional, Dict, Any
import json # Added for additional_metadata

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)

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
    # Determine the effective symbol name (could be 'symbol' or 'identifier_base' from continuous_group)
    # This 'item_symbol_name' is what gets stored in the 'symbol' column of symbol_metadata table.
    item_symbol_name = symbol_config.get('symbol') or symbol_config.get('identifier_base')
    
    # base_symbol is the root part, e.g., ES, NQ, VX. For equities/indices, it's the same as item_symbol_name.
    # For continuous_group, symbol_config['identifier_base'] is used. For individual futures, it's symbol_config['base_symbol'].
    effective_base_symbol = symbol_config.get('base_symbol', item_symbol_name)

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
        if asset_type == 'index' and effective_base_symbol == '$VIX.X' and interval_unit == 'daily':
            hist_script = 'src/scripts/market_data/vix/update_vix_index.py' 
            upd_script = 'src/scripts/market_data/vix/update_vix_index.py'
        elif asset_type == 'future_group' and effective_base_symbol == 'VX' and interval_unit == 'daily':
            # Daily VX futures: update from CBOE script, historical fetch from fetch_market_data (which reads CBOE table)
            hist_script = 'src/scripts/market_data/fetch_market_data.py' 
            upd_script = 'src/scripts/market_data/vix/update_vx_futures.py'
        else:
            # Fallback for other potential CBOE sources/intervals - needs specific handling if added
            logging.warning(f"Unhandled CBOE source case for {effective_base_symbol} {interval_unit} {interval_value}. Using default scripts.")
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
        if item_symbol_name == '@VX': # The generic @VX symbol, typically locally-generated.
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

def create_metadata_table_if_not_exists(conn: duckdb.DuckDBPyConnection):
    """Create the symbol_metadata table with the new schema if it doesn't exist."""
    try:
        # Check if table exists
        table_info = conn.execute(f"PRAGMA table_info('{METADATA_TABLE_NAME}')").fetchall()
        if table_info:
            # Basic check for a few key new/changed columns to infer if it's the new schema
            column_names = [col[1] for col in table_info]
            if 'symbol' in column_names and 'start_date' in column_names and 'additional_metadata' in column_names and 'base_symbol' in column_names:
                logging.info(f"Table '{METADATA_TABLE_NAME}' already exists with the expected new schema.")
                return # Assume schema is correct
            else: # Columns mismatch, could be old schema or something else.
                  # For safety, this script won't drop/recreate if main init should handle it.
                  # However, if run standalone and it's an old schema, it should be updated.
                  # Given user dropped it, this path might not be hit on first run.
                logging.warning(f"Table '{METADATA_TABLE_NAME}' exists but schema seems incorrect. "
                                f"Expected columns like 'symbol', 'start_date', 'additional_metadata', 'base_symbol'. Found: {column_names}. "
                                f"The table should have been created/updated by the main application's schema initialization. "
                                "This script will attempt to proceed but mismatches may cause errors.")
                # Optionally, one could re-run the CREATE TABLE IF NOT EXISTS, but that doesn't alter.
                # Or, raise an error here if strict schema adherence is required before populating.
                # conn.execute(f"DROP TABLE IF EXISTS {METADATA_TABLE_NAME};") # Risky if shared
                # logging.info(f"Dropped table '{METADATA_TABLE_NAME}' due to schema mismatch for re-creation.")
        else: # Table does not exist
             logging.info(f"Table '{METADATA_TABLE_NAME}' does not exist. Attempting to create.")

        # Create table (IF NOT EXISTS handles if it's already there and matches this exact schema)
        # This schema MUST match init_schema.sql
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {METADATA_TABLE_NAME} (
                symbol VARCHAR NOT NULL,         -- Actual symbol, e.g. @ES=102XC or SPY
                base_symbol VARCHAR,             -- Root symbol, e.g. ES or SPY
                description VARCHAR,
                exchange VARCHAR,
                asset_type VARCHAR NOT NULL,     -- e.g. 'future', 'continuous_future', 'equity'
                data_source VARCHAR,
                data_table VARCHAR NOT NULL,
                interval_unit VARCHAR NOT NULL,
                interval_value INTEGER NOT NULL,
                config_path VARCHAR,             -- Path to the main market_symbols.yaml
                start_date DATE,                 -- Earliest data date from config
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                historical_script_path VARCHAR,
                update_script_path VARCHAR,
                additional_metadata JSON,        -- Full item_config from YAML
                PRIMARY KEY (symbol, interval_unit, interval_value)
            );
        """)
        logging.info(f"Ensured table '{METADATA_TABLE_NAME}' is present (created if it didn't exist).")

    except duckdb.Error as e:
        logging.error(f"Database error during setup of {METADATA_TABLE_NAME}: {e}")
        raise

def populate_metadata(conn: duckdb.DuckDBPyConnection, config_file_path: str):
    """Read the config and populate the metadata table for each symbol/interval."""
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file_path}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_file_path}: {e}")
        return

    default_start_date_str = config.get('settings', {}).get('default_start_date', '2000-01-01')
    records_to_upsert = []

    symbol_categories = ['indices', 'futures', 'equities', 'forex', 'crypto'] # Add other categories if present in YAML

    for category in symbol_categories:
        if category not in config:
            continue

        for item_config in config[category]:
            if 'continuous_group' in item_config:
                # Handle continuous_group expansion
                group_details = item_config['continuous_group']
                id_base = group_details['identifier_base'] # e.g., "@VX"
                month_codes = group_details.get('month_codes', [])
                settings_code = group_details.get('settings_code', "") # e.g., "01XN"
                
                for mc_idx, month_code in enumerate(month_codes):
                    # Construct the specific continuous symbol name
                    # Example: @VX + = + 1 + 01XN -> @VX=101XN (if month_code_is_nth_month=false)
                    # Example: @VX + = + 1 (as nth month) + 01XN -> @VX=101XN
                    # The original YAML has 'month_codes' which seems to imply the actual Nth month identifier part
                    
                    # Default way to form symbol: @VX + = + month_code + settings_code
                    # If settings_code already contains type (like XN, XC), month_code is likely just the number part
                    # For "@VX=101XN", month_code is "1", settings_code is "01XN"
                    # Let's assume settings_code may already imply type (like XN)
                    # and month_code is the numeric part for Nth month
                    
                    # A more robust way might be needed if symbol construction varies greatly
                    # For "@VX=101XN", "1" is the month code here.
                    # The template for description suggests {nth_month}
                    
                    nth_month_str = get_ordinal_suffix(int(month_code)) # Assuming month_code is '1', '2' etc.
                    current_symbol_name = f"{id_base}={month_code}{settings_code}"
                    
                    current_description = group_details.get('description_template', "").format(nth_month=nth_month_str)
                    current_base_symbol = id_base # For @VX=101XN, base_symbol is @VX
                    
                    # Use group's settings, override with specifics if any (though continuous_group usually defines all)
                    # This constructed_item_config is what gets stored in additional_metadata
                    constructed_item_config = {**group_details, 
                                               'symbol': current_symbol_name, 
                                               'description': current_description,
                                               'base_symbol': current_base_symbol # Store the group's base_symbol
                                               }
                    # Remove keys that are not part of a standard item_config or are group-specific
                    constructed_item_config.pop('identifier_base', None)
                    constructed_item_config.pop('month_codes', None)
                    constructed_item_config.pop('settings_code', None)
                    constructed_item_config.pop('description_template', None)
                    
                    # Frequencies for the group apply to each generated symbol
                    yaml_frequencies = group_details.get('frequencies', [])
                    parsed_frequencies = _parse_symbol_frequencies(yaml_frequencies, current_symbol_name)
                    
                    item_start_date_str = group_details.get('start_date', default_start_date_str)
                    item_exchange = group_details.get('exchange')
                    # asset_type for continuous_group items is usually 'continuous_future'
                    item_asset_type = group_details.get('type', 'continuous_future').lower()

                    for interval_unit, interval_value, freq_specific_config in parsed_frequencies:
                        interval_metadata = determine_metadata_for_interval(constructed_item_config, interval_unit, interval_value, freq_specific_config)
                        record = (
                            current_symbol_name, # symbol
                            current_base_symbol, # base_symbol
                            current_description, # description
                            item_exchange,       # exchange
                            interval_metadata['asset_type'] or item_asset_type, # asset_type
                            interval_metadata['data_source'],
                            interval_metadata['data_table'],
                            interval_unit,
                            interval_value,
                            config_file_path, # config_path (main YAML)
                            item_start_date_str, # start_date
                            datetime.now(), # last_updated
                            interval_metadata['historical_script_path'],
                            interval_metadata['update_script_path'],
                            json.dumps(constructed_item_config) # additional_metadata (JSON of the constructed item_config)
                        )
                        records_to_upsert.append(record)
                        logging.debug(f"DEBUG_POPULATOR (continuous_group_item): Preparing to insert for {current_symbol_name} | {interval_unit} | {interval_value} with metadata: {interval_metadata}")

            else: # Handle individual symbol entries (not in a continuous_group)
                item_symbol_name = item_config.get('symbol')
                if not item_symbol_name:
                    logging.warning(f"Skipping item in {category} due to missing 'symbol': {item_config}")
                    continue

                # For non-group items, base_symbol is either specified or defaults to the symbol itself
                item_base_symbol = item_config.get('base_symbol', item_symbol_name)
                item_description = item_config.get('description')
                item_exchange = item_config.get('exchange')
                item_asset_type = item_config.get('type', 'unknown').lower() # Default to 'unknown' if not specified
                
                yaml_frequencies = item_config.get('frequencies', [])
                parsed_frequencies = _parse_symbol_frequencies(yaml_frequencies, item_symbol_name)

                item_start_date_str = item_config.get('start_date', default_start_date_str)

                for interval_unit, interval_value, freq_specific_config in parsed_frequencies:
                    interval_metadata = determine_metadata_for_interval(item_config, interval_unit, interval_value, freq_specific_config)
                    record = (
                        item_symbol_name,    # symbol
                        item_base_symbol,    # base_symbol
                        item_description,    # description
                        item_exchange,       # exchange
                        interval_metadata['asset_type'] or item_asset_type, # asset_type
                        interval_metadata['data_source'],
                        interval_metadata['data_table'],
                        interval_unit,
                        interval_value,
                        config_file_path,    # config_path (main YAML)
                        item_start_date_str, # start_date
                        datetime.now(),      # last_updated
                        interval_metadata['historical_script_path'],
                        interval_metadata['update_script_path'],
                        json.dumps(item_config) # additional_metadata (JSON of the item_config from YAML)
                    )
                    records_to_upsert.append(record)
                    logging.debug(f"DEBUG_POPULATOR (individual_symbol): Preparing to insert for {item_symbol_name} | {interval_unit} | {interval_value} with metadata: {interval_metadata}")

    if records_to_upsert:
        try:
            # Column names must match the CREATE TABLE statement
            column_names = [
                "symbol", "base_symbol", "description", "exchange", "asset_type", 
                "data_source", "data_table", "interval_unit", "interval_value",
                "config_path", "start_date", "last_updated", 
                "historical_script_path", "update_script_path", "additional_metadata"
            ]
            columns_str = ", ".join(column_names)
            placeholders = ", ".join(["?"] * len(column_names))

            # Prepare for ON CONFLICT: PK is (symbol, interval_unit, interval_value)
            # Columns to update: all except PK.
            # Create SET statements for each column except PK: col_name = excluded.col_name
            update_setters = []
            pk_columns = {"symbol", "interval_unit", "interval_value"}
            for col in column_names:
                if col not in pk_columns:
                    update_setters.append(f"{col} = excluded.{col}")
            update_setters_str = ", ".join(update_setters)
            
            upsert_sql = f"""
                INSERT INTO {METADATA_TABLE_NAME} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (symbol, interval_unit, interval_value) DO UPDATE SET
                {update_setters_str};
            """
            
            conn.executemany(upsert_sql, records_to_upsert)
            logging.info(f"Successfully populated/updated {len(records_to_upsert)} entries in {METADATA_TABLE_NAME}.")
        except duckdb.Error as e:
            logging.error(f"Error upserting records into {METADATA_TABLE_NAME}: {e}")
            logging.error(f"Sample record that may have caused error: {records_to_upsert[0] if records_to_upsert else 'No records'}")

def main():
    parser = argparse.ArgumentParser(description=f"Populate or update the {METADATA_TABLE_NAME} table.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the DuckDB database file.")
    parser.add_argument("--config-path", default=CONFIG_PATH, help="Path to the market symbols YAML configuration file.")
    args = parser.parse_args()

    try:
        conn = duckdb.connect(args.db_path, read_only=False)
        logging.info(f"Connected to database: {args.db_path}")
        
        # Ensure table exists with the correct schema
        create_metadata_table_if_not_exists(conn)
        
        # Populate metadata
        populate_metadata(conn, args.config_path)
        
    except duckdb.Error as e:
        logging.error(f"Database operation failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main() 