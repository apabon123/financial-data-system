#!/usr/bin/env python
"""
Update all market data using the new architecture.

This script orchestrates the entire market data update process using the new
architecture components. It includes:
1. Symbol metadata updates
2. Raw market data fetching
3. Continuous contract generation with Panama method
4. Data cleaning and validation

This is the successor to update_all_market_data.py, redesigned to work with
the new component-based architecture.
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)

# Application imports
from src.core.app import Application
from src.core.config import ConfigManager
from src.processors.continuous.registry import get_registry
from src.processors.cleaners.pipeline import DataCleaningPipeline
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner
from src.processors.continuous.panama import PanamaContractBuilder
from src.processors.continuous.unadjusted import UnadjustedContractBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Update all market data using the new architecture")
    
    # Basic options
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Path to config YAML")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to DuckDB database")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    # Update control options
    parser.add_argument("--verify", action="store_true", help="Verify data after update")
    parser.add_argument("--full-update", action="store_true", help="Force full data update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    
    # Component selection
    parser.add_argument("--skip-metadata", action="store_true", help="Skip symbol metadata update")
    parser.add_argument("--skip-vix", action="store_true", help="Skip VIX index update")
    parser.add_argument("--skip-futures", action="store_true", help="Skip VX futures update")
    parser.add_argument("--skip-es-nq", action="store_true", help="Skip ES/NQ futures update")
    parser.add_argument("--skip-continuous", action="store_true", help="Skip continuous contract generation")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-panama", action="store_true", help="Skip Panama continuous contract generation specifically")
    
    # Specific component options (These are now handled by YAML frequency configuration)
    # parser.add_argument("--update-active-es-15min", action="store_true", help="Update active ES 15min data")
    # parser.add_argument("--update-active-es-1min", action="store_true", help="Update active ES 1min data")
    
    # Advanced options
    parser.add_argument("--lookback-days", type=int, default=5, 
                       help="Days to look back for updates")
    parser.add_argument("--panama-ratio", type=float, default=0.75,
                       help="Panama method ratio (0-1, 0=forward adjustment, 1=back adjustment)")
    
    return parser.parse_args()

def main():
    """Main function to update all market data."""
    start_time = time.time()
    args = parse_arguments()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("-" * 80)
    logger.info(f"Starting market data update process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using config: {args.config_path}")
    logger.info(f"Using database: {args.db_path}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    app = None  # Initialize app to None
    try:
        app = Application(
            config_path=args.config_path
        )
        app.init_database(db_path_override=args.db_path, read_only=args.dry_run)
        
        # Step 1: Update symbol metadata
        if not args.skip_metadata:
            logger.info("Updating symbol metadata...")
            app.update_symbol_metadata()
            logger.info("Symbol metadata update completed")
        else:
            logger.info("Skipping symbol metadata update")
        
        # Step 2: Update raw market data
        if not args.skip_vix:
            logger.info("Updating VIX index data...")
            app.update_vix_index(force_full=args.full_update)
            logger.info("VIX index update completed")
        else:
            logger.info("Skipping VIX index update")
        
        # Step 2b: Update Raw Futures Data (ES, NQ, VX, CL, GC, etc.) based on YAML frequencies
        # This new section replaces the old specific calls to app.update_vx_futures(), app.update_es_futures(), app.update_nq_futures() for raw data fetching.
        logger.info("Updating raw futures data based on configured frequencies...")
        root_symbol_configs = app.get_continuous_contract_symbols()

        processed_roots_for_raw_update = set() # To avoid double processing if a root appears multiple times (e.g. VX)

        for root_cfg in root_symbol_configs:
            base_root_symbol = root_cfg.get('root_symbol')
            if not base_root_symbol or base_root_symbol in processed_roots_for_raw_update:
                if base_root_symbol in processed_roots_for_raw_update:
                    logger.debug(f"Root {base_root_symbol} already processed for raw data update via another config entry. Skipping.")
                continue

            # Check skip flags before processing frequencies for this root
            should_skip = False
            if base_root_symbol == 'VX':
                if args.skip_futures: # --skip-futures is historically for VIX futures
                    logger.info(f"Skipping raw data update for {base_root_symbol} due to --skip-futures flag.")
                    should_skip = True
            elif base_root_symbol in ['ES', 'NQ']:
                if args.skip_es_nq:
                    logger.info(f"Skipping raw data update for {base_root_symbol} due to --skip-es-nq flag.")
                    should_skip = True
            # Add other general future skip flags here if they exist, e.g. a hypothetical --skip-other-futures
            # For now, CL, GC, etc., will be updated if they have frequencies and no specific skip flag applies.

            if should_skip:
                processed_roots_for_raw_update.add(base_root_symbol)
                continue

            # Get frequencies for the RAW generic symbol (e.g. @ES) from the root_cfg
            update_frequencies = root_cfg.get('raw_data_update_frequencies', []) 
            if not update_frequencies:
                logger.info(f"No 'raw_data_update_frequencies' defined in root_cfg for {base_root_symbol}. Skipping its raw data update.")
            else:
                logger.info(f"Processing raw data updates for generic continuous symbol based on {base_root_symbol} (e.g. @{base_root_symbol}) across {len(update_frequencies)} potential frequencies.")
                
                parsed_frequencies_for_raw = []
                if isinstance(update_frequencies, list) and update_frequencies:
                    if isinstance(update_frequencies[0], str): # e.g., ['15min', 'daily']
                        for freq_name_str in update_frequencies:
                            interval, unit = None, None
                            if freq_name_str == 'daily': interval, unit = 1, 'daily'
                            elif freq_name_str == '1min': interval, unit = 1, 'minute'
                            elif freq_name_str.endswith('min'):
                                try: interval = int(freq_name_str[:-3]); unit = 'minute'
                                except ValueError: pass
                            # Add more parsers if needed (e.g., '1hour')
                            if interval and unit:
                                parsed_frequencies_for_raw.append({'name': freq_name_str, 'interval': interval, 'unit': unit})
                            elif freq_name_str: # Log only if non-empty and unparsed
                                logger.warning(f"Could not parse frequency string '{freq_name_str}' in raw_data_update_frequencies for {base_root_symbol}.")
                    elif isinstance(update_frequencies[0], dict): # Already in dict format
                        parsed_frequencies_for_raw = update_frequencies
                    else:
                        logger.error(f"Unsupported format for raw_data_update_frequencies for {base_root_symbol}: {update_frequencies}")
                
                if not parsed_frequencies_for_raw:
                    logger.info(f"No valid frequencies resolved for raw data update of {base_root_symbol}.")
                else:
                    for freq_detail in parsed_frequencies_for_raw: # Iterate over the parsed list of dicts
                        logger.info(f"Updating generic @{base_root_symbol} for interval: {freq_detail.get('interval', 'N/A')} {freq_detail.get('unit', 'N/A')}")
                        try:
                            # Use the new generic method in Application class
                            app.update_future_instrument_raw_data(
                                symbol_root=base_root_symbol,
                                interval_unit=freq_detail['unit'],
                                interval_value=freq_detail['interval'], # Note: YAML uses 'interval', App method uses 'interval_value'
                                force_full=args.full_update
                                # fetch_mode, lookback_days, roll_proximity_threshold_days will use defaults in app method
                            )
                        except KeyError as ke:
                            logger.error(f"Missing 'unit' or 'interval' in frequency detail for {base_root_symbol}: {freq_detail}. Error: {ke}", exc_info=True)
                        except Exception as e:
                            logger.error(f"Failed to update {base_root_symbol} for frequency {freq_detail}: {e}", exc_info=True)
            
            processed_roots_for_raw_update.add(base_root_symbol)

        # Step 2c: Update Individual Active Futures Contracts (e.g. ESM25, ESU25 for ES)
        logger.info("Updating data for active individual futures contracts (e.g., ESM25, ESU25)...")
        all_futures_configs = app.config.get_section('futures') # Get all future configs
        if all_futures_configs:
            for item_cfg in all_futures_configs:
                if item_cfg.get('asset_type') == 'future_group' and 'base_symbol' in item_cfg:
                    base_symbol = item_cfg['base_symbol']
                    logger.info(f"Processing individual active contracts for future_group: {base_symbol}")
                    
                    # Check skip flags before processing
                    should_skip_individual = False
                    if base_symbol == 'VX': # VX is often handled as 'futures' in skip flags
                        if args.skip_futures:
                            logger.info(f"Skipping individual active contracts for {base_symbol} due to --skip-futures flag.")
                            should_skip_individual = True
                    elif base_symbol in ['ES', 'NQ']:
                        if args.skip_es_nq:
                            logger.info(f"Skipping individual active contracts for {base_symbol} due to --skip-es-nq flag.")
                            should_skip_individual = True
                    # Add other specific skip flags if needed, e.g. args.skip_cl_futures for 'CL'

                    if should_skip_individual:
                        continue

                    try:
                        app.update_individual_active_futures(
                            symbol_root=base_symbol,
                            item_config=item_cfg, # Pass the full config for this future_group
                            force_full=args.full_update
                        )
                    except Exception as e_ind:
                        logger.error(f"Failed to update individual active contracts for {base_symbol}: {e_ind}", exc_info=True)
        else:
            logger.info("No 'futures' configuration found, skipping individual active contract updates.")
        
        # Step 2d: Update/Generate All Continuous Contracts
        logger.info("Processing all defined continuous future contracts (Step 2d)...")
        # Get the new structure from app.get_continuous_contract_symbols()
        all_root_product_configs = app.get_continuous_contract_symbols()

        if not args.skip_continuous: # Global skip for all continuous types
            if not all_root_product_configs:
                logger.info("No root product configurations returned by app.get_continuous_contract_symbols(). Skipping Step 2d.")
            else:
                for root_product_cfg in all_root_product_configs: # Iterate through roots like ES, NQ
                    root_s = root_product_cfg.get('root_symbol', 'UnknownRoot')
                    logger.info(f"-- Processing continuous contracts for root: {root_s} --")
                    
                    # Skip specific roots if flags are set
                    if root_s == 'VX' and args.skip_futures: 
                        logger.info(f"Skipping continuous contracts for root {root_s} due to --skip-futures.")
                        continue
                    if root_s in ['ES', 'NQ'] and args.skip_es_nq:
                        logger.info(f"Skipping continuous contracts for root {root_s} due to --skip-es-nq.")
                        continue
                    
                    # --- MODIFIED DEBUG LOGGING ---
                    temp_cc_list_for_debug = root_product_cfg.get('continuous_symbols_to_process', [])
                    if root_s == 'ES':
                        logger.debug(f"V2_DEBUG: ES root_product_cfg has continuous_symbols_to_process with len: {len(temp_cc_list_for_debug)}")
                        logger.debug(f"V2_DEBUG_ID: ES root_product_cfg object id: {id(root_product_cfg)}")
                        logger.debug(f"V2_DEBUG_ID: ES continuous_symbols_to_process from get() id: {id(temp_cc_list_for_debug)}")
                        # --- ADDED KEY CHECK LOGGING ---
                        key_exists = 'continuous_symbols_to_process' in root_product_cfg
                        logger.debug(f"V2_DEBUG_KEY_CHECK: 'continuous_symbols_to_process' key exists in ES root_product_cfg: {key_exists}")
                        if key_exists:
                            logger.debug(f"V2_DEBUG_KEY_CHECK: ID of existing list if key found: {id(root_product_cfg['continuous_symbols_to_process'])}")
                        # --- END ADDED KEY CHECK LOGGING ---
                        if not temp_cc_list_for_debug:
                             logger.debug(f"V2_DEBUG: ES root_product_cfg (full) when list is empty: {root_product_cfg}")
                    # --- END MODIFIED DEBUG LOGGING ---
                    continuous_contracts_list = temp_cc_list_for_debug 
                    if not continuous_contracts_list:
                        logger.info(f"No specific continuous contracts found under 'continuous_symbols_to_process' for root {root_s} in the processed config. Skipping specific cc processing for this root.")
                        # continue # Don't skip the whole root, might have other definitions like groups

                    # Initialize cc_method to a default value before the loop
                    cc_method = None

                    # --- Loop through specific continuous contracts defined directly under the root ---
                    for cc_item in continuous_contracts_list: # This is one specific continuous contract dict
                        cc_identifier = cc_item.get('identifier')
                        if not cc_identifier:
                            logger.warning(f"Continuous contract item for root {root_s} is missing 'identifier'. Skipping item: {cc_item}")
                            continue

                        cc_source = cc_item.get('default_source', root_product_cfg.get('default_source', 'unknown_source'))
                        cc_method = cc_item.get('method', 'none') # 'none', 'panama', 'backwards_ratio', etc.
                        cc_type = cc_item.get('type', 'continuous_future') # Default type

                        logger.info(f"Processing continuous symbol: {cc_identifier} (Source: {cc_source}, Method: {cc_method}) for root {root_s}")

                        # Get frequencies for this SPECIFIC continuous contract item
                        cc_frequencies_config = cc_item.get('frequencies', []) # This is typically a list of strings like ["15min", "daily"]
                        
                        # Determine target table (can be defined in cc_item or root_product_cfg)
                        if cc_source == 'inhouse_built':
                            default_target_table = 'continuous_contracts'
                        else:
                            default_target_table = cc_item.get('default_raw_table', root_product_cfg.get('default_raw_table', 'continuous_contracts'))

                        if cc_source == 'tradestation' and cc_method != 'panama': # Panama for TS is handled by a different mechanism
                            if not cc_frequencies_config:
                                logger.warning(f"No frequencies defined for TradeStation source continuous symbol {cc_identifier}. Skipping fetch.")
                                continue
                            
                            logger.info(f"Fetching data for TradeStation continuous symbol: {cc_identifier} for {len(cc_frequencies_config)} frequencies.")
                            
                            for freq_entry_loop_var in cc_frequencies_config: # Renamed variable here
                                logger.debug(f"Processing freq_entry_loop_var for {cc_identifier}: {freq_entry_loop_var}")
                                logger.debug(f"Relevant defaults for {cc_identifier}: cc_source='{cc_source}', default_target_table='{default_target_table}'")
                                
                                # Prepare freq_entry with defaults if it's a dictionary
                                current_freq_entry = freq_entry_loop_var
                                if isinstance(freq_entry_loop_var, dict):
                                    current_freq_entry = freq_entry_loop_var.copy()
                                    if 'source' not in current_freq_entry:
                                        current_freq_entry['source'] = cc_source
                                    if 'raw_table' not in current_freq_entry:
                                        current_freq_entry['raw_table'] = default_target_table
                                
                                logger.debug(f"Calling _parse_frequency_entry for {cc_identifier} with current_freq_entry: {current_freq_entry}")
                                # Corrected call to _parse_frequency_entry
                                parsed_freq_detail = app._parse_frequency_entry(
                                    freq_entry=current_freq_entry, 
                                    global_data_frequencies_map={}, 
                                    symbol_identifier=cc_identifier
                                )
                                logger.debug(f"Returned parsed_freq_detail for {cc_identifier}: {parsed_freq_detail}")

                                if not parsed_freq_detail or not parsed_freq_detail.get('unit') or not parsed_freq_detail.get('interval'):
                                    logger.warning(f"Invalid frequency detail for {cc_identifier}: {freq_entry_loop_var}. Parsed as: {parsed_freq_detail}. Skipping this frequency.")
                                    continue
                                
                                try:
                                    logger.info(f"Calling app.fetch_specific_symbol_data for {cc_identifier}, Freq: {parsed_freq_detail.get('name')}")
                                    app.fetch_specific_symbol_data(
                                        symbol_to_fetch=cc_identifier,
                                        interval_value=parsed_freq_detail['interval'],
                                        interval_unit=parsed_freq_detail['unit'],
                                        force_full=args.full_update
                                    )
                                except AttributeError as ae:
                                    logger.error(f"Application object does not have method 'fetch_specific_symbol_data' or there was an issue. Error: {ae}", exc_info=True)
                                    break 
                                except Exception as e_ts_cc:
                                    logger.error(f"Error updating TradeStation continuous contract {cc_identifier} for frequency {parsed_freq_detail.get('name')}: {e_ts_cc}", exc_info=True)
                        
                        elif cc_source == 'cboe': # Example: Direct CBOE downloads for some continuous (though less common for continuous)
                            # Similar parsing and calling logic for CBOE specific continuous data if needed
                            logger.info(f"CBOE source for continuous symbol {cc_identifier}. Processing not yet implemented in this loop.")
                            pass

                        elif cc_source == 'inhouse_built':
                            if not cc_frequencies_config:
                                logger.warning(f"No frequencies specified for in-house build of {cc_identifier}. Build will not be triggered based on frequencies here.")
                                # Build might be triggered by other mechanisms or not frequency-specific.
                            
                            logger.info(f"Build process for in-house continuous contract: {cc_identifier}")
                            # Logic for triggering in-house builds (e.g., from Panama or other methods)
                            # This section might call a different app method, e.g., app.build_inhouse_continuous_contract
                            
                            # If build is frequency specific and uses the same frequency list:
                            logger.info(f"Preparing to trigger build for {cc_identifier} for {len(cc_frequencies_config)} frequencies.")
                            for freq_entry_loop_var in cc_frequencies_config: # Renamed variable here
                                logger.debug(f"Processing freq_entry_loop_var for INHOUSE BUILD {cc_identifier}: {freq_entry_loop_var}")
                                logger.debug(f"Relevant defaults for INHOUSE BUILD {cc_identifier}: cc_source='{cc_source}', default_target_table='{default_target_table}'")
                                
                                # Prepare freq_entry with defaults if it's a dictionary
                                current_freq_entry = freq_entry_loop_var
                                if isinstance(freq_entry_loop_var, dict):
                                    current_freq_entry = freq_entry_loop_var.copy()
                                    if 'source' not in current_freq_entry:
                                        current_freq_entry['source'] = cc_source # Should be 'inhouse_built'
                                    if 'raw_table' not in current_freq_entry:
                                        current_freq_entry['raw_table'] = default_target_table # Target table for the built data

                                logger.debug(f"Calling _parse_frequency_entry for INHOUSE BUILD {cc_identifier} with current_freq_entry: {current_freq_entry}")
                                # Corrected call to _parse_frequency_entry
                                parsed_freq_detail = app._parse_frequency_entry(
                                    freq_entry=current_freq_entry, 
                                    global_data_frequencies_map={}, 
                                    symbol_identifier=cc_identifier
                                )
                                logger.debug(f"Returned parsed_freq_detail for INHOUSE BUILD {cc_identifier}: {parsed_freq_detail}")

                                if not parsed_freq_detail or not parsed_freq_detail.get('unit') or not parsed_freq_detail.get('interval'):
                                    logger.warning(f"Invalid frequency detail for built {cc_identifier}: {freq_entry_loop_var}. Skipping this frequency.")
                                    continue
                                try:
                                    # Build the in-house continuous contract using UnadjustedContractBuilder
                                    logger.info(f"Building in-house continuous contract for {cc_identifier} - Freq: {parsed_freq_detail.get('name')}")
                                    
                                    # Determine the root symbol (extract from cc_identifier)
                                    root_symbol_for_build = cc_identifier.split('=')[0].lstrip('@') if '=' in cc_identifier else cc_identifier.lstrip('@')
                                    
                                    # Create builder configuration
                                    builder_config = {
                                        'roll_calendar_table': cc_item.get('roll_calendar_table', 'futures_roll_dates'),
                                        'market_data_table': cc_item.get('market_data_table', 'market_data'),
                                        'continuous_data_table': default_target_table
                                    }
                                    
                                    # Create the builder directly
                                    builder = UnadjustedContractBuilder(db=app.db, config=builder_config)
                                    
                                    # Build the continuous series
                                    continuous_data = builder.build_continuous_series(
                                        root_symbol=root_symbol_for_build,
                                        continuous_symbol=cc_identifier,
                                        interval_unit=parsed_freq_detail['unit'],
                                        interval_value=parsed_freq_detail['interval'],
                                        force=args.full_update
                                    )
                                    
                                    # Store the continuous data
                                    if not continuous_data.empty:
                                        rows_stored = builder.store_continuous_data(cc_identifier, continuous_data)
                                        logger.info(f"Successfully built and stored {rows_stored} rows for {cc_identifier}")
                                    else:
                                        logger.warning(f"No continuous data generated for {cc_identifier}")
                                        
                                except AttributeError as ae:
                                    logger.error(f"Error with continuous contract builder: {ae}", exc_info=True)
                                    break
                                except Exception as e_build:
                                    logger.error(f"Error building in-house continuous contract {cc_identifier} for frequency {parsed_freq_detail.get('name')}: {e_build}", exc_info=True)
                        else:
                            logger.warning(f"Source '{cc_source}' for continuous contract {cc_identifier} is not explicitly handled in Step 2d main loop.")

                    # Panama method continuous contracts (often separate due to specific build logic)
                    if root_s == 'VX' and cc_method == 'panama':
                        logger.info(f"Skipping continuous contracts for {root_s} due to --skip-panama and method 'panama'.")
                        continue
                    if root_s in ['ES', 'NQ'] and cc_method == 'panama':
                        logger.info(f"Skipping continuous contracts for {root_s} due to --skip-panama and method 'panama'.")
                        continue

        else: # args.skip_continuous is True
            logger.info("Skipping all continuous contract updates/generation in Step 2d due to --skip-continuous flag.")

        # Step 3: Apply data cleaning pipeline
        if not args.skip_cleaning:
            logger.info("Running data cleaning pipeline...")

            # Ensure 'data_cleaning_runs' table exists
            data_cleaning_runs_table_sql = """
            CREATE TABLE IF NOT EXISTS data_cleaning_runs (
                run_id VARCHAR PRIMARY KEY,
                pipeline_name VARCHAR NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                records_count INTEGER,
                modifications_count INTEGER,
                elapsed_time DOUBLE,
                status VARCHAR, -- e.g., 'success', 'error', 'skipped'
                error_message TEXT, -- For storing error details if status is 'error'
                cleaners_applied TEXT, -- Comma-separated list of cleaner names
                fields_modified TEXT, -- Comma-separated list of modified fields
                log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- When this log entry was created
            );
            """
            try:
                if not app.db.table_exists('data_cleaning_runs'):
                    logger.info("Table 'data_cleaning_runs' does not exist. Attempting to create it now...")
                    app.db.execute(data_cleaning_runs_table_sql)
                    if app.db.table_exists('data_cleaning_runs'):
                        logger.info("Successfully created 'data_cleaning_runs' table.")
                    else:
                        logger.error("Failed to create 'data_cleaning_runs' table after execution attempt.")
                        # Depending on criticality, you might want to raise an error or sys.exit() here
                else:
                    logger.debug("'data_cleaning_runs' table already exists.")
            except Exception as e:
                logger.error(f"Error during check/creation of 'data_cleaning_runs' table: {e}", exc_info=True)
                # Handle error appropriately, perhaps exit if this table is critical for logging
            
            # Create data cleaning pipeline
            cleaning_pipeline = DataCleaningPipeline(
                name="market_data_cleaner",
                db_connector=app.db,
                config={
                    'track_performance': True,
                    'track_modifications': True,
                    'save_summary': True
                }
            )
            
            # Add cleaners to pipeline
            cleaning_pipeline.add_cleaner(VXZeroPricesCleaner(
                db_connector=app.db,
                config={
                    'interpolation_method': 'linear',
                    'max_gap_days': 5,
                    'log_all_modifications': True
                }
            ))
            
            # Process VX data
            if not args.skip_futures:
                # Get all VX symbols
                vx_symbols = app.get_vx_futures_symbols()
                
                # Process in batches
                for symbol in vx_symbols:
                    if args.dry_run:
                        logger.info(f"Would clean data for {symbol}")
                    else:
                        success, summary = cleaning_pipeline.process_symbol(
                            symbol=symbol,
                            interval_unit="daily",
                            interval_value=1
                        )
                        
                        if success:
                            logger.info(f"Cleaned data for {symbol}: "
                                     f"{summary.get('total_modifications', 0)} modifications")
                        else:
                            logger.warning(f"Failed to clean data for {symbol}")
            
            logger.info("Data cleaning completed")
        else:
            logger.info("Skipping data cleaning")
        
        # Step 4: Verify data if requested
        if args.verify:
            logger.info("Verifying data...")
            
            # Verify continuous contracts
            app.verify_continuous_contracts()
            
            # Verify raw data
            app.verify_market_data()
            
            logger.info("Data verification completed")
        
        # Clean up
        app.close()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Market data update completed in {elapsed_time:.2f} seconds")
        logger.info("-" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during market data update: {e}", exc_info=True)
        return 1
    finally:
        if app:
            app.close() # Ensure app is closed if it was initialized

if __name__ == "__main__":
    sys.exit(main())