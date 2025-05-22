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

            update_frequencies = root_cfg.get('update_frequencies', [])
            if not update_frequencies:
                logger.info(f"No 'update_frequencies' defined in config for {base_root_symbol}. Skipping its raw data update.")
            else:
                logger.info(f"Processing raw data updates for {base_root_symbol} across {len(update_frequencies)} frequencies.")
                for freq_detail in update_frequencies:
                    logger.info(f"Updating {base_root_symbol} for interval: {freq_detail.get('interval', 'N/A')} {freq_detail.get('unit', 'N/A')}")
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

        # Step 3: Generate continuous contracts with Panama method
        if not args.skip_continuous:
            logger.info("Generating continuous contracts with Panama method...")
            
            # Get symbols that need continuous contract generation
            # root_symbol_configs is a list of dicts, one for each base (ES, NQ etc.)
            root_symbol_configs = app.get_continuous_contract_symbols() 
            
            for root_cfg in root_symbol_configs: # e.g., root_cfg for 'ES'
                base_root_symbol = root_cfg.get('root_symbol')
                if not base_root_symbol:
                    logger.warning(f"Skipping root_cfg due to missing 'root_symbol': {root_cfg}")
                    continue
                
                logger.info(f"Processing Panama continuous contracts for root: {base_root_symbol}")

                # Iterate through each specific continuous symbol to be generated for this root
                specific_symbols_list = root_cfg.get('continuous_symbols_to_generate', [])

                if not specific_symbols_list:
                    logger.warning(f"No 'continuous_symbols_to_generate' found in config for {base_root_symbol}. Skipping Panama generation for it.")
                    continue

                for specific_symbol_detail in specific_symbols_list:
                    target_continuous_symbol = specific_symbol_detail.get('symbol')
                    
                    if not target_continuous_symbol:
                        logger.warning(f"Missing 'symbol' key in specific_symbol_detail for {base_root_symbol}. Details: {specific_symbol_detail}")
                        continue

                    # Check if this specific symbol is meant for Panama adjustment
                    if specific_symbol_detail.get('adjustment_method') != 'panama':
                        logger.info(f"Skipping {target_continuous_symbol} as its adjustment method is not 'panama'.")
                        continue
                        
                    logger.info(f"Generating Panama contract for: {target_continuous_symbol} (base: {base_root_symbol})")
                    
                    # Create a merged config for the builder:
                    # Start with a copy of the root_cfg (general settings for ES, NQ)
                    builder_config = root_cfg.copy() 
                    # Update/add specifics from the symbol detail (e.g. adjustment_method)
                    builder_config.update(specific_symbol_detail)

                    try:
                        generator = PanamaContractBuilder(
                            db=app.db,
                            config=builder_config 
                        )
                        
                        # Call the correct method: build_continuous_series
                        # Default interval is daily/1, start/end are None (handles full range)
                        generated_df = generator.build_continuous_series(
                            root_symbol=base_root_symbol,
                            continuous_symbol=target_continuous_symbol,
                            force=args.full_update 
                        )
                        if generated_df is not None and not generated_df.empty:
                            logger.info(f"Successfully generated/updated Panama series for {target_continuous_symbol}, {len(generated_df)} rows.")
                        elif generated_df is not None and generated_df.empty:
                            logger.info(f"Panama series generation for {target_continuous_symbol} returned empty DataFrame (might be up-to-date or no data).")
                        else: # None returned
                            logger.warning(f"Panama series generation for {target_continuous_symbol} returned None.")
                    except Exception as e_gen:
                        logger.error(f"Error generating Panama series for {target_continuous_symbol}: {e_gen}", exc_info=True)
            
            logger.info("Continuous contract generation completed")
        else:
            logger.info("Skipping continuous contract generation")
        
        # Step 4: Apply data cleaning pipeline
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
        
        # Step 5: Verify data if requested
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