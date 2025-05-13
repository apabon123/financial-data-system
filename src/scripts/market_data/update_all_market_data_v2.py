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
sys.path.append(project_root)

# Application imports
from src.core.app import Application
from src.core.config import ConfigManager
from src.processors.continuous.registry import get_registry
from src.processors.cleaners.pipeline import DataCleaningPipeline
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner

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
    
    # Specific component options
    parser.add_argument("--update-active-es-15min", action="store_true", help="Update active ES 15min data")
    parser.add_argument("--update-active-es-1min", action="store_true", help="Update active ES 1min data")
    
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
    
    # Initialize the application
    try:
        app = Application(
            config_path=args.config_path,
            db_path=args.db_path,
            read_only=args.dry_run
        )
        
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
        
        if not args.skip_futures:
            logger.info("Updating VX futures data...")
            app.update_vx_futures(force_full=args.full_update)
            logger.info("VX futures update completed")
        else:
            logger.info("Skipping VX futures update")
        
        if not args.skip_es_nq:
            logger.info("Updating ES/NQ futures data...")
            
            # Handle specific timeframes if requested
            if args.update_active_es_15min:
                app.update_es_futures(interval_unit="minute", interval_value=15, 
                                    only_active=True, force_full=args.full_update)
            
            if args.update_active_es_1min:
                app.update_es_futures(interval_unit="minute", interval_value=1, 
                                    only_active=True, force_full=args.full_update)
            
            # Always update daily data
            app.update_es_futures(interval_unit="daily", interval_value=1, 
                                force_full=args.full_update)
            
            app.update_nq_futures(force_full=args.full_update)
            logger.info("ES/NQ futures update completed")
        else:
            logger.info("Skipping ES/NQ futures update")
        
        # Step 3: Generate continuous contracts with Panama method
        if not args.skip_continuous:
            logger.info("Generating continuous contracts with Panama method...")
            
            # Get continuous contract registry
            registry = get_registry()
            
            # Get symbols that need continuous contract generation
            symbols = app.get_continuous_contract_symbols()
            
            for symbol_config in symbols:
                root_symbol = symbol_config['root_symbol']
                positions = symbol_config.get('positions', [1])
                
                logger.info(f"Processing continuous contracts for {root_symbol}")
                
                for position in positions:
                    # Create Panama contract generator
                    generator = registry.create(
                        'panama',
                        root_symbol=root_symbol,
                        position=position,
                        ratio_limit=args.panama_ratio,
                        db_connector=app.db_connector,
                        roll_strategy='volume'
                    )
                    
                    # Generate continuous contract
                    if generator:
                        # Determine date range for update
                        lookback_date = datetime.now() - timedelta(days=args.lookback_days)
                        lookback_str = lookback_date.strftime('%Y-%m-%d')
                        
                        if args.full_update:
                            # Use None for start_date to get full history
                            result_df = generator.generate()
                        else:
                            # Use lookback date for incremental update
                            result_df = generator.generate(start_date=lookback_str)
                        
                        if not args.dry_run and not result_df.empty:
                            # Save to database
                            app.save_continuous_contracts(result_df)
                            
                            logger.info(f"Generated {len(result_df)} data points for "
                                      f"{root_symbol} position {position}")
                    else:
                        logger.error(f"Failed to create Panama generator for {root_symbol}")
            
            logger.info("Continuous contract generation completed")
        else:
            logger.info("Skipping continuous contract generation")
        
        # Step 4: Apply data cleaning pipeline
        if not args.skip_cleaning:
            logger.info("Running data cleaning pipeline...")
            
            # Create data cleaning pipeline
            cleaning_pipeline = DataCleaningPipeline(
                name="market_data_cleaner",
                db_connector=app.db_connector,
                config={
                    'track_performance': True,
                    'track_modifications': True,
                    'save_summary': True
                }
            )
            
            # Add cleaners to pipeline
            cleaning_pipeline.add_cleaner(VXZeroPricesCleaner(
                db_connector=app.db_connector,
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

if __name__ == "__main__":
    sys.exit(main())