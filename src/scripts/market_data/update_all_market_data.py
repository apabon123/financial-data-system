#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Market Data Update Script

This script orchestrates the full update process for financial market data:
1. Updates the VIX Index
2. Updates active VX futures contracts
3. Updates continuous VX contracts
4. Fills historical gaps in VXc1 and VXc2 for 2004-2005 if full update requested

Run this script daily to keep the database up-to-date.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime, timedelta
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "data/financial_data.duckdb"
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"

def run_script(script_name, args=None):
    """Run a Python module with arguments."""
    cmd = [sys.executable, "-m", script_name]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully completed {script_name}")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def check_db_exists(db_path):
    """Check if the database file exists, create parent directories if needed."""
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        logger.info(f"Database file not found. Will be created at: {db_path}")
        return False
    return True

def update_vix_index(db_path):
    """Update the VIX Index data."""
    logger.info("=== Updating VIX Index ($VIX.X) ===")
    args = [f"--db-path={db_path}"]
    return run_script("src.scripts.market_data.update_vix_index", args)

def update_vx_futures(db_path, config_path, full_regen=False):
    """Update active VX futures contracts."""
    logger.info("=== Updating VX Futures Contracts ===")
    args = [f"--db-path={db_path}", f"--config-path={config_path}"]
    if full_regen:
        args.append("--full-regen")
    return run_script("src.scripts.market_data.update_vx_futures", args)

def update_continuous_contracts(db_path, config_path, start_date=None, end_date=None, force=False):
    """Generate continuous VX contracts."""
    logger.info("=== Updating VX Continuous Contracts ===")
    
    # Default to last 90 days if no start date provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Default to today if no end date provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    args = [
        f"--db-path={db_path}", 
        f"--config-path={config_path}",
        "--root-symbol=VX",
        f"--start-date={start_date}",
        f"--end-date={end_date}"
    ]
    
    if force:
        args.append("--force")
        
    return run_script("src.scripts.market_data.generate_continuous_futures", args)

def fill_historical_gaps(db_path):
    """Fill historical gaps in VXc1 and VXc2 for 2004-2005."""
    logger.info("=== Filling Historical Gaps in VXc1/VXc2 (2004-2005) ===")
    args = [f"--db-path={db_path}"]
    return run_script("src.scripts.market_data.fill_vx_continuous_gaps", args)

def verify_continuous_contracts(db_path, start_date=None, end_date=None):
    """Verify VX continuous contracts after update."""
    logger.info("=== Verifying VX Continuous Contracts ===")
    
    # Default to last 90 days if no start date provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Default to today if no end date provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Verify both VXc1 and VXc2
    for symbol in ["VXc1", "VXc2"]:
        args = [
            f"--symbol={symbol}",
            f"--start-date={start_date}",
            f"--end-date={end_date}",
            f"--db-path={db_path}"
        ]
        logger.info(f"Verifying {symbol}...")
        run_script("src.scripts.market_data.improved_verify_continuous", args)

def verify_data_counts(db_path):
    """Check data counts in the database for key tables."""
    logger.info("=== Verifying Data Counts ===")
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Query market_data table
        symbols = ['$VIX.X', 'VXc1', 'VXc2']
        for symbol in symbols:
            result = conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE Symbol = ?", 
                [symbol]
            ).fetchone()
            logger.info(f"{symbol} data count: {result[0]} rows")
        
        # Count active VX contracts
        result = conn.execute(
            "SELECT COUNT(DISTINCT Symbol) FROM market_data WHERE Symbol LIKE 'VX%' AND Symbol NOT LIKE 'VXc%'"
        ).fetchone()
        logger.info(f"VX contracts count: {result[0]} distinct symbols")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error verifying data counts: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete update workflow for financial market data.')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
    parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the market symbols config YAML file.')
    parser.add_argument('--full-update', action='store_true', help='Perform a full update including historical data.')
    parser.add_argument('--start-date', type=str, help='Start date for continuous contracts update (YYYY-MM-DD).')
    parser.add_argument('--end-date', type=str, help='End date for continuous contracts update (YYYY-MM-DD).')
    parser.add_argument('--verify', action='store_true', help='Run verification after update.')
    parser.add_argument('--skip-vix', action='store_true', help='Skip VIX Index update.')
    parser.add_argument('--skip-futures', action='store_true', help='Skip VX futures update.')
    parser.add_argument('--skip-continuous', action='store_true', help='Skip continuous contracts update.')
    parser.add_argument('--skip-historical', action='store_true', help='Skip historical gap filling.')
    
    args = parser.parse_args()
    
    # Ensure database directory exists
    check_db_exists(args.db_path)
    
    # Track success of each step
    success = True
    
    # Update VIX Index
    if not args.skip_vix:
        success = update_vix_index(args.db_path) and success
    else:
        logger.info("Skipping VIX Index update")
    
    # Update VX futures
    if not args.skip_futures:
        success = update_vx_futures(args.db_path, args.config_path, args.full_update) and success
    else:
        logger.info("Skipping VX futures update")
    
    # Update continuous contracts
    if not args.skip_continuous:
        if args.full_update:
            # For full update, we regenerate from the beginning of the data
            success = update_continuous_contracts(args.db_path, args.config_path, 
                                                start_date="2004-01-01", force=True) and success
        else:
            # For regular update, use the provided date range or default to last 90 days
            success = update_continuous_contracts(args.db_path, args.config_path, 
                                                args.start_date, args.end_date) and success
    else:
        logger.info("Skipping continuous contracts update")
    
    # Fill historical gaps for 2004-2005
    if args.full_update and not args.skip_historical:
        success = fill_historical_gaps(args.db_path) and success
    elif args.skip_historical:
        logger.info("Skipping historical gap filling")
    else:
        logger.info("Historical gap filling only runs with --full-update")
    
    # Verify the data if requested
    if args.verify:
        if args.full_update:
            verify_continuous_contracts(args.db_path, "2004-01-01", args.end_date)
        else:
            verify_continuous_contracts(args.db_path, args.start_date, args.end_date)
        
        verify_data_counts(args.db_path)
    
    if success:
        logger.info("=== All updates completed successfully ===")
    else:
        logger.warning("=== Updates completed with some errors ===")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 