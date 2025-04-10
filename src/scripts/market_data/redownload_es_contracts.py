#!/usr/bin/env python
"""
Redownload ES Contracts Script

This script redownloads the problematic ES contracts (ESM25, ESU25, ESZ25) to fix data issues.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def delete_contract_data(symbol):
    """Delete data for a specific contract from the database."""
    try:
        conn = duckdb.connect('data/financial_data.duckdb')
        conn.execute(f"DELETE FROM market_data WHERE symbol = '{symbol}'")
        logger.info(f"Deleted existing data for {symbol}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting data for {symbol}: {e}")
        return False

def download_contract_data(symbol, start_date=None, end_date=None):
    """Download data for a specific contract."""
    try:
        # Import the fetch_market_data module
        from src.scripts.fetch_market_data import fetch_market_data
        
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch the data
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        success = fetch_market_data(symbol, start_date, end_date)
        
        if success:
            logger.info(f"Successfully downloaded data for {symbol}")
            return True
        else:
            logger.error(f"Failed to download data for {symbol}")
            return False
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return False

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Redownload ES contracts to fix data issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Redownload all problematic ES contracts
  python redownload_es_contracts.py
  
  # Redownload a specific contract
  python redownload_es_contracts.py --symbol ESM25
  
  # Redownload with custom date range
  python redownload_es_contracts.py --start 2023-01-01 --end 2025-12-31
        """
    )
    parser.add_argument('--symbol', help='Specific contract to redownload (e.g., ESM25)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # List of problematic contracts
    problematic_contracts = ['ESM25', 'ESU25', 'ESZ25']
    
    # If a specific symbol is provided, only process that one
    if args.symbol:
        if args.symbol not in problematic_contracts:
            logger.warning(f"{args.symbol} is not in the list of problematic contracts")
            logger.warning(f"Proceeding anyway, but be aware this may not fix the issue")
        
        # Delete existing data
        if delete_contract_data(args.symbol):
            # Download new data
            download_contract_data(args.symbol, args.start, args.end)
    else:
        # Process all problematic contracts
        for symbol in problematic_contracts:
            # Delete existing data
            if delete_contract_data(symbol):
                # Download new data
                download_contract_data(symbol, args.start, args.end)
    
    logger.info("Redownload process completed")

if __name__ == '__main__':
    main() 