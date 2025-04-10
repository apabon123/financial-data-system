#!/usr/bin/env python

"""
Script to force update specific futures contracts.
This script is designed to update specific contracts with the force flag,
overwriting any existing data in the database.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

# Import MarketDataFetcher
from src.scripts.fetch_market_data import MarketDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")

def force_update_contracts(base_symbol, start_contract, end_contract, interval_value=15, interval_unit="minute"):
    """
    Force update specific futures contracts.
    
    Args:
        base_symbol: Base symbol (e.g., 'ES' for E-mini S&P 500 futures)
        start_contract: Starting contract (e.g., 'ESH20')
        end_contract: Ending contract (e.g., 'ESZ25')
        interval_value: Interval value (e.g., 15 for 15-minute data)
        interval_unit: Interval unit (e.g., 'minute' for minute data)
    """
    # Initialize fetcher
    config_path = os.path.join(project_root, "config", "market_symbols.yaml")
    fetcher = MarketDataFetcher(config_path=config_path, db_path=DEFAULT_DB_PATH)
    
    # Authenticate with TradeStation
    if not fetcher.ts_agent.authenticate():
        logger.error("Failed to authenticate with TradeStation API")
        return
    
    logger.info(f"Successfully authenticated with TradeStation API")
    
    # Generate list of contracts to update
    contracts = []
    current_contract = start_contract
    
    # Extract month codes and years
    start_month = start_contract[2]
    start_year = int(start_contract[3:])
    end_month = end_contract[2]
    end_year = int(end_contract[3:])
    
    # Map month codes to months for comparison
    month_map = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    # Generate all contracts between start and end
    for year in range(start_year, end_year + 1):
        for month_code, month_num in month_map.items():
            # Skip months before start month in start year
            if year == start_year and month_num < month_map[start_month]:
                continue
            # Skip months after end month in end year
            if year == end_year and month_num > month_map[end_month]:
                continue
                
            contract = f"{base_symbol}{month_code}{str(year)[-2:]}"
            contracts.append(contract)
    
    # Sort contracts chronologically
    contracts.sort()
    
    # Process each contract
    success_count = 0
    failed_count = 0
    
    for contract in contracts:
        logger.info(f"Force updating {contract} with interval {interval_value} {interval_unit}")
        
        try:
            # Delete existing data for this contract and interval
            fetcher.delete_existing_data(contract, interval_value, interval_unit)
            
            # Fetch and save new data
            fetcher.process_symbol(contract, update_history=True, force=True)
            
            logger.info(f"Successfully updated {contract}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error updating {contract}: {e}")
            failed_count += 1
    
    # Print summary
    logger.info(f"\nUpdate Summary:")
    logger.info(f"Successfully updated: {success_count} contracts")
    logger.info(f"Failed to update: {failed_count} contracts")

def main():
    """Main function to force update specific futures contracts."""
    parser = argparse.ArgumentParser(description='Force update specific futures contracts')
    parser.add_argument('base_symbol', help='Base symbol (e.g., ES for E-mini S&P 500 futures)')
    parser.add_argument('start_contract', help='Starting contract (e.g., ESH20)')
    parser.add_argument('end_contract', help='Ending contract (e.g., ESZ25)')
    parser.add_argument('--interval-value', type=int, default=15, help='Interval value (e.g., 15 for 15-minute data)')
    parser.add_argument('--interval-unit', default='minute', choices=['minute', 'daily', 'hour'], help='Interval unit')
    
    args = parser.parse_args()
    
    # Force update contracts
    force_update_contracts(
        args.base_symbol,
        args.start_contract,
        args.end_contract,
        args.interval_value,
        args.interval_unit
    )

if __name__ == "__main__":
    main() 