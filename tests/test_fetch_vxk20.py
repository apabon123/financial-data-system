#!/usr/bin/env python
"""
Test script to fetch data for VXK20 with different intervals.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

# Import MarketDataFetcher
from src.scripts.fetch_market_data import MarketDataFetcher

# Load environment variables from .env file
env_path = os.path.join(project_root, "config", ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_fetch_vxk20():
    """Test fetching data for VXK20 with different intervals."""
    # Load the configuration file
    config_path = os.path.join(project_root, "config", "market_symbols.yaml")
    
    # Create the fetcher with the config
    fetcher = MarketDataFetcher(config_path=config_path)
    
    # Authenticate with TradeStation
    if not fetcher.ts_agent.authenticate():
        print("Failed to authenticate with TradeStation API")
        return
    
    print("Successfully authenticated with TradeStation API")
    
    # Test symbol
    symbol = "VXK20"
    
    # Test intervals
    intervals = [
        {"value": 1, "unit": "daily", "name": "Daily"},
        {"value": 15, "unit": "minute", "name": "15-minute"},
        {"value": 1, "unit": "minute", "name": "1-minute"}
    ]
    
    # Test each interval
    for interval in intervals:
        print(f"\n{'='*50}")
        print(f"Testing {interval['name']} data for {symbol}")
        
        try:
            # Fetch data
            data = fetcher.fetch_data_since(
                symbol=symbol,
                interval=interval['value'],
                unit=interval['unit'],
                start_date='2020-01-01'
            )
            
            # Verify data
            if data is not None and not data.empty:
                print(f"Successfully fetched {len(data)} records")
                print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Save to test database
                fetcher.save_to_db(data)
                
                # Verify data was saved
                count = fetcher.conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM market_data 
                    WHERE symbol = '{symbol}'
                    AND interval_value = {interval['value']}
                    AND interval_unit = '{interval['unit']}'
                """).fetchone()[0]
                
                print(f"Verified {count} records in database")
            else:
                print(f"No data returned for {interval['name']}")
                
        except Exception as e:
            print(f"Error testing {symbol} with {interval['name']}: {e}")

if __name__ == "__main__":
    test_fetch_vxk20() 