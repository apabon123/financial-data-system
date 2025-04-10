#!/usr/bin/env python
"""
Test script to fetch data for VXF25 with different intervals.
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

def test_fetch_vxf25():
    """Test fetching data for VXF25 with different intervals."""
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
    symbol = "VXF25"
    
    # Test intervals
    intervals = [
        {"value": 1, "unit": "daily", "name": "Daily"},
        {"value": 15, "unit": "minute", "name": "15-minute"},
        {"value": 1, "unit": "minute", "name": "1-minute"}
    ]
    
    # Fetch data for each interval
    for interval in intervals:
        print(f"\n{'='*50}")
        print(f"Testing {interval['name']} data for {symbol}")
        
        try:
            print(f"Fetching {interval['name']} data for {symbol}...")
            
            # Fetch data using the fetcher's API
            data = fetcher.fetch_data_since(
                symbol=symbol,
                interval=interval['value'],
                unit=interval['unit'],
                start_date='2024-01-01'  # Start from beginning of 2024
            )
            
            if data is not None and not data.empty:
                print("Successfully fetched data")
                print(f"Records: {len(data)}")
                print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Print first few records
                print("\nFirst few records:")
                print(data.head().to_string())
            else:
                print("No data returned")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch_vxf25() 