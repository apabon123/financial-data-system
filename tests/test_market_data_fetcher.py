#!/usr/bin/env python
"""
Test script for market data fetching functionality.
Tests both futures contracts and equities with different intervals.
"""

import os
import sys
import logging
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

# Import MarketDataFetcher
from src.scripts.fetch_market_data import MarketDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TestMarketDataFetcher(unittest.TestCase):
    """Test cases for MarketDataFetcher class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load environment variables
        env_path = os.path.join(project_root, "config", ".env")
        load_dotenv(env_path)
        
        # Set up test database
        cls.test_db_path = os.path.join(project_root, "data", "test_financial_data.duckdb")
        cls.conn = duckdb.connect(cls.test_db_path)
        
        # Create market_data table for testing
        cls.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                up_volume BIGINT,
                down_volume BIGINT,
                source VARCHAR,
                interval_value INTEGER,
                interval_unit VARCHAR,
                adjusted BOOLEAN,
                quality INTEGER,
                PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
            )
        """)
        
        # Initialize fetcher
        config_path = os.path.join(project_root, "config", "market_symbols.yaml")
        cls.fetcher = MarketDataFetcher(config_path=config_path)
        cls.fetcher.db_path = cls.test_db_path
        
        # Authenticate with TradeStation
        if not cls.fetcher.ts_agent.authenticate():
            raise Exception("Failed to authenticate with TradeStation API")
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        try:
            # Drop the test table
            cls.conn.execute("DROP TABLE IF EXISTS market_data")
            cls.conn.close()
            
            # Delete the test database file
            if os.path.exists(cls.test_db_path):
                os.remove(cls.test_db_path)
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def setUp(self):
        """Set up for each test."""
        # Clear the market_data table before each test
        self.conn.execute("DELETE FROM market_data")
    
    def test_fetch_vx_futures(self):
        """Test fetching VX futures data."""
        # Test symbols
        symbols = ["VXZ24", "VXF25"]
        intervals = [
            {"value": 1, "unit": "daily", "name": "Daily"},
            {"value": 15, "unit": "minute", "name": "15-minute"},
            {"value": 1, "unit": "minute", "name": "1-minute"}
        ]
        
        for symbol in symbols:
            logging.info(f"\nTesting {symbol}")
            for interval in intervals:
                logging.info(f"\n{'='*50}")
                logging.info(f"Testing {interval['name']} data for {symbol}")
                
                try:
                    # Fetch data
                    data = self.fetcher.fetch_data_since(
                        symbol=symbol,
                        interval=interval['value'],
                        unit=interval['unit'],
                        start_date='2024-01-01'
                    )
                    
                    # Verify data
                    self.assertIsNotNone(data)
                    self.assertFalse(data.empty)
                    self.assertTrue('timestamp' in data.columns)
                    self.assertTrue('open' in data.columns)
                    self.assertTrue('high' in data.columns)
                    self.assertTrue('low' in data.columns)
                    self.assertTrue('close' in data.columns)
                    self.assertTrue('volume' in data.columns)
                    
                    # Log results
                    logging.info(f"Successfully fetched {len(data)} records")
                    logging.info(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                    
                    # Save to test database
                    self.fetcher.save_to_db(data)
                    
                    # Verify data was saved
                    count = self.conn.execute(f"""
                        SELECT COUNT(*) 
                        FROM market_data 
                        WHERE symbol = '{symbol}'
                        AND interval_value = {interval['value']}
                        AND interval_unit = '{interval['unit']}'
                    """).fetchone()[0]
                    
                    self.assertEqual(count, len(data))
                    
                except Exception as e:
                    self.fail(f"Error testing {symbol} with {interval['name']}: {e}")
    
    def test_fetch_equity(self):
        """Test fetching equity data."""
        # Test with SPY
        symbol = "SPY"
        intervals = [
            {"value": 1, "unit": "daily", "name": "Daily"},
            {"value": 15, "unit": "minute", "name": "15-minute"},
            {"value": 1, "unit": "minute", "name": "1-minute"}
        ]
        
        logging.info(f"\nTesting {symbol}")
        for interval in intervals:
            logging.info(f"\n{'='*50}")
            logging.info(f"Testing {interval['name']} data for {symbol}")
            
            try:
                # Fetch data
                data = self.fetcher.fetch_data_since(
                    symbol=symbol,
                    interval=interval['value'],
                    unit=interval['unit'],
                    start_date='2024-01-01'
                )
                
                # Verify data
                self.assertIsNotNone(data)
                self.assertFalse(data.empty)
                self.assertTrue('timestamp' in data.columns)
                self.assertTrue('open' in data.columns)
                self.assertTrue('high' in data.columns)
                self.assertTrue('low' in data.columns)
                self.assertTrue('close' in data.columns)
                self.assertTrue('volume' in data.columns)
                
                # Log results
                logging.info(f"Successfully fetched {len(data)} records")
                logging.info(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Save to test database
                self.fetcher.save_to_db(data)
                
                # Verify data was saved
                count = self.conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM market_data 
                    WHERE symbol = '{symbol}'
                    AND interval_value = {interval['value']}
                    AND interval_unit = '{interval['unit']}'
                """).fetchone()[0]
                
                self.assertEqual(count, len(data))
                
            except Exception as e:
                self.fail(f"Error testing {symbol} with {interval['name']}: {e}")

if __name__ == '__main__':
    unittest.main() 