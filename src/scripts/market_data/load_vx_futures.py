#!/usr/bin/env python3
"""
Script to load VX futures data into the database.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
import duckdb
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger
from src.config.settings import DEFAULT_DB_PATH

logger = setup_logger(__name__)

class VXFuturesLoader:
    """Class to load VX futures data into the database."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the VX futures loader."""
        self.db_path = db_path
        self.conn = self._connect_database()
        self._init_database()
        
    def _connect_database(self) -> duckdb.DuckDBPyConnection:
        """Connect to the database."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
            
    def _init_database(self):
        """Initialize the database tables if they don't exist."""
        try:
            # Create market_data table if it doesn't exist
            self.conn.execute("""
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
                adjusted BOOLEAN DEFAULT FALSE,
                quality INTEGER DEFAULT 100,
                PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
            )
            """)
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
            
    def _get_vx_contracts(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Get list of VX futures contracts for the given date range."""
        contracts = []
        current_date = start_date
        
        while current_date <= end_date:
            # VX futures expire on the third Wednesday of each month
            # Format: VX + month code + year (e.g., VXF24 for February 2024)
            month_codes = {
                1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
            }
            
            # Get the third Wednesday of the current month
            first_day = current_date.replace(day=1)
            first_wednesday = first_day + timedelta(days=((2 - first_day.weekday()) % 7))
            third_wednesday = first_wednesday + timedelta(weeks=2)
            
            # If we're past the expiry, move to next month
            if current_date > third_wednesday:
                current_date = (current_date + timedelta(days=32)).replace(day=1)
                continue
                
            # Create contract symbol
            month_code = month_codes[current_date.month]
            year_code = str(current_date.year)[-2:]
            contract = f"VX{month_code}{year_code}"
            
            if contract not in contracts:
                contracts.append(contract)
                
            current_date = (current_date + timedelta(days=32)).replace(day=1)
            
        return contracts
        
    def _fetch_contract_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data for a specific VX futures contract."""
        try:
            # Use yfinance to fetch the data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Format the data to match our database schema
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add required columns
            df['symbol'] = symbol
            df['source'] = 'yfinance'
            df['interval_unit'] = 'day'
            df['interval_value'] = 1
            df['up_volume'] = None
            df['down_volume'] = None
            df['adjusted'] = False
            df['quality'] = 100
            
            return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                      'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
                      'adjusted', 'quality']]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
    def load_data(self, start_date: datetime, end_date: datetime):
        """Load VX futures data for the given date range."""
        try:
            # Get list of contracts
            contracts = self._get_vx_contracts(start_date, end_date)
            logger.info(f"Found {len(contracts)} contracts to load: {contracts}")
            
            # Fetch and load data for each contract
            for contract in contracts:
                logger.info(f"Loading data for {contract}")
                df = self._fetch_contract_data(contract, start_date, end_date)
                
                if df is not None and not df.empty:
                    # Insert data into database
                    self.conn.execute("""
                    INSERT INTO market_data 
                    SELECT * FROM df
                    ON CONFLICT (timestamp, symbol, interval_value, interval_unit) DO UPDATE
                    SET open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        source = excluded.source,
                        quality = excluded.quality
                    """)
                    logger.info(f"Loaded {len(df)} rows for {contract}")
                else:
                    logger.warning(f"No data loaded for {contract}")
                    
        except Exception as e:
            logger.error(f"Error loading VX futures data: {e}")
            raise
            
def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Load VX futures data into the database")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the database file")
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        loader = VXFuturesLoader(db_path=args.db_path)
        loader.load_data(start_date, end_date)
        
    except Exception as e:
        logger.error(f"Error running script: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 