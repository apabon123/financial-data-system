#!/usr/bin/env python
"""
Fetch Market Data Script

This script fetches market data from TradeStation for symbols defined in the market_symbols.yaml file.
It first checks the database to see what data is already available, then fetches only the missing data.
"""

import os
import sys
import yaml
import time
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import requests
import duckdb
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from our project
from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_data_fetch.log')
    ]
)
logger = logging.getLogger(__name__)

# Get database path from environment variable
DATA_DIR = os.getenv('DATA_DIR', './data')
DEFAULT_DB_PATH = os.path.join(DATA_DIR, 'financial_data.duckdb')

class MarketDataFetcher:
    """Class to fetch market data from TradeStation and update the database."""
    
    def __init__(self, config_path=None):
        """Initialize the market data fetcher."""
        self.config = self._load_config(config_path) if config_path else {}
        self.logger = logging.getLogger(__name__)
        self.start_date = self.config.get('settings', {}).get('default_start_date', '2023-01-01')
        
        # Connect to database
        db_path = self.config.get('settings', {}).get('database_path', './data/financial_data.duckdb')
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        
        self.config_path = config_path
        self.ts_agent = TradeStationMarketDataAgent(database_path=db_path)
        
        # Add VALID_UNITS attribute to the ts_agent if it doesn't exist
        if not hasattr(self.ts_agent, 'VALID_UNITS'):
            self.ts_agent.VALID_UNITS = ['daily', 'minute', 'weekly', 'monthly']
            
    def _connect_database(self):
        """Connect to the DuckDB database."""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.config.get('settings', {}).get('database_path', DEFAULT_DB_PATH))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to the database
            conn = duckdb.connect(self.config.get('settings', {}).get('database_path', DEFAULT_DB_PATH))
            logger.info(f"Connected to database: {self.config.get('settings', {}).get('database_path', DEFAULT_DB_PATH)}")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
        
    def _load_config(self, config_path):
        """Load the configuration from the YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def get_latest_date_in_db(self, symbol):
        """
        Get the latest date for a symbol in the database.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            The latest date as a datetime object, or None if no data exists
        """
        try:
            query = f"""
            SELECT MAX(timestamp) as latest_date
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchone()
            
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting latest date for {symbol}: {e}")
            return None
    
    def fetch_data_since(self, symbol, interval, unit, start_date):
        """
        Fetch data since a specific timestamp.
        
        Args:
            symbol: The ticker symbol
            interval: Interval of data
            unit: Time unit ('Minute', 'Daily', etc.)
            start_date: The timestamp to start fetching data from
            
        Returns:
            List of bars
        """
        if unit not in self.ts_agent.VALID_UNITS:
            raise ValueError(f"Invalid unit '{unit}'. Valid units are: {', '.join(self.ts_agent.VALID_UNITS)}")

        all_bars = []
        last_date = start_date or datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

        logger.info(f"Fetching data for {symbol} from {last_date} onward...")
        bars_back = 50000  # Increased from 10000 to 50000 for efficiency
        min_chunk_size = 1000  # Increased minimum chunk size
        
        while bars_back > 0:
            # Calculate chunk size - start with 10000, but don't go below min_chunk_size
            chunk_size = max(min(bars_back, 10000), min_chunk_size)
            chunk = self.fetch_data(symbol, interval, unit, chunk_size, last_date)
            if not chunk:
                break

            all_bars.extend(chunk)
            logger.info(f"Retrieved {len(chunk)} bars. Remaining: {bars_back - len(chunk)}.")

            # Update last_date to the last retrieved bar
            last_date = chunk[-1]['TimeStamp']

            # Reduce bars_back by the number of bars fetched
            bars_back -= len(chunk)

            # Add a safeguard: If the API returns only 1 bar repeatedly, stop fetching
            if len(chunk) == 1:
                logger.warning("Only 1 bar returned. Stopping further requests to avoid infinite loop.")
                break

        return all_bars

    def fetch_data(self, symbol, interval, unit, bars_back=10000, last_date=None):
        """
        Fetch historical market data from TradeStation with input validation.
        
        Args:
            symbol: The ticker symbol
            interval: Interval of data
            unit: Time unit ('Minute', 'Daily', etc.)
            bars_back: Number of bars to fetch
            last_date: The last date to fetch data from
            
        Returns:
            List of bars
        """
        if not self.ts_agent.access_token:
            raise Exception("Error: No active connection. Please call 'connect' first.")

        # Validate 'unit' input
        if unit not in self.ts_agent.VALID_UNITS:
            logger.error(f"Error: Invalid unit '{unit}'. Valid units are: {', '.join(self.ts_agent.VALID_UNITS)}")
            return None

        all_bars = []
        last_date = last_date or datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

        logger.info(f"Fetching {bars_back} bars of {unit} data for {symbol}...")
        
        # Use a larger chunk size for initial requests
        chunk_size = min(bars_back, 50000)  # Maximum chunk size of 50000
        
        url = f"{self.ts_agent.base_url}/marketdata/barcharts/{symbol}?interval={interval}&unit={unit}&barsback={chunk_size}&lastdate={last_date}"
        headers = {'Authorization': f'Bearer {self.ts_agent.access_token}'}

        for attempt in range(self.ts_agent.max_retries):
            try:
                response = requests.get(url, headers=headers)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                json_data = response.json()

                if 'Bars' in json_data:
                    chunk = json_data['Bars']
                    if not chunk:
                        logger.info("No more data returned.")
                        return all_bars

                    all_bars.extend(chunk)
                    logger.info(f"Retrieved {len(chunk)} bars. Remaining: {bars_back - len(chunk)}.")
                    
                    # Add a small delay between requests to avoid rate limiting
                    time.sleep(0.5)
                    return all_bars  # Return immediately after successful fetch
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.ts_agent.retry_delay * (attempt + 1))  # Exponential backoff
        else:
            logger.error("Max retries reached. Ending fetch process.")
            return all_bars  # Return whatever we have so far
    
    def save_to_db(self, df: pd.DataFrame) -> None:
        """Save market data to the database."""
        if df is None or df.empty:
            self.logger.warning("No data to save to database")
            return

        try:
            # Ensure all required columns are present with correct types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symbol'] = df['symbol'].astype(str)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(int)
            df['up_volume'] = df.get('up_volume', 0).astype(int)
            df['down_volume'] = df.get('down_volume', 0).astype(int)
            df['source'] = 'tradestation'  # Set the data source
            df['interval_value'] = df['interval_value'].astype(int)
            df['interval_unit'] = df['interval_unit'].astype(str)
            df['adjusted'] = df.get('adjusted', False).astype(bool)
            df['quality'] = df.get('quality', 100).astype(int)

            # Process in chunks of 1000 rows
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                # Create a temporary table for this chunk
                self.conn.execute("DROP TABLE IF EXISTS temp_market_data")
                self.conn.execute("""
                    CREATE TEMP TABLE temp_market_data (
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
                        quality INTEGER
                    )
                """)
                
                # Insert data into temporary table with explicit column selection
                self.conn.execute("""
                    INSERT INTO temp_market_data (
                        timestamp, symbol, open, high, low, close, volume, 
                        up_volume, down_volume, source, interval_value, interval_unit,
                        adjusted, quality
                    )
                    SELECT 
                        timestamp, symbol, open, high, low, close, volume, 
                        up_volume, down_volume, source, interval_value, interval_unit,
                        adjusted, quality
                    FROM chunk
                """)
                
                # Insert from temporary table to main table
                self.conn.execute("""
                    INSERT INTO market_data (
                        timestamp, symbol, open, high, low, close, volume, 
                        up_volume, down_volume, source, interval_value, interval_unit,
                        adjusted, quality
                    )
                    SELECT 
                        timestamp, symbol, open, high, low, close, volume, 
                        up_volume, down_volume, source, interval_value, interval_unit,
                        adjusted, quality
                    FROM temp_market_data
                    ON CONFLICT (timestamp, symbol, interval_value, interval_unit) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        up_volume = EXCLUDED.up_volume,
                        down_volume = EXCLUDED.down_volume,
                        source = EXCLUDED.source,
                        adjusted = EXCLUDED.adjusted,
                        quality = EXCLUDED.quality
                """)
                
                self.logger.info(f"Inserted chunk {i//chunk_size + 1} ({len(chunk)} rows)")

            self.conn.commit()
            self.logger.info(f"Successfully saved {len(df)} rows to database")

        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")
            self.conn.rollback()
            raise
    
    def process_symbol(self, symbol: str, update_frequency: str = 'daily') -> None:
        """Process a single symbol."""
        logger.info(f"Processing {symbol}")
        
        # Set interval based on update frequency
        if update_frequency == 'daily':
            interval_value = 1
            unit = 'daily'
        elif update_frequency == 'hourly':
            interval_value = 60
            unit = 'minute'
        else:  # minute
            interval_value = 1
            unit = 'minute'

        # Get existing data for this symbol
        try:
            result = self.conn.execute(
                """
                SELECT MAX(timestamp) as last_timestamp
                FROM market_data 
                WHERE symbol = ? AND interval_value = ? AND interval_unit = ?
                """,
                (symbol, interval_value, unit)
            ).fetchone()
            
            last_timestamp = result[0] if result and result[0] else None
            
            if last_timestamp:
                logger.info(f"Found existing data for {symbol}, last timestamp: {last_timestamp}")
                data = self.fetch_data_since(symbol, last_timestamp, interval_value, unit)
            else:
                logger.info(f"No existing data for {symbol}, fetching from {self.start_date}")
                data = self.fetch_data(symbol, interval_value, unit)
                
            # Convert to DataFrame and prepare for saving
            if data:
                df = pd.DataFrame(data)
                
                # Rename columns to match schema
                column_mapping = {
                    'TimeStamp': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'TotalVolume': 'volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Add required columns
                df['symbol'] = symbol
                df['interval_value'] = interval_value
                df['interval_unit'] = unit
                df['up_volume'] = 0  # TradeStation doesn't provide this
                df['down_volume'] = 0  # TradeStation doesn't provide this
                df['adjusted'] = False
                df['quality'] = 100
                
                # Save to database
                self.save_to_db(df)
            else:
                logger.warning(f"No data fetched for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            raise
    
    def find_symbol_info(self, symbol):
        """
        Find symbol information in the config.
        
        Args:
            symbol: The symbol to find
            
        Returns:
            Tuple of (symbol_info, symbol_type) or (None, None) if not found
        """
        # Check futures
        for symbol_info in self.config.get('futures', []):
            if symbol_info['symbol'] == symbol:
                return symbol_info, 'futures'
        
        # Check equities
        for symbol_info in self.config.get('equities', []):
            if symbol_info['symbol'] == symbol:
                return symbol_info, 'equities'
        
        return None, None
    
    def run(self, symbol=None, start_date=None):
        """
        Run the data fetcher for all symbols in the config or a specific symbol.
        
        Args:
            symbol: Optional symbol to process (if None, process all symbols)
            start_date: Optional start date to override the config
        """
        try:
            # Connect to TradeStation
            if not self.ts_agent.authenticate():
                logger.error("Failed to authenticate with TradeStation API")
                return
            
            if symbol:
                # Process a single symbol
                symbol_info, symbol_type = self.find_symbol_info(symbol)
                if symbol_info:
                    self.process_symbol(symbol, symbol_info.get('update_frequency', self.config['settings']['default_update_frequency']))
                else:
                    logger.error(f"Symbol '{symbol}' not found in configuration")
            else:
                # Process all symbols
                for symbol_info in self.config.get('futures', []):
                    self.process_symbol(symbol_info['symbol'], symbol_info.get('update_frequency', self.config['settings']['default_update_frequency']))
                
                for symbol_info in self.config.get('equities', []):
                    self.process_symbol(symbol_info['symbol'], symbol_info.get('update_frequency', self.config['settings']['default_update_frequency']))
                
        except Exception as e:
            logger.error(f"Error running data fetcher: {e}")
        finally:
            # Close the database connection
            self.conn.close()

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Fetch market data from TradeStation')
    parser.add_argument('--config', default='src/config/market_symbols.yaml', help='Path to the market_symbols.yaml file')
    parser.add_argument('--db', help='Path to the DuckDB database file (optional, defaults to DATA_DIR/financial_data.duckdb)')
    parser.add_argument('--symbol', help='Specific symbol to fetch data for (optional, if not provided, fetches all symbols)')
    parser.add_argument('--start-date', help='Start date for data fetching (YYYY-MM-DD format, overrides config)')
    args = parser.parse_args()
    
    fetcher = MarketDataFetcher(args.config)
    fetcher.run(args.symbol, args.start_date)

if __name__ == '__main__':
    main()
