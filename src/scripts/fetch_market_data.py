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
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from our project
from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
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
            
        # Create a requests session for reuse
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1  # seconds
            
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
    
    def fetch_data_since(self, symbol: str, interval: int = 1, unit: str = 'daily', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical market data since a given date.
        
        Args:
            symbol: The symbol to fetch data for
            interval: The interval value (e.g. 1 for 1-minute or 1-day)
            unit: The interval unit (daily, minute, weekly, monthly)
            start_date: The start date in YYYY-MM-DD format
            end_date: The end date in YYYY-MM-DD format
            
        Returns:
            DataFrame with the fetched data
        """
        if start_date:
            # Convert start_date to datetime
            start_dt = pd.to_datetime(start_date)
            # Ensure start_dt is timezone-naive
            if start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            # Calculate days between start_date and today
            days_diff = (pd.Timestamp.now() - start_dt).days
            # Set bars_back based on days_diff, with a minimum of 1000
            bars_back = max(days_diff, 1000)
        else:
            bars_back = 10000
            
        # Ensure bars_back doesn't exceed 50000 (API limit)
        bars_back = min(bars_back, 50000)
        
        # Initialize empty list to store all data
        all_data = []
        total_bars = 0
        last_bar_count = None
        
        while True:
            try:
                # Construct the API endpoint
                endpoint = f"{self.ts_agent.base_url}/marketdata/barcharts/{symbol}"
                
                # Prepare query parameters
                params = {
                    'interval': interval,
                    'unit': unit,
                    'barsback': bars_back
                }
                
                # Add lastdate parameter if end_date is provided
                if end_date:
                    params['lastdate'] = f"{end_date}T00:00:00Z"
                
                # Make the API request with retries
                data = self.make_request_with_retry(endpoint, params)
                
                if not data:
                    break
                    
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Break if we got less than requested bars (means we hit the start)
                if len(df) < bars_back:
                    all_data.append(df)
                    break
                    
                # Check if we're getting the same number of bars repeatedly
                if last_bar_count == len(df):
                    logger.warning(f"Got same number of bars ({len(df)}) twice in a row, stopping to avoid infinite loop")
                    break
                    
                last_bar_count = len(df)
                
                # Add to our collection
                all_data.append(df)
                total_bars += len(df)
                
                logger.info(f"Retrieved {len(df)} bars. Remaining: {bars_back - len(df)}.")
                
                # Get the earliest timestamp
                earliest_ts = pd.to_datetime(df['TimeStamp'].min())
                # Ensure earliest_ts is timezone-naive for comparison
                if earliest_ts.tz is not None:
                    earliest_ts = earliest_ts.tz_localize(None)
                
                # If we've gone back far enough, break
                if start_date and earliest_ts <= start_dt:
                    break
                    
                # Update lastdate for next request
                params['lastdate'] = earliest_ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
                
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by date range if start_date is provided
            if start_date:
                combined_df['TimeStamp'] = pd.to_datetime(combined_df['TimeStamp'])
                # Ensure timestamps are timezone-naive for comparison
                if combined_df['TimeStamp'].dt.tz is not None:
                    combined_df['TimeStamp'] = combined_df['TimeStamp'].dt.tz_localize(None)
                combined_df = combined_df[combined_df['TimeStamp'] >= start_dt]
                
            # Rename columns to match schema
            column_mapping = {
                'TimeStamp': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'TotalVolume': 'volume'
            }
            combined_df = combined_df.rename(columns=column_mapping)
            
            # Add required columns
            combined_df['symbol'] = symbol
            combined_df['interval_value'] = interval
            combined_df['interval_unit'] = unit
            combined_df['up_volume'] = 0  # TradeStation doesn't provide this
            combined_df['down_volume'] = 0  # TradeStation doesn't provide this
            combined_df['adjusted'] = False
            combined_df['quality'] = 100
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp')
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
            
            return combined_df
            
        return pd.DataFrame()

    def fetch_data(self, symbol, interval, unit, bars_back=10000, last_date=None):
        """
        Fetch historical market data from TradeStation with input validation.
        
        Args:
            symbol: The ticker symbol
            interval: Interval of data
            unit: Time unit ('daily', 'minute', etc.)
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
    
    def make_request_with_retry(self, endpoint: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Make an API request with retries and exponential backoff.
        
        Args:
            endpoint: The API endpoint URL
            params: Query parameters for the request
            
        Returns:
            List of bar data if successful, None otherwise
        """
        headers = {'Authorization': f'Bearer {self.ts_agent.access_token}'}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(endpoint, params=params, headers=headers)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if 'Bars' in data:
                    return data['Bars']
                else:
                    logger.warning("No 'Bars' found in response")
                    return None
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
                
        logger.error("Max retries reached. Request failed.")
        return None
    
    def get_existing_data(self, symbol: str) -> pd.DataFrame:
        """
        Get existing data for a symbol from the database.
        
        Args:
            symbol: The symbol to get data for
            
        Returns:
            DataFrame with the existing data
        """
        try:
            query = f"""
            SELECT *
            FROM market_data
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            df = self.conn.execute(query).fetchdf()
            return df
        except Exception as e:
            logger.error(f"Error getting existing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def process_symbol(self, symbol: str) -> None:
        """Process a single symbol, retrieving existing data and fetching new data if none exists."""
        logger.info(f"Processing {symbol}")
        
        # Get the root symbol (e.g. 'ES' from 'ESH24')
        root_symbol = symbol[:2] if len(symbol) > 2 else symbol
        
        # Get the config for this symbol
        symbol_config = None
        for futures_config in self.config.get('futures', []):
            if futures_config['symbol'] == root_symbol:
                symbol_config = futures_config
                break
                
        if not symbol_config:
            for equity_config in self.config.get('equities', []):
                if equity_config['symbol'] == root_symbol:
                    symbol_config = equity_config
                    break
        
        if not symbol_config:
            logger.error(f"No configuration found for symbol {root_symbol}")
            return
        
        # Get the start date from config, defaulting to 2023-01-01
        start_date = symbol_config.get('start_date', '2023-01-01')
        
        # For futures contracts, determine the expiration date
        if len(symbol) > 2:
            month_code = symbol[2]
            year = int('20' + symbol[3:5]) if len(symbol) >= 5 else None
            
            if year:
                # Map month codes to their respective months
                month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
                month = month_map.get(month_code)
                
                if month:
                    # Set end date to the last day of the expiration month
                    end_date = datetime(year, month, 1) + relativedelta(months=1, days=-1)
                    end_date = end_date.strftime('%Y-%m-%d')
                else:
                    end_date = None
            else:
                end_date = None
        else:
            end_date = None

        # Get existing data for this symbol
        existing_data = self.get_existing_data(symbol)
        
        if existing_data.empty:
            logger.info(f"No existing data for {symbol}, fetching from {start_date} to {end_date}")
            # For daily data
            daily_data = self.fetch_data_since(
                symbol=symbol,
                interval=1,
                unit='daily',
                start_date=start_date,
                end_date=end_date
            )
            if not daily_data.empty:
                self.save_to_db(daily_data)
        else:
            logger.info(f"Found existing data for {symbol}")
            
        return None
    
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
    
    def generate_futures_contracts(self, root_symbol, start_date, end_date=None):
        """
        Generate futures contract symbols for a given root symbol.
        
        Args:
            root_symbol: The root symbol (e.g., 'ES')
            start_date: Start date as datetime or string 'YYYY-MM-DD'
            end_date: End date as datetime or string 'YYYY-MM-DD' (defaults to today)
            
        Returns:
            List of contract symbols (e.g., ['ESH23', 'ESM23', 'ESU23', 'ESZ23'])
        """
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        if end_date is None:
            end_date = pd.Timestamp.now()
            
        # Ensure dates are timezone-naive for comparison
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
            
        # Contract months for ES: H (March), M (June), U (Sept), Z (Dec)
        month_codes = ['H', 'M', 'U', 'Z']
        month_numbers = [3, 6, 9, 12]  # Corresponding month numbers
        
        contracts = []
        current_date = start_date
        
        while current_date <= end_date:
            year = current_date.year
            # Generate contracts for current year and next year
            for _ in range(2):  # Current year and next year
                for month_code, month_num in zip(month_codes, month_numbers):
                    contract_date = pd.Timestamp(datetime(year, month_num, 1))
                    # Only include contracts that expire after start_date and before end_date
                    if contract_date >= start_date and contract_date <= end_date:
                        # Format year as two digits
                        year_str = str(year)[-2:]
                        contract = f"{root_symbol}{month_code}{year_str}"
                        if contract not in contracts:
                            contracts.append(contract)
                year += 1
            # Move to next year
            current_date = pd.Timestamp(datetime(current_date.year + 1, 1, 1))
            
        return sorted(contracts)

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
                    if symbol_type == 'futures':
                        # Generate and process futures contracts
                        start = start_date or symbol_info.get('start_date', self.start_date)
                        contracts = self.generate_futures_contracts(symbol, start)
                        logger.info(f"Generated {len(contracts)} futures contracts for {symbol}")
                        for contract in contracts:
                            try:
                                self.process_symbol(contract)
                            except Exception as e:
                                logger.error(f"Error processing futures contract {contract}: {e}")
                                continue
                    else:
                        self.process_symbol(symbol)
                else:
                    logger.error(f"Symbol '{symbol}' not found in configuration")
            else:
                # Process all symbols
                # Process futures first
                for symbol_info in self.config.get('futures', []):
                    try:
                        # Generate and process futures contracts
                        start = start_date or symbol_info.get('start_date', self.start_date)
                        contracts = self.generate_futures_contracts(symbol_info['symbol'], start)
                        logger.info(f"Generated {len(contracts)} futures contracts for {symbol_info['symbol']}")
                        for contract in contracts:
                            try:
                                self.process_symbol(contract)
                            except Exception as e:
                                logger.error(f"Error processing futures contract {contract}: {e}")
                                continue
                    except Exception as e:
                        logger.error(f"Error processing futures symbol {symbol_info['symbol']}: {e}")
                        continue
                
                # Then process equities
                for symbol_info in self.config.get('equities', []):
                    try:
                        self.process_symbol(symbol_info['symbol'])
                    except Exception as e:
                        logger.error(f"Error processing equity symbol {symbol_info['symbol']}: {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Error running data fetcher: {e}")
        finally:
            # Close the database connection
            self.conn.close()

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Fetch market data for symbols from TradeStation API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for a specific symbol
  python fetch_market_data.py --symbol ES
  
  # Fetch data for all symbols defined in the config
  python fetch_market_data.py
  
  # Use a custom configuration file
  python fetch_market_data.py --config path/to/custom_config.yaml
        """
    )
    parser.add_argument('--symbol', type=str, help='Symbol to fetch data for (e.g., ES, NQ, CL)')
    parser.add_argument('--config', default='config/market_symbols.yaml', help='Path to the market_symbols.yaml configuration file (default: config/market_symbols.yaml)')
    args = parser.parse_args()
    
    try:
        fetcher = MarketDataFetcher(config_path=args.config)
        if args.symbol:
            logger.info(f"Fetching data for symbol: {args.symbol}")
            fetcher.run(symbol=args.symbol)
        else:
            logger.info("Fetching data for all symbols")
            fetcher.run()
    except Exception as e:
        logger.error(f"Error running market data fetcher: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
