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
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import re
import pandas_market_calendars as mcal

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
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

# --- Calendar Helper --- # Added
_calendar_cache = {}

def get_trading_calendar(calendar_name='NYSE'): # Default to NYSE if not specified
    """Gets a pandas_market_calendars object for the given exchange name."""
    if calendar_name not in _calendar_cache:
        try:
            logger.info(f"Loading trading calendar: {calendar_name}")
            _calendar_cache[calendar_name] = mcal.get_calendar(calendar_name)
            logger.info(f"Calendar {calendar_name} loaded successfully.")
        except RuntimeError as e: # Catch RuntimeError for invalid names
             # Check if the error message indicates an unregistered class
             if "is not one of the registered classes" in str(e):
                 logger.error(f"Invalid calendar name: '{calendar_name}'. Falling back to NYSE.")
             else:
                 # Re-raise if it's a different RuntimeError
                 logger.error(f"RuntimeError loading calendar '{calendar_name}': {e}. Falling back to NYSE.")
                 # raise e # Optional: re-raise unexpected RuntimeErrors
             # Fallback logic
             if 'NYSE' not in _calendar_cache:
                 _calendar_cache['NYSE'] = mcal.get_calendar('NYSE')
             _calendar_cache[calendar_name] = _calendar_cache['NYSE'] # Use NYSE as fallback
        except Exception as e:
            logger.error(f"Unexpected error loading calendar '{calendar_name}': {e}. Falling back to NYSE.")
            if 'NYSE' not in _calendar_cache:
                 _calendar_cache['NYSE'] = mcal.get_calendar('NYSE')
            _calendar_cache[calendar_name] = _calendar_cache['NYSE'] # Use NYSE as fallback
    return _calendar_cache[calendar_name]
# -----------------------

class MarketDataFetcher:
    """Class to fetch market data from TradeStation and update the database."""
    
    def __init__(self, config_path=None, start_date=None, db_path=None, existing_conn=None):
        """Initialize the market data fetcher.
        
        Args:
            config_path: Path to config file (optional)
            start_date: Start date for historical data (optional)
            db_path: Path to the database file (optional, used if existing_conn is None)
            existing_conn: An existing DuckDB connection object (optional)
        """
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Set start date
        self.start_date = pd.Timestamp(start_date) if start_date else pd.Timestamp('2010-01-01')
        
        # Set database path (needed even if connection is passed, for other logic potentially)
        self.db_path = db_path or self.config.get('settings', {}).get('database_path', DEFAULT_DB_PATH)
        
        # Initialize database connection
        if existing_conn:
            self.conn = existing_conn
            logger.info("Using existing database connection for MarketDataFetcher.")
        else:
            logger.info("No existing connection passed, MarketDataFetcher creating its own.")
            self.conn = self._connect_database()
            
        # Setup schema (safe to run even on existing connection)
        self._setup_database()
        
        # Initialize TradeStation agent
        self.ts_agent = TradeStationMarketDataAgent(database_path=':memory:', verbose=True)
        
        # Add VALID_UNITS attribute to the ts_agent if it doesn't exist
        if not hasattr(self.ts_agent, 'VALID_UNITS'):
            self.ts_agent.VALID_UNITS = ['daily', 'minute', 'weekly', 'monthly']
            
        # Create a requests session for reuse
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1  # seconds
            
    def set_connection(self, new_conn):
        """Update the internal database connection object."""
        if new_conn:
            self.conn = new_conn
            logger.info("MarketDataFetcher database connection updated.")
        else:
            logger.warning("Attempted to set an invalid connection in MarketDataFetcher.")
            
    def _connect_database(self):
        """Connect to the DuckDB database."""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to the database
            conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
        
    def _setup_database(self):
        """Set up the database schema if it doesn't exist."""
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
                adjusted BOOLEAN,
                quality INTEGER,
                PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
            )
            """)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            sys.exit(1)
        
    def _load_config(self, config_path):
        """Load the configuration from the YAML file."""
        try:
            # If config_path is relative, make it relative to project root
            if not os.path.isabs(config_path):
                config_path = os.path.join(project_root, config_path)
            
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
    
    def fetch_data_since(self, symbol, interval, unit, start_date=None, end_date=None):
        """
        Fetch historical market data from TradeStation since a specific date.
        
        Args:
            symbol: The ticker symbol
            interval: Interval of data
            unit: Time unit ('daily', 'minute', etc.)
            start_date: The start date to fetch data from (format: 'YYYY-MM-DD')
            end_date: The end date to fetch data to (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with market data
        """
        try:
            if not self.ts_agent.access_token:
                raise Exception("Error: No active connection. Please call 'connect' first.")

            logger.debug(f"Fetching data for {symbol} with interval={interval} {unit}")
            logger.debug(f"Date range: {start_date} to {end_date}")

            # Validate 'unit' input
            if unit not in self.ts_agent.VALID_UNITS:
                logger.error(f"Error: Invalid unit '{unit}'. Valid units are: {", ".join(self.ts_agent.VALID_UNITS)}")
                return None

            # Convert start_date to datetime if provided
            start_dt = None
            if start_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    logger.debug(f"Converted start_date to datetime: {start_dt}")
                except Exception as e:
                    logger.error(f"Error parsing start_date: {e}")
                    return None

            # Convert end_date to datetime if provided
            end_dt = None
            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date)
                    logger.debug(f"Converted end_date to datetime: {end_dt}")
                except Exception as e:
                    logger.error(f"Error parsing end_date: {e}")
                    return None

            # Calculate days difference if start_date is provided
            if start_dt:
                days_diff = (datetime.now() - start_dt).days
                logger.debug(f"Days difference: {days_diff}")
                # Use a reasonable number of bars based on the interval
                if unit == 'daily':
                    bars_back = max(days_diff, 1000)
                elif unit == 'minute':
                    if interval == 1:
                        bars_back = max(days_diff * 390, 50000)  # ~390 minutes per trading day
                    elif interval == 15:
                        bars_back = max(days_diff * 26, 50000)   # ~26 15-minute bars per trading day
                    else:
                        bars_back = max(days_diff * 100, 50000)  # Conservative estimate
                else:
                    bars_back = max(days_diff, 1000)
            else:
                bars_back = 50000
                
            # Ensure bars_back doesn't exceed 50000 (API limit)
            bars_back = min(bars_back, 50000)
            logger.debug(f"Requesting {bars_back} bars")
            
            # Initialize empty list to store all data
            all_data = []
            total_bars = 0
            # Start pagination from the end_date (or now if end_date is None)
            last_date_param = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
            logger.debug(f"Initial last_date for pagination: {last_date_param}")
            
            while True:
                try:
                    # Determine the number of bars for this attempt
                    current_bars_back = bars_back # Start with the originally calculated bars_back
                    
                    # Construct the API endpoint
                    endpoint = f"{self.ts_agent.base_url}/marketdata/barcharts/{symbol}"
                    
                    # Prepare query parameters
                    params = {
                        'interval': interval,
                        'unit': unit,
                        'barsback': current_bars_back, # Use current attempt size
                        'lastdate': last_date_param.strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                    
                    logger.debug(f"Requesting {current_bars_back} bars ending {params['lastdate']} for {symbol}")
                    
                    # Make the API request with internal retries
                    bars_data = self.make_request_with_retry(endpoint, params)
                    
                    # --- Check for failure and retry with smaller chunk --- 
                    if bars_data is None:
                        # If the first attempt (with original bars_back, assumed 50k or less) failed
                        if current_bars_back > 25000: # Only retry if we used a large chunk size
                            logger.warning(f"Initial request with {current_bars_back} bars failed. Retrying with 25000 bars.")
                            current_bars_back = 25000
                            params['barsback'] = current_bars_back # Update params for retry
                            
                            # Retry the request with the smaller chunk size
                            bars_data = self.make_request_with_retry(endpoint, params)
                            
                            if bars_data is None:
                                logger.error(f"Retry with {current_bars_back} bars also failed for {symbol} ending {params['lastdate']}. Stopping pagination.")
                                break # Both attempts failed, exit the while loop
                            # If retry succeeded, bars_data is now populated, loop continues below
                        else:
                             # Initial request was already small or the retry failed
                            logger.error(f"Request with {current_bars_back} bars failed for {symbol} ending {params['lastdate']} after retries. Stopping pagination.")
                            break # Exit the while loop
                    # ---------------------------------------------------
                        
                    # --- If we are here, bars_data is NOT None (either from initial or retry) --- 
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(bars_data)
                    
                    # Break if we got an empty DataFrame (means end of data)
                    if df.empty:
                        logger.info(f"Received empty DataFrame for {symbol} ending {params['lastdate']}. Assuming end of data.")
                        break # Exit the while loop
                    
                    # Add to our collection
                    all_data.append(df)
                    total_bars += len(df)
                    
                    logger.info(f"Retrieved {len(df)} bars for {symbol}. Total so far: {total_bars}")
                    
                    # Get the earliest timestamp from the current chunk
                    earliest_ts = pd.to_datetime(df['TimeStamp'].min())
                    if earliest_ts.tz is not None:
                        earliest_ts = earliest_ts.tz_localize(None) # Ensure naive
                    
                    logger.debug(f"Earliest timestamp in batch: {earliest_ts}")
                    
                    # If we've gone back far enough, break
                    if start_dt and earliest_ts <= start_dt:
                        logger.info(f"Reached or exceeded start date {start_dt}. Stopping pagination.")
                        break # Exit the while loop
                    
                    # Update last_date_param for the next request: use the earliest timestamp minus 1 sec
                    last_date_param = earliest_ts - timedelta(seconds=1)
                    logger.debug(f"New last_date for next request: {last_date_param}")
                    
                    # IMPORTANT: Use the *actual* number of bars requested in this iteration for the comparison
                    if len(df) < current_bars_back: 
                        logger.info(f"Got fewer bars ({len(df)}) than requested ({current_bars_back}). Assuming start of data history reached for {symbol}.")
                        break # Exit the while loop
                        
                except Exception as e:
                    logger.error(f"Unexpected error during pagination loop for {symbol}: {e}", exc_info=True)
                    break # Stop fetching for this symbol if an unexpected error occurs in the loop
            
            # Define the expected final columns REGARDLESS of whether data was fetched
            final_columns = [
                'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
                'adjusted', 'quality'
            ]
            
            # Combine all data if any was collected
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.debug(f"Combined {len(all_data)} batches into DataFrame with {len(combined_df)} rows for {symbol}")
                
                # Convert TimeStamp to datetime and ensure timezone-naive
                combined_df['TimeStamp'] = pd.to_datetime(combined_df['TimeStamp'])
                if combined_df['TimeStamp'].dt.tz is not None:
                    combined_df['TimeStamp'] = combined_df['TimeStamp'].dt.tz_localize(None)
                    
                # Sort by timestamp before dropping duplicates
                combined_df = combined_df.sort_values('TimeStamp')
                
                # Remove duplicates, keeping the first occurrence (earliest)
                combined_df = combined_df.drop_duplicates(subset=['TimeStamp'], keep='first')
                logger.debug(f"Dropped duplicates, {len(combined_df)} rows remain for {symbol}")
                
                # Filter by date range if start_date is provided
                if start_dt:
                    initial_len = len(combined_df)
                    combined_df = combined_df[combined_df['TimeStamp'] > start_dt]
                    logger.debug(f"Filtered {initial_len} to {len(combined_df)} rows to get data strictly after {start_dt} for {symbol}")
                    
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
                combined_df['source'] = 'tradestation'
                
                # Select only necessary columns in the correct order
                combined_df = combined_df[final_columns]
                logger.debug(f"Final DataFrame for {symbol} has {len(combined_df)} rows")
                return combined_df
            else:
                # If no data was collected (e.g., all API calls failed), return an empty DataFrame with correct columns
                logger.warning(f"No data collected for {symbol} in date range {start_date} to {end_date}")
                return pd.DataFrame(columns=final_columns)
                
        except Exception as e:
            logger.error(f"Error in fetch_data_since for {symbol}: {e}", exc_info=True)
            # Define final_columns here as well for the exception case return
            final_columns = [
                'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
                'adjusted', 'quality'
            ]
            return pd.DataFrame(columns=final_columns) # Return empty DF on error

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
            logger.error(f"Error: Invalid unit '{unit}'. Valid units are: {", ".join(self.ts_agent.VALID_UNITS)}")
            return None

        all_bars = []
        last_date_str = last_date or datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        if isinstance(last_date, datetime):
             last_date_str = last_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        logger.info(f"Fetching {bars_back} bars of {unit} data for {symbol} ending {last_date_str}...")
        
        # Use a larger chunk size for initial requests
        chunk_size = min(bars_back, 50000)  # Maximum chunk size of 50000
        
        url = f"{self.ts_agent.base_url}/marketdata/barcharts/{symbol}?interval={interval}&unit={unit}&barsback={chunk_size}&lastdate={last_date_str}"
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
                
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                json_data = response.json()

                if 'Bars' in json_data:
                    chunk = json_data['Bars']
                    if not chunk:
                        logger.info(f"No more data returned for {symbol} ending {last_date_str}.")
                        return all_bars # Return what we have if empty chunk received

                    all_bars.extend(chunk)
                    logger.info(f"Retrieved {len(chunk)} bars for {symbol}.")
                    
                    # Add a small delay between requests to avoid rate limiting
                    time.sleep(0.5)
                    return all_bars  # Return immediately after successful fetch
                else:
                    logger.warning(f"'Bars' key not found in response for {symbol}. Response: {json_data}")
                    return all_bars # Return empty or partial list
            except requests.exceptions.RequestException as e:
                 logger.error(f"HTTP Request failed for {symbol}: {e}")
                 time.sleep(self.ts_agent.retry_delay * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(self.ts_agent.retry_delay * (2 ** attempt))  # Exponential backoff
        else:
            logger.error(f"Max retries reached for {symbol}. Ending fetch process.")
            return all_bars  # Return whatever we have so far

    def save_to_db(self, df: pd.DataFrame) -> None:
        """Save market data to the database using DuckDB's direct DataFrame insertion.
           Handles potential conflicts by updating existing rows.
        """
        if df is None or df.empty:
            self.logger.warning("No data to save to database")
            return

        try:
            # Ensure all required columns are present with correct types
            # Convert timestamp to datetime64[us] which DuckDB handles well
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('datetime64[us]')
            df['symbol'] = df['symbol'].astype(str)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64') # Handle potential NaN
            df['up_volume'] = df.get('up_volume', 0).fillna(0).astype('int64')
            df['down_volume'] = df.get('down_volume', 0).fillna(0).astype('int64')
            df['source'] = df['source'].astype(str)
            df['interval_value'] = df['interval_value'].astype(int)
            df['interval_unit'] = df['interval_unit'].astype(str)
            df['adjusted'] = df.get('adjusted', False).astype(bool)
            df['quality'] = df.get('quality', 100).astype(int)

            # Select columns in the exact order of the table to be safe
            df_to_insert = df[[
                 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                 'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
                 'adjusted', 'quality'
            ]]

            # Use DuckDB's efficient UPSERT capability via temp view registration
            table_name = "market_data"
            temp_view_name = f"temp_{table_name}_view"
            
            self.conn.register(temp_view_name, df_to_insert)
            
            sql = f"""
            INSERT INTO {table_name} (
                 timestamp, symbol, open, high, low, close, volume,
                 up_volume, down_volume, source, interval_value, interval_unit,
                 adjusted, quality 
            )
            SELECT * FROM {temp_view_name}
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
            """
            
            self.conn.execute(sql)
            self.conn.unregister(temp_view_name) # Clean up the view
            self.conn.commit()
            self.logger.info(f"Successfully upserted {len(df_to_insert)} rows into {table_name}")

        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")
            self.conn.rollback()
            import traceback
            traceback.print_exc()
            # raise # Optional: re-raise the exception if needed
    
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
                
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                
                if 'Bars' in data:
                    return data['Bars']
                else:
                    # Log specific error if provided by TradeStation
                    if "error" in data:
                        logger.warning(f"API returned error for {params.get('symbol')}: {data['error']}")
                    else:
                         logger.warning(f"No 'Bars' found in response for {params.get('symbol')}: {data}")
                    return None # Return None if no bars or error
                    
            except requests.exceptions.RequestException as e:
                 logger.error(f"HTTP Request failed: {e}")
                 if attempt < self.max_retries - 1:
                     time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                 continue # Retry
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                continue # Retry
                
        logger.error(f"Max retries reached for {endpoint} with params {params}. Request failed.")
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
    
    def calculate_expiration_date(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Calculate the expiration date for a futures contract using pandas_market_calendars.
        
        Args:
            symbol: The futures contract symbol (e.g., 'ESH24')
            
        Returns:
            pd.Timestamp: The expiration date, or None if invalid symbol/config.
        """
        if len(symbol) < 4:
            logger.warning(f"Invalid futures symbol format: {symbol}")
            return None
        
        # Extract components (handle 1 or 2 digit year)
        match = re.match(r"([A-Z]{1,2})([FGHJKMNQUVXZ])([0-9]{1,2})$", symbol)
        if not match:
             logger.warning(f"Could not parse futures symbol: {symbol}")
             return None
        base_symbol, month_code, year_code = match.groups()
        
        logger.debug(f"Calculating expiration for {symbol} (base: {base_symbol}, month: {month_code}, year: {year_code})")
        
        # Get base symbol config
        symbol_config = self._get_symbol_config(symbol) # Use the existing logic to get base config
        if not symbol_config:
             logger.warning(f"No configuration found for base symbol derived from {symbol}")
             return None

        # Get calendar
        calendar_name = symbol_config.get('calendar', 'NYSE') # Default to NYSE
        calendar = get_trading_calendar(calendar_name)

        # Convert year code to full year (handle century correctly)
        year_int = int(year_code)
        current_year = datetime.now().year
        current_century = (current_year // 100) * 100
        if len(year_code) == 1: # Assume 202X for single digit year
             year = current_century + year_int if year_int <= (current_year % 100 + 10) else current_century - 100 + year_int
        else: # 2-digit year
             year = (current_century - 100 + year_int) if year_int > (current_year % 100 + 10) else (current_century + year_int)

        # --- Reuse logic from calculate_volume_roll_dates.py's get_expiry_date --- 
        rule = symbol_config.get('expiry_rule', {})
        month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                     'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
        contract_month = month_map.get(month_code)
        if not contract_month:
            logger.error(f"Invalid contract month code: {month_code} in {symbol}")
            return None

        month_start = pd.Timestamp(f'{year}-{contract_month:02d}-01')
        search_start = month_start - timedelta(days=5)
        search_end = month_start + timedelta(days=40)
        valid_days = calendar.valid_days(start_date=search_start.strftime('%Y-%m-%d'),
                                         end_date=search_end.strftime('%Y-%m-%d'))

        # Rule: Nth specific weekday
        if rule.get('day_type') in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] and rule.get('day_number'):
            day_name_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}
            target_weekday = day_name_map[rule['day_type']]
            occurrence = int(rule['day_number'])
            month_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == year)]
            target_days = month_days[month_days.weekday == target_weekday]
            if len(target_days) >= occurrence:
                expiry = target_days[occurrence - 1]
                logger.debug(f"Calculated expiration date for {symbol} (Nth Weekday): {expiry.date()}")
                return expiry
            else:
                logger.warning(f"Could not find {occurrence} {rule['day_type']}s in {year}-{contract_month} for {symbol}. Rule: {rule}")
                return None # Indicate failure

        # Rule: N business days before reference day
        elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_day'):
            days_before = int(rule['days_before'])
            reference_day_num = int(rule['reference_day'])
            try:
                 reference_date = pd.Timestamp(f'{year}-{contract_month:02d}-{reference_day_num:02d}').normalize()
            except ValueError: # Handle invalid day like Feb 30
                 logger.warning(f"Invalid reference day {reference_day_num} for {year}-{contract_month}. Using last day of month.")
                 reference_date = pd.Timestamp(f'{year}-{contract_month:02d}-01') + pd.offsets.MonthEnd(0)

            days_strictly_before_ref = valid_days[valid_days < reference_date]
            if len(days_strictly_before_ref) >= days_before:
                 expiry = days_strictly_before_ref[-days_before]
                 logger.debug(f"Calculated expiration date for {symbol} (Days Before Ref): {expiry.date()}")
                 return expiry
            else:
                 logger.warning(f"Not enough trading days found before {reference_date.date()} for {days_before} days rule for {symbol}. Rule: {rule}")
                 return None

        # Rule: N business days before last business day
        elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_point') == 'last_business_day':
            days_before = int(rule['days_before'])
            month_trading_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == year)]
            if len(month_trading_days) >= days_before + 1:
                expiry = month_trading_days[-(days_before + 1)]
                logger.debug(f"Calculated expiration date for {symbol} (Days Before Last): {expiry.date()}")
                return expiry
            else:
                logger.warning(f"Not enough trading days in {year}-{contract_month} for {days_before} days before last rule for {symbol}. Rule: {rule}")
                return None

        # Rule: Special VX expiry
        elif rule.get('special_rule') == 'VX_expiry':
            next_month = contract_month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_month_start = pd.Timestamp(f'{next_year}-{next_month:02d}-01')
            search_end_spx = next_month_start + timedelta(days=35)
            valid_days_spx = calendar.valid_days(start_date=next_month_start.strftime('%Y-%m-%d'),
                                                 end_date=search_end_spx.strftime('%Y-%m-%d'))
            next_month_days = valid_days_spx[(valid_days_spx.month == next_month) & (valid_days_spx.year == next_year)]
            fridays_next_month = next_month_days[next_month_days.weekday == 4]
            if len(fridays_next_month) >= 3:
                spx_expiry_friday = fridays_next_month[2]
                target_date = spx_expiry_friday - timedelta(days=30)
                potential_expiry = target_date
                while potential_expiry.weekday() != 2: # Wednesday
                    potential_expiry -= timedelta(days=1)
                if calendar.is_session(potential_expiry.strftime('%Y-%m-%d')):
                     expiry = potential_expiry.normalize()
                     logger.debug(f"Calculated expiration date for {symbol} (VX Rule): {expiry.date()}")
                     return expiry
                else:
                     schedule = calendar.schedule(start_date=(potential_expiry - timedelta(days=5)).strftime('%Y-%m-%d'),
                                                end_date=potential_expiry.strftime('%Y-%m-%d'))
                     if not schedule.empty:
                         expiry = schedule.index[-1].normalize()
                         logger.debug(f"Calculated expiration date for {symbol} (VX Rule, adjusted): {expiry.date()}")
                         return expiry
                     else:
                          logger.warning(f"Could not find previous trading day for VX calculated expiry {potential_expiry.date()} for {symbol}.")
                          return None
            else:
                logger.warning(f"Could not find 3rd Friday in {next_year}-{next_month} for VX expiry calc for {symbol}. Rule: {rule}")
                return None

        # Fallback if no rule matched
        else:
            logger.warning(f"No specific expiry rule matched for {symbol}. Rule: {rule}. Returning None.")
            return None
        # --- End reuse --- 

    def process_symbol(self, symbol: str, update_history: bool = False, force: bool = False) -> None:
        """Process a single symbol, fetching and storing its data.
        
        Args:
            symbol: The symbol to process (can be equity, index, base future, or specific contract)
            update_history: If True, fetch from start_date in config to current date, skipping existing data
            force: If True, overwrite existing data in the database
        """
        try:
            logger.info(f"Processing {symbol}")
            
            # Find symbol configuration using the updated logic
            symbol_info = self._get_symbol_config(symbol)
            if not symbol_info:
                logger.error(f"No configuration found for symbol {symbol}")
                return
                
            # Determine the actual symbol to use for API calls and DB storage
            # If it was a specific contract, use that. Otherwise, use the config symbol.
            actual_symbol = symbol_info.get('specific_contract', symbol_info.get('symbol', symbol))
            logger.debug(f"Found configuration for {symbol}. Actual symbol for fetch/store: {actual_symbol}")
            
            # Get latest date from DB for the actual symbol
            latest_date = self.get_latest_date_in_db(actual_symbol)
            logger.debug(f"Latest date in DB for {actual_symbol}: {latest_date}")
            
            # Determine start date for fetching
            start_date = None
            config_start_date_str = symbol_info.get('start_date', self.start_date.strftime('%Y-%m-%d'))
            config_start_date = pd.Timestamp(config_start_date_str)

            # If it's a specific futures contract, calculate its typical trading start (~9 months before expiry)
            is_specific_contract = 'specific_contract' in symbol_info
            if is_specific_contract:
                expiry_date = self.calculate_expiration_date(actual_symbol) # Use actual symbol (e.g. ESH25)
                logger.debug(f"Calculated expiry date for {actual_symbol}: {expiry_date}")
                if expiry_date:
                    # Set a sensible start, e.g., ~9 months before expiry, but not before config start
                    contract_start_calc = expiry_date - timedelta(days=270) 
                    # Ensure both are naive before comparing
                    start_date = max(contract_start_calc.tz_localize(None), config_start_date)
                    logger.info(f"Calculated fetch start date for specific contract {actual_symbol}: {start_date.date()}")
                else:
                    logger.warning(f"Could not calculate expiry for {actual_symbol}, using config start date {config_start_date.date()}")
                    start_date = config_start_date
            else:
                # For base symbols or non-futures, use the config start date
                 start_date = config_start_date
            
            # Adjust start date based on mode and existing data
            if latest_date and not force and not update_history:
                # Normal mode: fetch only new data since last timestamp in DB
                fetch_start_date = latest_date # Use the actual last timestamp as the boundary
                logger.info(f"Found existing data for {actual_symbol}, last timestamp: {latest_date}. Fetching backward to include data since this time.")
            elif update_history:
                # Update history mode: fetch from the determined start_date to now
                fetch_start_date = start_date
                logger.info(f"Update history mode: fetching {actual_symbol} from {fetch_start_date.date()} to current date")
            elif force:
                 # Force mode: fetch from the determined start_date to now, will overwrite
                 fetch_start_date = start_date
                 logger.info(f"Force mode: overwriting {actual_symbol} data from {fetch_start_date.date()} to current date")
            else: # No existing data
                 fetch_start_date = start_date
                 logger.info(f"No existing data for {actual_symbol}, fetching from {fetch_start_date.date()} to current date")
            
            # Ensure fetch_start_date is not in the future
            if fetch_start_date > pd.Timestamp.now():
                 logger.info(f"Fetch start date {fetch_start_date} is in the future. Skipping {actual_symbol}.")
                 return
                 
            # Get frequencies from symbol_info, defaulting to daily if none specified
            frequencies = symbol_info.get('frequencies', ['daily'])
            if not frequencies:
                frequencies = ['daily']
                
            logger.debug(f"Processing frequencies for {actual_symbol}: {frequencies}")
            
            # Fetch data for each frequency
            for freq_name in frequencies:
                try:
                    # Map frequency name to interval and unit
                    interval = None
                    unit = None
                    if freq_name == 'daily':
                        interval = 1
                        unit = 'daily'
                    elif freq_name.endswith('min'):
                         try:
                             interval = int(freq_name[:-3])
                             unit = 'minute'
                         except ValueError:
                             logger.warning(f"Could not parse interval from frequency '{freq_name}' for {actual_symbol}, skipping")
                             continue
                    # Add other frequency mappings if needed (e.g., 'weekly', 'monthly')
                    else:
                        logger.warning(f"Unsupported frequency '{freq_name}' for {actual_symbol}, skipping")
                        continue
                        
                    logger.info(f"Fetching {freq_name} ({interval} {unit}) data for {actual_symbol} from {fetch_start_date}")
                    
                    # Use the actual symbol for fetching
                    data = self.fetch_data_since(actual_symbol, interval, unit, start_date=fetch_start_date)
                    
                    if data is not None and not data.empty:
                        logger.info(f"Retrieved {len(data)} rows of {freq_name} data for {actual_symbol}")
                        if force:
                            # In force mode, delete existing data for this symbol and frequency
                            logger.info(f"Force mode: Deleting existing {freq_name} data for {actual_symbol}")
                            self.delete_existing_data(actual_symbol, interval, unit)
                        self.save_to_db(data)
                    else:
                        logger.warning(f"No new {freq_name} data retrieved for {actual_symbol} from {fetch_start_date}")
                except Exception as e:
                    logger.error(f"Error processing {actual_symbol} for frequency {freq_name}: {str(e)}", exc_info=True)
                    continue # Continue to next frequency
        except Exception as e:
            logger.error(f"General error processing symbol {symbol}: {str(e)}", exc_info=True)
            # Decide whether to raise or just log and continue with other symbols
            # raise # Uncomment to stop execution on error for a symbol

    def delete_existing_data(self, symbol: str, interval: int, unit: str) -> None:
        """Delete existing data for a symbol and frequency from the database.
        
        Args:
            symbol: The symbol to delete data for
            interval: The interval value
            unit: The interval unit
        """
        try:
            query = f"""
            DELETE FROM market_data 
            WHERE symbol = ? 
            AND interval_value = ? 
            AND interval_unit = ?
            """
            # Use parameters to prevent SQL injection
            self.conn.execute(query, [symbol, interval, unit])
            deleted_count = self.conn.fetchone()[0] # DuckDB DELETE returns count
            logger.info(f"Deleted {deleted_count} existing rows for {symbol} with interval {interval} {unit}")
        except Exception as e:
            logger.error(f"Error deleting data for {symbol}: {str(e)}")
            
    def _get_symbol_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the configuration for a symbol.

        Handles base symbols (ES, NQ, SPY), indices ($VIX.X), specific
        futures contracts (ESH25, NQM25, VXF24), and continuous contracts (@BASE=ID<Letter>Suf)
        by looking up their base config.
        """
        # --- ADDED DEBUGGING ---
        self.logger.debug(f"_get_symbol_config called for: {symbol}")
        self.logger.debug(f"Current self.config object: {self.config}")
        # --- END ADDED DEBUGGING ---

        # Check indices first
        for index in self.config.get('indices', []):
            if index.get('symbol') == symbol:
                return index

        # Check equities
        for equity in self.config.get('equities', []):
            if equity.get('symbol') == symbol:
                return equity

        # Check for continuous contract pattern (@BASE=ID<Letter>Suf)
        # Example: @ES=101XN, @VX=101IN
        continuous_pattern = re.compile(r"^(@[A-Z]{1,3})=[0-9]+[A-Z][A-Z]+$")
        cont_match = continuous_pattern.match(symbol)
        if cont_match:
            base_symbol = cont_match.group(1) # Extract @BASE (e.g., @VX)
            logger.debug(f"Detected continuous contract pattern: {symbol}, Base: {base_symbol}")
            # Find config for the base symbol (e.g., find config for @VX)
            for future in self.config.get('futures', []):
                if future.get('base_symbol') == base_symbol:
                    config = future.copy()
                    config['specific_contract'] = symbol # Store original full symbol
                    logger.debug(f"Found config for continuous base {base_symbol}")
                    return config
            logger.warning(f"Found continuous pattern {symbol} but no config for base {base_symbol}")

        # Check if it looks like a standard futures contract (e.g., ESH25, NQM25)
        futures_pattern = re.compile(r"^[A-Z]{1,2}[FGHJKMNQUVXZ][0-9]{1,2}$")
        fut_match = futures_pattern.match(symbol)
        if fut_match:
            # Extract base symbol more carefully
            if len(symbol) >= 4 and symbol[-2:].isdigit() and symbol[-3].isalpha(): # Common case like ESH24
                base_symbol = symbol[:-3]
            elif len(symbol) >= 3 and symbol[-1:].isdigit() and symbol[-2].isalpha(): # Case like ESH4
                 base_symbol = symbol[:-2]
            else:
                 base_symbol = None # Couldn't reliably extract base
            
            if base_symbol:
                logger.debug(f"Detected standard futures pattern: {symbol}, Base: {base_symbol}")
                # Find the config for the base symbol
                futures_configs = self.config.get('futures', [])
                logger.debug(f"Available future base symbols in config: {[fc.get('base_symbol') for fc in futures_configs]}") # DEBUG LOG
                for future in futures_configs:
                    future_base = future.get('base_symbol') # Get the base from config
                    logger.debug(f"Checking config base: '{future_base}' against extracted base: '{base_symbol}'") # DEBUG LOG
                    if future_base == base_symbol: 
                        config = future.copy()
                        config['specific_contract'] = symbol # Store the specific contract
                        logger.debug(f"Found config for standard futures base {base_symbol}")
                        return config
                logger.warning(f"Found standard future {symbol} but no config for base {base_symbol}")

        # Fallback: Check if the provided symbol itself is a base future symbol
        for future in self.config.get('futures', []):
             if future.get('base_symbol') == symbol:
                 return future # Return the base config directly

        # If no match found
        logger.warning(f"No configuration match found for symbol: {symbol}")
        # --- ADDED DEBUGGING ---
        self.logger.debug(f"Config object when '{symbol}' not found: {self.config}")
        # --- END ADDED DEBUGGING ---
        return None
    
    def generate_futures_contracts(self, root_symbol, start_date, end_date=None):
        """
        Generate futures contract symbols for a given root symbol based on YAML configuration.
        
        Args:
            root_symbol: The root symbol (e.g., 'ES', 'VIX')
            start_date: Start date as datetime or string 'YYYY-MM-DD'
            end_date: End date as datetime or string 'YYYY-MM-DD' (defaults to today)
            
        Returns:
            List of contract symbols (e.g., ['ESH23', 'ESM23', 'ESU23', 'ESZ23'] for ES)
        """
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        elif isinstance(start_date, datetime):
            start_date = pd.Timestamp(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        elif isinstance(end_date, datetime):
            end_date = pd.Timestamp(end_date)
            
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        # Ensure dates are timezone-naive for comparison
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        # Get symbol config
        symbol_config = self._get_symbol_config(root_symbol)
        if not symbol_config:
            logger.error(f"No configuration found for futures symbol {root_symbol} in generate_futures_contracts")
            return []
        
        # Get contract configuration
        # Use defaults more safely
        hist_contracts_config = symbol_config.get('historical_contracts', {})
        month_patterns = hist_contracts_config.get('patterns', [])
        start_year_config = hist_contracts_config.get('start_year', datetime.now().year - 10) # Default to 10 years back
        num_active_contracts = symbol_config.get('num_active_contracts', 3) # Default to 3 active contracts
        # Assuming cycle_type from patterns (quarterly if H,M,U,Z, monthly otherwise)
        cycle_type = 'quarterly' if set(month_patterns) == {'H', 'M', 'U', 'Z'} else 'monthly' 
        
        if not month_patterns:
             logger.error(f"No month patterns defined for {root_symbol} in config.")
             return []
             
        # Map month codes to months
        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        month_map_inv = {v: k for k, v in month_map.items()}
        
        contracts = set() # Use set to avoid duplicates
        current_dt = pd.Timestamp.now().normalize()

        # Generate contracts from start_year up to end_date year + lookahead
        start_loop_year = max(start_year_config, start_date.year)
        end_loop_year = end_date.year + (num_active_contracts // len(month_patterns)) + 2 # Look ahead a bit

        for year in range(start_loop_year, end_loop_year):
            yr_code = str(year)[-2:]
            for month_code in month_patterns:
                contract_month = month_map.get(month_code)
                if not contract_month: continue
                
                # Estimate expiry date to check if contract is relevant
                expiry_estimate = self.calculate_expiration_date(f"{root_symbol}{month_code}{yr_code}")
                
                if expiry_estimate:
                    # Include contracts whose expiry is after our overall start_date
                    # and whose potential *start* (e.g., expiry - 9mo) is before our overall end_date
                    potential_start = expiry_estimate - timedelta(days=270) # Approx 9 months
                    if expiry_estimate >= start_date and potential_start <= end_date:
                         contracts.add(f"{root_symbol}{month_code}{yr_code}")
                else:
                    # If expiry calculation fails, maybe still include based on year/month range? Risky.
                    logger.warning(f"Could not estimate expiry for {root_symbol}{month_code}{yr_code}, including based on year/month.")
                    # Crude check: include if the contract month is within range
                    contract_date_est = pd.Timestamp(f'{year}-{contract_month:02d}-01')
                    if contract_date_est >= start_date and contract_date_est <= end_date + pd.DateOffset(months=num_active_contracts*2):
                        contracts.add(f"{root_symbol}{month_code}{yr_code}")

        logger.info(f"Generated {len(contracts)} potential contracts for {root_symbol} between {start_date.date()} and {end_date.date()}")
        return sorted(list(contracts))

    def run(self, symbol=None, update_history=False, force=False):
        """
        Run the data fetcher for all symbols in the config or a specific symbol.
        
        Args:
            symbol: Optional symbol to process (if None, process all symbols)
            update_history: If True, fetch from start_date in config to current date
            force: If True, overwrite existing data in the database
        """
        try:
            # Connect to TradeStation
            if not self.ts_agent.authenticate():
                logger.error("Failed to authenticate with TradeStation API")
                return
                
            if symbol:
                # Process the single specified symbol (could be base, index, equity, or specific contract)
                self.process_symbol(symbol, update_history, force)
            else:
                # Process all symbols from config
                logger.info("Processing all symbols defined in the configuration...")
                processed_symbols = set()
                
                # Process equities
                for equity in self.config.get('equities', []):
                    sym = equity['symbol']
                    if sym not in processed_symbols:
                        self.process_symbol(sym, update_history, force)
                        processed_symbols.add(sym)
                    
                # Process indices
                for index in self.config.get('indices', []):
                    sym = index['symbol']
                    if sym not in processed_symbols:
                        self.process_symbol(sym, update_history, force)
                        processed_symbols.add(sym)
                        
                # Process futures (by generating contracts for each base symbol)
                for future_config in self.config.get('futures', []):
                    base_symbol = future_config['base_symbol']
                    if base_symbol in processed_symbols:
                        continue # Skip if base symbol was processed individually
                        
                    logger.info(f"--- Processing base future: {base_symbol} ---")
                    # Determine start date for contract generation
                    gen_start_date_str = future_config.get('start_date', self.start_date.strftime('%Y-%m-%d'))
                    gen_start_date = pd.Timestamp(gen_start_date_str)
                    
                    # If updating history, use the config start date, otherwise use a shorter lookback for efficiency?
                    # For now, always generate based on config start date
                    contracts_to_process = self.generate_futures_contracts(
                        base_symbol,
                        gen_start_date,
                        datetime.now() # Generate up to today + lookahead
                    )
                    logger.info(f"Generated {len(contracts_to_process)} contracts for {base_symbol} to potentially process.")
                    
                    for contract in contracts_to_process:
                         if contract not in processed_symbols:
                            self.process_symbol(contract, update_history, force)
                            processed_symbols.add(contract)
                         else:
                            logger.debug(f"Skipping already processed contract: {contract}")
                            
                    processed_symbols.add(base_symbol) # Mark base as processed
                        
                logger.info("Finished processing all symbols.")
                
        except Exception as e:
            logger.error(f"Error running data fetcher: {e}", exc_info=True)
            # Decide whether to raise or let main handle
            raise 
        finally:
             # Ensure DB connection is closed
             if self.conn:
                 try:
                     self.conn.close()
                     logger.info("Database connection closed.")
                 except Exception as e:
                     logger.error(f"Error closing database connection in main: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fetch market data from TradeStation')
    parser.add_argument('--symbol', help='Symbol to fetch (e.g., ES or ESH20)')
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'config', 'market_symbols.yaml'),
                      help='Path to market symbols configuration file')
    parser.add_argument('--updatehistory', action='store_true', help='Update historical data')
    parser.add_argument('--force', action='store_true', help='Force update data')
    parser.add_argument('--interval-value', type=int, default=None, help='Interval value (e.g., 15 for 15-minute data)')
    parser.add_argument('--interval-unit', choices=['minute', 'daily', 'hour'], default=None, help='Interval unit')
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Set the logging level')
    parser.add_argument('--db-path', help='Path to database file', default=None)
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.loglevel))
    
    fetcher = None # Initialize fetcher to None
    try:
        # Initialize fetcher with config path and db path
        fetcher = MarketDataFetcher(config_path=args.config, db_path=args.db_path)
        
        # Run the fetcher
        fetcher.run(args.symbol, args.updatehistory, args.force)
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        # Exit with error status if needed
        # sys.exit(1) 

if __name__ == "__main__":
    main()
