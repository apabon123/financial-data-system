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
import io

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

# --- Expiry Calendar Helper --- #
# Placeholder definition - THIS WILL BE REMOVED
# class ExpiryCalendar: ... 
# --- End Expiry Calendar Helper --- #

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
        self.logger.setLevel(logging.INFO)
        
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
        self.ts_agent = TradeStationMarketDataAgent(database_path=':memory:', verbose=False)
        
        # Add VALID_UNITS attribute to the ts_agent if it doesn't exist
        if not hasattr(self.ts_agent, 'VALID_UNITS'):
            self.ts_agent.VALID_UNITS = ['daily', 'minute', 'weekly', 'monthly']
            
        # Create a requests session for reuse
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Store config path for potential reloads or use by other methods
        self._config_path = config_path 
            
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

    def _fetch_cboe_vx_daily(self, symbol: str, interval: int, unit: str, fetch_start_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch daily VX data directly from CBOE CSV for a specific contract."""
        self.logger.info(f"Fetching daily data for {symbol} directly from CBOE website.")
        
        final_columns = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'settle', 'open_interest',
            'source', 'interval_value', 'interval_unit',
            'adjusted', 'quality'
        ]
        
        # 1. Calculate Expiration/Settlement Date to build URL
        settlement_date = self.calculate_expiration_date(symbol)
        if not settlement_date:
            self.logger.error(f"Could not determine settlement date for {symbol} to build CBOE URL. Cannot fetch.")
            return pd.DataFrame(columns=final_columns)

        # 2. Construct CBOE URL based on settlement date year
        settlement_year = settlement_date.year
        settlement_date_str = settlement_date.strftime('%Y-%m-%d')
        
        if settlement_year < 2014:
            # Use archive URL format: https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_MYY_VX.csv
            self.logger.info(f"Settlement year {settlement_year} is before 2014. Using CBOE archive URL format.")
            # Extract month code and 2-digit year
            match = re.match(r"([A-Z]{1,2})([FGHJKMNQUVXZ])([0-9]{1,2})$", symbol)
            if not match:
                 self.logger.error(f"Could not parse month code/year from symbol {symbol} for archive URL.")
                 return pd.DataFrame(columns=final_columns)
            base_symbol, month_code, year_code = match.groups()
            yy = settlement_date.strftime('%y') # Get 2-digit year
            # Ensure yy matches the parsed year_code for consistency if needed?
            # For now, directly use calculated yy and parsed month_code
            
            cboe_url = f"https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{month_code}{yy}_VX.csv"
            # Note: Archive CSV format might differ, parsing below might need adjustment.
            # Define expected columns for archive files based on observed header
            archive_expected_cols = ['Trade Date','Futures','Open','High','Low','Close','Settle','Change','Total Volume','EFP','Open Interest']
            is_archive = True # Flag to use specific parsing args later
        else:
            # Use current/recent data URL format: https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_YYYY-MM-DD.csv
            logger.info(f"Settlement year {settlement_year} is 2014 or later. Using current CBOE data URL format.")
            # Extract base symbol (e.g., VX from VXU06) - Although usually just VX for these files
            base_symbol_match = re.match(r"([A-Z]{1,2})", symbol)
            base_symbol = base_symbol_match.group(1) if base_symbol_match else "VX" # Should generally be VX
            
            # Construct the URL using the base symbol and settlement date
            # Corrected path based on update_vx_futures.py logs
            cboe_url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/{base_symbol}/{base_symbol}_{settlement_date_str}.csv"
            archive_expected_cols = None # Not needed for modern files
            is_archive = False # Flag
            
        self.logger.info(f"Attempting download from CBOE URL: {cboe_url}")

        # 3. Download CSV data
        try:
            response = self.session.get(cboe_url, timeout=30) # Use the shared session
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            csv_content = response.text
            # --- REMOVED DEBUG --- #
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading CBOE data for {symbol} from {cboe_url}: {e}")
            # Specifically check for 404 Not Found
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                 self.logger.warning(f"CBOE URL returned 404 (Not Found). Data for {symbol} (settlement {settlement_date_str}) may not be available.")
            return pd.DataFrame(columns=final_columns)
        except Exception as e:
            self.logger.error(f"Unexpected error during CBOE download for {symbol}: {e}")
            return pd.DataFrame(columns=final_columns)

        # 4. Parse CSV content
        try:
            # Use io.StringIO to treat the string as a file
            csv_file = io.StringIO(csv_content)
            
            # --- Dynamically determine header row --- #
            header_row_index = 0 # Default assumption
            try:
                # Peek at the first two lines without consuming the reader
                line1 = csv_file.readline().strip()
                line2 = csv_file.readline().strip()
                csv_file.seek(0) # IMPORTANT: Reset reader to start
                
                # Check if line 2 starts like the expected header
                if line2.startswith('Trade Date'):
                    header_row_index = 1
                    logger.debug(f"Detected header on line {header_row_index + 1}. Skipping disclaimer.")
                elif line1.startswith('Trade Date'):
                    header_row_index = 0
                    logger.debug(f"Detected header on line {header_row_index + 1}. No disclaimer found.")
                else:
                    logger.warning("Could not detect 'Trade Date' header in first two lines. Assuming header is on line 1.")
            except Exception as e_peek:
                 logger.warning(f"Error peeking at header lines: {e_peek}. Assuming header is on line 1.")
                 csv_file.seek(0) # Ensure reader is reset even on error
            # ------------------------------------------ #

            # --- MODIFIED: Use specific args for archive files --- #
            if is_archive:
                logger.debug(f"Parsing archive file: header={header_row_index}, usecols={archive_expected_cols}")
                df = pd.read_csv(csv_file, 
                                 header=header_row_index, # Use determined header row (0 or 1)
                                 usecols=archive_expected_cols, # Use only expected columns
                                 on_bad_lines='skip' # Skip rows with too many fields (handles trailing comma)
                                 )
            else:
                 # Default parsing for modern files, skipping potential disclaimer
                 logger.debug(f"Parsing modern file: skiprows={header_row_index}")
                 df = pd.read_csv(csv_file, skiprows=header_row_index)
            # ------------------------------------------------------ #

            # Check if essential columns exist (use a common subset required for our final df)
            required_csv_cols = ['Trade Date', 'Open', 'High', 'Low', 'Close', 'Total Volume', 'Settle', 'Open Interest']
            if not all(col in df.columns for col in required_csv_cols):
                 self.logger.error(f"Downloaded CBOE CSV for {symbol} is missing required columns. Columns found: {df.columns.tolist()}")
                 return pd.DataFrame(columns=final_columns)

            # 5. Rename and Select Columns
            df = df.rename(columns={
                'Trade Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Total Volume': 'volume',
                'Settle': 'settle',
                'Open Interest': 'open_interest'
            })
            
            # Select only the needed columns after renaming
            # Ensure order matches final_columns as much as possible before adding metadata
            selected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'settle', 'open_interest']
            df = df[selected_cols]
            
            # Convert timestamp and numeric types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'settle', 'timestamp'])
            df['volume'] = df['volume'].fillna(0).astype('int64')
            df['open_interest'] = df['open_interest'].fillna(0).astype('int64')
            
        except Exception as e:
            self.logger.error(f"Error parsing CBOE CSV content for {symbol}: {e}", exc_info=True)
            return pd.DataFrame(columns=final_columns)

        # 6. Add Standard Metadata Columns
        df['symbol'] = symbol # Use the specific contract symbol
        df['source'] = 'cboe'
        df['interval_value'] = interval
        df['interval_unit'] = unit
        df['adjusted'] = False
        df['quality'] = 100 # Assume direct download is high quality

        # 7. Filter by fetch_start_date
        # Ensure timestamp is naive before comparison
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        # Ensure fetch_start_date is also naive for comparison
        naive_fetch_start_date = fetch_start_date
        if fetch_start_date.tz is not None:
            naive_fetch_start_date = fetch_start_date.tz_localize(None)
            
        df_filtered = df[df['timestamp'] > naive_fetch_start_date].copy()
        
        self.logger.info(f"Successfully parsed {len(df)} rows from CBOE CSV for {symbol}. Filtered to {len(df_filtered)} rows after {naive_fetch_start_date.date()}.")
        
        # Return in the correct column order
        return df_filtered[final_columns]

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

    def save_to_db(self, df: pd.DataFrame, table_name: str = "market_data") -> None:
        """Save market data to the specified database table using DuckDB's direct DataFrame insertion.
           Handles potential conflicts by updating existing rows.
        """
        if df is None or df.empty:
            self.logger.warning(f"No data to save to database table {table_name}")
            return

        try:
            # --- Define known columns for each target table --- #
            market_data_cols = [
                'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                'open_interest', 'up_volume', 'down_volume', 'source',
                'interval_value', 'interval_unit', 'adjusted', 'quality', 'settle',
                'changed', 'UnderlyingSymbol'
            ]
            market_data_cboe_cols = [
                'timestamp', 'symbol', 'open', 'high', 'low', 'settle',
                'interval_value', 'interval_unit', 'source', 'close', 'volume',
                'open_interest'
            ]
            
            if table_name == "market_data":
                valid_columns_for_table = market_data_cols
            elif table_name == "market_data_cboe":
                valid_columns_for_table = market_data_cboe_cols
            else:
                self.logger.error(f"Unknown target table '{table_name}' in save_to_db. Cannot save.")
                return
            # ---------------------------------------------------- #

            # --- Prepare DataFrame based on TARGET TABLE schema --- #
            
            # 1. Identify columns present in the input df AND valid for the target table
            cols_to_keep = [col for col in df.columns if col in valid_columns_for_table]
            df_filtered = df[cols_to_keep].copy() # Work with a filtered copy

            # 2. Perform type conversions on the filtered DataFrame
            # Ensure correct types for columns that exist in df_filtered
            if 'timestamp' in df_filtered.columns: df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp']).astype('datetime64[us]')
            if 'symbol' in df_filtered.columns: df_filtered['symbol'] = df_filtered['symbol'].astype(str)
            if 'open' in df_filtered.columns: df_filtered['open'] = df_filtered['open'].astype(float)
            if 'high' in df_filtered.columns: df_filtered['high'] = df_filtered['high'].astype(float)
            if 'low' in df_filtered.columns: df_filtered['low'] = df_filtered['low'].astype(float)
            if 'close' in df_filtered.columns: df_filtered['close'] = df_filtered['close'].astype(float)
            if 'volume' in df_filtered.columns: df_filtered['volume'] = pd.to_numeric(df_filtered['volume'], errors='coerce').fillna(0).astype('int64')
            if 'up_volume' in df_filtered.columns: df_filtered['up_volume'] = df_filtered['up_volume'].fillna(0).astype('int64')
            if 'down_volume' in df_filtered.columns: df_filtered['down_volume'] = df_filtered['down_volume'].fillna(0).astype('int64')
            if 'source' in df_filtered.columns: df_filtered['source'] = df_filtered['source'].astype(str)
            if 'interval_value' in df_filtered.columns: df_filtered['interval_value'] = df_filtered['interval_value'].astype(int)
            if 'interval_unit' in df_filtered.columns: df_filtered['interval_unit'] = df_filtered['interval_unit'].astype(str)
            if 'adjusted' in df_filtered.columns: df_filtered['adjusted'] = df_filtered.get('adjusted', False).astype(bool)
            if 'quality' in df_filtered.columns: df_filtered['quality'] = df_filtered.get('quality', 100).astype(int)
            if 'settle' in df_filtered.columns: df_filtered['settle'] = df_filtered['settle'].astype(float)
            if 'open_interest' in df_filtered.columns: df_filtered['open_interest'] = df_filtered['open_interest'].fillna(0).astype('int64')
            # Add conversions for 'changed', 'UnderlyingSymbol' if needed for market_data table?
            # Assuming they are handled correctly or not critical for now

            # 3. Prepare for Insertion
            df_to_insert = df_filtered # Use the filtered and type-converted df

            if df_to_insert.empty:
                 self.logger.warning(f"DataFrame is empty after filtering for target table '{table_name}'. No data to save.")
                 return

            # Use DuckDB's efficient UPSERT capability via temp view registration
            temp_view_name = f"temp_{table_name}_view"
            
            self.conn.register(temp_view_name, df_to_insert)
            
            # Dynamically generate column names and EXCLUDED placeholders for UPSERT
            # Use columns from df_to_insert which are guaranteed to be in the target table
            column_names_str = ", ".join([f'"{col}"' for col in df_to_insert.columns])
            select_cols_str = ", ".join([f'"{col}"' for col in df_to_insert.columns])
            
            # Define primary key columns based on the target table (assuming same PK structure)
            pk_columns = ['timestamp', 'symbol', 'interval_value', 'interval_unit']
            update_setters_list = []
            for col in df_to_insert.columns:
                 if col not in pk_columns:
                     update_setters_list.append(f'"{col}" = EXCLUDED."{col}"')
            update_setters = ", ".join(update_setters_list)
            
            conflict_target = f'({" , ".join([f'"{col}"' for col in pk_columns])})'
            
            # Check if there are any columns to update before adding SET clause
            if not update_setters:
                 # If only PK columns are inserted, use DO NOTHING
                 sql = f'''
                 INSERT INTO "{table_name}" ({column_names_str}) 
                 SELECT {select_cols_str} FROM {temp_view_name}
                 ON CONFLICT {conflict_target} DO NOTHING
                 '''
                 self.logger.warning(f"Only primary key columns present for upsert into {table_name}. Using ON CONFLICT DO NOTHING.")
            else:
                 # Use DO UPDATE SET if there are non-PK columns
                 sql = f'''
                 INSERT INTO "{table_name}" ({column_names_str}) 
                 SELECT {select_cols_str} FROM {temp_view_name}
                 ON CONFLICT {conflict_target} DO UPDATE SET 
                     {update_setters}
                 '''
            
            self.conn.execute(sql)
            self.conn.unregister(temp_view_name) # Clean up the view
            self.conn.commit()
            self.logger.info(f"Successfully upserted {len(df_to_insert)} rows into {table_name}")

        except Exception as e:
            self.logger.error(f"Error saving to database table {table_name}: {str(e)}")
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
    
    # --- NEW Expiry Calculation Method (Adapted from calculate_volume_roll_dates.py) --- #
    def _calculate_expiry_date_from_config(self, calendar, symbol_config, contract_year, contract_month_code):
        """
        Calculates the expiry date using pandas_market_calendars and rules from config.
        Internal helper method.
        """
        rule = symbol_config.get('expiry_rule', {})
        month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                     'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
        contract_month = month_map.get(contract_month_code)
        if not contract_month:
            raise ValueError(f"Invalid contract month code: {contract_month_code}")

        # Define start and end of the contract month for calendar searching
        month_start = pd.Timestamp(f'{contract_year}-{contract_month:02d}-01')
        search_start = month_start - timedelta(days=5)
        search_end = month_start + timedelta(days=40)

        # Get valid trading days within the potential range
        try:
            valid_days = calendar.valid_days(start_date=search_start.strftime('%Y-%m-%d'),
                                         end_date=search_end.strftime('%Y-%m-%d'))
            # --- ADDED: Ensure valid_days is timezone-naive for comparisons ---
            if valid_days.tz is not None:
                 self.logger.debug(f"Converting valid_days from {valid_days.tz} to timezone-naive.")
                 valid_days = valid_days.tz_localize(None)
            # --------------------------------------------------------------------
        except Exception as e:
            self.logger.error(f"Error getting valid days from calendar for {symbol_config.get('base_symbol')} {contract_year}-{contract_month_code}: {e}")
            return None # Cannot proceed without valid days

        # --- Apply Specific Expiry Rules ---
        base_sym_for_log = symbol_config.get('base_symbol', 'UnknownSymbol') # For logging

        # Rule: Nth specific weekday of the month (e.g., 3rd Friday)
        if rule.get('day_type') in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] and rule.get('day_number'):
            day_name_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}
            target_weekday = day_name_map[rule['day_type']]
            occurrence = int(rule['day_number'])
            month_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == contract_year)]
            target_days = month_days[month_days.weekday == target_weekday]
            if len(target_days) >= occurrence:
                return target_days[occurrence - 1].normalize() # Return Timestamp
            else:
                self.logger.warning(f"Could not find {occurrence} occurrences of {rule['day_type']} in {contract_year}-{contract_month} for {base_sym_for_log}. Rule: {rule}")
                return None

        # Rule: N business days before a specific day of the month (e.g., 3 days before 25th for CL)
        elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_day'):
            days_before = int(rule['days_before'])
            reference_day_num = int(rule['reference_day'])
            reference_date = pd.Timestamp(f'{contract_year}-{contract_month:02d}-{reference_day_num:02d}').normalize()
            days_strictly_before_ref = valid_days[valid_days < reference_date]
            if len(days_strictly_before_ref) >= days_before:
                 return days_strictly_before_ref[-days_before].normalize() # Return Timestamp
            else:
                 self.logger.warning(f"Not enough trading days found before {reference_date.date()} to satisfy {days_before} days_before rule for {base_sym_for_log}. Rule: {rule}")
                 return None

        # Rule: N business days before the last business day of the month (e.g., GC)
        elif rule.get('day_type') == 'business_day' and rule.get('days_before') and rule.get('reference_point') == 'last_business_day':
            days_before = int(rule['days_before'])
            month_trading_days = valid_days[(valid_days.month == contract_month) & (valid_days.year == contract_year)]
            if len(month_trading_days) >= days_before + 1:
                return month_trading_days[-(days_before + 1)].normalize() # Return Timestamp
            else:
                self.logger.warning(f"Not enough trading days in {contract_year}-{contract_month} to find {days_before} days before last business day for {base_sym_for_log}. Rule: {rule}")
                return None

        # Rule: Special VX expiry (Wednesday 30 days prior to 3rd Friday of *following* month)
        elif rule.get('special_rule') == 'VX_expiry':
            next_month = contract_month + 1
            next_year = contract_year
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_month_start = pd.Timestamp(f'{next_year}-{next_month:02d}-01')
            search_end_spx = next_month_start + timedelta(days=35)
            try:
                 valid_days_spx = calendar.valid_days(start_date=next_month_start.strftime('%Y-%m-%d'),
                                                      end_date=search_end_spx.strftime('%Y-%m-%d'))
            except Exception as e:
                 self.logger.error(f"Error getting valid days from calendar for VX rule next month ({next_year}-{next_month}): {e}")
                 return None
            
            next_month_days = valid_days_spx[(valid_days_spx.month == next_month) & (valid_days_spx.year == next_year)]
            fridays_next_month = next_month_days[next_month_days.weekday == 4]
            if len(fridays_next_month) >= 3:
                spx_expiry_friday = fridays_next_month[2]
                target_date = spx_expiry_friday - timedelta(days=30)
                potential_expiry = target_date
                while potential_expiry.weekday() != 2:
                    potential_expiry -= timedelta(days=1)
                try:
                    # --- MODIFIED CHECK: Use valid_days index instead of is_session ---
                    # Ensure potential_expiry is normalized and timezone-naive for comparison
                    potential_expiry_normalized = potential_expiry.normalize()
                    # We need valid_days covering the *potential_expiry* date's month, not just the *next* month.
                    # Re-fetch valid_days around the potential_expiry date.
                    check_start = potential_expiry_normalized - timedelta(days=5)
                    check_end = potential_expiry_normalized + timedelta(days=5)
                    valid_days_check = calendar.valid_days(start_date=check_start.strftime('%Y-%m-%d'),
                                                           end_date=check_end.strftime('%Y-%m-%d'))
                    if valid_days_check.tz is not None:
                         valid_days_check = valid_days_check.tz_localize(None)

                    if potential_expiry_normalized in valid_days_check:
                    # -----------------------------------------------------------------
                        return potential_expiry_normalized # Return Timestamp
                    else:
                        # Fallback: get previous trading day before non-session Wednesday
                        # Use the schedule from the original `calendar` object
                        schedule = calendar.schedule(start_date=(potential_expiry_normalized - timedelta(days=5)).strftime('%Y-%m-%d'),
                                                    end_date=potential_expiry_normalized.strftime('%Y-%m-%d'))
                        if not schedule.empty:
                             # Ensure the index is timezone-naive before returning
                             last_trading_day = schedule.index[-1]
                             if last_trading_day.tz is not None:
                                  last_trading_day = last_trading_day.tz_localize(None)
                             return last_trading_day.normalize() # Return Timestamp
                        else:
                            self.logger.warning(f"Could not find previous trading day for VX calculated expiry {potential_expiry_normalized.date()} for {base_sym_for_log}.")
                            return None # Fallback failed
                except Exception as e:
                     self.logger.error(f"Error checking calendar session for VX expiry {potential_expiry.date()}: {e}", exc_info=True)
                     return None
            else:
                self.logger.warning(f"Could not find 3rd Friday in {next_year}-{next_month} for VX expiry calculation for {base_sym_for_log}. Rule: {rule}")
                return None

        # --- Fallback for unhandled rules ---
        else:
            self.logger.warning(f"Expiry rule calculation not implemented or rule invalid for {base_sym_for_log}: {rule}. Using fallback.")
            # Fallback might be just None, or a very simple estimate like mid-month
            # Returning None is safer as it signals calculation failure.
            return None
    # --- END NEW Expiry Calculation Method --- #

    def calculate_expiration_date(self, symbol: str) -> Optional[pd.Timestamp]:
        """Calculate the expiration date for a specific futures contract symbol.
           Uses the detailed _calculate_expiry_date_from_config method.
        """
        # 1. Parse symbol to get base, month code, year
        match = re.match(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$", symbol)
        if not match:
            self.logger.warning(f"Cannot parse futures contract symbol format: {symbol}")
            return None
        base_symbol, month_code, year_code = match.groups()
        
        # Convert year code to full year (assuming 20xx)
        try:
            year_int = int(year_code)
            contract_year = 2000 + year_int
            # Add simple validation for sensible year range if needed
            current_year = pd.Timestamp.now().year
            if not (2000 <= contract_year <= current_year + 5): # Allow some future years
                 self.logger.warning(f"Parsed year {contract_year} from symbol {symbol} seems unlikely. Check symbol format.")
                 # Continue for now, but could return None
        except ValueError:
             self.logger.error(f"Could not convert year code '{year_code}' to integer for symbol {symbol}")
             return None

        # 2. Get config for the base symbol
        symbol_config = self._get_symbol_config(base_symbol)
        if not symbol_config:
            self.logger.warning(f"No configuration found for base symbol '{base_symbol}' when calculating expiry for {symbol}.")
            return None
            
        # 3. Get the correct trading calendar
        calendar_name = symbol_config.get('calendar', 'NYSE') # Default to NYSE if not specified
        try:
             calendar = get_trading_calendar(calendar_name) # Use the helper function in this file
        except Exception as e:
             self.logger.error(f"Failed to get trading calendar '{calendar_name}' for {symbol}: {e}")
             return None
             
        # 4. Call the detailed calculation method
        try:
             expiry_date = self._calculate_expiry_date_from_config(calendar, symbol_config, contract_year, month_code)
             # Method returns normalized Timestamp or None
             return expiry_date 
        except Exception as e:
             self.logger.error(f"Error during detailed expiry calculation for {symbol}: {e}", exc_info=True)
             return None

    def process_symbol(self, symbol: str, update_history: bool = False, force: bool = False, interval_value: Optional[int] = None, interval_unit: Optional[str] = None) -> None:
        """Process a single symbol, fetching and storing its data.
        
        Args:
            symbol: The symbol to process (can be equity, index, base future, or specific contract)
            update_history: If True, fetch from start_date in config to current date, skipping existing data
            force: If True, overwrite existing data in the database
            interval_value: Optional specific interval value to fetch.
            interval_unit: Optional specific interval unit to fetch.
        """
        try:
            logger.info(f"Processing {symbol}")
            
            symbol_info = self._get_symbol_config(symbol)
            if not symbol_info:
                logger.error(f"No configuration found for symbol {symbol}")
                return
                
            actual_symbol = symbol_info.get('specific_contract', symbol_info.get('symbol', symbol))
            logger.debug(f"Found configuration for {symbol}. Actual symbol for fetch/store: {actual_symbol}")
            
            latest_date = self.get_latest_date_in_db(actual_symbol)
            logger.debug(f"Latest date in DB for {actual_symbol}: {latest_date}")
            
            # Determine start date for fetching
            start_date = None
            config_start_date_str = symbol_info.get('start_date', self.start_date.strftime('%Y-%m-%d'))
            config_start_date = pd.Timestamp(config_start_date_str)
            is_specific_contract = 'specific_contract' in symbol_info
            if is_specific_contract:
                expiry_date = self.calculate_expiration_date(actual_symbol)
                logger.debug(f"Calculated expiry date for {actual_symbol}: {expiry_date}")
                if expiry_date:
                    contract_start_calc = expiry_date - timedelta(days=270) 
                    start_date = max(contract_start_calc.tz_localize(None), config_start_date)
                    logger.info(f"Calculated fetch start date for specific contract {actual_symbol}: {start_date.date()}")
                else:
                    logger.warning(f"Could not calculate expiry for {actual_symbol}, using config start date {config_start_date.date()}")
                    start_date = config_start_date
            else:
                 start_date = config_start_date
            
            # Adjust start date based on mode and existing data
            if latest_date and not force and not update_history:
                fetch_start_date = latest_date
                logger.info(f"Found existing data for {actual_symbol}, last timestamp: {latest_date}. Fetching backward to include data since this time.")
            elif update_history:
                fetch_start_date = start_date
                logger.info(f"Update history mode: fetching {actual_symbol} from {fetch_start_date.date()} to current date")
            elif force:
                 fetch_start_date = start_date
                 logger.info(f"Force mode: overwriting {actual_symbol} data from {fetch_start_date.date()} to current date")
            else: 
                 fetch_start_date = start_date
                 logger.info(f"No existing data for {actual_symbol}, fetching from {fetch_start_date.date()} to current date")
            
            if fetch_start_date > pd.Timestamp.now():
                 logger.info(f"Fetch start date {fetch_start_date} is in the future. Skipping {actual_symbol}.")
                 return
                 
            # --- Get Frequencies to Process (Handles simple list and list of dicts) --- #
            config_frequencies_setting = symbol_info.get('frequencies', [])
            frequencies_to_process = []
            # Get symbol-level defaults, falling back to hardcoded script defaults
            default_source = symbol_info.get('default_source', 'tradestation')
            default_raw_table = symbol_info.get('default_raw_table', 'market_data')

            # Check if a specific interval was passed via CLI args
            if interval_value is not None and interval_unit is not None:
                # --- Handle CLI Override --- #
                # Standardize unit for matching/name generation
                cli_unit_standardized = interval_unit
                if interval_unit == 'day': cli_unit_standardized = 'daily'
                
                freq_name = f"{interval_value}{cli_unit_standardized}" # Basic name
                if cli_unit_standardized == 'daily': freq_name = 'daily'
                elif cli_unit_standardized == 'minute': freq_name = f"{interval_value}min"
                
                source = default_source # Start with default
                target_table = default_raw_table # Start with default
                
                # Check if config_frequencies is list of dicts to find specific override
                if isinstance(config_frequencies_setting, list) and config_frequencies_setting and isinstance(config_frequencies_setting[0], dict):
                    for freq_dict in config_frequencies_setting:
                        if freq_dict.get('interval') == interval_value and freq_dict.get('unit') == cli_unit_standardized:
                            source = freq_dict.get('source', default_source) # Use specific, fallback to default
                            target_table = freq_dict.get('raw_table', default_raw_table) # Use specific, fallback to default
                            freq_name = freq_dict.get('name', freq_name) # Use configured name if available
                            logger.info(f"Found matching frequency config for CLI args: Name='{freq_name}', Source='{source}', Table='{target_table}'")
                            break # Found match
                    else: # If loop completes without break
                         logger.warning(f"CLI interval {interval_value} {cli_unit_standardized} not explicitly in config dict for {actual_symbol}. Using symbol defaults: Source='{source}', Table='{target_table}'")
                else:
                    logger.warning(f"Frequencies for {actual_symbol} not a list of dicts. Using symbol defaults for CLI args: Source='{source}', Table='{target_table}'")
                    
                frequencies_to_process.append({
                    'name': freq_name,
                    'interval': interval_value,
                    'unit': cli_unit_standardized,
                    'source': source,
                    'raw_table': target_table # Pass the determined table
                })
                logger.info(f"Processing ONLY specific interval from command line: {freq_name}")
                
            # --- Handle Config Frequencies (No CLI override) --- #
            elif isinstance(config_frequencies_setting, list):
                if not config_frequencies_setting: # Handle empty list
                     logger.warning(f"No frequencies defined in config for {actual_symbol}. Nothing to process.")
                
                elif isinstance(config_frequencies_setting[0], str): # Check if it's the simple list format
                    logger.debug(f"Processing simple frequency list for {actual_symbol} using defaults (Source: {default_source}, Table: {default_raw_table})")
                    for freq_name_str in config_frequencies_setting:
                        interval, unit = None, None
                        # Parse interval/unit from string
                        if freq_name_str == 'daily':
                            interval, unit = 1, 'daily'
                        elif freq_name_str.endswith('min'):
                            try:
                                interval = int(freq_name_str[:-3])
                                unit = 'minute'
                            except ValueError:
                                logger.warning(f"Could not parse interval from legacy frequency '{freq_name_str}', skipping.")
                                continue
                        # Add more parsing logic here if other string formats are used (e.g., '1hour')
                        
                        if interval is not None and unit is not None:
                            frequencies_to_process.append({
                                'name': freq_name_str,
                                'interval': interval,
                                'unit': unit,
                                'source': default_source, # Use symbol default
                                'raw_table': default_raw_table # Use symbol default
                            })
                        else:
                            logger.warning(f"Could not parse frequency string '{freq_name_str}', skipping.")
                            
                elif isinstance(config_frequencies_setting[0], dict): # Check if it's the list of dictionaries format
                    logger.debug(f"Processing frequency list of dictionaries for {actual_symbol}")
                    for freq_dict in config_frequencies_setting:
                        # Validate required keys
                        if not all(k in freq_dict for k in ('name', 'interval', 'unit')):
                            logger.warning(f"Skipping frequency dict due to missing keys: {freq_dict}")
                            continue
                        
                        # Get source and raw_table, falling back to defaults
                        source = freq_dict.get('source', default_source)
                        raw_table = freq_dict.get('raw_table', default_raw_table)
                        
                        frequencies_to_process.append({
                            'name': freq_dict['name'],
                            'interval': freq_dict['interval'],
                            'unit': freq_dict['unit'],
                            'source': source,
                            'raw_table': raw_table
                        })
                else:
                     logger.error(f"Invalid format for 'frequencies' in config for {actual_symbol}: {config_frequencies_setting}")
            else:
                 logger.error(f"Invalid type for 'frequencies' in config for {actual_symbol}: {type(config_frequencies_setting)}")
            # -------------------------------------------------------------------------- #
            
            # --- Process Each Determined Frequency --- #
            logger.info(f"Starting fetch loop for {len(frequencies_to_process)} frequencies for {actual_symbol}")
            for freq_info in frequencies_to_process:
                try:
                    # --- Get frequency details (already resolved with defaults) --- #
                    freq_name = freq_info['name']
                    interval = freq_info['interval']
                    unit = freq_info['unit']
                    source = freq_info['source']
                    target_table = freq_info['raw_table'] # Use the resolved raw_table
                    # ------------------------------------------------------------------ #

                    logger.info(f"Fetching {freq_name} ({interval} {unit}) data for {actual_symbol} from {source} (start: {fetch_start_date}) -> Target Table: {target_table}")
                    
                    data = None # Initialize data DataFrame
                    # --- Source-specific Fetching --- #
                    if source == 'cboe':
                        # --- MODIFIED: Call direct CBOE download for daily ---
                        if unit == 'daily':
                            data = self._fetch_cboe_vx_daily(actual_symbol, interval, unit, fetch_start_date)
                        else:
                            # Handle non-daily CBOE sources if they ever exist
                            logger.error(f"Fetching non-daily data from CBOE source is not currently supported for {actual_symbol} ({interval} {unit}). Skipping.")
                            continue # Skip this frequency
                    elif source == 'tradestation':
                        # Check TradeStation authentication before attempting fetch
                        if not self.ts_agent.access_token:
                             if not self.ts_agent.authenticate(): # Try to authenticate if needed
                                 logger.error("Failed to authenticate with TradeStation API for tradestation source. Skipping.")
                                 continue
                             
                        data = self.fetch_data_since(actual_symbol, interval, unit, start_date=fetch_start_date)
                    else:
                        logger.error(f"Unsupported source '{source}' defined for {actual_symbol} frequency {freq_name}. Skipping.")
                        continue # Skip this frequency
                    # -------------------------------- #
                    
                    if data is not None and not data.empty:
                        # Overwrite the 'source' column in the dataframe with the correct source from config
                        data['source'] = source 
                        logger.info(f"Retrieved {len(data)} rows of {freq_name} data for {actual_symbol} from {source}")
                        
                        if force:
                            logger.info(f"Force mode: Deleting existing {freq_name} data for {actual_symbol} from {target_table}")
                            self.delete_existing_data(actual_symbol, interval, unit, target_table)
                        # Pass target_table to save_to_db
                        self.save_to_db(data, target_table)
                    else:
                        logger.warning(f"No new {freq_name} data retrieved for {actual_symbol} from {source} starting {fetch_start_date}")
                except Exception as e:
                    logger.error(f"Error processing {actual_symbol} for frequency {freq_name}: {str(e)}", exc_info=True)
                    continue # Continue to next frequency
        except Exception as e:
            logger.error(f"General error processing symbol {symbol}: {str(e)}", exc_info=True)

    def delete_existing_data(self, symbol: str, interval: int, unit: str, table_name: str = "market_data") -> None:
        """Delete existing data for a symbol and frequency from the specified database table.
        
        Args:
            symbol: The symbol to delete data for
            interval: The interval value
            unit: The interval unit
            table_name: The table to delete from (defaults to market_data)
        """
        try:
            query = f"""
            DELETE FROM "{table_name}" 
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
        futures contracts (ESH25, NQM25, VXF24), and continuous contracts (@BASE=...)
        by looking up their base config.
        """
        self.logger.debug(f"_get_symbol_config checking for: {symbol}")
        
        # 0. Handle specific continuous contracts explicitly if needed (e.g. @VX=101XN)
        # If these have their own top-level entries in the YAML
        for future in self.config.get('futures', []):
            if future.get('symbol') == symbol and future.get('asset_type') == 'continuous_future':
                 self.logger.debug(f"Found explicit continuous config for {symbol}")
                 return future
        # Add checks for indices/equities if they can be continuous

        # 1. Check for continuous contract pattern (@BASE=...)
        # Example: @ES=102XC, @VX=101XN
        continuous_pattern = re.compile(r"^(@[A-Z]{1,3})(?:=.*)?$") # Match @BASE or @BASE=...
        cont_match = continuous_pattern.match(symbol)
        base_symbol_from_cont = None
        if cont_match:
            base_symbol_from_cont = cont_match.group(1).lstrip('@') # Extract BASE (e.g., ES, VX)
            self.logger.debug(f"Detected continuous pattern: {symbol}, Extracted Base: {base_symbol_from_cont}")
            # Find config for the extracted base symbol (ES, NQ, VX...)
            for future in self.config.get('futures', []):
                if future.get('base_symbol') == base_symbol_from_cont:
                    config = future.copy()
                    config['_derived_from_continuous'] = True # Mark that this config was found via a continuous pattern
                    config['_original_symbol'] = symbol # Store original full symbol for reference
                    self.logger.debug(f"Found base config for continuous symbol '{symbol}' using base '{base_symbol_from_cont}'")
                    return config
            # If base config not found, log warning but continue to check other types
            self.logger.warning(f"Found continuous pattern {symbol} but no base config found for '{base_symbol_from_cont}'")

        # 2. Check indices directly
        for index in self.config.get('indices', []):
            if index.get('symbol') == symbol:
                self.logger.debug(f"Found index config for {symbol}")
                return index

        # 3. Check equities directly
        for equity in self.config.get('equities', []):
            if equity.get('symbol') == symbol:
                self.logger.debug(f"Found equity config for {symbol}")
                return equity

        # 4. Check if it looks like a standard futures contract (e.g., ESH25, NQM25)
        # This pattern might need adjustment depending on exact symbol formats used
        futures_pattern = re.compile(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$")
        fut_match = futures_pattern.match(symbol)
        base_symbol_from_fut = None
        if fut_match:
            base_symbol_from_fut = fut_match.group(1) # Extract Base (e.g., ES, VX)
            self.logger.debug(f"Detected standard futures pattern: {symbol}, Extracted Base: {base_symbol_from_fut}")
            # Find the config for the base symbol
            for future in self.config.get('futures', []):
                if future.get('base_symbol') == base_symbol_from_fut:
                    config = future.copy()
                    config['_derived_from_standard_future'] = True # Mark how config was found
                    config['_original_symbol'] = symbol
                    self.logger.debug(f"Found base config for standard future '{symbol}' using base '{base_symbol_from_fut}'")
                    return config
            # If base config not found, log warning but continue
            self.logger.warning(f"Found standard future {symbol} but no base config found for '{base_symbol_from_fut}'")

        # 5. Fallback: Check if the provided symbol itself is a base future symbol in the config
        for future in self.config.get('futures', []):
             if future.get('base_symbol') == symbol:
                 self.logger.debug(f"Found base future config matching symbol directly: {symbol}")
                 return future # Return the base config directly

        # If no match found after all checks
        self.logger.error(f"No configuration match found for symbol: {symbol}")
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
        # --- ADDED: Get exclusion list --- #
        exclude_list = set(hist_contracts_config.get('exclude_contracts', []))
        # --------------------------------- #
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
                
                # --- ADDED: Create symbol and check exclusion list --- #
                contract_symbol = f"{root_symbol}{month_code}{yr_code}"
                if contract_symbol in exclude_list:
                    logger.debug(f"Skipping excluded contract: {contract_symbol}")
                    continue # Skip this contract
                # ---------------------------------------------------- #

                # Estimate expiry date to check if contract is relevant
                expiry_estimate = self.calculate_expiration_date(contract_symbol)
                
                if expiry_estimate:
                    # Include contracts whose expiry is after our overall start_date
                    # and whose potential *start* (e.g., expiry - 9mo) is before our overall end_date
                    potential_start = expiry_estimate - timedelta(days=270) # Approx 9 months
                    if expiry_estimate >= start_date and potential_start <= end_date:
                         contracts.add(contract_symbol)
                else:
                    # If expiry calculation fails, maybe still include based on year/month range? Risky.
                    logger.warning(f"Could not estimate expiry for {contract_symbol}, including based on year/month.")
                    # Crude check: include if the contract month is within range
                    contract_date_est = pd.Timestamp(f'{year}-{contract_month:02d}-01')
                    if contract_date_est >= start_date and contract_date_est <= end_date + pd.DateOffset(months=num_active_contracts*2):
                        contracts.add(contract_symbol)

        logger.info(f"Generated {len(contracts)} potential contracts for {root_symbol} between {start_date.date()} and {end_date.date()}")
        return sorted(list(contracts))

    def run(self, symbol=None, update_history=False, force=False, interval_value: Optional[int] = None, interval_unit: Optional[str] = None):
        """
        Run the data fetcher for all symbols in the config or a specific symbol.
        
        Args:
            symbol: Optional symbol to process (if None, process all symbols)
            update_history: If True, fetch from start_date in config to current date
            force: If True, overwrite existing data in the database
            interval_value: Optional specific interval value to fetch.
            interval_unit: Optional specific interval unit to fetch.
        """
        try:
            # Connect to TradeStation
            if not self.ts_agent.authenticate():
                logger.error("Failed to authenticate with TradeStation API")
                return
                
            if symbol:
                # Process the single specified symbol, passing interval args
                self.process_symbol(symbol, update_history, force, interval_value, interval_unit)
            else:
                # Process all symbols from config
                # If specific intervals are given, warn that it only applies if a single symbol is specified
                if interval_value is not None or interval_unit is not None:
                     logger.warning("Interval arguments (--interval-value, --interval-unit) are ignored when processing all symbols (no --symbol specified).")
                     
                logger.info("Processing all symbols defined in the configuration...")
                processed_symbols = set()
                
                # Process equities (without passing intervals)
                for equity in self.config.get('equities', []):
                    sym = equity['symbol']
                    if sym not in processed_symbols:
                        self.process_symbol(sym, update_history, force)
                        processed_symbols.add(sym)
                    
                # Process indices (without passing intervals)
                for index in self.config.get('indices', []):
                    sym = index['symbol']
                    if sym not in processed_symbols:
                        self.process_symbol(sym, update_history, force)
                        processed_symbols.add(sym)
                        
                # Process futures (without passing intervals)
                for future_config in self.config.get('futures', []):
                    base_symbol = future_config['base_symbol']
                    if base_symbol in processed_symbols:
                        continue 
                        
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
                            # Process specific contract without interval override
                            self.process_symbol(contract, update_history, force)
                            processed_symbols.add(contract)
                         else:
                            logger.debug(f"Skipping already processed contract: {contract}")
                            
                    processed_symbols.add(base_symbol) # Mark base as processed
                        
                logger.info("Finished processing all symbols.")
                
        except Exception as e:
            logger.error(f"Error running data fetcher: {e}", exc_info=True)
            raise 
        finally:
             # Ensure DB connection is closed
             if self.conn:
                 try:
                     self.conn.close()
                     logger.info("Database connection closed.")
                 except Exception as e:
                     logger.error(f"Error closing database connection in main: {e}")

    def get_active_futures_symbols(self, base_symbol: str, item_config: Dict[str, Any]) -> List[str]:
        """Determine the currently active futures contract symbols based on config.

        Args:
            base_symbol: The base symbol (e.g., "ES", "NQ").
            item_config: The configuration dictionary for this base symbol from market_symbols.yaml.

        Returns:
            A list of the N active contract symbols (e.g., ['ESH24', 'ESM24']),
            ordered by proximity of expiry date (nearest first).
            Returns an empty list if configuration is missing or invalid.
        """
        if not item_config:
            logger.error(f"Missing item_config for {base_symbol} in get_active_futures_symbols.")
            return []
            
        num_active = item_config.get('num_active_contracts', 1)
        if num_active <= 0:
            logger.warning(f"num_active_contracts is {num_active} for {base_symbol}. Returning empty list.")
            return []

        # Generate potential contracts for the current and next year
        today = pd.Timestamp.now().normalize()
        start_gen_date = today - pd.Timedelta(days=90) # Look back slightly to catch recent contracts
        end_gen_date = today + pd.Timedelta(days=540) # Look ahead ~1.5 years
        
        try:
            # Ensure generate_futures_contracts exists and works
            potential_symbols = self.generate_futures_contracts(
                base_symbol, 
                start_date=start_gen_date, 
                end_date=end_gen_date
            )
        except AttributeError:
             logger.error(f"MarketDataFetcher does not have 'generate_futures_contracts' method.")
             return []
        except Exception as e:
             logger.error(f"Error generating potential contracts for {base_symbol}: {e}", exc_info=True)
             return []

        if not potential_symbols:
            logger.warning(f"No potential symbols generated for {base_symbol} around {today.date()}")
            return []

        # Calculate expiry dates for potential contracts
        contracts_with_expiry = []
        for symbol in potential_symbols:
            try:
                # Ensure calculate_expiration_date exists and works
                expiry_date = self.calculate_expiration_date(symbol)
                if expiry_date and expiry_date >= today:
                    # Ensure expiry_date is a Timestamp for comparison/sorting
                    if isinstance(expiry_date, (datetime, date)):
                         expiry_date = pd.Timestamp(expiry_date)
                    else: # Skip if conversion fails or type is wrong
                         logger.warning(f"Expiry date for {symbol} is not a valid date type: {type(expiry_date)}. Skipping.")
                         continue 
                    contracts_with_expiry.append((symbol, expiry_date))
            except AttributeError:
                 logger.error(f"MarketDataFetcher does not have 'calculate_expiration_date' method.")
                 # Cannot proceed without this method, return empty or raise?
                 return []
            except Exception as e:
                 logger.warning(f"Could not calculate expiry for potential symbol {symbol}: {e}")
        
        # Sort contracts by expiry date (nearest first)
        contracts_with_expiry.sort(key=lambda x: x[1])

        # Select the top N active contracts
        active_symbols = [symbol for symbol, expiry in contracts_with_expiry[:num_active]]
        
        if len(active_symbols) < num_active:
            logger.warning(f"Found only {len(active_symbols)} active contracts for {base_symbol}, expected {num_active}. Check generation range or config.")
            
        return active_symbols

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
        
        # Pass interval args to run method
        fetcher.run(args.symbol, args.updatehistory, args.force, args.interval_value, args.interval_unit)
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        # Exit with error status if needed
        # sys.exit(1) 

if __name__ == "__main__":
    main()
