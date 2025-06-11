#!/usr/bin/env python
"""
Fetch Market Data Script

This script fetches market data from TradeStation for symbols defined in the market_symbols.yaml file.
It first checks the database to see what data is already available, then fetches only the missing data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Union, Tuple
import yaml
import duckdb
from pathlib import Path
import pandas_market_calendars as mcal
from dotenv import load_dotenv
import requests
import re
import time # Added import
import io # Added import, was implicitly used in _fetch_cboe_vx_daily

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
# if project_root not in sys.path: # Removed this block
#     sys.path.insert(0, project_root)

# Import from our project
from src.agents.market import TradeStationMarketDataAgent

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
    """Fetches market data from various sources."""
    
    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None, db_connector: Optional[Any] = None, existing_conn: Optional[Any] = None): # Added existing_conn
        """
        Initialize the MarketDataFetcher.
        
        Args:
            db_path: Path to the database file
            config_path: Path to the configuration file
            db_connector: Optional existing database connection
            existing_conn: Alias for db_connector
        """
        # Load main configuration
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'market_symbols.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Main configuration (market_symbols.yaml) loaded successfully from {config_path}.")
        except Exception as e:
            logger.error(f"Error loading main configuration from {config_path}: {e}")
            self.config = {}
        
        # Load futures.yaml configuration
        futures_yaml_path = os.path.join(project_root, 'config', 'futures.yaml')
        try:
            with open(futures_yaml_path, 'r') as f:
                self.futures_config = yaml.safe_load(f)
            logger.info(f"Futures configuration (futures.yaml) loaded successfully from {futures_yaml_path}.")
        except Exception as e:
            logger.error(f"Error loading futures configuration from {futures_yaml_path}: {e}")
            self.futures_config = {}
        
        # Initialize database connection
        conn_to_use = db_connector if db_connector else existing_conn # Prioritize db_connector
        if conn_to_use:
            self.conn = conn_to_use
            logger.info("Using provided database connection.")
        else:
            logger.info("No existing connection passed, MarketDataFetcher creating its own.")
            if db_path is None:
                db_path = os.path.join(project_root, 'data', 'financial_data.duckdb')
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        
        # Initialize database schema
        self._setup_database()
        logger.info("Database schema initialized")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize TradeStation agent
        self.ts_agent = TradeStationMarketDataAgent(database_path=':memory:', verbose=False)
        try:
            self.ts_agent.connect()
            logger.info("TradeStation agent connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect TradeStation agent: {e}")
            # Depending on the desired behavior, you might want to raise the exception
            # or handle it by setting a flag that prevents TS-dependent operations.
            # For now, we log the error and the ts_agent will not have an access_token.

        # Add VALID_UNITS attribute to the ts_agent if it doesn't exist
        if not hasattr(self.ts_agent, 'VALID_UNITS'):
            self.ts_agent.VALID_UNITS = ['daily', 'minute', 'weekly', 'monthly']
            
        self.default_start_date = pd.Timestamp("2000-01-01") # Added default start date
        
        # Create a requests session for reuse
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Store config path for potential reloads or use by other methods
        self._config_path = config_path 
        self._futures_specific_config_path = futures_yaml_path # Store this path too
            
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
        """Initialize the database schema."""
        try:
            # Create tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    open_interest BIGINT,
                    up_volume BIGINT,
                    down_volume BIGINT,
                    source VARCHAR,
                    interval_value INTEGER,
                    interval_unit VARCHAR,
                    adjusted BOOLEAN DEFAULT FALSE,
                    quality INTEGER DEFAULT 100,
                    PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
                );
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS futures_contracts (
                    symbol VARCHAR NOT NULL,
                    expiration_date DATE NOT NULL,
                    roll_date DATE,
                    is_active BOOLEAN DEFAULT TRUE,
                    PRIMARY KEY (symbol, expiration_date)
                );
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS continuous_contracts (
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR NOT NULL,
                    underlying_symbol VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    open_interest BIGINT,
                    up_volume BIGINT,
                    down_volume BIGINT,
                    source VARCHAR,
                    built_by VARCHAR,
                    interval_value INTEGER,
                    interval_unit VARCHAR,
                    adjusted BOOLEAN DEFAULT FALSE,
                    quality INTEGER DEFAULT 100,
                    settle DOUBLE,
                    PRIMARY KEY (symbol, timestamp, interval_value, interval_unit)
                );
            """)
            
            # Create indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_continuous_contracts_symbol ON continuous_contracts(symbol);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_continuous_contracts_timestamp ON continuous_contracts(timestamp);")
            
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise
        
    def _load_yaml_file(self, file_path: str, description: str) -> Dict[str, Any]:
        """Loads a single YAML file and returns its content or an empty dict on error."""
        try:
            # If file_path is relative, make it relative to project root
            abs_file_path = Path(file_path)
            if not abs_file_path.is_absolute():
                abs_file_path = Path(project_root) / file_path
            
            if not abs_file_path.exists():
                self.logger.warning(f"{description} file not found at {abs_file_path}. Returning empty config.")
                return {}
            
            with open(abs_file_path, 'r') as f:
                content = yaml.safe_load(f)
                self.logger.info(f"{description} loaded successfully from {abs_file_path}.")
                return content if content else {} # Ensure return {} if file is empty
        except Exception as e:
            self.logger.error(f"Error loading {description} from {file_path}: {e}")
            return {} # Return empty dict on any error
    
    def get_latest_date_in_db(self, symbol: str, interval_value: Optional[int] = None, interval_unit: Optional[str] = None):
        """
        Get the latest date for a symbol in the database, optionally filtered by interval.
        
        Args:
            symbol: The symbol to check
            interval_value: Optional interval value to filter by
            interval_unit: Optional interval unit to filter by
            
        Returns:
            The latest date as a datetime object, or None if no data exists
        """
        try:
            query = f"""
            SELECT MAX(timestamp) as latest_date
            FROM market_data
            WHERE symbol = ?
            """
            params = [symbol]
            
            if interval_value is not None and interval_unit is not None:
                query += " AND interval_value = ? AND interval_unit = ?"
                params.extend([interval_value, interval_unit])
            
            result = self.conn.execute(query, params).fetchone()
            
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
    
    def _get_contract_details_from_symbol(self, symbol: str) -> Optional[Tuple[str, str, int, int]]:
        """Parse a futures contract symbol to extract its components.

        Args:
            symbol: The futures contract symbol (e.g., ESH25, VXU24, ZWZ23).

        Returns:
            A tuple (root_symbol, month_code, year, month_number) or None if parsing fails.
        """
        # Regex to capture root (1-3 letters), month code (1 letter), and year (1 or 2 digits)
        match = re.fullmatch(r"([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$", symbol)
        if not match:
            self.logger.debug(f"Could not parse futures symbol: {symbol} with regex.")
            return None

        root_symbol, month_code, year_str = match.groups()

        # Convert year_str to a full 4-digit year
        current_year = datetime.now().year
        current_century = (current_year // 100) * 100
        year_int = int(year_str)

        if len(year_str) == 1: # e.g., ZCH5 -> ZCH2025 (assuming current decade)
            # This logic might need refinement if single-digit years span across decade changes frequently.
            # For now, assume it's current decade or next if single digit implies roll-over.
            year = (current_year // 10) * 10 + year_int 
            if year < current_year - 2: # Heuristic: if calculated year is too far in past, assume next decade
                year += 10
        elif len(year_str) == 2: # e.g., ESH25 -> ESH2025
            # Heuristic: if year_int is e.g. 99 and current year is 2023, it's 1999.
            # If year_int is 03 and current year is 2023, it's 2003.
            # If year_int is 25 and current year is 2023, it's 2025.
            # If year_int is (current_year % 100 + 5) % 100 (e.g. 23+5=28 for 2023)
            # it's likely current century. If it's much smaller, (e.g. 01 for 2023) it's current century.
            # If it's much larger (e.g. 98 for 2003), it's previous century.
            if year_int + current_century > current_year + 70: # Heuristic for previous century
                year = current_century - 100 + year_int
            else:
                year = current_century + year_int
        else:
            # Should not happen with the regex, but as a fallback
            self.logger.error(f"Unexpected year string format '{year_str}' for symbol {symbol}")
            return None

        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        month_number = month_map.get(month_code)

        if month_number is None:
            self.logger.error(f"Invalid month code '{month_code}' for symbol {symbol}")
            return None

        # self.logger.debug(f"Parsed {symbol} -> Root: {root_symbol}, Code: {month_code}, Year: {year}, MonthNum: {month_number}")
        return root_symbol, month_code, year, month_number

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
                    
                    # Check if potential_expiry_normalized is a valid trading day
                    if not trading_calendar.valid_days(start_date=potential_expiry_normalized, end_date=potential_expiry_normalized).empty:
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
        """
        Calculate the expiration date of a given futures contract symbol.
        This method relies on rules defined in 'futures.yaml' and resolved via _get_symbol_config.
        """
        logger.debug(f"Calculating expiration for symbol: {symbol}")

        contract_details = self._get_contract_details_from_symbol(symbol)
        if not contract_details:
            logger.warning(f"Could not parse contract details from symbol: {symbol}")
            return None
        
        root_symbol, contract_month_code, contract_year, contract_month_num = contract_details
        
        # Get the fully resolved configuration for the root symbol (includes inherited templates)
        future_product_config = self._get_symbol_config(root_symbol) 
        if not future_product_config:
            logger.warning(f"No configuration (after merge) found for root symbol '{root_symbol}' when calculating expiration for {symbol}. Source: {getattr(future_product_config, '_config_source', 'N/A')}")
            return None
        
        # --- ADDED DEBUG LOGGING ---
        logger.debug(f"[{symbol}] future_product_config keys: {list(future_product_config.keys())}")
        logger.debug(f"[{symbol}] Full future_product_config: {future_product_config}")
        # --- END ADDED DEBUG LOGGING ---
            
        # Get the expiry_rule dictionary from the resolved future_product_config
        # This expiry_rule should contain the actual rule definition (e.g., 'rule', 'special_rule') and calendar info
        expiry_rule_config = future_product_config.get('expiry_rule')
        
        # --- ADDED DEBUG LOGGING ---
        logger.debug(f"[{symbol}] expiry_rule_config from future_product_config.get('expiry_rule'): {expiry_rule_config}")
        # --- END ADDED DEBUG LOGGING ---

        if not expiry_rule_config or not isinstance(expiry_rule_config, dict):
            logger.warning(f"No 'expiry_rule' dictionary found in the resolved configuration for root symbol '{root_symbol}' (contract {symbol}). Config keys: {list(future_product_config.keys()) if future_product_config else 'None'}. Source: {future_product_config.get('_config_source', 'N/A')}")
            return None

        # Extract the rule definition and calendar from the expiry_rule_config
        actual_rule_definition = expiry_rule_config.get('rule') 
        special_rule_type = expiry_rule_config.get('special_rule')
        # Calendar can be in expiry_rule_config, or at future_product_config level, or default to NYSE
        calendar_name = expiry_rule_config.get('calendar', future_product_config.get('calendar', 'NYSE'))
        
        try:
            trading_calendar = get_trading_calendar(calendar_name)
        except ValueError as e:
            logger.error(f"Failed to get trading calendar '{calendar_name}' for {symbol}: {e}")
            return None

        expiry_date = None
        
        # --- Get the actual rule details ---
        # The 'rule' could be a string (e.g., "third_friday") or a dictionary (e.g., for VX_expiry or nth_weekday)
        # The 'expiry_rule' variable here will hold the actual rule definition.
        # 'expiry_rule_config' is the dict that *contains* the rule definition (and calendar).
        
        actual_rule_definition = expiry_rule_config.get('rule') # This might be None if rule is defined by other keys like 'special_rule'
        
        # Check for special_rule (like VX_expiry) first if actual_rule_definition is not a simple string
        # or if the structure implies a special rule (e.g., presence of 'special_rule' key)
        
        special_rule_type = expiry_rule_config.get('special_rule') # e.g., "VX_expiry"

        # Rule: Third Friday of the contract month
        if actual_rule_definition == "third_friday":
            # Find the first day of the contract month
            first_day_of_month = date(contract_year, contract_month_num, 1)
            
            # Determine the day of the week for the first day (0=Monday, 4=Friday)
            day_of_week_first_day = first_day_of_month.weekday()
            
            # Calculate days needed to reach the first Friday
            # If first day is Friday (4), days_to_first_friday is 0.
            # If first day is Saturday (5), days_to_first_friday is 6 (next Friday).
            # If first day is Sunday (6), days_to_first_friday is 5.
            # Formula: (4 - day_of_week_first_day + 7) % 7
            days_to_first_friday = (4 - day_of_week_first_day + 7) % 7
            first_friday = first_day_of_month + timedelta(days=days_to_first_friday)
            
            # The third Friday is 14 days after the first Friday
            expiry_dt = first_friday + timedelta(weeks=2)
            expiry_date = pd.Timestamp(expiry_dt)

        # Rule: Nth specific weekday of the month (e.g., 2nd Tuesday)
        elif isinstance(actual_rule_definition, dict) and "nth_weekday" in actual_rule_definition:
            try:
                n = actual_rule_definition["nth_weekday"]["n"] # e.g., 3 for third
                weekday_target = actual_rule_definition["nth_weekday"]["weekday"] # e.g., "Wednesday" or 2 (0=Mon)
                
                if isinstance(weekday_target, str):
                    weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                    weekday_int = weekday_map.get(weekday_target.lower())
                    if weekday_int is None:
                        logger.error(f"Invalid weekday name '{weekday_target}' in expiry rule for {symbol}.")
                        return None
                elif isinstance(weekday_target, int):
                    weekday_int = weekday_target
                else:
                    logger.error(f"Invalid weekday format '{weekday_target}' in expiry rule for {symbol}.")
                    return None

                first_day_of_month = date(contract_year, contract_month_num, 1)
                day_of_week_first_day = first_day_of_month.weekday() # 0=Monday

                days_to_first_target_weekday = (weekday_int - day_of_week_first_day + 7) % 7
                first_target_weekday_date = first_day_of_month + timedelta(days=days_to_first_target_weekday)
                
                # Add (n-1) weeks to the first target weekday
                nth_target_weekday_date = first_target_weekday_date + timedelta(weeks=n - 1)
                
                # Check if this date is still in the contract month
                if nth_target_weekday_date.month == contract_month_num:
                    expiry_date = pd.Timestamp(nth_target_weekday_date)
                else:
                    # This can happen if e.g. 5th Friday is requested but month doesn't have one
                    logger.warning(f"Calculated {n}th {weekday_target} ({nth_target_weekday_date}) is not in contract month {contract_month_num} for {symbol}. Trying prior week.")
                    # Fallback: try (n-1)th target weekday (which is (n-2) weeks from first)
                    # This logic might need refinement based on specific exchange rules for such edge cases.
                    # For now, let's assume it means the last such weekday in the month.
                    # A simpler approach for "last X day" would be to find first of NEXT month, subtract days.
                    # However, this rule is for Nth weekday. If 5th is requested and doesn't exist, it usually means the contract spec is different.
                    # For now, this will likely fail or give an unexpected result if n is too large.
                    expiry_date = pd.Timestamp(first_target_weekday_date + timedelta(weeks=n-2)) # try one week earlier
                    if expiry_date.month != contract_month_num:
                         logger.error(f"Could not determine {n}th {weekday_target} for {symbol} within contract month after fallback.")
                         return None
            except KeyError as ke:
                logger.error(f"Missing key {ke} in 'nth_weekday' expiry rule for {symbol}.")
                return None
            except Exception as e_nth:
                logger.error(f"Error processing 'nth_weekday' rule for {symbol}: {e_nth}")
                return None

        # Rule: Tuesday prior to the week of the third Wednesday of the month
        # Used by VIX Futures (VX) - This was a direct string check, now superseded by special_rule
        # elif actual_rule_definition == "tuesday_before_third_wednesday_week":
            # ... (logic for this was here, now handled by vx_style/VX_expiry below)

        # Rule: Business day prior to the business day prior to the 25th day of the month prior to the contract month
        # (Example logic for a complex rule like some CL contracts, needs verification)
        elif actual_rule_definition == "cl_style_approx": # This is an example, actual CL rules are complex
            month_prior = contract_month_num - 1
            year_prior = contract_year
            if month_prior == 0:
                month_prior = 12
                year_prior -= 1
            
            target_day = date(year_prior, month_prior, 25)
            
            # Get schedule around target_day
            schedule = trading_calendar.schedule(start_date=target_day - timedelta(days=10), end_date=target_day + timedelta(days=2))
            if schedule.empty:
                logger.warning(f"Trading calendar '{calendar_name}' returned empty schedule around {target_day} for {symbol}")
                return None

            # Business day strictly prior to the 25th (if 25th is not a business day) or the 25th if it is
            # Find first trading day <= 25th
            actual_25th_or_prior_trading_day = schedule[schedule.index <= pd.Timestamp(target_day)].index.max()
            
            # Business day prior to that
            idx_loc = schedule.index.get_loc(actual_25th_or_prior_trading_day)
            if idx_loc > 0:
                day_minus_1 = schedule.index[idx_loc - 1]
                # Business day prior to THAT (day_minus_2)
                if idx_loc > 1:
                    day_minus_2 = schedule.index[idx_loc - 2]
                    expiry_date = day_minus_2
                else:
                    logger.warning(f"Not enough trading days before {actual_25th_or_prior_trading_day} for CL style rule for {symbol}")
                    return None
            else:
                logger.warning(f"Cannot find trading day prior to {actual_25th_or_prior_trading_day} for CL style rule for {symbol}")
                return None

        # Rule: Specific day of the month (e.g., 15th), if it's a business day, else prior business day.
        elif isinstance(actual_rule_definition, dict) and "day_of_month" in actual_rule_definition:
            try:
                day = actual_rule_definition["day_of_month"]["day"]
                settlement_logic = actual_rule_definition["day_of_month"].get("settlement", "same_day_or_prior_bday") # 'same_day_or_prior_bday' or 'exact_day'

                # Tentative expiry date
                tentative_expiry_dt = date(contract_year, contract_month_num, day)

                if settlement_logic == "exact_day":
                    expiry_date = pd.Timestamp(tentative_expiry_dt)
                elif settlement_logic == "same_day_or_prior_bday":
                    # Check if this day is a trading day
                    schedule = trading_calendar.schedule(start_date=tentative_expiry_dt - timedelta(days=7), end_date=tentative_expiry_dt + timedelta(days=7))
                    if schedule.empty:
                         logger.warning(f"Trading calendar '{calendar_name}' returned empty schedule around {tentative_expiry_dt} for {symbol}")
                         return None
                    
                    # Find the closest trading session at or before the tentative_expiry_dt
                    valid_expiry_dates = schedule.index[schedule.index <= pd.Timestamp(tentative_expiry_dt)]
                    if not valid_expiry_dates.empty:
                        expiry_date = valid_expiry_dates.max()
                    else:
                        # Should not happen if calendar is correct and range is sufficient
                        logger.error(f"Could not find a valid trading day at or before {tentative_expiry_dt} for {symbol} on calendar {calendar_name}")
                        return None
                else:
                    logger.error(f"Unknown settlement logic '{settlement_logic}' for day_of_month rule for {symbol}")
                    return None

            except KeyError as ke:
                logger.error(f"Missing key {ke} in 'day_of_month' expiry rule for {symbol}.")
                return None
            except ValueError as ve: # e.g. day is 31 for a 30 day month
                logger.error(f"Date value error for 'day_of_month' rule for {symbol} (Year {contract_year}, Month {contract_month_num}, Day {day}): {ve}")
                return None


        # Rule: Specific date (YYYY-MM-DD) - for overrides or very specific contracts
        elif isinstance(actual_rule_definition, str) and re.match(r"\d{4}-\d{2}-\d{2}", actual_rule_definition):
            try:
                expiry_date = pd.Timestamp(actual_rule_definition)
                if expiry_date.year != contract_year or expiry_date.month != contract_month_num:
                    logger.warning(f"Specified expiry date {actual_rule_definition} for {symbol} does not match contract year/month {contract_year}-{contract_month_num}. Using specified date anyway.")
            except ValueError:
                logger.error(f"Invalid specific date format '{actual_rule_definition}' for {symbol}.")
                return None
        
        # Rule for energy futures (e.g., CL): N business days before a specific calendar day of the month
        elif isinstance(actual_rule_definition, dict) and "business_days_before_day_of_month" in actual_rule_definition:
            try:
                rule_details = actual_rule_definition["business_days_before_day_of_month"]
                days_b = int(rule_details["days_before"])
                ref_day_num = int(rule_details["day_of_month"])

                # Tentative reference date: the specified day in the contract month
                tentative_ref_date = pd.Timestamp(date(contract_year, contract_month_num, ref_day_num))

                # Find the last trading day at or before this tentative_ref_date
                schedule_around_ref = trading_calendar.schedule(start_date=tentative_ref_date - timedelta(days=10), 
                                                                end_date=tentative_ref_date + timedelta(days=2))
                if schedule_around_ref.empty:
                    logger.warning(f"Trading calendar '{calendar_name}' returned empty schedule around {tentative_ref_date.date()} for {symbol} (energy rule).")
                    return None
                
                actual_ref_trading_day = schedule_around_ref[schedule_around_ref.index <= tentative_ref_date].index.max()
                if pd.isna(actual_ref_trading_day):
                    logger.warning(f"Could not find actual reference trading day at or before {tentative_ref_date.date()} for {symbol} (energy rule).")
                    return None

                # Now find `days_b` business days before this actual_ref_trading_day
                # Get the index of actual_ref_trading_day in a broader schedule
                # We need a schedule that definitely includes days_b prior to actual_ref_trading_day
                extended_search_start = actual_ref_trading_day - timedelta(days=days_b + 15) # More buffer
                full_schedule = trading_calendar.schedule(start_date=extended_search_start, end_date=actual_ref_trading_day)
                
                if actual_ref_trading_day not in full_schedule.index:
                    logger.error(f"Logic error: actual_ref_trading_day {actual_ref_trading_day} not in its own schedule for {symbol}")
                    return None

                ref_day_idx = full_schedule.index.get_loc(actual_ref_trading_day)

                if ref_day_idx >= days_b:
                    expiry_date = full_schedule.index[ref_day_idx - days_b]
                else:
                    logger.warning(f"Not enough trading days ({ref_day_idx + 1}) before {actual_ref_trading_day.date()} to find {days_b} prior business days for {symbol} (energy rule).")
                    return None

            except KeyError as ke:
                logger.error(f"Missing key {ke} in 'business_days_before_day_of_month' rule for {symbol}.")
                return None
            except ValueError as ve: # For int() conversions or date creation
                logger.error(f"Value error in 'business_days_before_day_of_month' rule for {symbol}: {ve}")
                return None

        # Rule for metal futures (e.g., GC): N business days before the last business day of the contract month
        elif isinstance(actual_rule_definition, dict) and "business_days_before_last_bday_of_month" in actual_rule_definition:
            try:
                rule_details = actual_rule_definition["business_days_before_last_bday_of_month"]
                days_b = int(rule_details["days_before"])

                # Find all trading days in the contract month
                first_day_cm = date(contract_year, contract_month_num, 1)
                # To get the last day, go to first of next month and subtract one day
                if contract_month_num == 12:
                    last_day_cm = date(contract_year, 12, 31)
                else:
                    last_day_cm = date(contract_year, contract_month_num + 1, 1) - timedelta(days=1)
                
                month_schedule = trading_calendar.schedule(start_date=first_day_cm, end_date=last_day_cm)
                if month_schedule.empty:
                    logger.warning(f"Trading calendar '{calendar_name}' returned empty schedule for {contract_year}-{contract_month_num} for {symbol} (metal rule).")
                    return None
                
                # Last business day of the month is the last day in this schedule
                last_bday_of_month = month_schedule.index.max()

                # Now find `days_b` business days before this last_bday_of_month
                # The month_schedule itself can be used if it has enough entries
                last_bday_idx = month_schedule.index.get_loc(last_bday_of_month)

                if last_bday_idx >= days_b:
                    expiry_date = month_schedule.index[last_bday_idx - days_b]
                else:
                    logger.warning(f"Not enough trading days in {contract_year}-{contract_month_num} (found {last_bday_idx + 1} up to last bday) to find {days_b} prior business days for {symbol} (metal rule).")
                    return None

            except KeyError as ke:
                logger.error(f"Missing key {ke} in 'business_days_before_last_bday_of_month' rule for {symbol}.")
                return None
            except ValueError as ve: # For int() conversions or date creation
                logger.error(f"Value error in 'business_days_before_last_bday_of_month' rule for {symbol}: {ve}")
                return None

        # Rule for VX style futures based on CBOE VIX options expiry
        # This now checks the special_rule_type from the YAML config
        elif special_rule_type == "VX_expiry": 
            # This rule is complex: Settlement date is usually the Wednesday prior to or on the 3rd Friday of the *next* month's SPX options expiry.
            # CFE Rule: "Expire on the Wednesday that is 30 days prior to the third Friday of the calendar month immediately following the month in which the contract expires"
            # Example: VXH24 (Mar 2024 contract)
            # 1. Month immediately following contract month: April 2024
            # 2. Third Friday of April 2024: April 19, 2024
            # 3. 30 days prior to April 19, 2024: March 20, 2024 (This is a Wednesday) -> This is the settlement date.
            # The actual expiry/last trading day can be this settlement day or the day before. For simplicity, we'll use this as the expiry.

            # Determine the month immediately following the contract month
            following_month_num = contract_month_num + 1
            following_month_year = contract_year
            if following_month_num > 12:
                following_month_num = 1
                following_month_year += 1

            # Find the first day of that following month
            first_day_of_following_month = date(following_month_year, following_month_num, 1)

            # Find the third Friday of that following month
            day_of_week_first_day_fm = first_day_of_following_month.weekday()
            days_to_first_friday_fm = (4 - day_of_week_first_day_fm + 7) % 7
            first_friday_fm = first_day_of_following_month + timedelta(days=days_to_first_friday_fm)
            third_friday_fm = first_friday_fm + timedelta(weeks=2)

            # Expiry is 30 days prior to this third Friday
            expiry_dt = third_friday_fm - timedelta(days=30)
            
            # Ensure this expiry_dt is a Wednesday. If not, something is off or rule needs adjustment.
            if expiry_dt.weekday() != 2: # 2 = Wednesday
                 logger.warning(f"VX_STYLE calculated expiry {expiry_dt.strftime('%Y-%m-%d')} for {symbol} is not a Wednesday (it's {expiry_dt.strftime('%A')}). Check rule or calendar logic for {root_symbol}.")
                 # Optional: Adjust to nearest prior Wednesday if that's the convention
                 # expiry_dt = expiry_dt - timedelta(days=(expiry_dt.weekday() - 2 + 7) % 7)


            expiry_date = pd.Timestamp(expiry_dt)


        else:
            logger.warning(f"Unknown or unsupported expiry rule. Rule definition from YAML: '{actual_rule_definition}', Special rule type: '{special_rule_type}' for root symbol '{root_symbol}' (contract {symbol}).")
            return None

        if expiry_date:
            # Ensure the date is not on a weekend or holiday according to the specific calendar
            # Some rules might naturally fall on non-trading days if not careful.
            # Most rules above try to land on a specific weekday or use calendar logic.
            # Final check: if expiry_date is not a session on its calendar, roll back to previous session.
            
            # Make expiry_date timezone-naive for comparison with calendar session dates
            expiry_date_naive = expiry_date.tz_localize(None) if expiry_date.tz is not None else expiry_date

            # if not trading_calendar.is_session(expiry_date_naive):
            # Replacement for is_session:
            if trading_calendar.valid_days(start_date=expiry_date_naive.date(), end_date=expiry_date_naive.date()).empty:
                logger.debug(f"Calculated expiry {expiry_date_naive.strftime('%Y-%m-%d')} for {symbol} is not a trading session on {calendar_name}. Finding prior trading day.")
                # Get previous valid trading day using the calendar
                # Search a small window before the calculated expiry_date_naive
                search_end = expiry_date_naive
                search_start = expiry_date_naive - timedelta(days=7) # Look back up to 7 days
                
                # Get the schedule and find the last valid session <= expiry_date_naive
                schedule = trading_calendar.schedule(start_date=search_start, end_date=search_end)
                valid_days = schedule.index[schedule.index <= expiry_date_naive]
                
                if not valid_days.empty:
                    actual_expiry_day = valid_days.max()
                    if actual_expiry_day != expiry_date_naive:
                        logger.info(f"Adjusted expiry for {symbol} from {expiry_date_naive.strftime('%Y-%m-%d')} to previous trading day {actual_expiry_day.strftime('%Y-%m-%d')} based on calendar {calendar_name}.")
                        expiry_date = actual_expiry_day
                else:
                    logger.warning(f"Could not find a valid prior trading day for non-session expiry {expiry_date_naive.strftime('%Y-%m-%d')} for {symbol} on {calendar_name}.")
                    # Keep original calculated date, or handle as error? For now, keep.

            # Log the rule that was actually used
            rule_log_info = f"rule '{actual_rule_definition}'" if actual_rule_definition else f"special_rule '{special_rule_type}'"
            logger.info(f"Calculated expiration for {symbol} ({root_symbol}): {expiry_date.strftime('%Y-%m-%d')} based on {rule_log_info} and calendar '{calendar_name}'.")
            return expiry_date.normalize() # Normalize to remove time component, keep as pd.Timestamp
            
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
                
            # Prioritize 'specific_contract' if available, otherwise use 'symbol', finally fallback to original symbol.
            actual_symbol = symbol_info.get('specific_contract') or symbol_info.get('symbol') or symbol
            logger.debug(f"Found configuration for {symbol}. Actual symbol for fetch/store: {actual_symbol}")
            
            # Initial start date from config or script default
            config_start_date_str = symbol_info.get('start_date', self.default_start_date.strftime('%Y-%m-%d')) # Use default_start_date
            config_start_date = pd.Timestamp(config_start_date_str)

            # Determine specific contract start date if applicable
            # Corrected multi-line assignment with walrus operator
            is_specific_contract = (
                'specific_contract' in symbol_info or 
                (
                    (fut_match := re.compile(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$").match(actual_symbol)) is not None
                )
            )

            calculated_contract_start_date = config_start_date # Default to overall config start date
            if is_specific_contract:
                expiry_date = self.calculate_expiration_date(actual_symbol)
                logger.debug(f"Calculated expiry date for {actual_symbol}: {expiry_date}")
                if expiry_date:
                    # Ensure timezone-naive for comparison if expiry_date is tz-aware
                    expiry_date_naive = expiry_date.tz_localize(None) if expiry_date.tzinfo is not None else expiry_date
                    contract_start_calc = expiry_date_naive - timedelta(days=270) 
                    calculated_contract_start_date = max(contract_start_calc, config_start_date)
                    logger.info(f"Calculated base fetch start date for specific contract {actual_symbol}: {calculated_contract_start_date.date()}")
                else:
                    logger.warning(f"Could not calculate expiry for {actual_symbol}, using config start date {config_start_date.date()} as base for contract.")
            
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
                    target_table = freq_info['raw_table']
                    # ------------------------------------------------------------------ #

                    # Determine fetch_start_date for THIS specific frequency
                    latest_date_for_this_frequency = self.get_latest_date_in_db(actual_symbol, interval, unit)
                    logger.debug(f"Latest date in DB for {actual_symbol} ({freq_name}): {latest_date_for_this_frequency}")

                    # Base start date for this symbol (either specific contract start or general config start)
                    base_symbol_start_date = calculated_contract_start_date if is_specific_contract else config_start_date

                    fetch_start_date_for_this_frequency = None
                    if latest_date_for_this_frequency and not force and not update_history:
                        fetch_start_date_for_this_frequency = latest_date_for_this_frequency # Start from day after last data
                        # For intraday, we want to ensure we get the current day if latest_date is from a previous session.
                        # For daily, starting from latest_date (which is a date) means we'll fetch data *after* that date.
                        # If latest_date is today, fetch_data_since should handle getting today's new bars correctly.
                        logger.info(f"Found existing data for {actual_symbol} ({freq_name}) up to {latest_date_for_this_frequency}. Will fetch new data since then.")
                    elif update_history:
                        fetch_start_date_for_this_frequency = base_symbol_start_date
                        logger.info(f"Update history mode for {actual_symbol} ({freq_name}): fetching from {fetch_start_date_for_this_frequency.date()}")
                    elif force:
                        fetch_start_date_for_this_frequency = base_symbol_start_date
                        logger.info(f"Force mode for {actual_symbol} ({freq_name}): overwriting data from {fetch_start_date_for_this_frequency.date()}")
                    else: # No existing data for this frequency
                        fetch_start_date_for_this_frequency = base_symbol_start_date
                        logger.info(f"No existing data for {actual_symbol} ({freq_name}), fetching from {fetch_start_date_for_this_frequency.date()}")

                    if fetch_start_date_for_this_frequency > pd.Timestamp.now(tz='UTC').tz_localize(None): # Ensure timezone-naive comparison
                        logger.info(f"Fetch start date {fetch_start_date_for_this_frequency} for {actual_symbol} ({freq_name}) is in the future. Skipping this frequency.")
                        continue # Skip this frequency

                    logger.info(f"Fetching {freq_name} ({interval} {unit}) data for {actual_symbol} from {source} (effective start: {fetch_start_date_for_this_frequency}) -> Target Table: {target_table}")
                    
                    data = None # Initialize data DataFrame
                    # --- Source-specific Fetching --- #
                    if source == 'cboe':
                        # --- MODIFIED: Call direct CBOE download for daily ---
                        if unit == 'daily':
                            data = self._fetch_cboe_vx_daily(actual_symbol, interval, unit, fetch_start_date_for_this_frequency)
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
                             
                        data = self.fetch_data_since(actual_symbol, interval, unit, start_date=fetch_start_date_for_this_frequency)
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
                        logger.warning(f"No new {freq_name} data retrieved for {actual_symbol} from {source} starting {fetch_start_date_for_this_frequency}")
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
        """Get the configuration for a symbol, merging market_symbols.yaml (defaults) 
           and futures.yaml (overrides and continuous contract definitions).
        """
        self.logger.debug(f"_get_symbol_config checking for: {symbol}")
        final_config = {}

        # Identify if symbol is a specific continuous contract (e.g., @ES=101XN, @ES=101XN_d)
        # or a standard future (ESH25), or a base/root (ES, $VIX.X, SPY).
        is_specific_continuous_future = symbol.startswith('@') and '=' in symbol
        is_standard_future_contract = bool(re.match(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])([0-9]{1,2})$", symbol))
        
        base_symbol_for_futures_lookup = None
        if is_specific_continuous_future:
            base_symbol_for_futures_lookup = symbol.split('=')[0].lstrip('@') # e.g., ES from @ES=101XN
        elif is_standard_future_contract:
            # --- MODIFIED: Use _get_contract_details_from_symbol to get the correct root ---
            contract_details = self._get_contract_details_from_symbol(symbol)
            if contract_details:
                base_symbol_for_futures_lookup = contract_details[0] # e.g., VX from VXM25
            else:
                # Fallback, though should ideally not be needed if _get_contract_details_from_symbol is robust
                self.logger.warning(f"Could not parse contract details for '{symbol}' to determine base for futures lookup. Falling back to regex.")
                match = re.match(r"^([A-Z]{1,3})", symbol) 
                if match:
                    base_symbol_for_futures_lookup = match.group(1)
            # --- END MODIFICATION ---
        # else, symbol might be a base future root itself (ES), an index ($VIX.X), or equity (SPY)

        # 1. Handle specific continuous futures defined in futures.yaml
        if is_specific_continuous_future and base_symbol_for_futures_lookup and self.futures_config:
            future_product_config = self.futures_config.get('futures', {}).get(base_symbol_for_futures_lookup, {})
            for cc_def in future_product_config.get('continuous_contracts', []):
                if cc_def.get('identifier') == symbol:
                    self.logger.debug(f"Found explicit continuous config for '{symbol}' in futures.yaml.")
                    # This is the authoritative config for this specific continuous symbol.
                    # We might still want to layer some root defaults from market_symbols for exchange/calendar if not in cc_def.
                    base_defaults = {}
                    for item_cfg in self.config.get('futures', []):
                        if item_cfg.get('base_symbol') == base_symbol_for_futures_lookup and item_cfg.get('asset_type') == 'future_group':
                            base_defaults = item_cfg.copy()
                            break
                    merged_cc_def = {**base_defaults, **cc_def} # cc_def overrides base_defaults
                    merged_cc_def['_config_source'] = 'futures.yaml (continuous_contract list)'
                    return merged_cc_def
            # Also check continuous_contract_group if applicable (e.g. @VX=101XN)
            group_cfg = future_product_config.get('continuous_contract_group')
            if group_cfg and group_cfg.get('identifier_base') == symbol.split('=')[0]:
                # Regenerate to find match (simplified, actual generation is in Application class)
                month_codes = group_cfg.get('month_codes', [])
                settings_template = group_cfg.get('settings_code', '')
                for pos_str in month_codes:
                    # This logic for final_identifier needs to be identical to Application class one.
                    gen_id = f"{group_cfg['identifier_base']}={pos_str}{settings_template[1:]}" if settings_template.startswith('0') else f"{group_cfg['identifier_base']}={pos_str}{settings_template}"
                    if gen_id == symbol:
                        self.logger.debug(f"Found continuous config for '{symbol}' via group in futures.yaml.")
                        base_defaults = {}
                        for item_cfg in self.config.get('futures', []):
                            if item_cfg.get('base_symbol') == base_symbol_for_futures_lookup and item_cfg.get('asset_type') == 'future_group':
                                base_defaults = item_cfg.copy()
                                break
                        # Construct a cc_def like structure from group_cfg for this specific symbol
                        group_match_config = { 
                            **base_defaults, # Start with market_symbols defaults for the root
                            'identifier': symbol, 
                            'description': group_cfg.get('description_template', '').format(nth_month=pos_str),
                            'type': group_cfg.get('type', 'continuous_future'),
                            'default_source': group_cfg.get('default_source'),
                            'exchange': group_cfg.get('exchange', base_defaults.get('exchange')),
                            'calendar': group_cfg.get('calendar', base_defaults.get('calendar')),
                            'frequencies': group_cfg.get('frequencies', []),
                            'start_date': group_cfg.get('start_date'),
                            'method': group_cfg.get('method'),
                            'position': int(pos_str) if pos_str.isdigit() else None,
                            '_config_source': 'futures.yaml (continuous_contract_group)'
                        }
                        return group_match_config

        # 2. Handle indices & equities (from market_symbols.yaml only)
        for index_cfg in self.config.get('indices', []):
            if index_cfg.get('symbol') == symbol:
                self.logger.debug(f"Found index config for {symbol} in market_symbols.yaml")
                final_config = index_cfg.copy()
                final_config['_config_source'] = 'market_symbols.yaml (indices)'
                return final_config
        for equity_cfg in self.config.get('equities', []):
            if equity_cfg.get('symbol') == symbol:
                self.logger.debug(f"Found equity config for {symbol} in market_symbols.yaml")
                final_config = equity_cfg.copy()
                final_config['_config_source'] = 'market_symbols.yaml (equities)'
                return final_config

        # 3. Handle future_group defaults (from market_symbols.yaml) and overrides (from futures.yaml)
        # This applies if `symbol` is a base future root (e.g., "ES"), a standard contract (e.g., "ESH25"),
        # or a generic continuous future (e.g., "@ES" from market_symbols.yaml).

        # Determine the root symbol we are working with for lookup.
        # If `symbol` is "@ES", its base_symbol is "ES".
        # If `symbol` is "ESH25", its base_symbol is "ES".
        # If `symbol` is "ES", its base_symbol is "ES".
        effective_base_symbol = symbol 
        is_generic_continuous_at_symbol = symbol.startswith('@') and not is_specific_continuous_future # e.g. @ES, @NQ

        if is_standard_future_contract:
            effective_base_symbol = base_symbol_for_futures_lookup # ES from ESH25
        elif is_generic_continuous_at_symbol:
            effective_base_symbol = symbol.lstrip('@') # ES from @ES
        
        # Step 3a: Get defaults from market_symbols.yaml for the effective_base_symbol
        market_symbols_defaults = None
        for item_cfg in self.config.get('futures', []):
            condition1 = (item_cfg.get('base_symbol') == effective_base_symbol and item_cfg.get('asset_type') == 'future_group')
            condition2 = (item_cfg.get('symbol') == symbol and is_generic_continuous_at_symbol)
            if condition1 or condition2:
                market_symbols_defaults = item_cfg.copy()
                self.logger.debug(f"Found market_symbols.yaml defaults for base '{effective_base_symbol}' (or generic 'symbol={symbol}'): {list(market_symbols_defaults.keys())}")
                break
        
        # Ensure final_config is initialized based on whether defaults were found
        if market_symbols_defaults:
            final_config = market_symbols_defaults # It's already a copy
            final_config['_config_source'] = 'market_symbols.yaml (futures defaults)'
        else:
            final_config = {} # Initialize as empty dict if no defaults
            final_config['_config_source'] = 'market_symbols.yaml (no specific future_group defaults found)'

        # --- MODIFICATION FOR TEMPLATE INHERITANCE ---
        def deep_merge_dicts(base, override):
            """Recursively merge override dict into base dict."""
            merged = base.copy() # Start with a copy of the base
            for key, value in override.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = deep_merge_dicts(merged[key], value)
                else:
                    merged[key] = value
            return merged

        # Step 3b: Get overrides from futures.yaml for the effective_base_symbol
        # This now also handles template inheritance
        if self.futures_config and effective_base_symbol in self.futures_config.get('futures', {}):
            product_specific_config = self.futures_config['futures'][effective_base_symbol].copy()
            self.logger.debug(f"Found product-specific config in futures.yaml for '{effective_base_symbol}': {list(product_specific_config.keys())}")

            template_config = {}
            if 'inherit' in product_specific_config:
                template_name = product_specific_config.pop('inherit') # Remove 'inherit' key
                if template_name in self.futures_config.get('templates', {}):
                    template_config = self.futures_config['templates'][template_name].copy()
                    self.logger.debug(f"Loaded template '{template_name}' for '{effective_base_symbol}': {list(template_config.keys())}")
                    final_config['_config_source'] += f" (template: {template_name})"
                else:
                    self.logger.warning(f"Template '{template_name}' specified by '{effective_base_symbol}' not found in futures.yaml templates.")
            
            # Merge order: market_symbols_defaults -> template_config -> product_specific_config
            # Start with market_symbols_defaults (already in final_config)
            if template_config:
                final_config = deep_merge_dicts(final_config, template_config) # Merge template into market_symbols defaults
            
            # Now merge product_specific_config, which takes highest precedence
            final_config = deep_merge_dicts(final_config, product_specific_config)
            
            final_config['_config_source'] += ' + futures.yaml (specific product config / overrides)'
            
        elif not market_symbols_defaults and not is_specific_continuous_future: # Neither market_symbols nor futures.yaml had the base
             self.logger.warning(f"No base future config found in market_symbols.yaml or futures.yaml for: {effective_base_symbol} (derived from symbol {symbol})")
             return None # Cannot proceed if no base config for a future type

        # Step 3c: If the original symbol was a specific standard contract (e.g., ESH25), adjust final_config
        if is_standard_future_contract:
            final_config['symbol'] = symbol  # Ensure 'symbol' is the specific contract like ESH25
            final_config['specific_contract'] = symbol # Add a dedicated key for the specific contract
            final_config['_derived_from_standard_future'] = True
            final_config['_config_source'] += ' (adjusted for specific contract)'
            self.logger.debug(f"Adjusted config for standard future '{symbol}'. Final keys: {list(final_config.keys())}")
        
        # Step 3d: If symbol was a generic @ES type from market_symbols.yaml (already handled by final_config using market_symbols_defaults)
        elif is_generic_continuous_at_symbol:
            # The final_config should already be based on the market_symbols.yaml entry for "@ES"
            # and potentially overridden by an "ES:" block in futures.yaml if relevant keys were there.
            # Ensure `symbol` field in final_config is indeed the generic @Symbol.
            final_config['symbol'] = symbol 
            self.logger.debug(f"Final config for generic continuous '{symbol}'. Final keys: {list(final_config.keys())}")

        if not final_config: # Should not happen if logic is correct and symbol is valid type
            self.logger.error(f"_get_symbol_config: No configuration ultimately resolved for symbol: {symbol}")
            return None

        return final_config

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
        except AttributeError as ae:
             logger.error(f"MarketDataFetcher does not have 'generate_futures_contracts' method. Error: {ae}")
             logger.error(f"Attributes of self (MarketDataFetcher instance): {dir(self)}")
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
