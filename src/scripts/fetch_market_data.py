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
    
    def get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> datetime:
        """
        Get the nth occurrence of a weekday in a month.
        
        Args:
            year: The year
            month: The month (1-12)
            weekday: The day of week (0=Monday, 6=Sunday)
            n: Which occurrence (1=1st, 2=2nd, etc., negative counts from end)
            
        Returns:
            datetime: The date of the nth weekday
        """
        # Create first day of month
        first_day = datetime(year, month, 1)
        
        # Find first occurrence of weekday
        first_occurrence = first_day + timedelta(days=((weekday - first_day.weekday()) % 7))
        
        if n > 0:
            # Count forward from beginning
            target_date = first_occurrence + timedelta(weeks=n-1)
            # If we've gone into next month, there weren't enough occurrences
            if target_date.month != month:
                return None
        else:
            # Count backward from end
            # Find last occurrence first
            if first_occurrence.month != month:
                first_occurrence -= timedelta(days=7)
            while (first_occurrence + timedelta(days=7)).month == month:
                first_occurrence += timedelta(days=7)
            # Now count backward
            target_date = first_occurrence + timedelta(weeks=n+1)
        
        return target_date

    def is_holiday(self, date: datetime, calendar: str = None) -> bool:
        """
        Check if a date is a market holiday using the specified calendar.
        
        Args:
            date: The date to check
            calendar: Optional calendar name to use (defaults to config default)
            
        Returns:
            bool: True if holiday, False otherwise
        """
        # Get calendar to use
        if not calendar:
            calendar = self.config.get('settings', {}).get('holiday_calendar', 'NYSE')
        
        # Check if it's a weekend
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return True
        
        # Get holiday definitions for this calendar
        holiday_defs = self.config.get('holidays', {}).get(calendar, {})
        if not holiday_defs:
            logger.warning(f"No holiday definitions found for calendar: {calendar}")
            return False
        
        # Check fixed date holidays (MM-DD format)
        fixed_dates = holiday_defs.get('fixed_dates', [])
        date_str = date.strftime('%m-%d')
        if date_str in fixed_dates:
            return True
        
        # Check relative date holidays
        relative_dates = holiday_defs.get('relative_dates', [])
        for holiday in relative_dates:
            # Skip if not in the right month
            if holiday['month'] != date.month:
                continue
            
            # Get the target day of week (0=Monday, 6=Sunday)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_weekday = day_map.get(holiday['day_type'].lower())
            
            if target_weekday is None:
                logger.error(f"Invalid day_type in holiday definition: {holiday['day_type']}")
                continue
            
            # Get the occurrence (1=1st, 2=2nd, etc., -1=last)
            occurrence = holiday['occurrence']
            
            # Calculate the target date
            target_date = self.get_nth_weekday(date.year, date.month, target_weekday, occurrence)
            
            # Check if this is the target date
            if target_date and target_date.date() == date.date():
                return True
        
        return False

    def get_previous_business_day(self, date: datetime, calendar: str = None) -> datetime:
        """
        Get the previous business day before a date using the specified calendar.
        
        Args:
            date: The reference date
            calendar: Optional calendar name to use (defaults to config default)
            
        Returns:
            datetime: The previous business day
        """
        current = date
        while True:
            current = current - timedelta(days=1)
            if not self.is_holiday(current, calendar):
                return current

    def calculate_expiration_date(self, symbol: str) -> Optional[datetime]:
        """
        Calculate the expiration date for a futures contract based on config rules.
        
        Args:
            symbol: The futures contract symbol (e.g., 'ESH24')
            
        Returns:
            datetime: The expiration date, or None if invalid symbol
        """
        if len(symbol) < 4:
            return None
        
        # Extract components
        base_symbol = symbol[:-3]  # e.g., ES
        month_code = symbol[-3]    # e.g., H
        year_code = symbol[-2:]    # e.g., 24
        
        # Map month codes to months
        month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
        month = month_map.get(month_code)
        if not month:
            return None
        
        # Convert year code to full year
        year = 2000 + int(year_code)
        
        # Get contract config
        symbol_info = None
        for futures_config in self.config.get('futures', []):
            if futures_config['symbol'] == base_symbol:
                symbol_info = futures_config
                break
        
        if not symbol_info or 'expiry_rule' not in symbol_info:
            logger.warning(f"No expiry rule found for {base_symbol}, using default last day of month")
            # Default to last day of month
            next_month = datetime(year, month % 12 + 1, 1)
            return next_month - timedelta(days=1)
        
        expiry_rule = symbol_info['expiry_rule']
        calendar = symbol_info.get('holiday_calendar')
        
        # Handle business day rules (like CL and GC)
        if expiry_rule['day_type'] == 'business_day':
            if 'days_before' in expiry_rule:
                # Rule like CL: 3 business days before 25th of next month
                ref_month = month % 12 + 1
                ref_year = year if ref_month > month else year + 1
                ref_date = datetime(ref_year, ref_month, expiry_rule['reference_day'])
                target_date = ref_date
                for _ in range(expiry_rule['days_before']):
                    target_date = self.get_previous_business_day(target_date, calendar)
                return target_date
            elif 'days_from_end' in expiry_rule:
                # Rule like GC: 3rd last business day of month
                next_month = datetime(year, month % 12 + 1, 1)
                target_date = next_month - timedelta(days=1)
                for _ in range(expiry_rule['days_from_end']):
                    target_date = self.get_previous_business_day(target_date, calendar)
                return target_date
        
        # Convert day type to weekday number (0=Monday, 6=Sunday)
        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        weekday = day_map.get(expiry_rule['day_type'].lower())
        
        if weekday is None:
            logger.error(f"Invalid day_type in expiry rule for {base_symbol}")
            return None
        
        # If this is a reference-based rule (like VIX)
        if 'reference_day_type' in expiry_rule:
            ref_weekday = day_map.get(expiry_rule['reference_day_type'].lower())
            ref_number = expiry_rule['reference_day_number']
            # Get reference date (e.g., 3rd Friday)
            reference_date = self.get_nth_weekday(year, month, ref_weekday, ref_number)
            if not reference_date:
                return None
            # Calculate target date based on offset
            target_date = reference_date + timedelta(days=expiry_rule['day_number'])
        else:
            # Direct rule (like ES 3rd Friday)
            target_date = self.get_nth_weekday(year, month, weekday, expiry_rule['day_number'])
            if not target_date:
                return None
        
        # Adjust for holidays if specified
        if expiry_rule.get('adjust_for_holiday', False):
            while self.is_holiday(target_date, calendar):
                target_date = self.get_previous_business_day(target_date, calendar)
        
        return target_date

    def process_symbol(self, symbol: str, update_history: bool = False, force: bool = False) -> None:
        """Process a single symbol, fetching and storing its data.
        
        Args:
            symbol: The symbol to process
            update_history: If True, fetch from start_date in config to current date, skipping existing data
            force: If True, overwrite existing data in the database
        """
        logger.info(f"Processing {symbol}")
        
        # Find symbol configuration
        symbol_info = self.find_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"No configuration found for symbol {symbol}")
            return
            
        # Get existing data
        latest_date = self.get_latest_date_in_db(symbol)
        
        if latest_date and not force and not update_history:
            # Normal mode: fetch only new data since last date
            logger.info(f"Found existing data for {symbol}, last date: {latest_date}")
            start_date = latest_date + timedelta(days=1)
        elif update_history:
            # Update history mode: fetch from start_date in config to current date
            start_date = symbol_info.get('start_date', self.start_date)
            logger.info(f"Update history mode: fetching {symbol} from {start_date} to current date")
        else:
            # Force mode or no existing data: fetch from start_date in config
            start_date = symbol_info.get('start_date', self.start_date)
            if force:
                logger.info(f"Force mode: overwriting {symbol} data from {start_date} to current date")
            else:
                logger.info(f"No existing data for {symbol}, fetching from {start_date} to None")
        
        # Fetch data for each frequency
        for freq in symbol_info.get('frequencies', []):
            if freq == 'daily':
                interval = 1
                unit = 'daily'
            elif freq == '1min':
                interval = 1
                unit = 'minute'
            elif freq == '15min':
                interval = 15
                unit = 'minute'
            else:
                continue
                
            try:
                # Use the full contract symbol for fetching
                data = self.fetch_data_since(symbol, interval, unit, start_date)
                if data is not None and not data.empty:
                    if force:
                        # In force mode, delete existing data for this symbol and frequency
                        self.delete_existing_data(symbol, interval, unit)
                    self.save_to_db(data)
            except Exception as e:
                logger.error(f"Error processing {symbol} for {freq}: {str(e)}")
                continue
                
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
            WHERE symbol = '{symbol}' 
            AND interval_value = {interval} 
            AND interval_unit = '{unit}'
            """
            self.conn.execute(query)
            logger.info(f"Deleted existing data for {symbol} with interval {interval} {unit}")
        except Exception as e:
            logger.error(f"Error deleting data for {symbol}: {str(e)}")
            
    def find_symbol_info(self, symbol):
        """Find configuration info for a symbol."""
        # First check if it's a futures contract (e.g., ESH24, RTYM25)
        if len(symbol) > 2:
            # Try 3-character root first
            root_symbol = symbol[:3]
            for futures_config in self.config.get('futures', []):
                if futures_config.get('base_symbol') == root_symbol:
                    return dict(futures_config)
            
            # Then try 2-character root
            root_symbol = symbol[:2]
            for futures_config in self.config.get('futures', []):
                if futures_config.get('base_symbol') == root_symbol:
                    return dict(futures_config)
        
        # Then check equities
        for equity_config in self.config.get('equities', []):
            if equity_config.get('symbol') == symbol:
                return dict(equity_config)
                
        # Finally check futures again for root symbols
        for futures_config in self.config.get('futures', []):
            if futures_config.get('base_symbol') == symbol:
                return dict(futures_config)
                
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
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        # Ensure dates are timezone-naive for comparison
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        # Get symbol config
        symbol_config = None
        for futures_config in self.config.get('futures', []):
            if futures_config.get('base_symbol') == root_symbol:
                symbol_config = futures_config
                break
        
        if not symbol_config:
            logger.error(f"No configuration found for futures symbol {root_symbol}")
            return []
        
        # Get contract configuration
        num_active_contracts = symbol_config.get('num_active_contracts', 3)
        contract_config = symbol_config.get('historical_contracts', {})
        month_patterns = contract_config.get('patterns', [])
        cycle_type = contract_config.get('cycle_type', 'quarterly')  # quarterly, monthly, etc.
        
        # Map month codes to months
        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        
        # Get current date
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Find the current contract month based on cycle type
        current_contract_month = None
        
        if cycle_type == 'quarterly':
            # For quarterly contracts (e.g., ES: H,M,U,Z)
            for month_code in month_patterns:
                month_num = month_map.get(month_code)
                if month_num and month_num > current_month:
                    current_contract_month = month_code
                    break
        elif cycle_type == 'monthly':
            # For monthly contracts (e.g., VIX: all months)
            # Find the next contract month
            for month_code in month_patterns:
                month_num = month_map.get(month_code)
                if month_num and month_num > current_month:
                    current_contract_month = month_code
                    break
        
        # If no future month found, use the first month in the pattern
        if not current_contract_month:
            current_contract_month = month_patterns[0]
            if cycle_type == 'monthly':
                # For monthly contracts, we might need to increment the year
                current_year += 1
        
        # Generate contracts
        contracts = []
        
        # Add current contract
        year_str = str(current_year)[-2:]
        contracts.append(f"{root_symbol}{current_contract_month}{year_str}")
        
        # Add next contracts based on cycle type and num_active_contracts
        month_index = month_patterns.index(current_contract_month)
        for i in range(1, num_active_contracts):
            if cycle_type == 'quarterly':
                # For quarterly contracts, move to next quarter
                next_month_index = (month_index + i) % len(month_patterns)
                next_month_code = month_patterns[next_month_index]
                
                # If we've wrapped around to the beginning of the year, increment the year
                if next_month_index < month_index:
                    year_str = str(current_year + 1)[-2:]
                else:
                    year_str = str(current_year)[-2:]
            else:  # monthly
                # For monthly contracts, move to next month
                next_month_index = (month_index + i) % len(month_patterns)
                next_month_code = month_patterns[next_month_index]
                
                # Calculate the year based on how many months we've moved
                months_ahead = i
                year_offset = (current_month + months_ahead - 1) // 12
                year_str = str(current_year + year_offset)[-2:]
            
            contracts.append(f"{root_symbol}{next_month_code}{year_str}")
        
        # Add historical contracts if needed
        if start_date < current_date:
            # Generate contracts from start_date to current date
            historical_contracts = []
            historical_date = start_date
            
            while historical_date < current_date:
                year = historical_date.year
                month = historical_date.month
                
                # Find the contract month for this date based on cycle type
                contract_month = None
                if cycle_type == 'quarterly':
                    # For quarterly contracts, find the next quarter
                    for month_code in month_patterns:
                        month_num = month_map.get(month_code)
                        if month_num and month_num > month:
                            contract_month = month_code
                            break
                else:  # monthly
                    # For monthly contracts, use the next month
                    for month_code in month_patterns:
                        month_num = month_map.get(month_code)
                        if month_num and month_num > month:
                            contract_month = month_code
                            break
                
                # If no future month found, use the first month in the pattern
                if not contract_month:
                    contract_month = month_patterns[0]
                    if cycle_type == 'monthly':
                        year += 1
                
                # Format year as two digits
                year_str = str(year)[-2:]
                contract = f"{root_symbol}{contract_month}{year_str}"
                
                if contract not in contracts and contract not in historical_contracts:
                    historical_contracts.append(contract)
                
                # Move to next month
                if month == 12:
                    historical_date = datetime(year + 1, 1, 1)
                else:
                    historical_date = datetime(year, month + 1, 1)
            
            # Add historical contracts to the list
            contracts.extend(historical_contracts)
        
        return sorted(contracts)

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
                # Process a single symbol
                symbol_info = self.find_symbol_info(symbol)
                if symbol_info:
                    # Check if it's a futures contract
                    if symbol_info.get('type') == 'futures' or symbol_info.get('base_symbol'):
                        # Generate and process futures contracts
                        start = symbol_info.get('start_date', self.start_date)
                        contracts = self.generate_futures_contracts(symbol, start)
                        logger.info(f"Generated {len(contracts)} futures contracts for {symbol}")
                        for contract in contracts:
                            try:
                                self.process_symbol(contract, update_history, force)
                            except Exception as e:
                                logger.error(f"Error processing futures contract {contract}: {e}")
                                continue
                    else:
                        self.process_symbol(symbol, update_history, force)
                else:
                    logger.error(f"Symbol '{symbol}' not found in configuration")
            else:
                # Process all symbols
                # Process futures first
                for symbol_info in self.config.get('futures', []):
                    try:
                        # Generate and process futures contracts
                        start = symbol_info.get('start_date', self.start_date)
                        contracts = self.generate_futures_contracts(symbol_info['base_symbol'], start)
                        logger.info(f"Generated {len(contracts)} futures contracts for {symbol_info['base_symbol']}")
                        for contract in contracts:
                            try:
                                self.process_symbol(contract, update_history, force)
                            except Exception as e:
                                logger.error(f"Error processing futures contract {contract}: {e}")
                                continue
                    except Exception as e:
                        logger.error(f"Error processing futures symbol {symbol_info['base_symbol']}: {e}")
                        continue
                
                # Then process equities
                for symbol_info in self.config.get('equities', []):
                    try:
                        self.process_symbol(symbol_info['symbol'], update_history, force)
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
  
  # Update history for all symbols (from start_date to current)
  python fetch_market_data.py --updatehistory
  
  # Force update (overwrite existing data)
  python fetch_market_data.py --force
  
  # Force update for a specific symbol
  python fetch_market_data.py --symbol ES --force
        """
    )
    parser.add_argument('--symbol', type=str, help='Symbol to fetch data for (e.g., ES, NQ, CL)')
    parser.add_argument('--config', default='config/market_symbols.yaml', help='Path to the market_symbols.yaml configuration file (default: config/market_symbols.yaml)')
    parser.add_argument('--updatehistory', action='store_true', help='Update history from start_date in config to current date, skipping existing data')
    parser.add_argument('--force', action='store_true', help='Force update: overwrite existing data in the database')
    args = parser.parse_args()
    
    try:
        fetcher = MarketDataFetcher(config_path=args.config)
        if args.symbol:
            logger.info(f"Fetching data for symbol: {args.symbol}")
            fetcher.run(symbol=args.symbol, update_history=args.updatehistory, force=args.force)
        else:
            logger.info("Fetching data for all symbols")
            fetcher.run(update_history=args.updatehistory, force=args.force)
    except Exception as e:
        logger.error(f"Error running market data fetcher: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
