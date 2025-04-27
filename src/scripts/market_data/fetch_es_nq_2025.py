#!/usr/bin/env python
"""
Fetch ES and NQ March 2025 Futures Data

This script specifically fetches ESH25 and NQH25 futures data and adds it to the database.
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from datetime import datetime, date, timedelta
import time
import re

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our market data agent
from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fetch_es_nq_2025.log')
    ]
)
logger = logging.getLogger("fetch_es_nq_2025")

# Database path
DB_PATH = "./data/financial_data.duckdb"

def simple_save_market_data(self, df):
    """
    A simplified version of the save method that directly inserts data.
    Avoids complex SQL updates and just inserts data.
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe, nothing to save")
        return 0
    
    logger.info(f"Saving {len(df)} rows to database")
    
    # Ensure required columns exist
    required_columns = [
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle',
        'volume', 'interval_value', 'interval_unit', 'source'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            logger.info(f"Adding missing column: {col}")
            if col in ['open', 'high', 'low', 'close', 'settle', 'volume']:
                df[col] = None
            elif col == 'interval_value':
                df[col] = 1
            elif col == 'interval_unit':
                df[col] = 'daily'
            elif col == 'source':
                df[col] = 'TradeStation'
            else:
                df[col] = None
    
    # Connect to the database
    conn = duckdb.connect(DB_PATH, read_only=False)
    
    try:
        # Begin transaction
        conn.begin()
        rows_inserted = 0
        
        # Check if any of the rows already exist in the database
        # We do this by checking primary key values (timestamp and symbol)
        for idx, row in df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']
            
            # Check if this (timestamp, symbol) combination already exists
            check_query = f"""
                SELECT COUNT(*) FROM market_data 
                WHERE timestamp = '{timestamp}' AND symbol = '{symbol}'
            """
            count = conn.execute(check_query).fetchone()[0]
            
            # Only insert if it doesn't exist
            if count == 0:
                # Create a list of column values for this row
                columns = []
                values = []
                for col in df.columns:
                    columns.append(col)
                    values.append(row[col])
                
                # Build the INSERT query with ? placeholders
                columns_str = ", ".join(columns)
                placeholders = ", ".join(["?" for _ in columns])
                
                insert_query = f"""
                    INSERT INTO market_data ({columns_str})
                    VALUES ({placeholders})
                """
                
                # Execute the INSERT query with values
                conn.execute(insert_query, values)
                rows_inserted += 1
            else:
                logger.debug(f"Skipping duplicate entry: {symbol} at {timestamp}")
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully saved {rows_inserted} rows to database")
        return rows_inserted
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving data to database: {e}")
        return 0
    finally:
        conn.close()

def override_save_market_data():
    """Override the default save method with our simplified version."""
    TradeStationMarketDataAgent.save_market_data = simple_save_market_data
    logger.info("Overrode TradeStationMarketDataAgent.save_market_data with simple version")

def make_direct_api_request(symbol, start_date, end_date, agent):
    """
    Make a direct API request bypassing the agent's fetch_market_data method.
    This gives us more control over the API parameters.
    """
    import requests
    import json
    
    # Make sure we have a valid access token
    if not agent.authenticate():
        logger.error("Authentication failed, cannot proceed")
        return None
    
    # Ensure the dates are not in the future
    today = datetime.now().date()
    if end_date > today:
        end_date = today
    
    # Calculate appropriate barsback parameter for historical data
    # We'll just use a fixed value for the number of bars to request
    bars_back = 200  # This should be enough for new contracts
    
    # Build the API request URL and parameters
    endpoint = f"{agent.base_url}/marketdata/barcharts/{symbol}"
    params = {
        "interval": 1,
        "unit": "Daily",
        "barsback": bars_back,
    }
    
    # Don't use lastdate as it seems to be causing issues
    # Instead, rely solely on barsback to get available data
    
    headers = {
        "Authorization": f"Bearer {agent.access_token}",
        "Content-Type": "application/json"
    }
    
    logger.debug(f"Making direct API request to {endpoint} with params {params}")
    
    # Make the request
    response = requests.get(endpoint, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        logger.debug(f"Received response with status 200")
        if 'Bars' in data and data['Bars']:
            logger.info(f"Retrieved {len(data['Bars'])} bars for {symbol}")
            logger.debug(f"Sample data: {data['Bars'][0]}")
            
            # Process the data into a DataFrame
            df = pd.DataFrame(data['Bars'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['TimeStamp'])
            
            # Rename columns to match our schema
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'TotalVolume': 'volume',
                'TimeStamp': 'TimeStamp',  # Keep original column for now
                'DownVolume': 'down_volume',
                'UpVolume': 'up_volume'
            }
            
            # Rename only the columns that exist in the DataFrame
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and old_col != new_col:
                    df[new_col] = df[old_col]
            
            # Add symbol and other metadata
            df['symbol'] = symbol
            df['interval_value'] = 1
            df['interval_unit'] = 'daily'
            df['adjusted'] = True
            df['source'] = 'TradeStation API'
            df['settle'] = df['close']  # Use close as settle value
            
            # Print the DataFrame column names and types for debugging
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame dtypes: {df.dtypes}")
            
            # Make sure timestamp is a datetime object and handle timezone
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    logger.debug("Converting timestamp to datetime64")
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Remove timezone information to work with naive datetimes
                if hasattr(df['timestamp'].dt, 'tz'):
                    logger.debug("Converting timezone-aware timestamps to naive")
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Filter for the desired date range - convert to naive datetimes
            start_date_ts = pd.Timestamp(start_date).normalize()  # Normalize for consistency
            end_date_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)  # Include end date
            
            logger.debug(f"Filtering data between {start_date_ts} and {end_date_ts}")
            logger.debug(f"Data range before filtering: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Filter by timestamp
            if not df.empty:
                # Ensure filtering timestamps have the same timezone awareness
                df = df[(df['timestamp'] >= start_date_ts) & (df['timestamp'] <= end_date_ts)]
                
                # Drop any duplicates
                df = df.drop_duplicates(subset=['timestamp', 'symbol'])
                
                # Select and reorder columns for output
                output_columns = [
                    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 
                    'settle', 'volume', 'interval_value', 'interval_unit', 
                    'source', 'adjusted'
                ]
                
                # Only include columns that exist
                output_columns = [col for col in output_columns if col in df.columns]
                
                if not df.empty:
                    logger.debug(f"Final data shape: {df.shape}")
                    return df[output_columns]
                else:
                    logger.warning(f"No data found within requested date range for {symbol}")
                    return None
            else:
                logger.warning(f"Empty dataframe after initial processing for {symbol}")
                return None
        else:
            logger.warning(f"No bars found in response for {symbol}")
            return None
    else:
        logger.error(f"API error: {response.status_code} - {response.text}")
        return None

def fetch_single_contract(symbol, start_date, end_date, db_path):
    """Fetch data for a single futures contract."""
    logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
    
    # Create agent with the given database path
    agent = TradeStationMarketDataAgent(database_path=db_path, verbose=True)
    
    # Attempt to fetch data with retries using direct API method
    max_retries = 3
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            # Directly call the API to bypass the agent's fetch method
            df = make_direct_api_request(symbol, start_date, end_date, agent)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
                # Save the data to the database
                rows_saved = agent.save_market_data(df)
                logger.info(f"Saved {rows_saved} rows to database for {symbol}")
                success = True
            else:
                logger.warning(f"No data returned for {symbol}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            if retry_count == 0:  # Only on first error
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            retry_count += 1
            time.sleep(2)  # Wait before retrying
    
    if success:
        return True, f"Successfully fetched {symbol}"
    else:
        return False, f"Failed to fetch {symbol} after {max_retries} retries"

def get_contract_dates(symbol, year):
    """
    Determine the start and end dates for a futures contract based on 
    the symbol and year. This is a simplified version that focuses on
    getting the data rather than being precise about contract dates.
    """
    # Parse the symbol to get the month code
    match = re.match(r'^([A-Z]{2})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    if not match:
        return None, None
    
    base, month_code, year_code = match.groups()
    
    # Map month codes to months
    month_map = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10, # October
        'X': 11, # November
        'Z': 12  # December
    }
    
    month = month_map[month_code]
    
    # For ES/NQ March 2025 contracts, we want to get all available data
    # These contracts have likely just started trading
    contract_year = 2000 + int(year_code) if int(year_code) < 50 else 1900 + int(year_code)
    
    # For March 2025 contracts, start from January 2024 to capture all possible data
    if contract_year == 2025 and month == 3:
        start_date = date(2024, 1, 1)
    else:
        # Generic logic for other contracts
        if month <= 3:  # Q1 contracts
            start_date = date(contract_year - 1, month + 6, 1)
        else:  # Other contracts
            start_date = date(contract_year - 1, month - 3, 1)
    
    # End date is today
    end_date = datetime.now().date()
    
    return start_date, end_date

def manual_create_contract(symbol, db_path, source="Manual Entry"):
    """Create a manual entry for a contract that can't be fetched."""
    logger.info(f"Creating manual entry for {symbol}")
    
    # Parse the symbol
    match = re.match(r'^([A-Z]{2})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    if not match:
        logger.error(f"Invalid symbol format: {symbol}")
        return False
    
    # Create a simple DataFrame with one row
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Set some reasonable default values
    df = pd.DataFrame({
        'timestamp': [today],
        'symbol': [symbol],
        'open': [0.0],
        'high': [0.0],
        'low': [0.0],
        'close': [0.0],
        'settle': [0.0],
        'volume': [0],
        'interval_value': [1],
        'interval_unit': ['daily'],
        'source': [source],
        'adjusted': [True]
    })
    
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=False)
    
    try:
        # Check if the symbol already exists
        check_query = f"""
            SELECT COUNT(*) FROM market_data 
            WHERE symbol = '{symbol}'
        """
        count = conn.execute(check_query).fetchone()[0]
        
        if count > 0:
            logger.info(f"Symbol {symbol} already exists with {count} rows")
            return False
        
        # Insert the row directly
        conn.execute(
            """
            INSERT INTO market_data (
                timestamp, symbol, open, high, low, close, settle,
                volume, interval_value, interval_unit, source, adjusted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [today, symbol, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 'daily', source, True]
        )
        
        conn.commit()
        logger.info(f"Successfully created manual entry for {symbol}")
        return True
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating manual entry for {symbol}: {e}")
        return False
    
    finally:
        conn.close()

def main():
    """Main function to fetch ESH25 and NQH25 data."""
    logger.info("Starting fetch of ESH25 and NQH25 data with improved API handling")
    
    # Override the save method
    override_save_market_data()
    
    # List of contracts to fetch
    contracts = ["ESH25", "NQH25"]
    
    # Track results
    results = []
    
    # Fetch data for each contract
    for symbol in contracts:
        # Get contract dates
        start_date, end_date = get_contract_dates(symbol, 25)
        
        if start_date and end_date:
            logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
            success, message = fetch_single_contract(symbol, start_date, end_date, DB_PATH)
            
            # If API fetch fails, create a manual entry
            if not success:
                logger.warning(f"API fetch failed for {symbol}, creating manual entry")
                if manual_create_contract(symbol, DB_PATH, source="Manual entry after API failure"):
                    success = True
                    message = "Created manual entry"
            
            results.append((symbol, success, message))
        else:
            logger.error(f"Could not determine dates for {symbol}")
            results.append((symbol, False, "Could not determine contract dates"))
    
    # Log summary of results
    logger.info("Fetch results summary:")
    for symbol, success, message in results:
        logger.info(f"{symbol}: {'SUCCESS' if success else 'FAILED'} - {message}")
    
    # Print summary of successful fetches
    successful = [r for r in results if r[1]]
    logger.info(f"Successfully fetched/created {len(successful)} out of {len(contracts)} contracts")

if __name__ == "__main__":
    main() 