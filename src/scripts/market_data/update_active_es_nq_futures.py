#!/usr/bin/env python
"""
Update Active ES and NQ Futures Contracts

This script identifies and updates only the active ES and NQ futures contracts.
It determines which contracts are currently active based on trading calendar
and fetches the latest data for those contracts only.
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from datetime import datetime, date, timedelta
import re
import time

# Add project root to Python path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import market data agent
from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_active_es_nq_futures.log')
    ]
)
logger = logging.getLogger("update_active_es_nq_futures")

# Database path
DB_PATH = "./data/financial_data.duckdb"

# Month code mapping
MONTH_CODES = {
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

# Reverse mapping for generating contract symbols
MONTH_CODES_REVERSE = {v: k for k, v in MONTH_CODES.items()}

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

def get_active_contracts(base_symbols=['ES', 'NQ'], num_active=2):
    """
    Determine the active futures contracts based on the current date.
    Only returns contracts in the quarterly cycle: H, M, U, Z (March, June, September, December)
    
    Args:
        base_symbols: List of base symbols to get active contracts for
        num_active: Number of active contracts to return per symbol
        
    Returns:
        Dictionary mapping base symbols to lists of active contract symbols
    """
    today = datetime.now().date()
    current_month = today.month
    current_year = today.year
    
    # Define CME quarterly cycle months
    quarterly_months = [3, 6, 9, 12]  # H, M, U, Z (March, June, September, December)
    month_codes = {
        3: 'H',  # March
        6: 'M',  # June
        9: 'U',  # September
        12: 'Z'  # December
    }
    
    active_contracts = {}
    
    for base_symbol in base_symbols:
        contracts = []
        
        # Find the next active contract month in the quarterly cycle
        # The active contracts are the next two quarterly contracts
        # If we're past the 3rd Friday of the current quarterly month, move to the next one
        
        # First, find the nearest quarterly month
        next_quarterly_idx = 0
        for i, month in enumerate(quarterly_months):
            if month > current_month:
                next_quarterly_idx = i
                break
        
        # Get the next num_active quarterly contracts
        contract_count = 0
        idx = next_quarterly_idx
        year = current_year
        
        while contract_count < num_active:
            if idx >= len(quarterly_months):
                idx = 0
                year += 1
            
            month = quarterly_months[idx]
            month_code = month_codes[month]
            year_str = str(year)[-2:]
            
            # Check if we're past the 3rd Friday of the current month
            # If so and this is the first contract, we should skip to the next quarterly contract
            if contract_count == 0 and month == quarterly_months[next_quarterly_idx] and year == current_year:
                # Calculate the 3rd Friday of the month
                first_day = date(year, month, 1)
                # Find the first Friday
                days_until_friday = (4 - first_day.weekday()) % 7
                first_friday = first_day.replace(day=1 + days_until_friday)
                # Find the third Friday
                third_friday = first_friday.replace(day=first_friday.day + 14)
                
                # If today is past the 3rd Friday, move to the next quarterly contract
                if today > third_friday:
                    idx += 1
                    if idx >= len(quarterly_months):
                        idx = 0
                        year += 1
                    month = quarterly_months[idx]
                    month_code = month_codes[month]
                    year_str = str(year)[-2:]
            
            contract = f"{base_symbol}{month_code}{year_str}"
            contracts.append(contract)
            
            contract_count += 1
            idx += 1
        
        active_contracts[base_symbol] = contracts
    
    return active_contracts

def get_contract_dates(symbol):
    """
    Determine the start and end dates for a futures contract.
    For an active contract, we look back 90 days to ensure we don't miss data
    if the script hasn't been run in a while.
    """
    # Parse the symbol
    match = re.match(r'^([A-Z]{2})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    if not match:
        logger.error(f"Invalid symbol format: {symbol}")
        return None, None
    
    base, month_code, year_code = match.groups()
    
    month = MONTH_CODES[month_code]
    year = 2000 + int(year_code) if int(year_code) < 50 else 1900 + int(year_code)
    
    # For active contracts, we fetch data from 90 days ago to ensure we don't miss any
    # if the script hasn't been run in a while
    today = datetime.now().date()
    start_date = today - timedelta(days=90)
    
    return start_date, today

def make_direct_api_request(symbol, start_date, end_date, agent):
    """
    Make a direct API request to fetch market data.
    """
    import requests
    
    # Make sure we have a valid access token
    if not agent.authenticate():
        logger.error("Authentication failed, cannot proceed")
        return None
    
    # Ensure the dates are not in the future
    today = datetime.now().date()
    if end_date > today:
        end_date = today
    
    # Calculate appropriate barsback parameter for historical data
    # We'll use a fixed value for recent data
    bars_back = 20  # This should be enough for recent data
    
    # Build the API request URL and parameters
    endpoint = f"{agent.base_url}/marketdata/barcharts/{symbol}"
    params = {
        "interval": 1,
        "unit": "Daily",
        "barsback": bars_back,
    }
    
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
            
            # Make sure timestamp is a datetime object and handle timezone
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Remove timezone information to work with naive datetimes
                if hasattr(df['timestamp'].dt, 'tz'):
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Filter for the desired date range
            start_date_ts = pd.Timestamp(start_date).normalize()
            end_date_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
            
            # Filter by timestamp
            if not df.empty:
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
    
    # Attempt to fetch data with retries
    max_retries = 3
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            # Directly call the API
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
            retry_count += 1
            time.sleep(2)  # Wait before retrying
    
    if success:
        return True, f"Successfully fetched {symbol} ({len(df)} rows)"
    else:
        return False, f"Failed to fetch {symbol} after {max_retries} retries"

def check_if_contract_exists(symbol, db_path):
    """Check if a contract already exists in the database."""
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        query = f"""
            SELECT COUNT(*) FROM market_data
            WHERE symbol = '{symbol}'
        """
        count = conn.execute(query).fetchone()[0]
        conn.close()
        
        return count > 0
    except Exception as e:
        logger.error(f"Error checking if contract exists: {e}")
        conn.close()
        return False

def main():
    """Main function to update active ES and NQ futures contracts."""
    logger.info("Starting update of active ES and NQ futures contracts")
    
    # Override the save method
    override_save_market_data()
    
    # Get active contracts
    active_contracts = get_active_contracts(['ES', 'NQ'], num_active=2)
    
    # Flatten the contracts list
    all_contracts = []
    for base_symbol, contracts in active_contracts.items():
        all_contracts.extend(contracts)
    
    # Log the contracts we'll be updating
    logger.info(f"Updating the following contracts: {', '.join(all_contracts)}")
    
    # Track results
    results = []
    
    # Fetch data for each contract
    for symbol in all_contracts:
        # Get contract dates
        start_date, end_date = get_contract_dates(symbol)
        
        if start_date and end_date:
            # Check if the contract exists in the database
            if check_if_contract_exists(symbol, DB_PATH):
                logger.info(f"Contract {symbol} exists, updating recent data")
            else:
                logger.info(f"Contract {symbol} is new, fetching all available data")
            
            # Fetch the data
            success, message = fetch_single_contract(symbol, start_date, end_date, DB_PATH)
            results.append((symbol, success, message))
        else:
            logger.error(f"Could not determine dates for {symbol}")
            results.append((symbol, False, "Could not determine contract dates"))
    
    # Log summary of results
    logger.info("Update results summary:")
    for symbol, success, message in results:
        logger.info(f"{symbol}: {'SUCCESS' if success else 'FAILED'} - {message}")
    
    # Print summary of successful fetches
    successful = [r for r in results if r[1]]
    logger.info(f"Successfully updated {len(successful)} out of {len(all_contracts)} contracts")
    
    # Additional final message with next steps
    if len(successful) > 0:
        logger.info("Active contracts have been updated. You can now run view_futures_contracts.py to see the latest data.")
    else:
        logger.error("Failed to update any contracts. Check the error messages above for more information.")

if __name__ == "__main__":
    main() 