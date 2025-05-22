#!/usr/bin/env python
"""
Update Active Futures Contracts

This script identifies and updates active futures contracts based on configuration.
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
import yaml

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
        logging.FileHandler('update_active_futures.log')
    ]
)
logger = logging.getLogger("update_active_futures")

# Database path
DB_PATH = "./data/financial_data.duckdb"

def load_futures_config():
    """Load futures configuration from YAML files."""
    try:
        with open('config/market_symbols.yaml', 'r') as f:
            market_symbols = yaml.safe_load(f)
        
        with open('config/futures.yaml', 'r') as f:
            futures_config = yaml.safe_load(f)
        
        # Get futures from market_symbols.yaml
        futures = [f for f in market_symbols.get('futures', []) if 'base_symbol' in f]
        
        # Enrich with contract specs from futures.yaml
        for future in futures:
            base_symbol = future['base_symbol']
            if base_symbol in futures_config.get('futures', {}):
                contract_specs = futures_config['futures'][base_symbol]
                # Add contract specs if not already present
                if 'contract_specs' not in future:
                    future['contract_specs'] = {
                        'multiplier': contract_specs.get('multiplier'),
                        'point_value': contract_specs.get('point_value'),
                        'tick_size': contract_specs.get('tick_size'),
                        'tick_value': contract_specs.get('tick_value'),
                        'settlement_type': contract_specs.get('settlement_type')
                    }
        
        return futures
    except Exception as e:
        logger.error(f"Error loading futures configuration: {e}")
        return []

def get_active_contracts(future_config):
    """
    Determine the active futures contracts based on the current date.
    
    Args:
        future_config: Dictionary containing future configuration
        
    Returns:
        List of active contract symbols
    """
    base_symbol = future_config['base_symbol']
    patterns = future_config['historical_contracts']['patterns']
    num_active = future_config['num_active_contracts']
    
    today = datetime.now().date()
    current_month = today.month
    current_year = today.year
    
    # Map month codes to numbers
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    # Get valid patterns for this future
    valid_patterns = [p for p in patterns if p in month_codes]
    
    # Find the next active contract month
    next_contracts = []
    year = current_year
    month_idx = 0
    
    while len(next_contracts) < num_active:
        if month_idx >= len(valid_patterns):
            month_idx = 0
            year += 1
        
        month_code = valid_patterns[month_idx]
        month = month_codes[month_code]
        
        # Check if we're past the expiry for this month
        if month == current_month and year == current_year:
            # Calculate expiry date based on the future's expiry rule
            expiry_date = calculate_expiry_date(month, year, future_config['expiry_rule'])
            if today > expiry_date:
                month_idx += 1
                continue
        
        # Create contract symbol
        year_str = str(year)[-2:]
        contract = f"{base_symbol}{month_code}{year_str}"
        next_contracts.append(contract)
        
        month_idx += 1
    
    return next_contracts

def calculate_expiry_date(month, year, expiry_rule):
    """Calculate the expiry date based on the expiry rule."""
    if expiry_rule['day_type'] == 'friday':
        # Find the nth Friday of the month
        first_day = date(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day.replace(day=1 + days_until_friday)
        nth_friday = first_friday.replace(day=first_friday.day + (expiry_rule['day_number'] - 1) * 7)
        
        if expiry_rule.get('adjust_for_holiday', False):
            # Adjust for holidays (simplified - would need actual holiday calendar)
            while nth_friday.weekday() > 4:  # If it's a weekend
                nth_friday = nth_friday.replace(day=nth_friday.day - 1)
        
        return nth_friday
    
    elif expiry_rule['day_type'] == 'business_day':
        # For business day rules (like CL, GC)
        reference_day = date(year, month, expiry_rule['reference_day'])
        days_before = expiry_rule['days_before']
        
        # Move back the specified number of business days
        current_date = reference_day
        business_days = 0
        while business_days < days_before:
            current_date = current_date - timedelta(days=1)
            if current_date.weekday() < 5:  # Monday to Friday
                business_days += 1
        
        return current_date
    
    elif expiry_rule['day_type'] == 'wednesday' and expiry_rule.get('special_rule') == 'VX_expiry':
        # Special rule for VX futures
        # Expire 30 days before the 3rd Friday of the following month
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        
        first_day = date(next_year, next_month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day.replace(day=1 + days_until_friday)
        third_friday = first_friday.replace(day=first_friday.day + 14)
        
        return third_friday - timedelta(days=30)
    
    return None

def make_direct_api_request(symbol, start_date, end_date, agent, frequency='daily'):
    """
    Make a direct API request to fetch market data.
    
    Args:
        symbol: The contract symbol
        start_date: Start date for data
        end_date: End date for data
        agent: TradeStationMarketDataAgent instance
        frequency: Data frequency ('1min', '15min', or 'daily')
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

    # Map frequency to API parameters
    frequency_map = {
        '1min': {'interval': 1, 'unit': 'Minute'},
        '15min': {'interval': 15, 'unit': 'Minute'},
        'daily': {'interval': 1, 'unit': 'Daily'}
    }
    
    if frequency not in frequency_map:
        logger.error(f"Unsupported frequency: {frequency}")
        return None

    # Calculate barsback based on frequency and date range
    delta_days = (end_date - start_date).days + 1
    if frequency == 'daily':
        barsback = delta_days
    elif frequency == '1min':
        # Assume 6.5 trading hours per day = 390 minutes
        barsback = delta_days * 390
    elif frequency == '15min':
        # 6.5 hours = 26 bars per day
        barsback = delta_days * 26
    else:
        barsback = 2000  # fallback

    # Build the API request URL and parameters
    endpoint = f"{agent.base_url}/marketdata/barcharts/{symbol}"
    params = {
        **frequency_map[frequency],
        "barsback": barsback,
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
            df['interval_value'] = frequency_map[frequency]['interval']
            df['interval_unit'] = frequency_map[frequency]['unit'].lower()
            df['adjusted'] = True
            df['source'] = 'tradestation'
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

def fetch_single_contract(symbol, start_date, end_date, db_path, frequencies=None):
    """Fetch data for a single futures contract."""
    if frequencies is None:
        frequencies = ['daily']  # Default to daily if no frequencies specified
        
    logger.info(f"Fetching {symbol} data from {start_date} to {end_date} for frequencies: {frequencies}")
    
    # Create agent with the given database path
    agent = TradeStationMarketDataAgent(database_path=db_path, verbose=True)
    
    # Track results for each frequency
    results = []
    
    for frequency in frequencies:
        # Attempt to fetch data with retries
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Directly call the API with the specified frequency
                df = make_direct_api_request(symbol, start_date, end_date, agent, frequency)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {len(df)} rows for {symbol} ({frequency})")
                    # Save the data to the database
                    rows_saved = agent.save_market_data(df)
                    logger.info(f"Saved {rows_saved} rows to database for {symbol} ({frequency})")
                    success = True
                    results.append((frequency, True, f"Successfully fetched {len(df)} rows"))
                else:
                    logger.warning(f"No data returned for {symbol} ({frequency})")
                    retry_count += 1
                    time.sleep(2)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error fetching {symbol} ({frequency}): {e}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
        
        if not success:
            results.append((frequency, False, f"Failed to fetch after {max_retries} retries"))
    
    return results

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
    """Main function to update active futures contracts."""
    logger.info("Starting update of active futures contracts")
    
    # Override the save method
    override_save_market_data()
    
    # Load futures configuration
    futures_config = load_futures_config()
    if not futures_config:
        logger.error("Failed to load futures configuration")
        return
    
    # Track results
    results = []
    
    # Process each future
    for future_config in futures_config:
        base_symbol = future_config['base_symbol']
        logger.info(f"Processing {base_symbol} futures")
        
        # Get active contracts for this future
        active_contracts = get_active_contracts(future_config)
        logger.info(f"Active contracts for {base_symbol}: {', '.join(active_contracts)}")
        
        # Get frequencies from config
        frequencies = future_config.get('frequencies', ['daily'])
        
        # Fetch data for each contract
        for symbol in active_contracts:
            # Get contract dates (90 days back for active contracts)
            start_date = datetime.now().date() - timedelta(days=90)
            end_date = datetime.now().date()
            
            # Check if the contract exists in the database
            if check_if_contract_exists(symbol, DB_PATH):
                logger.info(f"Contract {symbol} exists, updating recent data")
            else:
                logger.info(f"Contract {symbol} is new, fetching all available data")
            
            # Fetch the data for all frequencies
            frequency_results = fetch_single_contract(symbol, start_date, end_date, DB_PATH, frequencies)
            results.extend([(symbol, freq, success, msg) for freq, success, msg in frequency_results])
    
    # Log summary of results
    logger.info("Update results summary:")
    for symbol, frequency, success, message in results:
        logger.info(f"{symbol} ({frequency}): {'SUCCESS' if success else 'FAILED'} - {message}")
    
    # Print summary of successful fetches
    successful = [r for r in results if r[2]]
    logger.info(f"Successfully updated {len(successful)} out of {len(results)} frequency/contract combinations")
    
    # Additional final message with next steps
    if len(successful) > 0:
        logger.info("Active contracts have been updated. You can now run view_futures_contracts.py to see the latest data.")
    else:
        logger.error("Failed to update any contracts. Check the error messages above for more information.")

if __name__ == "__main__":
    main() 