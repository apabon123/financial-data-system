#!/usr/bin/env python
"""
Fix March 2025 Contracts

This script specifically fixes the ESH25 and NQH25 (March 2025) contracts,
removing incorrect data and redownloading them with the proper date range ending on March 21, 2025.
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from datetime import datetime, date
import re
import time

# Add project root to Python path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import market data agent
from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent

# Database path
DB_PATH = "./data/financial_data.duckdb"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_march_contracts.log')
    ]
)
logger = logging.getLogger("fix_march_contracts")

def check_contract_data(contract):
    """Check current data for the contract."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    try:
        query = f"""
            SELECT 
                COUNT(*) as row_count,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date
            FROM market_data
            WHERE symbol = '{contract}'
                AND interval_value = 1 
                AND interval_unit = 'daily'
        """
        result = conn.execute(query).fetchone()
        
        if result and result[0] > 0:
            logger.info(f"Contract {contract} exists with {result[0]} rows")
            logger.info(f"Date range: {result[1]} to {result[2]}")
            return True, result[0], result[1], result[2]
        else:
            logger.info(f"Contract {contract} not found in database")
            return False, 0, None, None
    except Exception as e:
        logger.error(f"Error checking contract data: {e}")
        return False, 0, None, None
    finally:
        conn.close()

def delete_contract_data(contract):
    """Delete all data for the specified contract."""
    conn = duckdb.connect(DB_PATH, read_only=False)
    
    try:
        # Begin transaction
        conn.begin()
        
        # Delete data
        query = f"""
            DELETE FROM market_data
            WHERE symbol = '{contract}'
                AND interval_value = 1 
                AND interval_unit = 'daily'
        """
        conn.execute(query)
        
        # Commit transaction
        conn.commit()
        logger.info(f"Successfully deleted all data for {contract}")
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting contract data: {e}")
        return False
    finally:
        conn.close()

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
                  AND interval_value = 1 AND interval_unit = 'daily'
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

def make_direct_api_request(symbol, start_date, end_date, agent, bars_back=None):
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
    
    # Build the API request URL and parameters
    endpoint = f"{agent.base_url}/marketdata/barcharts/{symbol}"
    
    if bars_back:
        # Use bars_back if specified
        params = {
            "interval": 1,
            "unit": "Daily",
            "barsback": bars_back,
        }
    else:
        # Use start and end dates otherwise
        params = {
            "interval": 1,
            "unit": "Daily",
            "barsback": 200,  # Use a large value to get all data in the date range
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

def fetch_contract(symbol, start_date, end_date):
    """Fetch data for a specific contract with proper date range."""
    logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
    
    # Create agent
    agent = TradeStationMarketDataAgent(database_path=DB_PATH, verbose=True)
    
    # Override save method
    agent.save_market_data = lambda df: simple_save_market_data(agent, df)
    
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

def main():
    """Main function to fix the March 2025 contracts."""
    logger.info("Starting fix for March 2025 contracts (ESH25 and NQH25)")
    
    # Contracts to fix
    contracts = ["ESH25", "NQH25"]
    
    # Define correct date range for March 2025 contracts
    # March contracts expire on the third Friday of March
    # For 2025, that's March 21st
    start_date = date(2024, 6, 1)  # Start from June 2024 to get complete history
    end_date = date(2025, 3, 21)   # Correct end date is March 21, 2025
    
    # Process each contract
    for contract in contracts:
        # Check current data
        exists, row_count, first_date, last_date = check_contract_data(contract)
        
        if exists:
            # Delete existing data
            logger.info(f"Deleting existing data for {contract}")
            if delete_contract_data(contract):
                logger.info(f"Successfully deleted {row_count} rows for {contract}")
            else:
                logger.error(f"Failed to delete data for {contract}, skipping")
                continue
        
        # Fetch correct data
        logger.info(f"Fetching correct data for {contract} from {start_date} to {end_date}")
        success, message = fetch_contract(contract, start_date, end_date)
        
        if success:
            logger.info(f"Successfully fixed {contract}: {message}")
        else:
            logger.error(f"Failed to fix {contract}: {message}")
        
        # Important: Wait a bit between operations to ensure connections are closed
        time.sleep(2)
    
    # Verify the fix after all operations are complete
    time.sleep(3)  # Wait to ensure all connections are properly closed
    
    logger.info("\nVerifying fix:")
    for contract in contracts:
        exists, row_count, first_date, last_date = check_contract_data(contract)
        if exists:
            logger.info(f"{contract}: {row_count} rows, date range {first_date} to {last_date}")
            
            # Check if the end date is correct
            if last_date and last_date.date() <= end_date:
                logger.info(f"✓ {contract} now has correct end date!")
            else:
                logger.warning(f"✗ {contract} end date still incorrect: {last_date}")
        
        # Wait between checks
        time.sleep(1)
    
    logger.info("Fix complete")

if __name__ == "__main__":
    main() 