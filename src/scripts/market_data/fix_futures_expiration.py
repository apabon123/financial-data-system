#!/usr/bin/env python
"""
Fix Futures Expiration Dates

This script checks and fixes expiration dates for ES and NQ futures contracts
in the financial_data.duckdb database. It removes any data points that extend
beyond the contract's proper expiration date.
"""

import os
import logging
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_futures_expiration.log')
    ]
)
logger = logging.getLogger("fix_futures_expiration")

# Database path
DB_PATH = "./data/financial_data.duckdb"

# Contract expiration mapping (month codes to month numbers)
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

def parse_futures_symbol(symbol):
    """Parse a futures symbol into components."""
    if len(symbol) < 3:
        return None, None, None
    
    # Extract base symbol, month code, and year code
    if len(symbol) == 5:  # e.g., "ESH25"
        base = symbol[:2]
        month_code = symbol[2]
        year_code = symbol[3:]
    elif len(symbol) == 6:  # e.g., "MESH25"
        base = symbol[:3]
        month_code = symbol[3]
        year_code = symbol[4:]
    else:
        return None, None, None
    
    return base, month_code, year_code

def calculate_expiration_date(symbol):
    """Calculate the expiration date for a futures contract."""
    base, month_code, year_code = parse_futures_symbol(symbol)
    
    if not all([base, month_code, year_code]):
        return None
    
    # Convert year code to full year
    if year_code.isdigit() and len(year_code) == 2:
        year = 2000 + int(year_code)
    else:
        return None
    
    # Get month number
    if month_code not in MONTH_CODES:
        return None
    
    month = MONTH_CODES[month_code]
    
    # For ES and NQ futures, expiration is typically the third Friday of the contract month
    if base in ('ES', 'NQ'):
        # Get the first day of the month
        first_day = datetime(year, month, 1)
        # Calculate days until first Friday (Friday is 4 in Python's weekday())
        days_until_friday = (4 - first_day.weekday()) % 7
        # First Friday of the month
        first_friday = first_day.day + days_until_friday
        # Third Friday is 14 days later
        third_friday = first_friday + 14
        
        # In case the third Friday calculation puts us into the next month
        if third_friday > 31:  # Simplified check
            return None
        
        return datetime(year, month, third_friday)
    
    return None

def fix_contract_expiration(symbol, conn):
    """Fix contract data by removing points after expiration date."""
    expiration_date = calculate_expiration_date(symbol)
    
    if not expiration_date:
        logger.warning(f"Could not calculate expiration date for {symbol}")
        return False
    
    logger.info(f"Calculated expiration date for {symbol}: {expiration_date.strftime('%Y-%m-%d')}")
    
    # Get current data for the contract
    query = f"""
        SELECT * FROM market_data 
        WHERE symbol = '{symbol}' 
        AND interval_value = 1 
        AND interval_unit = 'daily'
        ORDER BY timestamp
    """
    
    try:
        df = conn.execute(query).fetchdf()
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return False
        
        logger.info(f"Found {len(df)} rows for {symbol}")
        
        # Check for data beyond expiration date
        beyond_expiration = df[df['timestamp'] > expiration_date]
        
        if beyond_expiration.empty:
            logger.info(f"No data beyond expiration date for {symbol}")
            return False
        
        logger.warning(f"Found {len(beyond_expiration)} rows beyond expiration date for {symbol}")
        
        # Delete data beyond expiration date
        delete_query = f"""
            DELETE FROM market_data 
            WHERE symbol = '{symbol}' 
            AND interval_value = 1 
            AND interval_unit = 'daily'
            AND timestamp > '{expiration_date.strftime('%Y-%m-%d')}'
        """
        
        conn.execute(delete_query)
        conn.commit()
        
        # Verify deletion
        check_query = f"""
            SELECT COUNT(*) FROM market_data 
            WHERE symbol = '{symbol}' 
            AND interval_value = 1 
            AND interval_unit = 'daily'
            AND timestamp > '{expiration_date.strftime('%Y-%m-%d')}'
        """
        
        remaining = conn.execute(check_query).fetchone()[0]
        
        if remaining == 0:
            logger.info(f"Successfully removed {len(beyond_expiration)} rows beyond expiration date for {symbol}")
            return True
        else:
            logger.error(f"Failed to remove all data beyond expiration date for {symbol}")
            return False
        
    except Exception as e:
        logger.error(f"Error fixing contract expiration for {symbol}: {e}")
        return False

def check_all_contracts(conn):
    """Check all ES and NQ contracts for expiration date issues."""
    query = """
        SELECT DISTINCT symbol
        FROM market_data
        WHERE (symbol LIKE 'ES%' OR symbol LIKE 'NQ%')
        AND LENGTH(symbol) = 5
        AND interval_value = 1
        AND interval_unit = 'daily'
    """
    
    try:
        symbols = conn.execute(query).fetchdf()
        
        if symbols.empty:
            logger.warning("No ES or NQ contracts found in database")
            return
        
        logger.info(f"Found {len(symbols)} ES and NQ contracts to check")
        
        fixed_count = 0
        for _, row in symbols.iterrows():
            symbol = row['symbol']
            if fix_contract_expiration(symbol, conn):
                fixed_count += 1
        
        logger.info(f"Fixed {fixed_count} contracts with data beyond expiration date")
        
    except Exception as e:
        logger.error(f"Error checking contracts: {e}")

def main():
    """Main function to run the expiration date fix."""
    logger.info("Starting futures expiration date fix")
    
    try:
        # Check if database exists
        db_path = Path(DB_PATH)
        if not db_path.exists():
            logger.error(f"Database not found at {DB_PATH}")
            return
        
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        
        # Fix specific contracts
        logger.info("Fixing ESH25 contract")
        fix_contract_expiration("ESH25", conn)
        
        logger.info("Fixing NQH25 contract")
        fix_contract_expiration("NQH25", conn)
        
        # Optionally check all contracts
        check_all = input("Check all ES and NQ contracts for expiration issues? (y/n): ").lower() == 'y'
        if check_all:
            check_all_contracts(conn)
        
        # Close connection
        conn.close()
        
        logger.info("Futures expiration date fix complete")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main() 