#!/usr/bin/env python
"""
Script to build back-adjusted continuous futures contracts for ES and NQ futures.

The script determines the active contract based on:
1. Being within 7 trading days of the third Friday of the contract month
2. If the volume of the second future is greater than the nearest contract,
   the second contract becomes the first contract

Then it back-adjusts the prices to create a continuous futures contract.
"""

import os
import re
import sys
import logging
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('build_continuous_contracts.log')
    ]
)
logger = logging.getLogger('continuous_futures')

# Database path
DB_PATH = './data/financial_data.duckdb'

# Month code mapping
MONTH_CODES = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

# Reverse mapping for month codes
MONTH_TO_CODE = {v: k for k, v in MONTH_CODES.items()}

# Dictionary to store US market holidays
us_holidays = []  # Will be populated by get_us_holidays

def get_us_holidays(start_year=2000, end_year=None):
    """Get US market holidays for a given range of years."""
    if not end_year:
        end_year = datetime.now().year + 1
    
    # Use DuckDB to get holidays if available, otherwise use a placeholder implementation
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        query = f"""
        SELECT holiday_date 
        FROM holidays 
        WHERE calendar = 'NYSE' 
        AND holiday_date BETWEEN DATE '{start_year}-01-01' AND DATE '{end_year}-12-31'
        ORDER BY holiday_date
        """
        holidays_df = conn.execute(query).fetchdf()
        holidays = [row['holiday_date'] for row in holidays_df.to_dict('records')]
        return holidays
    except Exception as e:
        logger.warning(f"Could not retrieve holidays from database: {e}")
        # Return empty list if database query fails
        return []
    finally:
        conn.close()

def get_third_friday(year, month):
    """Get the third Friday of the given month and year."""
    # Start with the first day of the month
    d = date(year, month, 1)
    
    # Find the first Friday
    while d.weekday() != 4:  # 4 represents Friday
        d += timedelta(days=1)
    
    # Add two more weeks to get the third Friday
    d += timedelta(days=14)
    
    return d

def get_trading_days(start_date, end_date, holidays):
    """Get all trading days between start_date and end_date, excluding weekends and holidays."""
    # Generate all calendar days
    all_days = pd.date_range(start=start_date, end=end_date)
    
    # Filter out weekends and holidays
    trading_days = [day.date() for day in all_days if day.weekday() < 5 and day.date() not in holidays]
    
    return trading_days

def get_nth_trading_day(reference_date, n, holidays, forward=True):
    """
    Get the nth trading day before or after the reference date.
    
    Args:
        reference_date: The reference date
        n: Number of trading days to move
        holidays: List of holiday dates
        forward: If True, move forward n trading days, otherwise move backward
    
    Returns:
        The date that is n trading days before or after the reference date
    """
    trading_days = []
    
    if forward:
        # Generate dates starting from reference_date going forward
        current_date = reference_date
        while len(trading_days) <= n:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5 and current_date not in holidays:
                trading_days.append(current_date)
    else:
        # Generate dates starting from reference_date going backward
        current_date = reference_date
        while len(trading_days) <= n:
            current_date -= timedelta(days=1)
            if current_date.weekday() < 5 and current_date not in holidays:
                trading_days.append(current_date)
    
    return trading_days[n - 1] if trading_days else reference_date

def parse_contract_symbol(symbol):
    """Parse a futures contract symbol into its components."""
    match = re.match(r'^([A-Z]{2})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    base, month_code, year_code = match.groups()
    
    month = MONTH_CODES[month_code]
    year = 2000 + int(year_code) if int(year_code) < 50 else 1900 + int(year_code)
    
    return base, month, year, month_code, year_code

def get_contract_expiration(symbol, holidays):
    """
    Determine the expiration date for a futures contract.
    For ES and NQ, this is typically the third Friday of the contract month.
    """
    base, month, year, month_code, year_code = parse_contract_symbol(symbol)
    
    # Get the third Friday of the contract month
    third_friday = get_third_friday(year, month)
    
    # If the third Friday is a holiday, adjust to the previous trading day
    if third_friday in holidays:
        third_friday = get_nth_trading_day(third_friday, 1, holidays, forward=False)
    
    return third_friday

def get_volume_based_active_contract(symbols, date, conn, lookback_days=7):
    """
    Determine the active contract based on trading volume over lookback_days.
    
    Args:
        symbols: List of contract symbols ordered by expiration (nearest first)
        date: The date to check
        conn: DuckDB connection
        lookback_days: Number of trading days to consider for volume comparison
    
    Returns:
        The symbol of the active contract
    """
    if len(symbols) < 2:
        return symbols[0] if symbols else None
    
    nearest_contract = symbols[0]
    second_contract = symbols[1]
    
    # Get average volume over the lookback period for both contracts
    start_date = get_nth_trading_day(date, lookback_days, us_holidays, forward=False)
    
    query = f"""
    SELECT symbol, AVG(volume) as avg_volume
    FROM market_data
    WHERE symbol IN ('{nearest_contract}', '{second_contract}')
    AND timestamp BETWEEN DATE '{start_date}' AND DATE '{date}'
    AND interval_value = 1
    AND interval_unit = 'day'
    GROUP BY symbol
    """
    
    volume_df = conn.execute(query).fetchdf()
    
    # If either contract is missing, return the nearest contract
    if len(volume_df) < 2:
        return nearest_contract
    
    # Get volumes
    nearest_volume = volume_df.loc[volume_df['symbol'] == nearest_contract, 'avg_volume'].iloc[0]
    second_volume = volume_df.loc[volume_df['symbol'] == second_contract, 'avg_volume'].iloc[0]
    
    # If the second contract has more volume, it becomes the active contract
    if second_volume > nearest_volume:
        logger.info(f"Volume switchover on {date}: {second_contract} ({second_volume:.0f}) > {nearest_contract} ({nearest_volume:.0f})")
        return second_contract
    
    return nearest_contract

def determine_active_contract(base_symbol, date, conn, holidays):
    """
    Determine the active contract for a given date using the specified criteria:
    1. Within 7 trading days of the third Friday of the contract month
    2. If volume of second contract > nearest contract, second becomes active
    
    Args:
        base_symbol: The base symbol (ES or NQ)
        date: The date to check
        conn: DuckDB connection
        holidays: List of holiday dates
    
    Returns:
        The symbol of the active contract
    """
    # Get all contracts for this base symbol
    query = f"""
    SELECT DISTINCT symbol
    FROM market_data
    WHERE symbol LIKE '{base_symbol}[FGHJKMNQUVXZ][0-9][0-9]'
    AND interval_value = 1
    AND interval_unit = 'day'
    AND timestamp <= DATE '{date}'
    """
    
    contracts_df = conn.execute(query).fetchdf()
    
    if contracts_df.empty:
        logger.warning(f"No contracts found for {base_symbol} on {date}")
        return None
    
    # Get contract expiration dates
    contracts = []
    for symbol in contracts_df['symbol']:
        try:
            expiration = get_contract_expiration(symbol, holidays)
            contracts.append((symbol, expiration))
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
    
    # Sort by expiration date
    contracts.sort(key=lambda x: x[1])
    
    # Filter out expired contracts
    active_contracts = [c for c in contracts if c[1] >= date]
    
    if not active_contracts:
        logger.warning(f"No active contracts found for {base_symbol} on {date}")
        return None
    
    # Get contracts that expire after the current date
    potential_contracts = [c[0] for c in active_contracts]
    
    # Check if we are within 7 trading days of the third Friday of the nearest contract
    nearest_contract = active_contracts[0]
    
    # Get the 7th trading day before expiration
    seven_days_before = get_nth_trading_day(nearest_contract[1], 7, holidays, forward=False)
    
    # If the current date is after seven_days_before, we're within 7 trading days of expiration
    if date >= seven_days_before:
        # Check volume-based criteria if we have at least 2 active contracts
        if len(potential_contracts) >= 2:
            return get_volume_based_active_contract(potential_contracts, date, conn)
        else:
            return potential_contracts[0]
    
    return potential_contracts[0]

def back_adjust_continuous_contract(contracts_data, roll_dates):
    """
    Create a back-adjusted continuous contract from individual contract data.
    
    Args:
        contracts_data: Dictionary mapping contract symbols to DataFrames with price data
        roll_dates: Dictionary mapping dates to (old_contract, new_contract) tuples
    
    Returns:
        DataFrame with back-adjusted continuous data
    """
    # Create a list of all dates across all contracts
    all_dates = set()
    for df in contracts_data.values():
        all_dates.update(df.index)
    
    # Sort dates
    all_dates = sorted(all_dates)
    
    # Create a DataFrame with all dates
    continuous_df = pd.DataFrame(index=all_dates)
    continuous_df.index.name = 'timestamp'
    
    # Add columns for adjusted and unadjusted prices
    price_columns = ['open', 'high', 'low', 'close', 'settle']
    for col in price_columns:
        continuous_df[f'adj_{col}'] = np.nan
        continuous_df[col] = np.nan
    
    continuous_df['symbol'] = None
    continuous_df['adj_factor'] = 1.0  # Initialize adjustment factor
    
    # Sort roll dates in reverse chronological order (newest first)
    sorted_roll_dates = sorted(roll_dates.keys(), reverse=True)
    
    # Start with the latest contract (unadjusted)
    if not sorted_roll_dates:
        logger.warning("No roll dates found, cannot create continuous contract")
        return continuous_df
    
    latest_date = sorted_roll_dates[0]
    latest_roll = roll_dates[latest_date]
    current_contract = latest_roll[1]  # New contract after the latest roll
    
    # Get the dates for the latest contract
    mask = (pd.to_datetime(all_dates) >= pd.to_datetime(latest_date))
    latest_dates = [d for d in all_dates if mask[all_dates.index(d)]]
    
    # Fill in the latest contract data (no adjustment needed)
    if current_contract in contracts_data:
        latest_data = contracts_data[current_contract]
        for date in latest_dates:
            if date in latest_data.index:
                continuous_df.loc[date, 'symbol'] = current_contract
                for col in price_columns:
                    if col in latest_data.columns:
                        continuous_df.loc[date, col] = latest_data.loc[date, col]
                        continuous_df.loc[date, f'adj_{col}'] = latest_data.loc[date, col]
    
    # Process each roll date from newest to oldest
    adjustment_factor = 1.0
    for i in range(len(sorted_roll_dates)):
        roll_date = sorted_roll_dates[i]
        old_contract, new_contract = roll_dates[roll_date]
        
        # Calculate adjustment factor for this roll
        if old_contract in contracts_data and new_contract in contracts_data:
            old_data = contracts_data[old_contract]
            new_data = contracts_data[new_contract]
            
            if roll_date in old_data.index and roll_date in new_data.index:
                # Calculate adjustment between old and new contract at roll date
                settlement_diff = new_data.loc[roll_date, 'settle'] - old_data.loc[roll_date, 'settle']
                adjustment_factor += settlement_diff
                logger.info(f"Roll on {roll_date}: {old_contract} to {new_contract}, adjustment: {settlement_diff:.2f}")
            else:
                logger.warning(f"Missing price data for roll on {roll_date}")
                continue
        else:
            logger.warning(f"Missing contract data for roll on {roll_date}")
            continue
        
        # Get the date range for this contract
        start_date = sorted_roll_dates[i]
        end_date = sorted_roll_dates[i-1] if i > 0 else all_dates[-1]
        
        contract_dates = [d for d in all_dates if pd.to_datetime(start_date) <= pd.to_datetime(d) < pd.to_datetime(end_date)]
        
        # Fill in the contract data with adjustment
        for date in contract_dates:
            if date in old_data.index:
                continuous_df.loc[date, 'symbol'] = old_contract
                continuous_df.loc[date, 'adj_factor'] = adjustment_factor
                for col in price_columns:
                    if col in old_data.columns:
                        continuous_df.loc[date, col] = old_data.loc[date, col]
                        continuous_df.loc[date, f'adj_{col}'] = old_data.loc[date, col] + adjustment_factor
    
    # Fill in any missing dates with forward fill
    continuous_df = continuous_df.sort_index()
    
    return continuous_df

def build_continuous_contract(base_symbol, start_date, end_date, conn):
    """
    Build a continuous futures contract for a given base symbol.
    
    Args:
        base_symbol: The base symbol (ES or NQ)
        start_date: Start date for the continuous contract
        end_date: End date for the continuous contract
        conn: DuckDB connection
    
    Returns:
        DataFrame with the continuous contract data
    """
    logger.info(f"Building continuous contract for {base_symbol} from {start_date} to {end_date}")
    
    # Generate a list of dates to process
    trading_days = get_trading_days(start_date, end_date, us_holidays)
    
    # Track active contracts and roll dates
    current_active_contract = None
    active_contracts = {}  # date -> active_contract
    roll_dates = {}  # roll_date -> (old_contract, new_contract)
    
    # Determine active contract for each date
    for day in trading_days:
        active_contract = determine_active_contract(base_symbol, day, conn, us_holidays)
        
        if active_contract:
            active_contracts[day] = active_contract
            
            # Check for contract roll
            if current_active_contract and current_active_contract != active_contract:
                roll_dates[day] = (current_active_contract, active_contract)
                logger.info(f"Roll detected on {day}: {current_active_contract} -> {active_contract}")
            
            current_active_contract = active_contract
    
    # Get data for all contracts involved
    all_contracts = set([c for c in active_contracts.values()] + 
                       [roll[0] for roll in roll_dates.values()] + 
                       [roll[1] for roll in roll_dates.values()])
    
    contracts_data = {}
    for contract in all_contracts:
        query = f"""
        SELECT timestamp, symbol, open, high, low, close, settle, volume
        FROM market_data
        WHERE symbol = '{contract}'
        AND interval_value = 1
        AND interval_unit = 'day'
        AND timestamp BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY timestamp
        """
        
        df = conn.execute(query).fetchdf()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            contracts_data[contract] = df
    
    # Create the back-adjusted continuous contract
    continuous_df = back_adjust_continuous_contract(contracts_data, roll_dates)
    
    # Add the base symbol to the DataFrame
    continuous_df['base_symbol'] = base_symbol
    
    # Generate continuous symbol
    continuous_symbol = f"{base_symbol}c1"
    continuous_df['continuous_symbol'] = continuous_symbol
    
    return continuous_df

def save_continuous_contract(continuous_df, conn):
    """
    Save the continuous contract data to the database.
    
    Args:
        continuous_df: DataFrame with continuous contract data
        conn: DuckDB connection
    
    Returns:
        Number of rows inserted
    """
    # Reset index to get timestamp as a column
    df = continuous_df.reset_index()
    
    # Create continuous_contracts table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS continuous_contracts (
        timestamp DATE,
        continuous_symbol VARCHAR,
        base_symbol VARCHAR,
        underlying_symbol VARCHAR,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        settle DOUBLE,
        adj_open DOUBLE,
        adj_high DOUBLE,
        adj_low DOUBLE,
        adj_close DOUBLE,
        adj_settle DOUBLE,
        adj_factor DOUBLE,
        PRIMARY KEY (timestamp, continuous_symbol)
    )
    """)
    
    # Rename columns to match database schema
    column_mapping = {
        'timestamp': 'timestamp',
        'continuous_symbol': 'continuous_symbol',
        'base_symbol': 'base_symbol',
        'symbol': 'underlying_symbol',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'settle': 'settle',
        'adj_open': 'adj_open',
        'adj_high': 'adj_high',
        'adj_low': 'adj_low',
        'adj_close': 'adj_close',
        'adj_settle': 'adj_settle',
        'adj_factor': 'adj_factor'
    }
    
    # Select and rename columns
    insert_df = df[[col for col in column_mapping.keys() if col in df.columns]]
    insert_df.columns = [column_mapping[col] for col in insert_df.columns]
    
    # Delete existing data for this continuous symbol
    continuous_symbol = insert_df['continuous_symbol'].iloc[0]
    conn.execute(f"DELETE FROM continuous_contracts WHERE continuous_symbol = '{continuous_symbol}'")
    
    # Insert data
    rows_inserted = 0
    try:
        conn.execute("BEGIN TRANSACTION")
        for _, row in insert_df.iterrows():
            columns = ', '.join(row.index)
            placeholders = ', '.join(['?' for _ in range(len(row))])
            query = f"INSERT INTO continuous_contracts ({columns}) VALUES ({placeholders})"
            conn.execute(query, tuple(row))
            rows_inserted += 1
        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"Error inserting data: {e}")
        raise
    
    return rows_inserted

def main():
    """Main function to build continuous contracts for ES and NQ futures."""
    # Get US holidays
    global us_holidays
    us_holidays = get_us_holidays(2000)
    
    # Connect to the database
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Define parameters
        base_symbols = ['ES', 'NQ']
        start_date = date(2004, 1, 1)
        end_date = datetime.now().date()
        
        # Build continuous contracts for each symbol
        for base_symbol in base_symbols:
            continuous_df = build_continuous_contract(base_symbol, start_date, end_date, conn)
            
            # Save to database
            rows_inserted = save_continuous_contract(continuous_df, conn)
            logger.info(f"Saved {rows_inserted} rows for {base_symbol}c1")
            
            # Optionally save to CSV for inspection
            csv_path = f"{base_symbol}_continuous.csv"
            continuous_df.to_csv(csv_path)
            logger.info(f"Saved CSV to {csv_path}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main() 