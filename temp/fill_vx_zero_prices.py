import duckdb
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime, timedelta
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('fill_vx_zero_prices')

def connect_to_db(db_path):
    """Connect to the DuckDB database."""
    logger.info(f"Connecting to database: {db_path}")
    try:
        conn = duckdb.connect(db_path, read_only=False)
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def load_vx_data(conn, start_date='2004-01-01', end_date='2006-12-31'):
    """Load VIX and VX continuous contract data from the database."""
    logger.info(f"Loading data from {start_date} to {end_date}...")
    
    # Query to get the data for all symbols
    query = f"""
    SELECT 
        timestamp AS date,
        symbol,
        settle,
        source,
        changed
    FROM market_data
    WHERE (symbol IN ('$VIX.X', 'VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5'))
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    AND interval_unit = 'day'
    ORDER BY timestamp, symbol
    """
    
    try:
        # Load the data
        df = conn.execute(query).fetchdf()
        logger.info(f"Loaded {len(df)} data points.")
        
        # Pivot the data to wide format
        settle_df = df.pivot(index='date', columns='symbol', values='settle')
        source_df = df.pivot(index='date', columns='symbol', values='source')
        changed_df = df.pivot(index='date', columns='symbol', values='changed')
        
        # Rename columns for clarity in source and changed dataframes
        source_df.columns = [f"{col}_source" for col in source_df.columns]
        changed_df.columns = [f"{col}_changed" for col in changed_df.columns]
        
        # Combine the dataframes
        result_df = pd.concat([settle_df, source_df, changed_df], axis=1)
        
        # Reset index to make date a column
        result_df = result_df.reset_index()
        
        return result_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def find_zero_price_days(df):
    """Find days where any of the VXc2-VXc5 contracts have a zero price."""
    zero_days = {}
    for contract in ['VXc2', 'VXc3', 'VXc4', 'VXc5']:
        if contract in df.columns:
            # Find days where the contract price is zero or NaN
            zero_mask = (df[contract] == 0) | df[contract].isna()
            zero_days[contract] = df.loc[zero_mask, 'date'].tolist()
            logger.info(f"Found {len(zero_days[contract])} days with zero or NaN prices for {contract}")
    
    return zero_days

def derive_price(row, contract, df):
    """
    Derive a price for a zero value using various methods.
    
    Methods (in order of preference):
    1. Linear interpolation between previous and next valid price
    2. Using previous day's price
    3. Using next day's price
    4. Deriving from VIX using a ratio based on other contracts
    5. Deriving from another continuous contract with a valid price
    
    Returns derived price and source information.
    """
    date = row['date']
    date_idx = df[df['date'] == date].index[0]
    
    # Method 1: Linear interpolation between previous and next valid price
    if date_idx > 0 and date_idx < len(df) - 1:
        # Find the nearest previous valid price
        prev_idx = date_idx - 1
        while prev_idx >= 0 and (df.loc[prev_idx, contract] == 0 or pd.isna(df.loc[prev_idx, contract])):
            prev_idx -= 1
        
        # Find the nearest next valid price
        next_idx = date_idx + 1
        while next_idx < len(df) and (df.loc[next_idx, contract] == 0 or pd.isna(df.loc[next_idx, contract])):
            next_idx += 1
        
        # If we found both previous and next valid prices, interpolate
        if prev_idx >= 0 and next_idx < len(df):
            prev_price = df.loc[prev_idx, contract]
            next_price = df.loc[next_idx, contract]
            prev_date = df.loc[prev_idx, 'date']
            next_date = df.loc[next_idx, 'date']
            
            # Calculate days between
            days_between = (next_date - prev_date).days
            days_from_prev = (date - prev_date).days
            
            # Interpolate
            interpolated_price = prev_price + (next_price - prev_price) * (days_from_prev / days_between)
            return interpolated_price, 'DERIVED_INTERPOLATION'
    
    # Method 2: Using previous day's price
    if date_idx > 0:
        prev_idx = date_idx - 1
        while prev_idx >= 0:
            prev_price = df.loc[prev_idx, contract]
            if prev_price != 0 and not pd.isna(prev_price):
                return prev_price, 'DERIVED_PREV_DAY'
            prev_idx -= 1
    
    # Method 3: Using next day's price
    if date_idx < len(df) - 1:
        next_idx = date_idx + 1
        while next_idx < len(df):
            next_price = df.loc[next_idx, contract]
            if next_price != 0 and not pd.isna(next_price):
                return next_price, 'DERIVED_NEXT_DAY'
            next_idx += 1
    
    # Method 4: Derive from VIX using ratio
    if '$VIX.X' in df.columns and not pd.isna(row['$VIX.X']):
        vix_price = row['$VIX.X']
        
        # Calculate the historical ratio between this contract and VIX
        valid_mask = (df[contract] > 0) & (~df[contract].isna()) & (~df['$VIX.X'].isna())
        if valid_mask.sum() > 10:  # Require at least 10 valid data points
            ratio = df.loc[valid_mask, contract].mean() / df.loc[valid_mask, '$VIX.X'].mean()
            derived_price = vix_price * ratio
            return derived_price, 'DERIVED_VIX_RATIO'
    
    # Method 5: Derive from another continuous contract
    for other_contract in ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']:
        if other_contract == contract:
            continue
        
        if other_contract in df.columns and not pd.isna(row[other_contract]) and row[other_contract] > 0:
            # Calculate the historical ratio between this contract and the other contract
            valid_mask = (df[contract] > 0) & (~df[contract].isna()) & (~df[other_contract].isna()) & (df[other_contract] > 0)
            if valid_mask.sum() > 10:  # Require at least 10 valid data points
                ratio = df.loc[valid_mask, contract].mean() / df.loc[valid_mask, other_contract].mean()
                derived_price = row[other_contract] * ratio
                return derived_price, f'DERIVED_{other_contract}_RATIO'
    
    # If all methods fail, return NaN and indicate no method worked
    return np.nan, 'DERIVED_FAILED'

def fill_zero_prices(df):
    """Fill zero prices with derived values."""
    # Make a copy of the original dataframe to avoid modifying during iteration
    df_copy = df.copy()
    
    # Track changes for reporting and database updates
    changes = []
    
    # Process each contract
    for contract in ['VXc2', 'VXc3', 'VXc4', 'VXc5']:
        if contract not in df.columns:
            logger.warning(f"Contract {contract} not found in the data")
            continue
            
        logger.info(f"Processing {contract}...")
        
        # Process each day with a zero price
        for date in find_zero_price_days(df).get(contract, []):
            row = df[df['date'] == date].iloc[0]
            
            # Derive a price
            derived_price, source = derive_price(row, contract, df)
            
            if not pd.isna(derived_price):
                logger.debug(f"Derived price for {contract} on {date.strftime('%Y-%m-%d')}: {derived_price:.2f} (source: {source})")
                
                changes.append({
                    'date': date,
                    'symbol': contract,
                    'original_value': row[contract],
                    'new_value': derived_price,
                    'source': source
                })
                
                # Update the dataframe copy
                df_copy.loc[df_copy['date'] == date, contract] = derived_price
                df_copy.loc[df_copy['date'] == date, f"{contract}_source"] = source
                df_copy.loc[df_copy['date'] == date, f"{contract}_changed"] = True
    
    logger.info(f"Filled {len(changes)} zero prices.")
    
    return df_copy, changes

def update_database(conn, changes):
    """Update the database with the derived prices."""
    if not changes:
        logger.info("No changes to update in the database.")
        return
    
    logger.info(f"Updating {len(changes)} derived prices in the database...")
    
    # First, get the existing rows to preserve any other column values
    symbols = [change['symbol'] for change in changes]
    dates = [change['date'] for change in changes]
    
    # Format dates for SQL query
    date_strings = [f"'{date}'" for date in dates]
    symbol_strings = [f"'{symbol}'" for symbol in symbols]
    
    # Build query to get existing data
    query = f"""
    SELECT * FROM market_data
    WHERE symbol IN ({','.join(symbol_strings)})
    AND timestamp IN ({','.join(date_strings)})
    """
    
    try:
        # Try to get existing data (some rows might not exist yet)
        existing_data = conn.execute(query).fetchdf()
    except:
        existing_data = pd.DataFrame()
    
    # Prepare data for batch update
    update_data = []
    for change in changes:
        # Check if we have existing data for this record
        existing_row = existing_data[
            (existing_data['symbol'] == change['symbol']) & 
            (existing_data['timestamp'] == change['date'])
        ] if not existing_data.empty else None
        
        # Default values for required fields
        new_row = {
            'timestamp': change['date'],
            'symbol': change['symbol'],
            'settle': change['new_value'],
            'source': change['source'],
            'changed': True,
            'interval_value': 1,  # daily
            'interval_unit': 'day',
            'adjusted': False,
            'quality': 100  # assuming good quality
        }
        
        # If we have existing data, preserve other fields
        if existing_row is not None and not existing_row.empty:
            row = existing_row.iloc[0]
            for col in row.index:
                if col not in ['timestamp', 'symbol', 'settle', 'source', 'changed'] and not pd.isna(row[col]):
                    new_row[col] = row[col]
        
        update_data.append(new_row)
    
    # Convert to DataFrame for easier batch update
    update_df = pd.DataFrame(update_data)
    
    # Update the database
    try:
        # Register the DataFrame as a view
        conn.register('update_data', update_df)
        
        # Get the column names from the DataFrame
        columns = ", ".join(update_df.columns)
        select_columns = ", ".join(update_df.columns)
        
        # Use INSERT OR REPLACE to update the data
        update_query = f"""
        INSERT OR REPLACE INTO market_data ({columns})
        SELECT {select_columns} FROM update_data
        """
        
        conn.execute(update_query)
        logger.info("Database updated successfully.")
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise

def fill_vx_zero_prices(db_path=None, start_date=None, end_date=None, dry_run=False, report=None):
    """
    Programmatic entry point for filling zero prices in VX continuous contracts.
    Can be called from other scripts or the command line.
    
    Args:
        db_path: Path to the DuckDB database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dry_run: Run without updating the database
        report: Path to save changes report (CSV)
        
    Returns:
        Number of changes made
    """
    # Set default values if not provided
    db_path = db_path or 'data/financial_data.duckdb'
    start_date = start_date or '2004-01-01'
    end_date = end_date or '2006-12-31'
    
    # Connect to the database
    conn = connect_to_db(db_path)
    num_changes = 0
    
    try:
        # Load the data
        logger.info(f"Loading data from {start_date} to {end_date}...")
        df = load_vx_data(conn, start_date, end_date)
        
        # Fill zero prices
        df_filled, changes = fill_zero_prices(df)
        num_changes = len(changes)
        
        # Print summary of changes
        if changes:
            logger.info("\nSummary of changes:")
            for contract in ['VXc2', 'VXc3', 'VXc4', 'VXc5']:
                contract_changes = [c for c in changes if c['symbol'] == contract]
                if contract_changes:
                    logger.info(f"  {contract}: {len(contract_changes)} zeros filled")
        
        # Save changes report if requested
        if report and changes:
            changes_df = pd.DataFrame(changes)
            changes_df.to_csv(report, index=False)
            logger.info(f"Changes report saved to {report}")
        
        # Update the database
        if not dry_run and changes:
            update_database(conn, changes)
        elif dry_run and changes:
            logger.info("Dry run - database not updated.")
        
    finally:
        # Close the database connection
        try:
            conn.close()
            logger.info("Database connection closed.")
        except:
            pass
    
    return num_changes

def main():
    parser = argparse.ArgumentParser(description='Fill zero prices in VX continuous contracts')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2006-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Run without updating the database')
    parser.add_argument('--report', type=str, default=None, help='Path to save changes report (CSV)')
    parser.add_argument('--apply', action='store_true', help='Apply changes to the database (equivalent to not using --dry-run)')
    
    args = parser.parse_args()
    
    # Convert --apply to not dry_run for backward compatibility
    dry_run = not args.apply if args.apply else args.dry_run
    
    # Call the programmatic entry point
    num_changes = fill_vx_zero_prices(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=dry_run,
        report=args.report
    )
    
    # Return a success indicator (0 = success)
    return 0 if num_changes > 0 or args.dry_run else 1

if __name__ == "__main__":
    sys.exit(main()) 