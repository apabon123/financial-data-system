#!/usr/bin/env python
"""
Script to view continuous futures contracts in the database.
This allows inspection of the back-adjusted continuous contracts
and verification of the roll dates and adjustment factors.
"""

import os
import sys
import logging
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('view_continuous')

# Database path
DB_PATH = './data/financial_data.duckdb'

def get_continuous_contracts(conn):
    """
    Get the list of available continuous contracts in the database.
    
    Args:
        conn: DuckDB connection
    
    Returns:
        List of continuous contract symbols
    """
    query = """
    SELECT DISTINCT continuous_symbol
    FROM continuous_contracts
    ORDER BY continuous_symbol
    """
    
    df = conn.execute(query).fetchdf()
    
    return df['continuous_symbol'].tolist() if not df.empty else []

def get_continuous_data(conn, symbol, start_date=None, end_date=None):
    """
    Get data for a specific continuous contract or all contracts for a base symbol.
    
    Args:
        conn: DuckDB connection
        symbol: Continuous contract symbol (e.g., 'ESc1') or base symbol pattern ('@ES')
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        DataFrame with continuous contract data
    """
    # Determine filter type
    if symbol.startswith('@'):
        base_pattern = symbol + "=%" # Convert @ES -> @ES=%
        where_clause = f"WHERE continuous_symbol LIKE '{base_pattern}'"
        logger.info(f"Filtering continuous data for base symbol pattern: {base_pattern}")
    else:
        where_clause = f"WHERE continuous_symbol = '{symbol}'"
        logger.info(f"Fetching continuous data for specific symbol: {symbol}")
        
    query = f"""
    SELECT 
        timestamp, 
        continuous_symbol, 
        underlying_symbol, 
        open, high, low, close, settle,
        adj_open, adj_high, adj_low, adj_close, adj_settle,
        adj_factor
    FROM continuous_contracts
    {where_clause}
    """
    
    if start_date:
        query += f" AND timestamp >= DATE '{start_date}'"
    
    if end_date:
        query += f" AND timestamp <= DATE '{end_date}'"
    
    query += " ORDER BY timestamp"
    
    df = conn.execute(query).fetchdf()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def get_roll_dates(df):
    """
    Identify roll dates from continuous contract data.
    
    Args:
        df: DataFrame with continuous contract data
    
    Returns:
        List of (roll_date, old_contract, new_contract) tuples
    """
    if df.empty:
        return []
    
    # Find where the underlying symbol changes
    rolls = []
    prev_symbol = None
    for i, row in df.iterrows():
        if prev_symbol and row['underlying_symbol'] != prev_symbol:
            rolls.append((row['timestamp'].date(), prev_symbol, row['underlying_symbol']))
        prev_symbol = row['underlying_symbol']
    
    return rolls

def analyze_continuous_contract(df, rolls):
    """
    Analyze a continuous contract and print summary information.
    
    Args:
        df: DataFrame with continuous contract data
        rolls: List of roll date tuples
    """
    if df.empty:
        logger.warning("No data found for this continuous contract")
        return
    
    # Basic statistics
    first_date = df['timestamp'].min().date()
    last_date = df['timestamp'].max().date()
    days_span = (last_date - first_date).days
    trading_days = len(df)
    
    # Calculate returns
    df['daily_return'] = df['adj_close'].pct_change()
    df['daily_return_non_adj'] = df['close'].pct_change()
    
    # Volatility (annualized standard deviation of returns)
    ann_vol = df['daily_return'].std() * (252 ** 0.5) * 100  # Annualize and convert to percentage
    
    # Print summary
    logger.info(f"Continuous Contract: {df['continuous_symbol'].iloc[0]}")
    logger.info(f"Date Range: {first_date} to {last_date} ({days_span} days, {trading_days} trading days)")
    logger.info(f"Number of Contract Rolls: {len(rolls)}")
    logger.info(f"Annualized Volatility: {ann_vol:.2f}%")
    logger.info(f"Latest price (adjusted): {df['adj_close'].iloc[-1]:.2f}")
    
    # Price range
    min_price = df['adj_close'].min()
    max_price = df['adj_close'].max()
    logger.info(f"Price Range (adjusted): {min_price:.2f} to {max_price:.2f} (Spread: {max_price - min_price:.2f})")
    
    # Print roll dates
    if rolls:
        logger.info("\nRoll Dates:")
        for roll_date, old_contract, new_contract in rolls:
            # Get roll date data
            roll_data = df[df['timestamp'].dt.date == roll_date]
            if not roll_data.empty:
                row = roll_data.iloc[0]
                adj_factor = row['adj_factor']
                logger.info(f"  {roll_date}: {old_contract} to {new_contract} (Adj Factor: {adj_factor:.2f})")
            else:
                logger.info(f"  {roll_date}: {old_contract} to {new_contract}")
    
    # Calculate returns around roll dates
    if rolls:
        logger.info("\nReturns around roll dates:")
        
        for roll_date, old_contract, new_contract in rolls:
            # Get 5 days before and after roll date
            roll_idx = df[df['timestamp'].dt.date == roll_date].index
            if len(roll_idx) > 0:
                roll_idx = roll_idx[0]
                
                # Get data for the period around the roll date
                start_idx = max(0, roll_idx - 5)
                end_idx = min(len(df) - 1, roll_idx + 5)
                period_df = df.iloc[start_idx:end_idx + 1]
                
                # Calculate cumulative return before and after roll
                before_cum_return = period_df.iloc[:roll_idx - start_idx + 1]['daily_return'].cumsum().iloc[-1] * 100
                after_cum_return = period_df.iloc[roll_idx - start_idx:]['daily_return'].cumsum().iloc[-1] * 100
                
                logger.info(f"  {roll_date}: 5-day Before: {before_cum_return:.2f}%, 5-day After: {after_cum_return:.2f}%")

def plot_continuous_contract(df, rolls, output_file=None):
    """
    Plot a continuous contract with roll dates marked.
    
    Args:
        df: DataFrame with continuous contract data
        rolls: List of roll date tuples
        output_file: Optional file path to save the plot
    """
    if df.empty:
        logger.warning("No data found for plotting")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create a date index for the DataFrame
    df = df.set_index('timestamp')
    
    # Plot adjusted and unadjusted prices
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['adj_close'], label='Adjusted Close')
    plt.plot(df.index, df['close'], label='Unadjusted Close', alpha=0.5)
    
    # Add markers for roll dates
    for roll_date, _, _ in rolls:
        plt.axvline(x=pd.to_datetime(roll_date), color='r', linestyle='--', alpha=0.3)
    
    plt.title(f"Continuous Contract: {df['continuous_symbol'].iloc[0]}")
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot adjustment factors
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['adj_factor'], label='Adjustment Factor')
    
    # Add markers for roll dates
    for roll_date, _, _ in rolls:
        plt.axvline(x=pd.to_datetime(roll_date), color='r', linestyle='--', alpha=0.3)
    
    plt.ylabel('Adjustment Factor')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    """Main function to view continuous contracts."""
    # Connect to the database
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    try:
        # Check if continuous_contracts table exists
        try:
            conn.execute("SELECT 1 FROM continuous_contracts LIMIT 1")
        except Exception as e:
            logger.error(f"continuous_contracts table not found. Run build_continuous_contracts.py first.")
            return
        
        # Get available continuous contracts
        contracts = get_continuous_contracts(conn)
        
        if not contracts:
            logger.warning("No continuous contracts found in the database.")
            logger.info("Run build_continuous_contracts.py to generate continuous contracts.")
            return
        
        logger.info(f"Found {len(contracts)} continuous contract(s): {', '.join(contracts)}")
        
        # Process command-line arguments
        symbol = None
        start_date = None
        end_date = None
        plot = False
        
        if len(sys.argv) > 1:
            symbol = sys.argv[1]
            
            if len(sys.argv) > 2:
                start_date = sys.argv[2]
                
                if len(sys.argv) > 3:
                    end_date = sys.argv[3]
                    
                    if len(sys.argv) > 4 and sys.argv[4].lower() == '--plot':
                        plot = True
        
        # If no symbol specified, analyze all contracts
        if not symbol:
            for contract_symbol in contracts: # Use a different variable name
                logger.info(f"\n{'-' * 40}")
                # Fetch and analyze each specific contract
                df = get_continuous_data(conn, contract_symbol, start_date, end_date)
                if not df.empty:
                    rolls = get_roll_dates(df)
                    analyze_continuous_contract(df, rolls)
                    if plot:
                        plot_continuous_contract(df, rolls, f"{contract_symbol}_plot.png")
                else:
                    logger.warning(f"No data found for {contract_symbol} in the given date range.")
        else:
            # If a symbol IS specified (e.g., @ES or ESc1)
            # Validate symbol if it's specific (doesn't start with @)
            if not symbol.startswith('@') and symbol not in contracts:
                logger.error(f"Specific continuous contract '{symbol}' not found in the database.")
                logger.info(f"Available specific contracts: {', '.join(contracts)}")
                return
            
            # Get data (handles both specific and base symbols like @ES now)
            df_all = get_continuous_data(conn, symbol, start_date, end_date)
            
            if df_all.empty:
                logger.warning(f"No continuous data found matching symbol/pattern: {symbol}")
                return

            # If a base symbol (@ES) was given, analyze each resulting contract (ESc1, ESc2, ...) individually
            if symbol.startswith('@'):
                 unique_symbols_found = df_all['continuous_symbol'].unique()
                 logger.info(f"Found {len(unique_symbols_found)} contracts matching base symbol '{symbol}': {', '.join(unique_symbols_found)}")
                 for specific_symbol in unique_symbols_found:
                     logger.info(f"\n--- Analyzing: {specific_symbol} ---")
                     df_specific = df_all[df_all['continuous_symbol'] == specific_symbol]
                     rolls = get_roll_dates(df_specific)
                     analyze_continuous_contract(df_specific, rolls)
                     if plot:
                         plot_continuous_contract(df_specific, rolls, f"{specific_symbol}_plot.png")
            else:
                 # If a specific symbol (ESc1) was given, analyze just that one
                 rolls = get_roll_dates(df_all)
                 analyze_continuous_contract(df_all, rolls)
                 if plot:
                     plot_continuous_contract(df_all, rolls, f"{symbol}_plot.png")
            
    finally:
        conn.close()

if __name__ == "__main__":
    main() 