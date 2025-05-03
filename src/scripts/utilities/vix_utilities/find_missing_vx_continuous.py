#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyzes dates where VIX data exists but continuous futures contracts data is missing.
Helps identify patterns in missing data, such as holidays or specific time periods.
"""

import duckdb
import pandas as pd
import argparse
from datetime import datetime
import calendar
from tabulate import tabulate

def load_vix_data(db_path, start_date, end_date):
    """Load VIX from market_data_cboe and VX continuous contracts from continuous_contracts."""
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # --- Load Continuous Contracts --- 
        continuous_symbols = [f'@VX={i}01XN' for i in range(1, 6)]
        symbols_in_clause = ", ".join([f"'{s}'" for s in continuous_symbols])
        query_vx = f"""
            SELECT 
                timestamp AS date,
                symbol,
                settle,
                source
            FROM continuous_contracts
            WHERE symbol IN ({symbols_in_clause})
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp, symbol
        """
        vx_df = conn.execute(query_vx, [start_date, end_date]).fetchdf()
        
        # --- Load VIX Data --- 
        query_vix = f"""
            SELECT 
                timestamp AS date,
                settle, -- Use settle for VIX price
                source
            FROM market_data_cboe
            WHERE symbol = '$VIX.X'
            AND interval_unit = 'daily' -- Ensure we get daily VIX
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        vix_df = conn.execute(query_vix, [start_date, end_date]).fetchdf()

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame() # Return empty df on error
    finally:
        if conn:
            conn.close()

    # --- Prepare VIX Data --- 
    if not vix_df.empty:
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        vix_df = vix_df.rename(columns={'settle': 'VIX', 'source': 'VIX_Source'})
        vix_df = vix_df.set_index('date')
    else:
        print("Warning: No VIX data loaded.")
        vix_df = pd.DataFrame(columns=['VIX', 'VIX_Source']) # Create empty df with expected columns

    # --- Prepare and Pivot VX Data ---
    if not vx_df.empty:
        vx_df['date'] = pd.to_datetime(vx_df['date'])
        # Pivot to get contracts as columns, keep source info separate for now
        vx_pivot_settle = vx_df.pivot(index='date', columns='symbol', values='settle')
        vx_pivot_source = vx_df.pivot(index='date', columns='symbol', values='source')
        vx_pivot_source.columns = [f'{{col}}_source' for col in vx_pivot_source.columns] # Rename source columns
        # Combine settle and source pivots
        vx_pivot = pd.concat([vx_pivot_settle, vx_pivot_source], axis=1)
    else:
        print("Warning: No VX continuous contract data loaded.")
        vx_pivot = pd.DataFrame()

    # --- Merge VIX and VX data --- 
    # Use an outer join to keep all dates from both
    if not vix_df.empty and not vx_pivot.empty:
        # Ensure both have datetime index before joining
        vix_df.index = pd.to_datetime(vix_df.index)
        vx_pivot.index = pd.to_datetime(vx_pivot.index)
        result_df = vix_df.join(vx_pivot, how='outer')
    elif not vix_df.empty:
        result_df = vix_df
        result_df.index = pd.to_datetime(result_df.index) # Ensure index is datetime
    elif not vx_pivot.empty:
        result_df = vx_pivot
        result_df.index = pd.to_datetime(result_df.index) # Ensure index is datetime
    else:
        result_df = pd.DataFrame()
        
    # Reset index to make date a column ONLY if the DataFrame is not empty
    if not result_df.empty:
        result_df = result_df.reset_index().rename(columns={'date': 'Date'}) # Use 'date' from index name
    else:
        # If empty, ensure it has a Date column for downstream compatibility
        result_df = pd.DataFrame(columns=['Date']) 

    # --- Ensure all expected columns exist --- 
    # Define continuous symbols and their source columns
    continuous_symbols = [f'@VX={i}01XN' for i in range(1, 6)]
    expected_vx_cols = continuous_symbols + [f'{s}_source' for s in continuous_symbols]
    expected_cols = ['Date', 'VIX', 'VIX_Source'] + expected_vx_cols

    # Add missing columns, filling with NaN for data cols and 'N/A' for source cols
    for col in expected_cols:
        if col not in result_df.columns:
            if col.endswith('_source'):
                result_df[col] = 'N/A' # Fill missing source columns
            elif col != 'Date': # Don't overwrite Date if it exists
                 result_df[col] = pd.NA # Fill missing data columns (use pd.NA for consistency)
            elif 'Date' not in result_df.columns: # Ensure Date column if absolutely missing
                 result_df['Date'] = pd.NaT 

    # Ensure correct column order (optional but good practice)
    # Filter expected_cols to only those actually present in result_df to avoid errors
    present_expected_cols = [col for col in expected_cols if col in result_df.columns]
    result_df = result_df[present_expected_cols] 

    # --- Fill Missing Source Columns (redundant now but keep for safety) ---
    # Define continuous symbols again for this scope
    continuous_symbols = [f'@VX={i}01XN' for i in range(1, 6)]
    
    # Fill NaN source columns that might have been created or missing
    for symbol in continuous_symbols:
        col_name = f"{{symbol}}_source"
        if col_name in result_df.columns:
            result_df[col_name] = result_df[col_name].fillna('N/A')
        else: # Add column if it wasn't created 
             result_df[col_name] = 'N/A'
             
    if 'VIX_Source' in result_df.columns:
         result_df['VIX_Source'] = result_df['VIX_Source'].fillna('N/A')
    elif 'Date' in result_df.columns: # Add VIX_Source only if Date exists
         result_df['VIX_Source'] = 'N/A'

    return result_df

def categorize_missing_data(df):
    """Find and categorize dates where VIX data exists but continuous contracts data is missing."""
    # Convert Date to datetime for easier manipulation
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define continuous contracts to check - Use the new format
    contracts = [f'@VX={i}01XN' for i in range(1, 6)]
    
    # Initialize dictionary to hold missing data frames
    missing_data = {}
    
    # Filter for rows where VIX exists but continuous contracts are missing
    for contract in contracts:
        if contract in df.columns:
            missing_df = df[df['VIX'].notna() & df[contract].isna()].copy()
            
            if not missing_df.empty:
                # Add weekday and month information
                missing_df['Weekday'] = missing_df['Date'].dt.day_name()
                missing_df['Month'] = missing_df['Date'].dt.month_name()
                missing_df['Day'] = missing_df['Date'].dt.day
                missing_df['Year'] = missing_df['Date'].dt.year
                
                # Categorize by known US holidays
                conditions = [
                    (missing_df['Month'] == 'January') & (missing_df['Day'] == 1),
                    (missing_df['Month'] == 'January') & (missing_df['Weekday'] == 'Monday') & (missing_df['Day'] >= 15) & (missing_df['Day'] <= 21),
                    (missing_df['Month'] == 'February') & (missing_df['Weekday'] == 'Monday') & (missing_df['Day'] >= 15) & (missing_df['Day'] <= 21),
                    (missing_df['Month'] == 'May') & (missing_df['Weekday'] == 'Monday') & (missing_df['Day'] >= 25),
                    (missing_df['Month'] == 'June') & (missing_df['Day'] == 19) | ((missing_df['Month'] == 'June') & (missing_df['Weekday'] == 'Monday') & (missing_df['Day'] == 20)),
                    (missing_df['Month'] == 'July') & (missing_df['Day'] == 4),
                    (missing_df['Month'] == 'September') & (missing_df['Weekday'] == 'Monday') & (missing_df['Day'] <= 7),
                    (missing_df['Month'] == 'November') & (missing_df['Weekday'] == 'Thursday') & (missing_df['Day'] >= 22) & (missing_df['Day'] <= 28),
                    (missing_df['Month'] == 'December') & (missing_df['Day'] == 25)
                ]
                
                holiday_names = [
                    'New Year\'s Day',
                    'Martin Luther King Jr. Day',
                    'Presidents\' Day',
                    'Memorial Day',
                    'Juneteenth',
                    'Independence Day',
                    'Labor Day',
                    'Thanksgiving',
                    'Christmas Day'
                ]
                
                missing_df['Holiday'] = 'Not a Holiday'
                for condition, holiday_name in zip(conditions, holiday_names):
                    missing_df.loc[condition, 'Holiday'] = holiday_name
                
                # Format date back to string for display
                missing_df['Date'] = missing_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Store in dictionary
                missing_data[contract] = missing_df
            else:
                missing_data[contract] = pd.DataFrame()
        else:
            # If contract doesn't exist in data
            missing_data[contract] = pd.DataFrame(columns=['Date', 'VIX', 'VIX_Source'])
    
    return missing_data

def analyze_patterns(missing_data):
    """Analyze patterns in the missing data."""
    results = {}
    
    for contract, missing_df in missing_data.items():
        if not missing_df.empty:
            # Count by year
            results[f'{contract}_by_year'] = missing_df['Year'].value_counts().sort_index()
            
            # Count by holiday
            results[f'{contract}_by_holiday'] = missing_df['Holiday'].value_counts().sort_index()
            
            # Count by weekday
            results[f'{contract}_by_weekday'] = missing_df['Weekday'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
            
            # Count by month
            month_order = [calendar.month_name[i] for i in range(1, 13)]
            results[f'{contract}_by_month'] = missing_df['Month'].value_counts().reindex(month_order)
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find and analyze dates where VIX data exists but continuous contracts data is missing')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis of missing data patterns')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_vix_data(args.db_path, args.start_date, args.end_date)
    
    # Find and categorize missing data
    missing_data = categorize_missing_data(df)
    
    # Format VIX values for display
    for contract, missing_df in missing_data.items():
        if not missing_df.empty:
            missing_df['VIX'] = missing_df['VIX'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Print results
    print(f"\nAnalysis from {args.start_date} to {args.end_date}")
    print(f"Total dates analyzed: {len(df)}")
    
    # Check available contracts in data - Use new format
    available_contracts_in_df = [col for col in [f'@VX={i}01XN' for i in range(1, 6)] if col in df.columns]
    print(f"Available contracts in data: {', '.join(available_contracts_in_df)}")
    
    # Display missing data for each contract
    for contract in available_contracts_in_df:
        if contract in missing_data:
            missing_df = missing_data[contract]
            print(f"\nDates where VIX exists but {contract} is missing: {len(missing_df)}")
            if not missing_df.empty:
                display_cols = ['Date', 'VIX', 'VIX_Source', 'Weekday', 'Holiday']
                print(tabulate(missing_df[display_cols], headers='keys', tablefmt='pretty', showindex=False))
    
    # Show additional analysis if requested
    if args.detailed:
        patterns = analyze_patterns(missing_data)
        
        print("\n--- Detailed Analysis of Missing Data ---")
        
        for key, value in patterns.items():
            if "_by_year" in key:
                contract = key.split("_by_year")[0]
                print(f"\n{contract} Missing by Year:")
                print(value)
            
        for key, value in patterns.items():
            if "_by_holiday" in key:
                contract = key.split("_by_holiday")[0]
                print(f"\n{contract} Missing by Holiday:")
                print(value)
            
        for key, value in patterns.items():
            if "_by_weekday" in key:
                contract = key.split("_by_weekday")[0]
                print(f"\n{contract} Missing by Weekday:")
                print(value)
            
        for key, value in patterns.items():
            if "_by_month" in key:
                contract = key.split("_by_month")[0]
                print(f"\n{contract} Missing by Month:")
                print(value)
    
    # Save to CSV if output path specified
    if args.output:
        # Create a single DataFrame with all missing data
        all_missing = []
        for contract, missing_df in missing_data.items():
            if not missing_df.empty:
                missing_df['Contract'] = contract
                all_missing.append(missing_df)
        
        if all_missing:
            combined_df = pd.concat(all_missing, ignore_index=True)
            combined_df.to_csv(args.output, index=False)
            print(f"\nMissing data saved to {args.output}")
        else:
            print("\nNo missing data to save.")

if __name__ == "__main__":
    main() 