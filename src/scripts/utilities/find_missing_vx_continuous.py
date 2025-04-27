import duckdb
import pandas as pd
import argparse
from datetime import datetime
import calendar
from tabulate import tabulate

def load_vix_data(db_path, start_date, end_date):
    """Load VIX and VX continuous contract data from the database."""
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Query to get the data for all symbols
    query = f"""
    SELECT 
        timestamp AS date,
        symbol,
        settle,
        source
    FROM market_data
    WHERE symbol IN ('$VIX.X', 'VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5')
    AND timestamp >= '{start_date}'
    AND timestamp <= '{end_date}'
    ORDER BY timestamp, symbol
    """
    
    # Execute the query
    df = conn.execute(query).fetchdf()
    
    # Close the connection
    conn.close()
    
    # Pivot the data to wide format
    pivot_df = df.pivot(index=['date'], columns='symbol', values=['settle', 'source'])
    
    # Flatten the multi-level columns
    pivot_df.columns = [f"{col[1]}_{col[0]}" if col[0] == 'source' else col[1] for col in pivot_df.columns]
    
    # Reset index to make date a column
    result_df = pivot_df.reset_index()
    
    # Create a combined source column for each symbol
    for symbol in ['$VIX.X', 'VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']:
        col_name = f"{symbol}_source"
        if col_name in result_df.columns:
            result_df[col_name] = result_df[col_name].fillna('N/A')
    
    # Rename columns for clarity
    rename_dict = {
        'date': 'Date',
        '$VIX.X': 'VIX',
        '$VIX.X_source': 'VIX_Source'
    }
    
    # Don't rename VXcN columns as they're already named correctly
    result_df = result_df.rename(columns=rename_dict)
    
    return result_df

def categorize_missing_data(df):
    """Find and categorize dates where VIX data exists but continuous contracts data is missing."""
    # Convert Date to datetime for easier manipulation
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define continuous contracts to check
    contracts = ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']
    
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
    
    # Check available contracts in data
    available_contracts = [col for col in ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5'] if col in df.columns]
    print(f"Available contracts in data: {', '.join(available_contracts)}")
    
    # Display missing data for each contract
    for contract in available_contracts:
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
    
    # Output to CSV if specified
    if args.output:
        # Combine results from all contracts
        all_missing = pd.DataFrame()
        
        for contract, missing_df in missing_data.items():
            if not missing_df.empty:
                missing_df['Missing_Contract'] = contract
                all_missing = pd.concat([all_missing, missing_df])
        
        if not all_missing.empty:
            all_missing.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    # Separate missing data by pre-2006 and post-2005
    print("\n--- Analysis of Missing Data by Time Period ---")
    for contract, missing_df in missing_data.items():
        if not missing_df.empty:
            try:
                missing_df['Year'] = missing_df['Year'].astype(int)
                pre_2006 = missing_df[missing_df['Year'] < 2006]
                post_2005 = missing_df[missing_df['Year'] >= 2006]
                print(f"\n{contract} missing pre-2006: {len(pre_2006)} dates")
                print(f"{contract} missing post-2005: {len(post_2005)} dates")
            except:
                print(f"\nCould not analyze {contract} by year")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install required libraries: pip install duckdb pandas tabulate") 