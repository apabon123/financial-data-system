import duckdb
import pandas as pd
import argparse
from datetime import datetime
from tabulate import tabulate

def load_vix_data(db_path, start_date, end_date):
    """Load VIX, VXc1, and VXc2 data from the database."""
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Query to get the data for all three symbols
    query = f"""
    SELECT 
        timestamp AS date,
        symbol,
        settle,
        source
    FROM market_data
    WHERE symbol IN ('$VIX.X', 'VXc1', 'VXc2')
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
    
    # Create a combined source column
    result_df['VXc1_source'] = result_df['VXc1_source'].fillna('N/A')
    result_df['VXc2_source'] = result_df['VXc2_source'].fillna('N/A')
    result_df['$VIX.X_source'] = result_df['$VIX.X_source'].fillna('N/A')
    
    # Rename columns for clarity
    result_df = result_df.rename(columns={
        'date': 'Date',
        '$VIX.X': 'VIX',
        'VXc1': 'VXc1',
        'VXc2': 'VXc2',
        '$VIX.X_source': 'VIX_Source',
        'VXc1_source': 'VXc1_Source',
        'VXc2_source': 'VXc2_Source'
    })
    
    return result_df

def filter_valid_trading_days(df, require_vix=True, require_vxc1=True, require_vxc2=True):
    """
    Filter for valid trading days based on data availability.
    
    Parameters:
    - df: DataFrame with VIX, VXc1, and VXc2 data
    - require_vix: If True, only include days where VIX data exists
    - require_vxc1: If True, only include days where VXc1 data exists
    - require_vxc2: If True, only include days where VXc2 data exists
    
    Returns:
    - DataFrame with only valid trading days
    """
    # Start with all dates
    valid_days = df.copy()
    
    # Apply filters based on requirements
    if require_vix:
        valid_days = valid_days[valid_days['VIX'].notna()]
    
    if require_vxc1:
        valid_days = valid_days[valid_days['VXc1'].notna()]
    
    if require_vxc2:
        valid_days = valid_days[valid_days['VXc2'].notna()]
    
    # Add a weekday column for analysis
    valid_days['Weekday'] = pd.to_datetime(valid_days['Date']).dt.day_name()
    
    return valid_days

def analyze_data_by_year(df):
    """Analyze the data availability by year."""
    # Add a Year column
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    
    # Count days by year
    year_counts = df['Year'].value_counts().sort_index()
    
    return year_counts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a list of valid trading days with complete VIX, VXc1, and VXc2 data')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--require-vix', action='store_true', default=True, help='Require VIX data to be present')
    parser.add_argument('--require-vxc1', action='store_true', default=True, help='Require VXc1 data to be present')
    parser.add_argument('--require-vxc2', action='store_true', default=True, help='Require VXc2 data to be present')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_vix_data(args.db_path, args.start_date, args.end_date)
    
    # Filter for valid trading days
    valid_days = filter_valid_trading_days(
        df, 
        require_vix=args.require_vix, 
        require_vxc1=args.require_vxc1, 
        require_vxc2=args.require_vxc2
    )
    
    # Analyze data by year
    year_counts = analyze_data_by_year(valid_days)
    
    # Format dates as strings for display
    valid_days['Date'] = pd.to_datetime(valid_days['Date']).dt.strftime('%Y-%m-%d')
    
    # Print results
    print(f"\nAnalysis from {args.start_date} to {args.end_date}")
    print(f"Total dates with any data: {len(df)}")
    print(f"Total valid trading days: {len(valid_days)}")
    print(f"Missing days: {len(df) - len(valid_days)}")
    
    # Print year breakdown
    print("\nValid Trading Days by Year:")
    print(year_counts)
    
    # Print weekday distribution
    weekday_counts = valid_days['Weekday'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    print("\nValid Trading Days by Weekday:")
    print(weekday_counts)
    
    # Output to CSV if specified
    if args.output:
        # Save valid trading days to CSV
        valid_days_output = valid_days[['Date', 'VIX', 'VXc1', 'VXc2', 'VIX_Source', 'VXc1_Source', 'VXc2_Source', 'Weekday']]
        valid_days_output.to_csv(args.output, index=False)
        print(f"\nValid trading days saved to {args.output}")
        
        # Also save just the dates to a simple text file
        date_only_output = args.output.replace('.csv', '_dates_only.txt')
        with open(date_only_output, 'w') as f:
            for date in valid_days['Date']:
                f.write(f"{date}\n")
        print(f"Valid trading dates (date-only format) saved to {date_only_output}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install required libraries: pip install duckdb pandas tabulate") 