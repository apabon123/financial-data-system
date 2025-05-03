import duckdb
import pandas as pd
import argparse
from datetime import datetime

def load_vix_data(db_path, start_date, end_date, limit=None):
    """Load VIX data from the database and create a combined table."""
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
    
    # Format date as string in the desired format
    result_df['Date'] = result_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Apply limit if specified
    if limit:
        result_df = result_df.head(limit)
    
    return result_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display VIX data in a table format')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of rows displayed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load the data
    result_df = load_vix_data(args.db_path, args.start_date, args.end_date, args.limit)
    
    # Output to CSV if specified
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
    
    # Display the data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 4)
    
    # Show data count
    print(f"Data from {args.start_date} to {args.end_date}")
    print(f"Total rows: {len(result_df)}")
    print("\n--- First 20 rows ---")
    print(result_df[['Date', 'VIX', 'VXc1', 'VXc2', 'VIX_Source', 'VXc1_Source', 'VXc2_Source']].head(20))
    
    if len(result_df) > 20:
        print("\n--- Last 20 rows ---")
        print(result_df[['Date', 'VIX', 'VXc1', 'VXc2', 'VIX_Source', 'VXc1_Source', 'VXc2_Source']].tail(20))
    
    return result_df

if __name__ == "__main__":
    main() 