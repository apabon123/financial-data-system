import duckdb
import pandas as pd
import argparse
from datetime import datetime
from tabulate import tabulate

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
    
    # Calculate differences and premium/discount
    result_df['VXc1-VIX'] = result_df['VXc1'] - result_df['VIX']
    result_df['VXc2-VXc1'] = result_df['VXc2'] - result_df['VXc1']
    
    # Calculate percentage premium/discount
    result_df['VXc1/VIX%'] = (result_df['VXc1'] / result_df['VIX'] - 1) * 100
    result_df['VXc2/VXc1%'] = (result_df['VXc2'] / result_df['VXc1'] - 1) * 100
    
    # Apply limit if specified
    if limit:
        result_df = result_df.head(limit)
    
    return result_df

def format_output(df):
    """Format the data for display."""
    # Convert date to string format
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Format numeric columns
    for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Add source indicators to settle values
    df['VXc1_Display'] = df.apply(
        lambda row: f"{row['VXc1']} ({row['VXc1_Source'][0]})", axis=1
    )
    
    df['VXc2_Display'] = df.apply(
        lambda row: f"{row['VXc2']} ({row['VXc2_Source'][0]})", axis=1
    )
    
    # Select display columns
    display_cols = [
        'Date', 'VIX', 'VXc1_Display', 'VXc2_Display', 
        'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%'
    ]
    
    return df[display_cols]

def calculate_statistics(df):
    """Calculate and return summary statistics for the data."""
    # Original df for source counts
    orig_df = df.copy()
    
    # Convert string values back to floats for calculation
    for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: float(x) if x != "N/A" else None)
    
    # Calculate statistics
    stats = {
        "Count": len(df),
        "Mean": {col: df[col].mean() for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%'] if col in df.columns},
        "Min": {col: df[col].min() for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%'] if col in df.columns},
        "Max": {col: df[col].max() for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%'] if col in df.columns}
    }
    
    # Source counts
    source_counts = {}
    for col in ['VXc1_Source', 'VXc2_Source']:
        if col in orig_df.columns:
            counts = orig_df[col].value_counts()
            source_counts[col] = counts.to_dict()
    
    return stats, source_counts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display VIX, VXc1, and VXc2 data in a combined table')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of rows displayed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_vix_data(args.db_path, args.start_date, args.end_date, args.limit)
    
    # Output to CSV if specified
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
    
    # Format the data for display
    display_df = format_output(df)
    
    # Calculate statistics
    stats, source_counts = calculate_statistics(df)
    
    # Print the data
    print(f"\nVIX, VXc1, and VXc2 Data from {args.start_date} to {args.end_date}")
    print(f"Total rows: {len(df)}")
    print("\n--- Data Table ---")
    print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False))
    
    # Print statistics
    print("\n--- Summary Statistics ---")
    stats_table = []
    for stat, values in [("Mean", stats["Mean"]), ("Min", stats["Min"]), ("Max", stats["Max"])]:
        row = [stat]
        for key in ["VIX", "VXc1", "VXc2", "VXc1-VIX", "VXc2-VXc1", "VXc1/VIX%", "VXc2/VXc1%"]:
            if key in values:
                row.append(f"{values[key]:.2f}" if values[key] is not None else "N/A")
            else:
                row.append("N/A")
        stats_table.append(row)
    
    print(tabulate(stats_table, headers=["Stat", "VIX", "VXc1", "VXc2", "VXc1-VIX", "VXc2-VXc1", "VXc1/VIX%", "VXc2/VXc1%"], tablefmt='pretty'))
        
    # Print source distribution
    print("\n--- Source Distribution ---")
    for source, counts in source_counts.items():
        print(f"\n{source.replace('_Source', '')}:")
        for src, count in counts.items():
            print(f"  {src}: {count} rows ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install required libraries: pip install duckdb pandas tabulate") 