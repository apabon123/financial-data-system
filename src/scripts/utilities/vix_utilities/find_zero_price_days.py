import duckdb
import pandas as pd
from datetime import datetime
from tabulate import tabulate

def find_zero_price_days(db_path, start_date='2004-01-01', end_date=None):
    """
    Find days where any of the VXc1-VXc5 contracts have a zero price.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Query to get the data for all VXc1-VXc5 symbols
    query = f"""
    SELECT 
        timestamp AS date,
        symbol,
        settle,
        source
    FROM market_data
    WHERE symbol IN ('VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5')
    AND timestamp >= '{start_date}'
    AND timestamp <= '{end_date}'
    ORDER BY timestamp, symbol
    """
    
    # Execute the query
    df = conn.execute(query).fetchdf()
    
    # Close the connection
    conn.close()
    
    # Pivot the data to wide format
    pivot_df = df.pivot(index=['date'], columns='symbol', values='settle')
    
    # Reset index to make date a column
    result_df = pivot_df.reset_index()
    
    # Filter for days where any contract has a zero price
    zero_days = result_df[(result_df['VXc1'] == 0) | 
                          (result_df['VXc2'] == 0) | 
                          (result_df['VXc3'] == 0) | 
                          (result_df['VXc4'] == 0) | 
                          (result_df['VXc5'] == 0)]
    
    # Handle NaN values - if a contract wasn't traded yet, it would be NaN, not 0
    zero_days = zero_days.fillna(0)
    
    # Count zeros by contract
    vxc1_zeros = len(zero_days[zero_days['VXc1'] == 0])
    vxc2_zeros = len(zero_days[zero_days['VXc2'] == 0])
    vxc3_zeros = len(zero_days[zero_days['VXc3'] == 0])
    vxc4_zeros = len(zero_days[zero_days['VXc4'] == 0])
    vxc5_zeros = len(zero_days[zero_days['VXc5'] == 0])
    
    # Print summary
    print(f"Total days with zero prices for any contract: {len(zero_days)}")
    print(f"VXc1 zero days: {vxc1_zeros}")
    print(f"VXc2 zero days: {vxc2_zeros}")
    print(f"VXc3 zero days: {vxc3_zeros}")
    print(f"VXc4 zero days: {vxc4_zeros}")
    print(f"VXc5 zero days: {vxc5_zeros}")
    
    # Create a nicer display format for the zero days
    display_data = []
    for _, row in zero_days.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        vxc1 = row['VXc1'] if row['VXc1'] != 0 else "ZERO"
        vxc2 = row['VXc2'] if row['VXc2'] != 0 else "ZERO"
        vxc3 = row['VXc3'] if row['VXc3'] != 0 else "ZERO"
        vxc4 = row['VXc4'] if row['VXc4'] != 0 else "ZERO"
        vxc5 = row['VXc5'] if row['VXc5'] != 0 else "ZERO"
        display_data.append([date, vxc1, vxc2, vxc3, vxc4, vxc5])
    
    # Display the zero days
    if display_data:
        print("\nDays with zero prices:")
        print(tabulate(display_data, headers=['Date', 'VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5'], tablefmt='plain'))
    
    return zero_days

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find days with zero prices in VX continuous contracts')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the database')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (CSV)')
    
    args = parser.parse_args()
    
    zero_days_df = find_zero_price_days(args.db_path, args.start_date, args.end_date)
    
    # Save to CSV if output file specified
    if args.output and not zero_days_df.empty:
        zero_days_df.to_csv(args.output, index=False)
        print(f"\nZero price days saved to {args.output}") 