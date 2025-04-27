import duckdb
import pandas as pd
import argparse
import numpy as np
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

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
    
    # Flag source transitions
    result_df['Source_Change'] = False
    for col in ['VXc1_Source', 'VXc2_Source']:
        result_df[f'{col}_Changed'] = result_df[col].shift(1) != result_df[col]
        result_df.loc[result_df[f'{col}_Changed'], 'Source_Change'] = True
    
    # Apply limit if specified
    if limit:
        result_df = result_df.head(limit)
    
    return result_df

def format_date_column(df):
    """Format the date column as a string."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df

def highlight_source_changes(df, html_output=None):
    """Highlight rows where the data source changes."""
    styled_df = df.copy()
    
    # Format numeric columns
    for col in ['VIX', 'VXc1', 'VXc2', 'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%']:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Add source indicator
    styled_df['VXc1_Indicator'] = styled_df['VXc1_Source'].apply(
        lambda x: "D" if x == "DERIVED_VIX_RATIO" else 
                  "C" if x == "CBOE" else "N"
    )
    
    styled_df['VXc2_Indicator'] = styled_df['VXc2_Source'].apply(
        lambda x: "D" if x == "DERIVED_VIX_RATIO" else 
                  "C" if x == "CBOE" else "N"
    )
    
    # Add source indicators to settle values
    styled_df['VXc1_Display'] = styled_df.apply(
        lambda row: f"{row['VXc1']} ({row['VXc1_Indicator']})", axis=1
    )
    
    styled_df['VXc2_Display'] = styled_df.apply(
        lambda row: f"{row['VXc2']} ({row['VXc2_Indicator']})", axis=1
    )
    
    # Create HTML output if requested
    if html_output:
        html_df = df.copy()
        
        # Function to apply styling
        def style_row(row):
            if row['Source_Change']:
                return ['background-color: #FFFFCC'] * len(row)
            return [''] * len(row)
        
        # Apply styling and save HTML
        styled_html = html_df.style.apply(style_row, axis=1)
        styled_html.to_html(html_output)
    
    # Return display columns for console output
    display_cols = [
        'Date', 'VIX', 'VXc1_Display', 'VXc2_Display', 
        'VXc1-VIX', 'VXc2-VXc1', 'VXc1/VIX%', 'VXc2/VXc1%'
    ]
    return styled_df[display_cols]

def calculate_statistics(df):
    """Calculate and return statistics for the data."""
    # Only use rows with complete data
    complete_df = df.dropna(subset=['VIX', 'VXc1', 'VXc2'])
    
    if len(complete_df) == 0:
        return "No complete data available for statistics."
    
    # Calculate statistics
    stats = {
        "Count": len(complete_df),
        "Mean": {
            "VIX": complete_df['VIX'].mean(),
            "VXc1": complete_df['VXc1'].mean(),
            "VXc2": complete_df['VXc2'].mean(),
            "VXc1-VIX": complete_df['VXc1-VIX'].mean(),
            "VXc2-VXc1": complete_df['VXc2-VXc1'].mean(),
            "VXc1/VIX%": complete_df['VXc1/VIX%'].mean(),
            "VXc2/VXc1%": complete_df['VXc2/VXc1%'].mean(),
        },
        "Min": {
            "VIX": complete_df['VIX'].min(),
            "VXc1": complete_df['VXc1'].min(),
            "VXc2": complete_df['VXc2'].min(),
            "VXc1-VIX": complete_df['VXc1-VIX'].min(),
            "VXc2-VXc1": complete_df['VXc2-VXc1'].min(),
            "VXc1/VIX%": complete_df['VXc1/VIX%'].min(),
            "VXc2/VXc1%": complete_df['VXc2/VXc1%'].min(),
        },
        "Max": {
            "VIX": complete_df['VIX'].max(),
            "VXc1": complete_df['VXc1'].max(),
            "VXc2": complete_df['VXc2'].max(),
            "VXc1-VIX": complete_df['VXc1-VIX'].max(),
            "VXc2-VXc1": complete_df['VXc2-VXc1'].max(),
            "VXc1/VIX%": complete_df['VXc1/VIX%'].max(),
            "VXc2/VXc1%": complete_df['VXc2/VXc1%'].max(),
        }
    }
    
    # Prepare the stats table
    stats_table = []
    for stat, values in [("Mean", stats["Mean"]), ("Min", stats["Min"]), ("Max", stats["Max"])]:
        row = [stat]
        for key in ["VIX", "VXc1", "VXc2", "VXc1-VIX", "VXc2-VXc1", "VXc1/VIX%", "VXc2/VXc1%"]:
            row.append(f"{values[key]:.2f}")
        stats_table.append(row)
    
    # Source distribution
    source_counts = {}
    for col in ['VXc1_Source', 'VXc2_Source']:
        counts = df[col].value_counts()
        source_counts[col] = counts.to_dict()
    
    return {
        "table": stats_table,
        "headers": ["Stat", "VIX", "VXc1", "VXc2", "VXc1-VIX", "VXc2-VXc1", "VXc1/VIX%", "VXc2/VXc1%"],
        "source_counts": source_counts,
        "count": stats["Count"]
    }

def plot_data(df, output_dir=None):
    """Create plots of the data."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Plot VIX, VXc1, VXc2
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['VIX'], label='VIX')
    plt.plot(df['Date'], df['VXc1'], label='VXc1')
    plt.plot(df['Date'], df['VXc2'], label='VXc2')
    plt.title('VIX, VXc1, VXc2 Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vix_values.png'))
        plt.close()
    
    # Plot VXc1-VIX and VXc2-VXc1 differences
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['VXc1-VIX'], label='VXc1-VIX')
    plt.plot(df['Date'], df['VXc2-VXc1'], label='VXc2-VXc1')
    plt.title('VXc1-VIX and VXc2-VXc1 Differences')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vix_differences.png'))
        plt.close()
    
    # Plot percentage differences
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['VXc1/VIX%'], label='VXc1/VIX%')
    plt.plot(df['Date'], df['VXc2/VXc1%'], label='VXc2/VXc1%')
    plt.title('VXc1/VIX% and VXc2/VXc1% Percentage Differences')
    plt.xlabel('Date')
    plt.ylabel('Percentage Difference')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vix_percentage_differences.png'))
        plt.close()
    else:
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display VIX data in a formatted table')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of rows displayed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--html', type=str, default=None, help='Output HTML file path')
    parser.add_argument('--plots', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--tabulate', action='store_true', help='Use tabulate for pretty printing')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_vix_data(args.db_path, args.start_date, args.end_date, args.limit)
    
    # Format date column
    df = format_date_column(df)
    
    # Output to CSV if specified
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
    
    # Create plots if requested
    if args.plots:
        plot_data(df, args.plots)
        print(f"Plots saved to {args.plots}")
    
    # Show data count
    print(f"Data from {args.start_date} to {args.end_date}")
    print(f"Total rows: {len(df)}")
    
    # Format and highlight the data
    styled_df = highlight_source_changes(df, args.html)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Print the data
    if args.tabulate:
        print("\n--- Data Table ---")
        print(tabulate(styled_df, headers='keys', tablefmt='pretty', showindex=False))
    else:
        print("\n--- First 20 rows ---")
        print(styled_df.head(20))
        if len(styled_df) > 20:
            print("\n--- Last 20 rows ---")
            print(styled_df.tail(20))
    
    # Print statistics
    print("\n--- Statistics ---")
    if isinstance(stats, str):
        print(stats)
    else:
        print(f"Data points: {stats['count']}")
        print("\nSummary Statistics:")
        print(tabulate(stats['table'], headers=stats['headers'], tablefmt='pretty'))
        
        print("\nSource Distribution:")
        for source, counts in stats['source_counts'].items():
            print(f"\n{source.replace('_Source', '')}:")
            for src, count in counts.items():
                print(f"  {src}: {count} rows ({count/len(df)*100:.1f}%)")
    
    # If HTML output was specified, notify
    if args.html:
        print(f"\nHTML output saved to {args.html}")
    
    return df

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        # Try to import without tabulate
        print("Attempting to run without tabulate...")
        try:
            import sys
            sys.argv.append("--tabulate")  # Remove tabulate flag
            main()
        except Exception as e2:
            print(f"Error: {e2}")
            print("Please install required libraries: pip install tabulate matplotlib") 