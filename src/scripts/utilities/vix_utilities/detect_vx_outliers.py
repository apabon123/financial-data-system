import duckdb
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt

def load_vx_data(db_path, start_date, end_date):
    """Load VIX and VX continuous contract data from the database."""
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Query to get the data for all symbols
    # Define symbols using the new format
    continuous_symbols = [f'@VX={i}01XN' for i in range(1, 6)]
    # Add $VIX.X to the list of symbols to query
    symbols_to_query = continuous_symbols + ['$VIX.X']
    symbols_in_clause = ", ".join([f"'{s}'" for s in symbols_to_query])
    query = f"""
    SELECT 
        timestamp AS date,
        symbol,
        settle,
        source
    FROM continuous_contracts
    WHERE symbol IN ({symbols_in_clause})
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
    # Use the symbols_to_query list for iteration
    for symbol in symbols_to_query:
        col_name = f"{symbol}_source"
        if col_name in result_df.columns:
            result_df[col_name] = result_df[col_name].fillna('N/A')
    
    # Rename columns for clarity
    # Keep the VIX renaming, but the VXc renaming is no longer needed
    # as the columns will be named @VX=101XN etc.
    rename_dict = {
        'date': 'Date',
        '$VIX.X': 'VIX',
        '$VIX.X_source': 'VIX_Source'
    }
    result_df = result_df.rename(columns=rename_dict)
    
    return result_df

def detect_outliers(df, methods=None, zscore_threshold=4.0, iqr_multiplier=3.0, pct_change_threshold=30.0, min_value_threshold=1.0, max_value_threshold=100.0):
    """
    Detect outliers in VX continuous contracts data using multiple methods.
    
    Parameters:
    - df: DataFrame with VIX and VXc1-VXc5 data
    - methods: List of methods to use for outlier detection. Default is all methods.
    - zscore_threshold: Threshold for z-score based outliers
    - iqr_multiplier: Multiplier for IQR based outliers
    - pct_change_threshold: Threshold for percentage change outliers
    - min_value_threshold: Minimum valid value (values below are considered outliers)
    - max_value_threshold: Maximum valid value (values above are considered outliers)
    
    Returns:
    - DataFrame with outliers information
    """
    # Default methods
    if methods is None:
        methods = ['zscore', 'iqr', 'pct_change', 'absolute', 'vix_compare']
    
    # Ensure Date is datetime for sorting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Contracts to check - Use the new format
    contracts = [f'@VX={{i}}01XN' for i in range(1, 6)]
    contracts_in_df = [col for col in contracts if col in df.columns]
    
    # Store outliers
    outliers = []
    
    for contract in contracts_in_df:
        # Skip if contract not in data
        if contract not in df.columns:
            continue
        
        # Create temporary dataframe with the contract data (no NaN values)
        temp_df = df[['Date', contract]].dropna(subset=[contract]).copy()
        if len(temp_df) == 0:
            continue
        
        # Z-score method
        if 'zscore' in methods:
            temp_df['zscore'] = np.abs((temp_df[contract] - temp_df[contract].mean()) / temp_df[contract].std())
            zscore_outliers = temp_df[temp_df['zscore'] > zscore_threshold]
            
            for _, row in zscore_outliers.iterrows():
                outliers.append({
                    'Date': row['Date'],
                    'Contract': contract,
                    'Value': row[contract],
                    'Method': 'Z-score',
                    'Detail': f"Z-score: {row['zscore']:.2f} (threshold: {zscore_threshold})"
                })
        
        # IQR method
        if 'iqr' in methods:
            Q1 = temp_df[contract].quantile(0.25)
            Q3 = temp_df[contract].quantile(0.75)
            IQR = Q3 - Q1
            iqr_lower = Q1 - iqr_multiplier * IQR
            iqr_upper = Q3 + iqr_multiplier * IQR
            
            iqr_outliers = temp_df[(temp_df[contract] < iqr_lower) | (temp_df[contract] > iqr_upper)]
            
            for _, row in iqr_outliers.iterrows():
                outliers.append({
                    'Date': row['Date'],
                    'Contract': contract,
                    'Value': row[contract],
                    'Method': 'IQR',
                    'Detail': f"Outside range: [{iqr_lower:.2f}, {iqr_upper:.2f}]"
                })
        
        # Percentage change method
        if 'pct_change' in methods:
            temp_df['pct_change'] = temp_df[contract].pct_change() * 100
            pct_change_outliers = temp_df[abs(temp_df['pct_change']) > pct_change_threshold].dropna(subset=['pct_change'])
            
            for _, row in pct_change_outliers.iterrows():
                outliers.append({
                    'Date': row['Date'],
                    'Contract': contract,
                    'Value': row[contract],
                    'Method': 'Pct Change',
                    'Detail': f"Change: {row['pct_change']:.2f}% (threshold: ±{pct_change_threshold}%)"
                })
        
        # Absolute level method
        if 'absolute' in methods:
            absolute_outliers = temp_df[(temp_df[contract] < min_value_threshold) | (temp_df[contract] > max_value_threshold)]
            
            for _, row in absolute_outliers.iterrows():
                outliers.append({
                    'Date': row['Date'],
                    'Contract': contract,
                    'Value': row[contract],
                    'Method': 'Absolute',
                    'Detail': f"Outside range: [{min_value_threshold}, {max_value_threshold}]"
                })
        
        # VIX comparison method
        if 'vix_compare' in methods and 'VIX' in df.columns:
            # Create a temporary dataframe with both VIX and contract data
            vix_compare_df = df[['Date', 'VIX', contract]].dropna().copy()
            
            if len(vix_compare_df) > 0:
                # Calculate percentage difference
                vix_compare_df['diff_pct'] = ((vix_compare_df[contract] - vix_compare_df['VIX']) / vix_compare_df['VIX']) * 100
                vix_compare_threshold = 30.0
                vix_outliers = vix_compare_df[abs(vix_compare_df['diff_pct']) > vix_compare_threshold]
                
                for _, row in vix_outliers.iterrows():
                    outliers.append({
                        'Date': row['Date'],
                        'Contract': contract,
                        'Value': row[contract],
                        'Method': 'VIX Compare',
                        'Detail': f"Difference: {row['diff_pct']:.2f}% from VIX ({row['VIX']:.2f})"
                    })
    
    # Create DataFrame from outliers list
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        
        # Format Date as string
        outliers_df['Date'] = pd.to_datetime(outliers_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Sort by date and contract
        outliers_df = outliers_df.sort_values(['Date', 'Contract'])
        
        # Remove duplicates (same date, contract, value)
        outliers_df = outliers_df.drop_duplicates(subset=['Date', 'Contract', 'Value'])
        
        return outliers_df
    else:
        return pd.DataFrame(columns=['Date', 'Contract', 'Value', 'Method', 'Detail'])

def plot_data(df, outliers_df, output_dir=None):
    """Create plots of the data with outliers highlighted."""
    if 'Date' not in df.columns:
        return
        
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Contracts to plot - Use the new format
    contracts = [f'@VX={{i}}01XN' for i in range(1, 6)]
    contracts_in_df = [col for col in contracts if col in df.columns]
    
    # Plot all contracts
    plt.figure(figsize=(12, 8))
    
    for contract in contracts_in_df:
        plt.plot(df['Date'], df[contract], label=contract, alpha=0.7)
    
    # If we have outliers, plot them
    if not outliers_df.empty:
        for contract in contracts_in_df:
            contract_outliers = outliers_df[outliers_df['Contract'] == contract]
            if not contract_outliers.empty:
                dates = pd.to_datetime(contract_outliers['Date'])
                values = contract_outliers['Value']
                plt.scatter(dates, values, s=100, marker='o', edgecolors='red', facecolors='none', label=f"{contract} Outliers" if contract == contracts_in_df[0] else "")
    
    plt.title('VX Continuous Contracts with Potential Outliers')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(f"{output_dir}/vx_outliers_overview.png")
    else:
        plt.show()
    
    # Plot each contract separately with outliers
    for contract in contracts_in_df:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df[contract], label=contract, color='blue', alpha=0.7)
        
        contract_outliers = outliers_df[outliers_df['Contract'] == contract]
        if not contract_outliers.empty:
            dates = pd.to_datetime(contract_outliers['Date'])
            values = contract_outliers['Value']
            plt.scatter(dates, values, s=100, marker='o', edgecolors='red', facecolors='none', label=f"Outliers")
        
        plt.title(f'{contract} with Potential Outliers')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(f"{output_dir}/{contract}_outliers.png")
        else:
            plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect outliers in VX continuous contract data')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--plots', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--zscore', type=float, default=4.0, help='Z-score threshold (default: 4.0)')
    parser.add_argument('--iqr', type=float, default=3.0, help='IQR multiplier (default: 3.0)')
    parser.add_argument('--pct-change', type=float, default=30.0, help='Percentage change threshold (default: 30.0)')
    parser.add_argument('--min-value', type=float, default=1.0, help='Minimum valid value (default: 1.0)')
    parser.add_argument('--max-value', type=float, default=100.0, help='Maximum valid value (default: 100.0)')
    parser.add_argument('--methods', type=str, default='all', help='Comma-separated list of methods to use (zscore,iqr,pct_change,absolute,vix_compare) or "all"')
    parser.add_argument('--cutoff-year', type=int, default=None, help='Only analyze data from this year forward (e.g., 2007)')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_vx_data(args.db_path, args.start_date, args.end_date)
    
    # Apply cutoff year if specified
    if args.cutoff_year is not None:
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        df = df[df['Year'] >= args.cutoff_year]
        print(f"Applied cutoff year: {args.cutoff_year}+")
    
    # Determine which methods to use
    if args.methods.lower() == 'all':
        methods = ['zscore', 'iqr', 'pct_change', 'absolute', 'vix_compare']
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    # Detect outliers
    outliers_df = detect_outliers(
        df,
        methods=methods,
        zscore_threshold=args.zscore,
        iqr_multiplier=args.iqr,
        pct_change_threshold=args.pct_change,
        min_value_threshold=args.min_value,
        max_value_threshold=args.max_value
    )
    
    # Print results
    print(f"\nAnalysis from {args.start_date} to {args.end_date}")
    print(f"Total dates analyzed: {len(df)}")
    
    # Check available contracts in data - Use the new format
    available_contracts = [f'@VX={{i}}01XN' for i in range(1, 6)]
    available_contracts_in_df = [col for col in available_contracts if col in df.columns]
    print(f"Available contracts in data: {', '.join(available_contracts_in_df)}")
    
    # Print outliers summary
    if not outliers_df.empty:
        print(f"\nFound {len(outliers_df)} potential outliers:")
        print(tabulate(outliers_df, headers='keys', tablefmt='pretty', showindex=False))
        
        # Summarize by contract
        print("\nOutliers by Contract:")
        contract_summary = outliers_df['Contract'].value_counts().sort_index()
        for contract, count in contract_summary.items():
            print(f"  {contract}: {count} outliers")
        
        # Summarize by method
        print("\nOutliers by Method:")
        method_summary = outliers_df['Method'].value_counts().sort_index()
        for method, count in method_summary.items():
            print(f"  {method}: {count} outliers")
    else:
        print("\nNo outliers detected with the specified thresholds.")
    
    # Output to CSV if specified
    if args.output and not outliers_df.empty:
        outliers_df.to_csv(args.output, index=False)
        print(f"\nOutliers saved to {args.output}")
    
    # Create plots if requested
    if args.plots:
        plot_data(df, outliers_df, args.plots)
        print(f"Plots saved to {args.plots}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install required libraries: pip install duckdb pandas numpy matplotlib tabulate") 