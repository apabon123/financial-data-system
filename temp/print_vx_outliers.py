import duckdb
import pandas as pd
import argparse
from datetime import datetime
from detect_vx_outliers import load_vx_data, detect_outliers

def print_vx_outliers(db_path, start_date, end_date, output_file, zscore_threshold=4.0, 
                      iqr_multiplier=3.0, pct_change_threshold=30.0, min_value_threshold=0.0):
    """
    Print VXc1 and VXc2 outliers with VIX values to a file.
    """
    # Load the data
    df = load_vx_data(db_path, start_date, end_date)
    
    # Detect outliers
    outliers_df = detect_outliers(
        df,
        methods=['zscore', 'iqr', 'pct_change', 'vix_compare'],
        zscore_threshold=zscore_threshold,
        iqr_multiplier=iqr_multiplier,
        pct_change_threshold=pct_change_threshold,
        min_value_threshold=min_value_threshold
    )
    
    # Filter to only VXc1 and VXc2 outliers
    filtered_outliers = outliers_df[outliers_df['Contract'].isin(['VXc1', 'VXc2'])]
    
    # Get unique dates from the outliers
    outlier_dates = pd.to_datetime(filtered_outliers['Date']).unique()
    
    # Create output dataframe with required columns
    output_data = []
    for date in outlier_dates:
        # Get the row from the original data for this date
        date_data = df[df['Date'] == date]
        if not date_data.empty:
            row_data = date_data.iloc[0]
            vix_value = row_data.get('VIX', None)
            vxc1_value = row_data.get('VXc1', None)
            vxc2_value = row_data.get('VXc2', None)
            
            # Get outlier info for this date
            date_outliers = filtered_outliers[pd.to_datetime(filtered_outliers['Date']) == date]
            outlier_info = []
            for _, outlier in date_outliers.iterrows():
                outlier_info.append(f"{outlier['Contract']}: {outlier['Method']} - {outlier['Detail']}")
            
            output_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'VIX': vix_value,
                'VXc1': vxc1_value,
                'VXc2': vxc2_value,
                'Outlier_Info': '; '.join(outlier_info)
            })
    
    # Create dataframe and save to file
    output_df = pd.DataFrame(output_data)
    
    # Sort by date
    output_df = output_df.sort_values('Date')
    
    # Save to file
    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} outlier records to {output_file}")
    
    # Also print to console
    print("\nOutlier Data Preview:")
    print(output_df.head(10).to_string())
    print(f"\nTotal outlier dates: {len(output_df)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Print VXc1 and VXc2 outliers with VIX values')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', 
                        help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'), 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='vx_outliers.csv', 
                        help='Output CSV file path')
    parser.add_argument('--zscore', type=float, default=4.0, 
                        help='Z-score threshold (default: 4.0)')
    parser.add_argument('--iqr', type=float, default=3.0, 
                        help='IQR multiplier (default: 3.0)')
    parser.add_argument('--pct-change', type=float, default=30.0, 
                        help='Percentage change threshold (default: 30.0)')
    parser.add_argument('--min-value', type=float, default=0.0, 
                        help='Minimum valid value (default: 0.0)')
    
    args = parser.parse_args()
    
    print_vx_outliers(
        args.db_path,
        args.start_date,
        args.end_date,
        args.output,
        args.zscore,
        args.iqr,
        args.pct_change,
        args.min_value
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}") 