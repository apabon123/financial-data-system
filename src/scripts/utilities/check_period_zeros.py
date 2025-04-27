#!/usr/bin/env python
# -*- coding: utf-8 -*-

import duckdb
import pandas as pd
import argparse
from datetime import datetime

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Check for zero prices in a date range.")
    parser.add_argument("--start-date", type=str, default="2007-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2007-03-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Connect to database
    conn = duckdb.connect('data/financial_data.duckdb')
    
    # Define symbols to check
    symbols = ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']
    
    print(f"Checking for zero values between {args.start_date} and {args.end_date}:")
    print("=" * 50)
    
    for symbol in symbols:
        # Query for zero prices in the range
        zero_query = f"""
        SELECT COUNT(*) as count
        FROM market_data 
        WHERE symbol = '{symbol}' 
          AND (settle = 0.0 OR settle IS NULL)
          AND timestamp BETWEEN '{args.start_date}' AND '{args.end_date}'
        """
        
        zero_count = conn.execute(zero_query).fetchone()[0]
        
        # Get total rows for the range
        total_query = f"""
        SELECT COUNT(*) as count
        FROM market_data 
        WHERE symbol = '{symbol}'
          AND timestamp BETWEEN '{args.start_date}' AND '{args.end_date}'
        """
        
        total_count = conn.execute(total_query).fetchone()[0]
        
        print(f"{symbol}: {zero_count} zeros out of {total_count} rows")
        
        # If there are still zeros, show details
        if zero_count > 0:
            detail_query = f"""
            SELECT timestamp::DATE as date_value, open, high, low, close, settle, source
            FROM market_data 
            WHERE symbol = '{symbol}' 
              AND (settle = 0.0 OR settle IS NULL)
              AND timestamp BETWEEN '{args.start_date}' AND '{args.end_date}'
            ORDER BY timestamp
            """
            
            details = conn.execute(detail_query).fetchdf()
            
            print("\nZero value details:")
            print(details)
            print("")
    
    # Additional check: get sources for filled values
    print("\nSources for filled values:")
    print("=" * 50)
    
    filled_query = f"""
    SELECT 
        symbol, 
        source, 
        COUNT(*) as count
    FROM market_data 
    WHERE symbol IN ('VXc2', 'VXc3', 'VXc4', 'VXc5')
      AND source LIKE 'DERIVED_ZERO_FILLED%'
      AND timestamp BETWEEN '{args.start_date}' AND '{args.end_date}'
    GROUP BY symbol, source
    ORDER BY symbol, source
    """
    
    filled_results = conn.execute(filled_query).fetchdf()
    
    if not filled_results.empty:
        print(filled_results)
    else:
        print("No derived filled values found.")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main() 