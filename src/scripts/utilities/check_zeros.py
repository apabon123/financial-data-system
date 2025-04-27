#!/usr/bin/env python
# -*- coding: utf-8 -*-

import duckdb
import pandas as pd
from datetime import datetime

# Connect to database
conn = duckdb.connect('data/financial_data.duckdb')

# Define symbols to check
symbols = ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']

# Check zeros for each symbol
print(f"Zero values in continuous contracts:")
print("=" * 50)

for symbol in symbols:
    # Query for zero prices
    query = f"""
    SELECT COUNT(*) as count_zeros, 
           MIN(timestamp) as min_date, 
           MAX(timestamp) as max_date
    FROM market_data 
    WHERE symbol = '{symbol}' AND (settle = 0.0 OR settle IS NULL)
    """
    
    result = conn.execute(query).fetchdf()
    
    # Get total rows for percentage calculation
    total_query = f"""
    SELECT COUNT(*) as total
    FROM market_data 
    WHERE symbol = '{symbol}'
    """
    
    total = conn.execute(total_query).fetchone()[0]
    zeros = result['count_zeros'][0]
    
    # Format dates
    min_date = result['min_date'][0]
    max_date = result['max_date'][0]
    
    if zeros > 0:
        min_date_str = min_date.strftime('%Y-%m-%d') if min_date else 'N/A'
        max_date_str = max_date.strftime('%Y-%m-%d') if max_date else 'N/A'
        percent = (zeros / total * 100) if total > 0 else 0
        print(f"{symbol}: {zeros} zeros ({percent:.2f}%) - Range: {min_date_str} to {max_date_str}")
        
        # Show sample of zero dates
        sample_query = f"""
        SELECT timestamp::DATE as date_value
        FROM market_data 
        WHERE symbol = '{symbol}' AND (settle = 0.0 OR settle IS NULL)
        ORDER BY timestamp
        LIMIT 5
        """
        
        samples = conn.execute(sample_query).fetchdf()
        if not samples.empty:
            sample_dates = [d.strftime('%Y-%m-%d') for d in samples['date_value']]
            print(f"  Sample dates: {', '.join(sample_dates)}...")
    else:
        print(f"{symbol}: No zero values found (total rows: {total})")

print("\nRunning check for early 2004 data...")
print("=" * 50)

# Check for early 2004 data
for symbol in symbols:
    early_2004_query = f"""
    SELECT COUNT(*) as count
    FROM market_data 
    WHERE symbol = '{symbol}' AND timestamp BETWEEN '2004-01-01' AND '2004-03-31'
    """
    
    count = conn.execute(early_2004_query).fetchone()[0]
    
    if count > 0:
        print(f"{symbol}: {count} rows exist for Jan-Mar 2004")
    else:
        print(f"{symbol}: No data for Jan-Mar 2004")

# Close connection
conn.close() 