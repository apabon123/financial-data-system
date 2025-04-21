#!/usr/bin/env python
# -*- coding: utf-8 -*-

import duckdb
import pandas as pd
from datetime import datetime

# Connect to database
conn = duckdb.connect('data/financial_data.duckdb')

# Query for VIX data in early January 2007
vix_query = """
SELECT timestamp::DATE as date, settle 
FROM market_data 
WHERE symbol = '$VIX.X' 
  AND timestamp BETWEEN '2007-01-01' AND '2007-01-10'
ORDER BY timestamp
"""

vix_data = conn.execute(vix_query).fetchdf()
print("VIX data for early January 2007:")
print(vix_data)
print()

# Check if January 1st and 2nd are market holidays
print("Market holidays check:")
dates_to_check = ['2007-01-01', '2007-01-02']
for date_str in dates_to_check:
    # Check if there's any financial data for this date (across all symbols)
    holiday_query = f"""
    SELECT COUNT(*) as data_count
    FROM market_data
    WHERE timestamp::DATE = '{date_str}'
      AND settle IS NOT NULL
      AND settle != 0.0
    """
    
    data_count = conn.execute(holiday_query).fetchone()[0]
    
    if data_count > 0:
        print(f"{date_str}: {data_count} valid data points found - NOT a holiday")
    else:
        print(f"{date_str}: No valid data found - Likely a holiday")

# Get a sample of financial data for January 3, 2007 (first likely trading day)
sample_query = """
SELECT symbol, settle, source
FROM market_data
WHERE timestamp::DATE = '2007-01-03'
  AND settle IS NOT NULL
  AND settle != 0.0
ORDER BY symbol
LIMIT 10
"""

sample_data = conn.execute(sample_query).fetchdf()
print("\nSample data for 2007-01-03:")
print(sample_data)

# Close connection
conn.close() 