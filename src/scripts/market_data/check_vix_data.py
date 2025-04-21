#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check if VIX data is properly accessible and display the first few rows
"""

import duckdb
import pandas as pd

DB_PATH = "data/financial_data.duckdb"
VIX_SYMBOL = "$VIX.X"

def main():
    # Connect to database
    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # Check count of VIX rows
    count_query = f"SELECT COUNT(*) FROM market_data WHERE symbol = '{VIX_SYMBOL}'"
    count = conn.execute(count_query).fetchone()[0]
    print(f"VIX row count: {count}")

    # Check date range
    range_query = f"SELECT MIN(timestamp), MAX(timestamp) FROM market_data WHERE symbol = '{VIX_SYMBOL}'"
    date_range = conn.execute(range_query).fetchall()
    print(f"VIX date range: {date_range}")

    # Get the first 5 rows
    first_rows_query = f"SELECT timestamp, symbol, open, high, low, settle FROM market_data WHERE symbol = '{VIX_SYMBOL}' ORDER BY timestamp LIMIT 5"
    first_rows = conn.execute(first_rows_query).fetchdf()
    print("First 5 rows of VIX data:")
    print(first_rows)

    # Get VIX rows from early 2004
    early_2004_query = f"SELECT timestamp, symbol, open, high, low, settle FROM market_data WHERE symbol = '{VIX_SYMBOL}' AND timestamp BETWEEN '2004-01-01' AND '2004-03-31' ORDER BY timestamp LIMIT 5"
    early_2004 = conn.execute(early_2004_query).fetchdf()
    print("\nEarly 2004 VIX data:")
    print(early_2004)

    # Check any VIX data in 2004-2005
    period_query = f"SELECT COUNT(*) FROM market_data WHERE symbol = '{VIX_SYMBOL}' AND timestamp BETWEEN '2004-01-01' AND '2005-12-31'"
    period_count = conn.execute(period_query).fetchone()[0]
    print(f"\nVIX rows between 2004-01-01 and 2005-12-31: {period_count}")

    # Close connection
    conn.close()
    print("Database connection closed")

if __name__ == "__main__":
    main() 