#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to check roll calendar data
"""

import duckdb
import pandas as pd

# Database path
DB_PATH = "data/financial_data.duckdb"

def main():
    # Connect to database
    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # Get latest 10 roll calendar entries
    latest_query = """
    SELECT * FROM futures_roll_calendar 
    WHERE root_symbol = 'VX' 
    ORDER BY last_trading_day DESC 
    LIMIT 10
    """
    latest_entries = conn.execute(latest_query).fetchdf()
    print("\nLatest 10 roll calendar entries:")
    print(latest_entries)

    # Check roll calendar entries for December 2023 - January 2024
    period_query = """
    SELECT * FROM futures_roll_calendar 
    WHERE root_symbol = 'VX' 
    AND last_trading_day BETWEEN '2023-12-01' AND '2024-01-31'
    ORDER BY last_trading_day ASC
    """
    period_entries = conn.execute(period_query).fetchdf()
    print("\nRoll calendar entries for Dec 2023 - Jan 2024:")
    print(period_entries)

    # Close connection
    conn.close()
    print("\nDatabase connection closed")

if __name__ == "__main__":
    main() 