import duckdb
import pandas as pd
from datetime import datetime, timedelta

def check_continuous_data():
    # Connect to the database
    con = duckdb.connect('data/financial_data.duckdb')
    
    # Get summary statistics
    print("Summary of 15-minute continuous contract data:")
    summary = con.execute('''
        SELECT 
            symbol,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            COUNT(*) as record_count
        FROM continuous_contracts 
        WHERE (symbol LIKE '@ES%' OR symbol LIKE '@NQ%')
        AND interval_value = 15
        AND interval_unit = 'minute'
        GROUP BY symbol 
        ORDER BY symbol
    ''').fetchdf()
    print("\nData Summary:")
    print(summary.to_string())
    
    # Check for gaps in data
    print("\nChecking for gaps > 1 trading day:")
    gaps = con.execute('''
        WITH date_gaps AS (
            SELECT 
                symbol,
                timestamp,
                LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp
            FROM continuous_contracts
            WHERE (symbol LIKE '@ES%' OR symbol LIKE '@NQ%')
            AND interval_value = 15
            AND interval_unit = 'minute'
        )
        SELECT 
            symbol,
            timestamp,
            prev_timestamp,
            (timestamp - prev_timestamp) as gap_duration
        FROM date_gaps
        WHERE (timestamp - prev_timestamp) > INTERVAL '30 minutes'  -- Increased threshold for 15-min data
        ORDER BY symbol, timestamp
    ''').fetchdf()
    
    if len(gaps) > 0:
        print("\nFound gaps in data:")
        print(gaps.to_string())
    else:
        print("\nNo significant gaps found in the data.")
    
    # Show latest data points
    print("\nLatest 15-minute data points:")
    latest = con.execute('''
        SELECT *
        FROM continuous_contracts
        WHERE (symbol LIKE '@ES%' OR symbol LIKE '@NQ%')
        AND interval_value = 15
        AND interval_unit = 'minute'
        ORDER BY timestamp DESC
        LIMIT 5
    ''').fetchdf()
    print(latest.to_string())
    
    con.close()

if __name__ == "__main__":
    check_continuous_data() 