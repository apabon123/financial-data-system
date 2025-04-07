import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('./data/financial_data.duckdb')

# Check ES futures data
print("\nES Futures Data Summary:")
print("-----------------------")

# Get interval distribution
print("\nInterval Distribution:")
result = conn.execute("""
    SELECT DISTINCT 
        interval_value, 
        interval_unit, 
        COUNT(*) as count 
    FROM market_data 
    WHERE symbol LIKE 'ES%' 
    GROUP BY interval_value, interval_unit
    ORDER BY interval_unit, interval_value
""").fetchdf()
print(result)

# Get date range
print("\nDate Range:")
result = conn.execute("""
    SELECT 
        MIN(timestamp) as start_date,
        MAX(timestamp) as end_date,
        COUNT(*) as total_records
    FROM market_data 
    WHERE symbol LIKE 'ES%'
""").fetchdf()
print(result)

# Get contract distribution
print("\nContract Distribution:")
result = conn.execute("""
    SELECT 
        symbol,
        COUNT(*) as count,
        MIN(timestamp) as start_date,
        MAX(timestamp) as end_date
    FROM market_data 
    WHERE symbol LIKE 'ES%'
    GROUP BY symbol
    ORDER BY symbol
""").fetchdf()
print(result)

conn.close() 