import duckdb

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb')

# Query for distinct intervals
result = conn.execute("""
    SELECT DISTINCT interval_value, interval_unit, COUNT(*) as count
    FROM market_data 
    WHERE symbol = 'VXF10'
    GROUP BY interval_value, interval_unit
    ORDER BY interval_value, interval_unit
""").fetchdf()

print(result) 