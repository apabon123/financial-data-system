import duckdb

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Get all continuous contract symbols for November 13
print("=== All VX Continuous symbols for Nov 13, 2024 ===")
query = """
SELECT symbol
FROM continuous_contracts
WHERE timestamp = '2024-11-13' 
AND symbol LIKE '@VX=%'
ORDER BY symbol
"""
symbols = conn.execute(query).fetchdf()
print(symbols)

# Check which series are available for the entire month
print("\n=== VX Continuous Series Available in November 2024 ===")
series_query = """
SELECT SUBSTRING(symbol, 1, 8) as series, 
       COUNT(DISTINCT CAST(timestamp AS DATE)) as num_days
FROM continuous_contracts
WHERE EXTRACT(YEAR FROM timestamp) = 2024
AND EXTRACT(MONTH FROM timestamp) = 11
AND symbol LIKE '@VX=%'
GROUP BY series
ORDER BY series
"""
series = conn.execute(series_query).fetchdf()
print(series)

# Check if the first 5 series (101-501) are available for an earlier date
print("\n=== Checking for series 101-501 on Jan 2, 2024 ===")
jan_query = """
SELECT symbol
FROM continuous_contracts
WHERE timestamp = '2024-01-02'
AND symbol LIKE '@VX=%'
ORDER BY symbol
"""
jan_symbols = conn.execute(jan_query).fetchdf()
print(jan_symbols) 