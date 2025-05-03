import duckdb

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Check for November 13, 2024
print("=== VX series available for Nov 13, 2024 ===")
query = """
SELECT symbol 
FROM continuous_contracts 
WHERE timestamp = '2024-11-13' 
AND symbol LIKE '@VX=%' 
ORDER BY symbol
"""
symbols = conn.execute(query).fetchdf()
print(symbols)

# Check for a random day in November to verify consistency
print("\n=== VX series available for Nov 19, 2024 ===")
query2 = """
SELECT symbol 
FROM continuous_contracts 
WHERE timestamp = '2024-11-19' 
AND symbol LIKE '@VX=%' 
ORDER BY symbol
"""
symbols2 = conn.execute(query2).fetchdf()
print(symbols2)

# Verify the data behind a sample contract
print("\n=== Data for @VX=101XN on Nov 13, 2024 ===")
data_query = """
SELECT timestamp, symbol, underlying_symbol, open, high, low, close, settle, built_by
FROM continuous_contracts 
WHERE timestamp = '2024-11-13' 
AND symbol = '@VX=101XN'
"""
data = conn.execute(data_query).fetchdf()
print(data) 