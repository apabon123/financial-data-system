import duckdb
import pandas as pd
from datetime import datetime

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Check the VIX data for November 13, 2024
print("=== VIX Data for November 13, 2024 ===")
vix_query = """
SELECT timestamp, symbol, open, high, low, close, source
FROM market_data_cboe
WHERE symbol = '$VIX.X'
AND timestamp = '2024-11-13'
"""
vix_data = conn.execute(vix_query).fetchdf()
print(vix_data)

# Check if there's continuous data for November 13, 2024
print("\n=== VX Continuous Data for November 13, 2024 ===")
vx_query = """
SELECT timestamp, symbol, open, high, low, close, source
FROM continuous_contracts
WHERE symbol LIKE '@VX=%'
AND timestamp = '2024-11-13'
"""
vx_data = conn.execute(vx_query).fetchdf()
print(vx_data)

# List all November 2024 dates with continuous contracts
print("\n=== All November 2024 dates with VX continuous data ===")
nov_query = """
SELECT DISTINCT CAST(timestamp AS DATE) as date
FROM continuous_contracts
WHERE symbol LIKE '@VX=%'
AND EXTRACT(YEAR FROM timestamp) = 2024
AND EXTRACT(MONTH FROM timestamp) = 11
ORDER BY date
"""
nov_dates = conn.execute(nov_query).fetchdf()
print(nov_dates)

# List all VIX dates in November 2024
print("\n=== All November 2024 dates with VIX data ===")
vix_nov_query = """
SELECT DISTINCT CAST(timestamp AS DATE) as date
FROM market_data_cboe
WHERE symbol = '$VIX.X'
AND EXTRACT(YEAR FROM timestamp) = 2024
AND EXTRACT(MONTH FROM timestamp) = 11
ORDER BY date
"""
vix_nov_dates = conn.execute(vix_nov_query).fetchdf()
print(vix_nov_dates)

# Compare to find missing dates
print("\n=== Checking for missing dates in November 2024 ===")
vix_dates_set = set(vix_nov_dates['date'].tolist())
vx_dates_set = set(nov_dates['date'].tolist())
missing = sorted(list(vix_dates_set - vx_dates_set))

print(f"Total VIX dates in Nov 2024: {len(vix_dates_set)}")
print(f"Total VX continuous dates in Nov 2024: {len(vx_dates_set)}")
print(f"Total missing dates: {len(missing)}")

if missing:
    print("\nMissing dates:")
    for date in missing:
        print(date)
else:
    print("\nNo missing dates! All VIX dates have corresponding VX continuous data.") 