import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('./data/financial_data.duckdb')

# Check daily_bars table
print("Daily Bars Table - VIX-related symbols:")
daily_bars = conn.execute("SELECT DISTINCT symbol FROM daily_bars WHERE symbol LIKE '%VIX%' ORDER BY symbol").fetchall()
print(daily_bars)

# Check market_data table
print("\nMarket Data Table - VIX-related symbols:")
market_data = conn.execute("SELECT DISTINCT symbol FROM market_data WHERE symbol LIKE '%VIX%' ORDER BY symbol").fetchall()
print(market_data)

# Check continuous_contracts table
print("\nContinuous Contracts Table - VIX-related symbols:")
continuous_contracts = conn.execute("SELECT DISTINCT symbol FROM continuous_contracts WHERE symbol LIKE '%VIX%' ORDER BY symbol").fetchall()
print(continuous_contracts)

# Close the connection
conn.close() 