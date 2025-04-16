import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('./data/financial_data.duckdb')

# Check market_data table
print("Market Data Table:")
market_data = conn.execute("SELECT DISTINCT symbol FROM market_data WHERE symbol LIKE 'VIX%' OR symbol LIKE 'VX%'").fetchall()
print(market_data)

# Check continuous_contracts table
print("\nContinuous Contracts Table:")
continuous_contracts = conn.execute("SELECT DISTINCT symbol FROM continuous_contracts WHERE symbol LIKE 'VX%'").fetchall()
print(continuous_contracts)

# Check daily_bars table
print("\nDaily Bars Table:")
daily_bars = conn.execute("SELECT DISTINCT symbol FROM daily_bars WHERE symbol LIKE 'VIX%' OR symbol LIKE 'VX%'").fetchall()
print(daily_bars)

# Close the connection
conn.close() 