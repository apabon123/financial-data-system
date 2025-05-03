import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Check the symbols actually used in the continuous_contracts table for November 2024
print("=== Continuous contracts for VX in November 2024 ===")
query = """
SELECT DISTINCT symbol
FROM continuous_contracts
WHERE EXTRACT(YEAR FROM timestamp) = 2024
AND EXTRACT(MONTH FROM timestamp) = 11
AND symbol LIKE '@VX=%'
ORDER BY symbol
"""
symbols = conn.execute(query).fetchdf()
print(symbols)

# Look at the contract mapping
print("\n=== Contract mapping for November 13, 2024 ===")
mapping_query = """
SELECT continuous_symbol, underlying_symbol
FROM continuous_contract_mapping
WHERE date = '2024-11-13'
AND continuous_symbol LIKE '@VX=%'
ORDER BY continuous_symbol
"""
mappings = conn.execute(mapping_query).fetchdf()
print(mappings)

# Check what's in the generate_continuous_futures.py function
# Get a sample date and see the complete records for it
print("\n=== Full record data for Nov 13, 2024 ===")
data_query = """
SELECT *
FROM continuous_contracts
WHERE EXTRACT(YEAR FROM timestamp) = 2024
AND EXTRACT(MONTH FROM timestamp) = 11
AND EXTRACT(DAY FROM timestamp) = 13
AND symbol LIKE '@VX=%' 
ORDER BY symbol
"""
data = conn.execute(data_query).fetchdf()
print(data)

# Now run the fix_vx_continuous.py script for a sample date and see if it creates all 5 series
print("\n=== Investigating what get_active_contracts_from_mapping returns ===")
active_contracts_query = """
SELECT continuous_symbol, underlying_symbol
FROM continuous_contract_mapping
WHERE date = '2024-11-13'
AND continuous_symbol LIKE '@VX=%'
ORDER BY continuous_symbol
"""
active_contracts = conn.execute(active_contracts_query).fetchdf()
print("Active contracts from mapping:")
print(active_contracts) 