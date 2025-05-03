import duckdb

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Check contract mappings for November 13, 2024
print("=== Contract mappings for Nov 13, 2024 ===")
query = """
SELECT continuous_symbol, underlying_symbol 
FROM continuous_contract_mapping 
WHERE date = '2024-11-13' 
AND continuous_symbol LIKE '@VX=%' 
ORDER BY continuous_symbol
"""
mappings = conn.execute(query).fetchdf()
print(mappings)

# Check January 2024 mappings for comparison
print("\n=== Contract mappings for Jan 2, 2024 ===")
jan_query = """
SELECT continuous_symbol, underlying_symbol 
FROM continuous_contract_mapping 
WHERE date = '2024-01-02' 
AND continuous_symbol LIKE '@VX=%' 
ORDER BY continuous_symbol
"""
jan_mappings = conn.execute(jan_query).fetchdf()
print(jan_mappings)

# Check all distinct continuous symbols in the mappings for November 2024
print("\n=== All distinct continuous symbols for November 2024 ===")
all_symbols_query = """
SELECT DISTINCT continuous_symbol
FROM continuous_contract_mapping 
WHERE date >= '2024-11-01' 
AND date <= '2024-11-30'
AND continuous_symbol LIKE '@VX=%' 
ORDER BY continuous_symbol
"""
all_symbols = conn.execute(all_symbols_query).fetchdf()
print(all_symbols) 