import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Check for completeness of series for each date
print("=== Completeness check for VX series ===")
query = """
SELECT 
    CAST(timestamp AS DATE) as date, 
    COUNT(DISTINCT symbol) as series_count,
    STRING_AGG(symbol, ', ' ORDER BY symbol) as available_series
FROM continuous_contracts 
WHERE timestamp BETWEEN '2024-11-01' AND '2025-02-28' 
AND symbol LIKE '@VX=%' 
GROUP BY date 
ORDER BY date
"""
completeness = conn.execute(query).fetchdf()

# Check which dates have all required series
completeness['has_standard_series'] = completeness['available_series'].apply(
    lambda x: all(f'@VX={i}01XN' in x for i in range(1, 6))
)

print(f"Total dates with VX continuous data: {len(completeness)}")
print(f"Dates with standard series (101-501): {completeness['has_standard_series'].sum()}")

# Print out the first few rows to verify
print("\nSample of results:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(completeness.head(10))

# Check for any dates missing standard series
print("\n=== Dates missing standard series ===")
missing_standard = completeness[~completeness['has_standard_series']]
if not missing_standard.empty:
    print(missing_standard)
else:
    print("All dates have the standard series (101-501)!")

# Check for any remaining days with only higher series
print("\n=== Days with only higher series ===")
higher_only = conn.execute("""
WITH higher_series AS (
    SELECT DISTINCT CAST(timestamp AS DATE) as date
    FROM continuous_contracts
    WHERE symbol LIKE '@VX=%'
    AND (symbol LIKE '@VX=6%' OR symbol LIKE '@VX=7%' OR symbol LIKE '@VX=8%' OR symbol LIKE '@VX=9%')
    AND timestamp BETWEEN '2024-11-01' AND '2025-02-28'
),
standard_series AS (
    SELECT DISTINCT CAST(timestamp AS DATE) as date
    FROM continuous_contracts
    WHERE symbol LIKE '@VX=%'
    AND (symbol LIKE '@VX=1%' OR symbol LIKE '@VX=2%' OR symbol LIKE '@VX=3%' OR symbol LIKE '@VX=4%' OR symbol LIKE '@VX=5%')
    AND timestamp BETWEEN '2024-11-01' AND '2025-02-28'
)
SELECT * FROM higher_series
WHERE date NOT IN (SELECT date FROM standard_series)
ORDER BY date
""").fetchdf()

if not higher_only.empty:
    print(higher_only)
else:
    print("No days found with only higher series - All fixed!") 