import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb')

# Function to get counts by year
def get_counts_by_year(symbol):
    query = f"""
    SELECT 
        EXTRACT(YEAR FROM timestamp) AS year, 
        COUNT(*) AS count,
        MIN(timestamp) AS first_date,
        MAX(timestamp) AS last_date
    FROM market_data 
    WHERE Symbol = '{symbol}' 
    GROUP BY year 
    ORDER BY year
    """
    return conn.execute(query).fetchdf()

# Check counts for VIX index
print("=== $VIX.X Data Counts by Year ===")
vix_counts = get_counts_by_year('$VIX.X')
print(vix_counts)
print()

# Check counts for VXc1
print("=== VXc1 Data Counts by Year ===")
vxc1_counts = get_counts_by_year('VXc1')
print(vxc1_counts)
print()

# Check counts for VXc2
print("=== VXc2 Data Counts by Year ===")
vxc2_counts = get_counts_by_year('VXc2')
print(vxc2_counts)
print()

# Check total counts
print("=== Total Data Counts ===")
symbols = ['$VIX.X', 'VXc1', 'VXc2']
for symbol in symbols:
    count = conn.execute(f"SELECT COUNT(*) FROM market_data WHERE Symbol = '{symbol}'").fetchone()[0]
    print(f"{symbol}: {count} rows")

# Check individual VX contract counts
vx_count = conn.execute("SELECT COUNT(DISTINCT Symbol) FROM market_data WHERE Symbol LIKE 'VX%' AND Symbol NOT LIKE 'VXc%'").fetchone()[0]
print(f"VX contracts: {vx_count} distinct symbols")

# Close the connection
conn.close() 