import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('./data/financial_data.duckdb')

# Check continuous_contracts table for VXc1
print("VXc1 data for April 2, 2004:")
vxc1_data = conn.execute("""
    SELECT * FROM continuous_contracts 
    WHERE symbol = 'VXc1' 
    AND DATE_TRUNC('day', timestamp) = '2004-04-02'
""").fetchdf()
print(vxc1_data)

# Check daily_bars table for VXJ04
print("\nVXJ04 data for April 2, 2004:")
vxj04_data = conn.execute("""
    SELECT * FROM daily_bars 
    WHERE symbol = 'VXJ04' 
    AND date = '2004-04-02'
""").fetchdf()
print(vxj04_data)

# Check daily_bars table for VXK04
print("\nVXK04 data for April 2, 2004:")
vxk04_data = conn.execute("""
    SELECT * FROM daily_bars 
    WHERE symbol = 'VXK04' 
    AND date = '2004-04-02'
""").fetchdf()
print(vxk04_data)

# Check all VX contracts for April 2, 2004
print("\nAll VX contracts for April 2, 2004:")
all_vx_data = conn.execute("""
    SELECT * FROM daily_bars 
    WHERE symbol LIKE 'VX%' 
    AND date = '2004-04-02'
    ORDER BY symbol
""").fetchdf()
print(all_vx_data)

# Close the connection
conn.close() 