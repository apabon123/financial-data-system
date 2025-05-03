import duckdb
import pandas as pd
from datetime import datetime

# Connect to the database
conn = duckdb.connect('data/financial_data.duckdb', read_only=True)

# Function to check missing dates for a specific period
def check_missing_dates(start_date, end_date):
    print(f"\n=== Checking period from {start_date} to {end_date} ===")
    
    # Get all dates where VIX data exists
    vix_dates_query = f"""
    SELECT DISTINCT CAST(timestamp AS DATE) as date
    FROM market_data_cboe
    WHERE symbol = '$VIX.X'
    AND timestamp >= '{start_date}' 
    AND timestamp <= '{end_date}'
    ORDER BY date
    """
    vix_dates = conn.execute(vix_dates_query).fetchdf()

    # Get all dates where VX continuous data exists
    vx_dates_query = f"""
    SELECT DISTINCT CAST(timestamp AS DATE) as date
    FROM continuous_contracts
    WHERE symbol LIKE '@VX=%'
    AND timestamp >= '{start_date}' 
    AND timestamp <= '{end_date}'
    ORDER BY date
    """
    vx_dates = conn.execute(vx_dates_query).fetchdf()

    # Convert to sets for comparison
    vix_dates_set = set(vix_dates['date'].tolist())
    vx_dates_set = set(vx_dates['date'].tolist())

    # Find missing dates
    missing_dates = sorted(list(vix_dates_set - vx_dates_set))

    print(f"Total VIX dates: {len(vix_dates)}")
    print(f"Total VX continuous dates: {len(vx_dates)}")
    print(f"Total missing dates: {len(missing_dates)}")

    if missing_dates:
        print("\nMissing dates:")
        for date in missing_dates:
            print(date)
    else:
        print("\nNo missing dates found! All VIX dates have corresponding VX continuous data.")
    
    return missing_dates, vix_dates_set, vx_dates_set

# Check 2024-2025 data
missing_dates, vix_dates_set, vx_dates_set = check_missing_dates('2024-11-01', '2025-02-28')

# List all VIX dates to see if there actually are VIX values for these dates
print("\n=== Listing all VIX dates and values in this period ===")
vix_query = """
SELECT timestamp, symbol, close, source, 
    CASE DAYOFWEEK(timestamp)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_of_week,
    CASE 
        WHEN timestamp = '2024-11-28' THEN 'Thanksgiving'
        WHEN timestamp = '2024-12-25' THEN 'Christmas'
        WHEN timestamp = '2025-01-01' THEN 'New Year''s Day'
        WHEN timestamp = '2025-01-20' THEN 'Martin Luther King Jr. Day'
        WHEN timestamp = '2025-02-17' THEN 'Presidents'' Day'
        ELSE 'Not a Holiday'
    END as holiday
FROM market_data_cboe
WHERE symbol = '$VIX.X'
AND timestamp >= '2024-11-01' 
AND timestamp <= '2025-02-28'
ORDER BY timestamp
"""
vix_data = conn.execute(vix_query).fetchdf()
if not vix_data.empty:
    print(vix_data)
else:
    print("No VIX data found for this period.") 