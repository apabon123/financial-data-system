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

# Check 2024 data
missing_2024, vix_dates_2024, vx_dates_2024 = check_missing_dates('2024-01-01', '2024-08-01')

# Check 2023 data
missing_2023, vix_dates_2023, vx_dates_2023 = check_missing_dates('2023-01-01', '2023-12-31')

# Check US market holidays for 2023
us_holidays_2023 = [
    '2023-01-02',  # New Year's Day (observed)
    '2023-01-16',  # Martin Luther King Jr. Day
    '2023-02-20',  # Presidents' Day
    '2023-04-07',  # Good Friday
    '2023-05-29',  # Memorial Day
    '2023-06-19',  # Juneteenth
    '2023-07-04',  # Independence Day
    '2023-09-04',  # Labor Day
    '2023-11-23',  # Thanksgiving
    '2023-12-25',  # Christmas
]

print("\n=== Checking if missing dates are US market holidays ===")
for date_str in us_holidays_2023:
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    in_missing = date in [d.date() if hasattr(d, 'date') else d for d in missing_2023]
    print(f"{date}: Is a US holiday: True, In missing dates: {in_missing}")

# Check if the known missing dates are actually trading days
known_missing_2024 = ['2024-01-15', '2024-02-19', '2024-05-27', '2024-06-19', '2024-07-04']
print("\n=== Checking known missing dates in 2024 ===")
for date_str in known_missing_2024:
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    in_vix = date in vix_dates_2024
    in_vx_cont = date in vx_dates_2024
    print(f"{date}: In VIX data: {in_vix}, In VX continuous: {in_vx_cont}")

# Check for non-holiday missing dates in 2023 (November and December)
print("\n=== Checking November-December 2023 missing dates ===")
nov_dec_missing = [d for d in missing_2023 if (hasattr(d, 'month') and (d.month == 11 or d.month == 12)) or 
                   (isinstance(d, str) and (d.startswith('2023-11') or d.startswith('2023-12')))]
print(f"Total missing dates in Nov-Dec 2023: {len(nov_dec_missing)}")
for date in nov_dec_missing:
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    is_holiday = date_str in us_holidays_2023
    print(f"{date}: Is US holiday: {is_holiday}") 