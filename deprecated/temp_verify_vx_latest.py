import duckdb
from datetime import date

DB_PATH = "data/financial_data.duckdb"
SYMBOLS_TO_CHECK = ["VXZ25", "VXU25", "VXX25", "VXV25"] # Checking the last few contracts

print(f"Connecting to {DB_PATH} (read-only)")
conn = None
try:
    conn = duckdb.connect(database=DB_PATH, read_only=True)
    print("Connection successful.")

    for symbol in SYMBOLS_TO_CHECK:
        print(f"--- Checking {symbol} ---")
        query = """
            SELECT 
                MIN(timestamp)::DATE as first_date,
                MAX(timestamp)::DATE as last_date,
                COUNT(*) as row_count
            FROM market_data
            WHERE symbol = ? 
              AND interval_value = 1 
              AND interval_unit = 'day'
        """
        try:
            result = conn.execute(query, [symbol]).fetchone()
            if result and result[2] > 0: # Check if count > 0
                print(f"  Symbol: {symbol}")
                print(f"  Count: {result[2]}")
                print(f"  First Date: {result[0]}")
                print(f"  Last Date: {result[1]}")
            elif result:
                print(f"  Symbol: {symbol}")
                print(f"  Count: 0")
                print(f"  First Date: N/A")
                print(f"  Last Date: N/A")
            else:
                print(f"  No results found for {symbol}.")
                
        except Exception as e:
            print(f"  Error querying {symbol}: {e}")

except Exception as e:
    print(f"Failed to connect or query database: {e}")
finally:
    if conn:
        conn.close()
        print("Connection closed.") 