import duckdb
from pathlib import Path

def check_intervals(symbol):
    conn = duckdb.connect(str(Path('./data/financial_data.duckdb')))
    
    # Check if symbol exists
    count = conn.execute(
        "SELECT COUNT(*) FROM market_data WHERE symbol = ?",
        [symbol]
    ).fetchone()[0]
    
    print(f"\nChecking intervals for {symbol}:")
    print(f"Total records: {count}")
    
    if count > 0:
        # Get distinct intervals
        intervals = conn.execute(
            """
            SELECT DISTINCT interval_value, interval_unit 
            FROM market_data 
            WHERE symbol = ?
            ORDER BY interval_value, interval_unit
            """,
            [symbol]
        ).fetchall()
        
        print("\nAvailable intervals:")
        for interval in intervals:
            value, unit = interval
            count = conn.execute(
                """
                SELECT COUNT(*) 
                FROM market_data 
                WHERE symbol = ? 
                AND interval_value = ? 
                AND interval_unit = ?
                """,
                [symbol, value, unit]
            ).fetchone()[0]
            print(f"- {value} {unit}: {count} records")
    else:
        print("No data found for this symbol")

if __name__ == "__main__":
    # Check a few ES contracts
    for symbol in ["ESH25", "ESM25", "ESU25", "ESZ25"]:
        check_intervals(symbol) 