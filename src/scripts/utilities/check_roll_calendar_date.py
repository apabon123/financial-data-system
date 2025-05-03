import duckdb
import pandas as pd

def main():
    # Database path
    db_path = "data/financial_data.duckdb"
    
    # Connect to database
    print(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)
    
    # Check roll calendar entries for December 2023 - March 2024
    query = """
    SELECT * 
    FROM futures_roll_calendar 
    WHERE root_symbol = 'VX' 
    AND last_trading_day >= DATE '2023-12-01' 
    AND last_trading_day <= DATE '2024-03-31'
    ORDER BY last_trading_day ASC
    """
    
    period_entries = conn.execute(query).fetchdf()
    print("\nRoll calendar entries for Dec 2023 - Mar 2024:")
    print(period_entries)
    
    # Close connection
    conn.close()
    print("\nDatabase connection closed")

if __name__ == "__main__":
    main() 