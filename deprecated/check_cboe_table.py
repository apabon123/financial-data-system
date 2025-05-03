#!/usr/bin/env python
"""
Check CBOE Table Data

This script checks if there's data in the market_data_cboe table,
specifically looking for VX contracts.
"""

import os
import sys
from pathlib import Path
import duckdb
import pandas as pd

# Database path
DB_PATH = "data/financial_data.duckdb"

def main():
    """Main function to check the market_data_cboe table."""
    print(f"Checking database: {DB_PATH}")
    
    # Check if database file exists
    db_file = Path(DB_PATH).resolve()
    if not db_file.exists():
        print(f"Error: Database file not found at {db_file}")
        sys.exit(1)
    
    conn = None
    try:
        # Connect to the database
        conn = duckdb.connect(database=str(db_file), read_only=True)
        print(f"Connected to database: {db_file}")
        
        # Check if table exists
        try:
            table_check = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'market_data_cboe'").fetchdf()
            if table_check.empty:
                print("market_data_cboe table does not exist in the database.")
                return
            else:
                print("market_data_cboe table exists in the database.")
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return
        
        # Get count of all rows
        try:
            count_query = "SELECT COUNT(*) FROM market_data_cboe"
            total_count = conn.execute(count_query).fetchone()[0]
            print(f"Total rows in market_data_cboe: {total_count:,}")
        except Exception as e:
            print(f"Error counting rows: {e}")
            return
            
        if total_count == 0:
            print("The table exists but is empty.")
            return
        
        # Check for VX symbols
        try:
            vx_query = "SELECT COUNT(*) FROM market_data_cboe WHERE symbol LIKE 'VX%'"
            vx_count = conn.execute(vx_query).fetchone()[0]
            print(f"Rows with VX symbols: {vx_count:,}")
        except Exception as e:
            print(f"Error counting VX rows: {e}")
            return
            
        # Get distinct symbols and their counts
        try:
            symbol_query = """
                SELECT symbol, COUNT(*) as count, 
                       MIN(timestamp)::DATE as first_date, 
                       MAX(timestamp)::DATE as last_date,
                       interval_value, interval_unit
                FROM market_data_cboe
                GROUP BY symbol, interval_value, interval_unit
                ORDER BY symbol, interval_value, interval_unit
            """
            symbols_df = conn.execute(symbol_query).fetchdf()
            
            if not symbols_df.empty:
                print("\nSymbols in market_data_cboe table:")
                print(symbols_df.to_string(index=False))
            else:
                print("No symbols found in the table, which is unexpected since the count was non-zero.")
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return
            
        # Debug the exact same query used in view_futures_contracts.py for VX
        try:
            debug_query = """
                SELECT DISTINCT symbol
                FROM market_data_cboe
                WHERE symbol LIKE 'VX%' 
                    AND interval_value = 1 AND interval_unit = 'daily'
                ORDER BY symbol
            """
            print("\nTesting exact query from view_futures_contracts.py:")
            debug_df = conn.execute(debug_query).fetchdf()
            print(f"Query returned {len(debug_df)} rows")
            
            if not debug_df.empty:
                print("Symbols found:")
                print(debug_df.to_string(index=False))
            else:
                print("No VX symbols with interval_value=1 and interval_unit='daily' found.")
                
                # Check if these VX symbols exist with different intervals
                alt_query = """
                    SELECT DISTINCT symbol, interval_value, interval_unit, COUNT(*) as count
                    FROM market_data_cboe
                    WHERE symbol LIKE 'VX%'
                    GROUP BY symbol, interval_value, interval_unit
                    ORDER BY symbol, interval_value, interval_unit
                """
                alt_df = conn.execute(alt_query).fetchdf()
                
                if not alt_df.empty:
                    print("\nVX symbols with other intervals:")
                    print(alt_df.to_string(index=False))
                else:
                    print("No VX symbols found with any interval.")
        except Exception as e:
            print(f"Error running debug query: {e}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 