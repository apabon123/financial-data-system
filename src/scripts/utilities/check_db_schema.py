#!/usr/bin/env python
"""
Check Database Schema

This script connects to the financial database and displays the schema information,
focusing on the market_data table structure and primary key constraints.
"""

import duckdb
import os
import sys

# Database path
DB_PATH = "./data/financial_data.duckdb"

def check_table_schema():
    """Check the schema of the market_data table including primary key constraints."""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return False
    
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        
        # Check if market_data table exists
        tables_query = "SHOW TABLES;"
        tables = conn.execute(tables_query).fetchdf()
        print("Available tables in database:")
        print(tables)
        print()
        
        if 'market_data' not in tables['name'].values:
            print("Error: market_data table not found in database")
            return False
        
        # Get table schema
        schema_query = "DESCRIBE market_data;"
        schema = conn.execute(schema_query).fetchdf()
        print("Schema of market_data table:")
        print(schema)
        print()
        
        # Check for primary key constraints
        pk_query = """
            SELECT * FROM duckdb_constraints() 
            WHERE table_name = 'market_data' AND constraint_type = 'PRIMARY KEY';
        """
        
        try:
            pk_constraints = conn.execute(pk_query).fetchdf()
            if not pk_constraints.empty:
                print("Primary key constraints:")
                print(pk_constraints)
            else:
                print("No primary key constraints defined in the schema.")
                
                # Let's check for uniqueness by sampling a few rows
                check_uniqueness(conn)
        except Exception as e:
            print(f"Could not query constraints: {e}")
            check_uniqueness(conn)
        
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()
    
    return True

def check_uniqueness(conn):
    """Check for practical uniqueness in the data."""
    print("\nChecking for practical uniqueness in the data...")
    
    # Check if the combination timestamp+symbol is unique
    timestamp_symbol_query = """
        SELECT timestamp, symbol, COUNT(*) as count
        FROM market_data
        GROUP BY timestamp, symbol
        HAVING COUNT(*) > 1
        LIMIT 10;
    """
    
    duplicates = conn.execute(timestamp_symbol_query).fetchdf()
    if duplicates.empty:
        print("No duplicates found with timestamp+symbol combination")
    else:
        print(f"Found {len(duplicates)} cases where timestamp+symbol is not unique:")
        print(duplicates)
    
    # Check if timestamp+symbol+interval_value+interval_unit is unique
    full_key_query = """
        SELECT timestamp, symbol, interval_value, interval_unit, COUNT(*) as count
        FROM market_data
        GROUP BY timestamp, symbol, interval_value, interval_unit
        HAVING COUNT(*) > 1
        LIMIT 10;
    """
    
    full_duplicates = conn.execute(full_key_query).fetchdf()
    if full_duplicates.empty:
        print("\nNo duplicates found with timestamp+symbol+interval_value+interval_unit combination")
        print("This suggests this combination effectively serves as a primary key")
    else:
        print(f"\nFound {len(full_duplicates)} cases where the full combination is not unique:")
        print(full_duplicates)
    
    # Count total rows and check for nulls in key fields
    stats_query = """
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) as null_timestamp,
            SUM(CASE WHEN symbol IS NULL THEN 1 ELSE 0 END) as null_symbol,
            SUM(CASE WHEN interval_value IS NULL THEN 1 ELSE 0 END) as null_interval_value,
            SUM(CASE WHEN interval_unit IS NULL THEN 1 ELSE 0 END) as null_interval_unit
        FROM market_data;
    """
    
    stats = conn.execute(stats_query).fetchone()
    print(f"\nTotal rows: {stats[0]}")
    print(f"Null values in potential key fields:")
    print(f"  timestamp: {stats[1]}")
    print(f"  symbol: {stats[2]}")
    print(f"  interval_value: {stats[3]}")
    print(f"  interval_unit: {stats[4]}")
    
    # Sample distribution of interval values
    interval_query = """
        SELECT interval_value, interval_unit, COUNT(*) as count
        FROM market_data
        GROUP BY interval_value, interval_unit
        ORDER BY count DESC
        LIMIT 10;
    """
    
    intervals = conn.execute(interval_query).fetchdf()
    print("\nInterval distribution (top 10):")
    print(intervals)

if __name__ == "__main__":
    print("Checking database schema and primary key constraints...")
    check_table_schema() 