#!/usr/bin/env python
"""
Fix CBOE Interval Units

This script updates interval_unit from 'day' to 'daily' in the market_data_cboe table.
"""

import os
import sys
from pathlib import Path
import duckdb
import argparse

# Database path
DB_PATH = "data/financial_data.duckdb"

def main():
    """Main function to fix interval_unit values in the market_data_cboe table."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fix interval_unit values in market_data_cboe table')
    parser.add_argument('--check-only', action='store_true', help='Only check for issues, do not make changes')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate records with interval_unit=day')
    args = parser.parse_args()
    
    print(f"Checking database: {DB_PATH}")
    check_only = args.check_only
    remove_duplicates = args.remove_duplicates
    
    if check_only:
        print("Running in CHECK ONLY mode - no changes will be made")
    
    # Check if database file exists
    db_file = Path(DB_PATH).resolve()
    if not db_file.exists():
        print(f"Error: Database file not found at {db_file}")
        sys.exit(1)
    
    conn = None
    try:
        # Connect to the database (read-write mode)
        conn = duckdb.connect(database=str(db_file))
        print(f"Connected to database: {db_file}")
        
        # Count records with interval_unit = 'day'
        count_query = "SELECT COUNT(*) FROM market_data_cboe WHERE interval_unit = 'day'"
        day_count = conn.execute(count_query).fetchone()[0]
        print(f"Records with interval_unit = 'day': {day_count:,}")
        
        if day_count == 0:
            print("No records to update. All interval_unit values are already correct.")
            return
        
        # Check for potential duplicate conflicts
        duplicate_check_query = """
            SELECT a.timestamp, a.symbol, a.interval_value
            FROM market_data_cboe a
            JOIN market_data_cboe b ON 
                a.timestamp = b.timestamp AND
                a.symbol = b.symbol AND
                a.interval_value = b.interval_value
            WHERE a.interval_unit = 'day' AND b.interval_unit = 'daily'
        """
        duplicates_df = conn.execute(duplicate_check_query).fetchdf()
        duplicate_count = len(duplicates_df)
        
        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count:,} potential duplicate records")
            print("These records have the same timestamp, symbol, and interval_value but different interval_unit")
            print("This would cause primary key violations when trying to update interval_unit from 'day' to 'daily'")
            
            if remove_duplicates and not check_only:
                print(f"\nRemoving {duplicate_count:,} duplicate records with interval_unit='day'...")
                
                # Delete duplicates with interval_unit='day'
                delete_query = """
                    DELETE FROM market_data_cboe
                    WHERE (timestamp, symbol, interval_value) IN (
                        SELECT a.timestamp, a.symbol, a.interval_value
                        FROM market_data_cboe a
                        JOIN market_data_cboe b ON 
                            a.timestamp = b.timestamp AND
                            a.symbol = b.symbol AND
                            a.interval_value = b.interval_value
                        WHERE a.interval_unit = 'day' AND b.interval_unit = 'daily'
                    ) AND interval_unit = 'day'
                """
                conn.execute(delete_query)
                print("Duplicate records removed.")
            else:
                print("\nPlease run with --remove-duplicates to resolve this issue, or manually fix the duplicates.")
                if check_only:
                    print("Exiting without making changes (--check-only mode).")
                    return
                else:
                    print("Cannot continue with update until duplicates are resolved.")
                    return
        
        # Now get final count of records to update (after potential removal of duplicates)
        day_count = conn.execute(count_query).fetchone()[0]
        print(f"Records now left to update: {day_count:,}")
        
        if day_count == 0:
            print("No more records to update after removing duplicates.")
            return
        
        if not check_only:
            # Update the remaining records
            print(f"Updating {day_count:,} records from 'day' to 'daily'...")
            update_query = "UPDATE market_data_cboe SET interval_unit = 'daily' WHERE interval_unit = 'day'"
            conn.execute(update_query)
            
            # Verify the update
            verify_query = "SELECT COUNT(*) FROM market_data_cboe WHERE interval_unit = 'day'"
            remaining_count = conn.execute(verify_query).fetchone()[0]
            
            if remaining_count == 0:
                print(f"Success! All {day_count:,} records have been updated to interval_unit = 'daily'.")
            else:
                print(f"Warning: {remaining_count:,} records still have interval_unit = 'day'.")
        else:
            print(f"Would update {day_count:,} records from 'day' to 'daily' (not making changes in check-only mode)")
        
        # Show counts by interval_unit
        interval_query = """
            SELECT interval_unit, COUNT(*) as count
            FROM market_data_cboe
            GROUP BY interval_unit
            ORDER BY interval_unit
        """
        intervals_df = conn.execute(interval_query).fetchdf()
        print("\nCurrent interval_unit distribution:")
        print(intervals_df.to_string(index=False))
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 