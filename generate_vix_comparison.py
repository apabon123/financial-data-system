#!/usr/bin/env python
"""
Generate VIX Comparison CSV

This script generates a CSV file with daily data containing:
- date
- VIX close
- VXc1 close
- VXc2 close

The data is merged from different tables in the database.
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = './data/financial_data.duckdb'

def generate_vix_comparison(db_path, output_path, start_date=None, end_date=None):
    """Generate a CSV file with VIX, VXc1, and VXc2 data."""
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        
        # Check if we have VIX data in daily_bars
        vix_check = conn.execute("SELECT COUNT(*) as count FROM daily_bars WHERE symbol = '$VIX.X'").fetchone()[0]
        has_vix = vix_check > 0
        
        # Check if we have VXc1 and VXc2 in continuous_contracts
        vxc1_check = conn.execute("SELECT COUNT(*) as count FROM continuous_contracts WHERE symbol = 'VXc1'").fetchone()[0]
        has_vxc1 = vxc1_check > 0
        
        vxc2_check = conn.execute("SELECT COUNT(*) as count FROM continuous_contracts WHERE symbol = 'VXc2'").fetchone()[0]
        has_vxc2 = vxc2_check > 0
        
        # Check if we have VXc2 in daily_bars (it might be there but not in continuous_contracts)
        vxc2_daily_check = conn.execute("SELECT COUNT(*) as count FROM daily_bars WHERE symbol = 'VXc2'").fetchone()[0]
        has_vxc2_daily = vxc2_daily_check > 0
        
        # Build the query based on available data
        query = """
        WITH combined_data AS (
        """
        
        # Add VIX data if available
        if has_vix:
            query += """
            SELECT 
                date,
                'VIX' as symbol,
                close
            FROM daily_bars
            WHERE symbol = '$VIX.X'
            """
        
        # Add VXc1 data if available
        if has_vxc1:
            if has_vix:
                query += "\nUNION ALL\n"
            query += """
            SELECT 
                DATE_TRUNC('day', timestamp) as date,
                'VXc1' as symbol,
                close
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            """
        
        # Add VXc2 data if available
        if has_vxc2:
            if has_vix or has_vxc1:
                query += "\nUNION ALL\n"
            query += """
            SELECT 
                DATE_TRUNC('day', timestamp) as date,
                'VXc2' as symbol,
                close
            FROM continuous_contracts
            WHERE symbol = 'VXc2'
            """
        
        # Add VXc2 data from daily_bars if available and not in continuous_contracts
        if has_vxc2_daily and not has_vxc2:
            if has_vix or has_vxc1:
                query += "\nUNION ALL\n"
            query += """
            SELECT 
                date,
                'VXc2' as symbol,
                close
            FROM daily_bars
            WHERE symbol = 'VXc2'
            """
        
        # Close the WITH clause and aggregate by date
        query += """
        )
        SELECT 
            date,
            MAX(CASE WHEN symbol = 'VIX' THEN close END) as vix_close,
            MAX(CASE WHEN symbol = 'VXc1' THEN close END) as vxc1_close,
            MAX(CASE WHEN symbol = 'VXc2' THEN close END) as vxc2_close
        FROM combined_data
        """
        
        # Add date filters if provided
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"
            
        # Group by date and order
        query += " GROUP BY date ORDER BY date"
        
        # Execute the query
        result = conn.execute(query).fetchdf()
        
        if result.empty:
            logger.error("No data found for VIX, VXc1, or VXc2")
            return False
            
        # Save to CSV
        result.to_csv(output_path, index=False)
        logger.info(f"Exported {len(result)} rows to {output_path}")
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"Date Range: {result['date'].min()} to {result['date'].max()}")
        print(f"Total Rows: {len(result)}")
        print("\nFirst few rows:")
        print(result.head())
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating VIX comparison: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Generate VIX comparison CSV')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                      help='Path to DuckDB database file')
    parser.add_argument('--output', type=str, default='vix_comparison.csv',
                      help='Path to output CSV file')
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    generate_vix_comparison(args.db_path, args.output, args.start_date, args.end_date)

if __name__ == '__main__':
    main() 