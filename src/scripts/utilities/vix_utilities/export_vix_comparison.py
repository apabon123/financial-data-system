#!/usr/bin/env python
"""
Export VIX Comparison Script

This script exports VIX, VXc1, and VXc2 data to a CSV file for comparison.
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

def export_vix_comparison(db_path, output_path, start_date=None, end_date=None):
    """Export VIX and VXc1 data to a CSV file."""
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        
        # First check if we have VIX data
        vix_check = conn.execute("SELECT COUNT(*) as count FROM market_data WHERE symbol = 'VIX'").fetchone()[0]
        has_vix = vix_check > 0
        
        # Build the query based on available data
        if has_vix:
            query = """
            WITH combined_data AS (
                SELECT 
                    timestamp,
                    symbol,
                    close
                FROM market_data
                WHERE symbol = 'VIX'
                AND interval_unit = 'daily'
                
                UNION ALL
                
                SELECT 
                    timestamp,
                    symbol,
                    close
                FROM continuous_contracts
                WHERE symbol = 'VXc1'
            )
            SELECT 
                timestamp as date,
                MAX(CASE WHEN symbol = 'VIX' THEN close END) as vix_close,
                MAX(CASE WHEN symbol = 'VXc1' THEN close END) as vxc1_close
            FROM combined_data
            """
        else:
            query = """
            SELECT 
                timestamp as date,
                close as vxc1_close
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            """
        
        # Add date filters if provided
        if start_date:
            query += f" AND timestamp >= '{start_date}'"
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
            
        # Group by date and order
        if has_vix:
            query += " GROUP BY timestamp"
        query += " ORDER BY timestamp"
        
        # Execute the query
        result = conn.execute(query).fetchdf()
        
        if result.empty:
            logger.error("No data found for VIX or VXc1")
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
        logger.error(f"Error exporting VIX comparison: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Export VIX, VXc1, and VXc2 data to CSV')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                      help='Path to DuckDB database file')
    parser.add_argument('--output', type=str, default='vix_comparison.csv',
                      help='Path to output CSV file')
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    export_vix_comparison(args.db_path, args.output, args.start_date, args.end_date)

if __name__ == '__main__':
    main() 