#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verify VX Continuous Contract Rollovers

This script runs SQL queries to verify the rollover logic for VX continuous contracts
by comparing VXc1 data with the underlying futures contracts that should be used for each day.
"""

import os
import sys
import pandas as pd
import duckdb
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_database():
    """Connect to the DuckDB database."""
    db_path = os.path.join(project_root, 'data', 'financial_data.duckdb')
    logger.info(f"Connecting to database at {db_path}")
    return duckdb.connect(db_path)

def load_sql_query(query_name):
    """Load a SQL query from the sql directory."""
    sql_path = os.path.join(project_root, 'sql', f"{query_name}.sql")
    logger.info(f"Loading SQL query from {sql_path}")
    
    with open(sql_path, 'r') as f:
        return f.read()

def run_query(conn, query, description):
    """Run a SQL query and return the results as a DataFrame."""
    logger.info(f"Running query: {description}")
    try:
        return conn.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Error running query: {e}")
        return pd.DataFrame()

def verify_first_month(conn):
    """Verify the first month of VXc1 data (March 26, 2004 - April 30, 2004)."""
    query = load_sql_query("vx_rollover_verification")
    
    # Extract the first query (First Month Verification)
    first_month_query = query.split("-- 2. Rollover Day Verification")[0]
    
    df = run_query(conn, first_month_query, "First Month Verification")
    
    if df.empty:
        logger.warning("No data found for the first month verification")
        return
    
    logger.info(f"Found {len(df)} days of VXc1 data in the first month")
    
    # Display the results
    print("\n=== VXc1 First Month Data (March 26, 2004 - April 30, 2004) ===")
    print(f"Total days: {len(df)}")
    
    # Format the DataFrame for display
    display_df = df.copy()
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    for col in ['vxc1_open', 'vxc1_high', 'vxc1_low', 'vxc1_close', 'contract_close']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    # Display the first few rows
    print("\nFirst few days:")
    print(display_df.head().to_string(index=False))
    
    # Check for rollover days
    rollover_days = display_df[display_df['day_type'] == 'ROLLOVER']
    if not rollover_days.empty:
        print("\nRollover days in the first month:")
        print(rollover_days.to_string(index=False))
    else:
        print("\nNo rollover days found in the first month")

def verify_rollover_days(conn):
    """Verify the rollover days to ensure correct contract transitions."""
    query = load_sql_query("vx_rollover_verification")
    
    # Extract the second query (Rollover Day Verification)
    rollover_query = query.split("-- 2. Rollover Day Verification")[1].split("-- 3. Continuous Contract Consistency Check")[0]
    
    df = run_query(conn, rollover_query, "Rollover Day Verification")
    
    if df.empty:
        logger.warning("No rollover days found")
        return
    
    logger.info(f"Found {len(df)} rollover days")
    
    # Display the results
    print("\n=== VXc1 Rollover Days ===")
    print(f"Total rollover days: {len(df)}")
    
    # Format the DataFrame for display
    display_df = df.copy()
    display_df['rollover_date'] = pd.to_datetime(display_df['rollover_date']).dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    for col in ['vxc1_close', 'prev_contract_close', 'new_contract_close', 
                'vxc1_prev_close', 'prev_contract_prev_close', 
                'vxc1_next_close', 'new_contract_next_close']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    # Display all rollover days
    print("\nAll rollover days:")
    print(display_df.to_string(index=False))
    
    # Calculate price differences
    if not display_df.empty:
        print("\nPrice differences on rollover days:")
        display_df['prev_day_diff'] = (display_df['vxc1_prev_close'] - display_df['prev_contract_prev_close']).round(2)
        display_df['rollover_day_diff'] = (display_df['vxc1_close'] - display_df['new_contract_close']).round(2)
        display_df['next_day_diff'] = (display_df['vxc1_next_close'] - display_df['new_contract_next_close']).round(2)
        
        print(display_df[['rollover_date', 'prev_contract', 'active_contract', 
                          'prev_day_diff', 'rollover_day_diff', 'next_day_diff']].to_string(index=False))

def check_consistency(conn):
    """Check if VXc1 data is consistent with the underlying futures contracts."""
    query = load_sql_query("vx_rollover_verification")
    
    # Extract the third query (Continuous Contract Consistency Check)
    consistency_query = query.split("-- 3. Continuous Contract Consistency Check")[1]
    
    df = run_query(conn, consistency_query, "Continuous Contract Consistency Check")
    
    if df.empty:
        logger.info("No inconsistencies found in VXc1 data")
        return
    
    logger.warning(f"Found {len(df)} days with significant price differences")
    
    # Display the results
    print("\n=== VXc1 Consistency Check ===")
    print(f"Days with significant price differences: {len(df)}")
    
    # Format the DataFrame for display
    display_df = df.copy()
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    for col in ['vxc1_close', 'contract_close', 'price_difference']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    # Display all inconsistencies
    print("\nDays with significant price differences:")
    print(display_df.to_string(index=False))

def main():
    """Main function to run all verification queries."""
    try:
        conn = connect_to_database()
        
        # Verify the first month of VXc1 data
        verify_first_month(conn)
        
        # Verify rollover days
        verify_rollover_days(conn)
        
        # Check consistency
        check_consistency(conn)
        
        conn.close()
        logger.info("Verification completed successfully")
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 