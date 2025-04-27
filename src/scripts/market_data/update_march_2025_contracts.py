#!/usr/bin/env python
"""
Update March 2025 Contracts

This script updates both the ESH25 and NQH25 (March 2025) contracts with the 
correct end date of March 21, 2025. It then verifies the data in the database.
"""

import os
import sys
import logging
import duckdb
import subprocess
import time
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_march_2025_contracts.log')
    ]
)
logger = logging.getLogger("update_march_2025")

# Database path
DB_PATH = "./data/financial_data.duckdb"

def run_script(script_name):
    """Run a Python script and return success status."""
    logger.info(f"Running {script_name}...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Log the output from the script
        if result.stdout:
            logger.info(f"Output from {script_name}:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Errors from {script_name}:\n{result.stderr}")
            
        if result.returncode == 0:
            logger.info(f"Successfully ran {script_name}")
            return True
        else:
            logger.error(f"Failed to run {script_name}, return code: {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False

def verify_contracts():
    """Verify the March 2025 contracts in the database."""
    logger.info("Verifying contract data in database")
    try:
        # Connect to the database
        conn = duckdb.connect(DB_PATH, read_only=True)
        
        # Check both contracts
        contracts = ["ESH25", "NQH25"]
        all_valid = True
        
        for symbol in contracts:
            query = f"""
                SELECT 
                    COUNT(*) as row_count,
                    MIN(timestamp) as first_date,
                    MAX(timestamp) as last_date
                FROM market_data
                WHERE symbol = '{symbol}'
                  AND interval_value = 1
                  AND interval_unit = 'daily'
            """
            
            result = conn.execute(query).fetchone()
            
            if result and result[0] > 0:
                logger.info(f"{symbol}: {result[0]} rows, from {result[1]} to {result[2]}")
                
                # Check if last date is <= March 21, 2025
                last_date = pd.Timestamp(result[2])
                correct_end_date = pd.Timestamp('2025-03-21')
                
                if last_date <= correct_end_date:
                    logger.info(f"{symbol}: Last date {last_date} is within expected range")
                else:
                    logger.warning(f"{symbol}: Last date {last_date} is AFTER expected end date {correct_end_date}")
                    all_valid = False
                    
                # Run an additional query to get data stats
                stats_query = f"""
                    SELECT 
                        MIN(open) as min_open,
                        MAX(open) as max_open,
                        AVG(high-low) as avg_range,
                        AVG(volume) as avg_volume
                    FROM market_data
                    WHERE symbol = '{symbol}'
                      AND interval_value = 1
                      AND interval_unit = 'daily'
                """
                
                stats = conn.execute(stats_query).fetchone()
                logger.info(f"{symbol} stats: Min Open: {stats[0]}, Max Open: {stats[1]}, Avg Range: {stats[2]:.2f}, Avg Volume: {stats[3]:.2f}")
                
            else:
                logger.warning(f"No data found for {symbol}")
                all_valid = False
        
        conn.close()
        return all_valid
        
    except Exception as e:
        logger.error(f"Error verifying contracts: {e}")
        return False

def clean_existing_data():
    """Clean existing March 2025 contract data."""
    logger.info("Cleaning existing March 2025 contract data")
    try:
        conn = duckdb.connect(DB_PATH, read_only=False)
        
        # Delete existing data for both contracts
        for symbol in ["ESH25", "NQH25"]:
            delete_query = f"""
                DELETE FROM market_data
                WHERE symbol = '{symbol}'
                  AND interval_value = 1
                  AND interval_unit = 'daily'
            """
            
            result = conn.execute(delete_query)
            deleted_count = result.fetchone()[0]
            logger.info(f"Deleted {deleted_count} existing rows for {symbol}")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning existing data: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting update of March 2025 contracts")
    
    # First, clean any existing data
    clean_existing_data()
    
    # Run both scripts
    es_success = run_script("fetch_es_march_2025.py")
    # Brief pause between scripts
    time.sleep(2)
    nq_success = run_script("fetch_nq_march_2025.py")
    
    # Verify the results
    if es_success and nq_success:
        logger.info("Both scripts completed successfully")
    else:
        logger.warning("One or more scripts failed")
    
    # Check the data in the database
    if verify_contracts():
        logger.info("✅ All contract data verified with correct date ranges")
    else:
        logger.warning("⚠️ One or more contracts have issues with the data")
    
    logger.info("Update complete") 