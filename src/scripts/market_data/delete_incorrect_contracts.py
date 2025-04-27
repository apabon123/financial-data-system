#!/usr/bin/env python
"""
Delete Incorrect March 2025 Contracts

This script removes the ESH25 and NQH25 (March 2025) contracts that have
incorrect expiration dates past March 21, 2025.
"""

import duckdb
import logging
from datetime import date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "./data/financial_data.duckdb"

def check_contract(conn, symbol):
    """Check data for a specific contract."""
    try:
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
            logger.info(f"Contract {symbol}: {result[0]} rows, {result[1]} to {result[2]}")
            return True, result[0], result[1], result[2]
        else:
            logger.info(f"Contract {symbol} not found")
            return False, 0, None, None
    except Exception as e:
        logger.error(f"Error checking contract: {e}")
        return False, 0, None, None

def delete_contract(conn, symbol):
    """Delete all data for a contract."""
    try:
        delete_query = f"""
            DELETE FROM market_data
            WHERE symbol = '{symbol}'
                AND interval_value = 1
                AND interval_unit = 'daily'
        """
        conn.execute(delete_query)
        logger.info(f"Deleted all data for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Error deleting contract {symbol}: {e}")
        return False

def main():
    """Main function to delete incorrect contracts."""
    contracts = ["ESH25", "NQH25"]
    
    # The correct expiration date for March 2025 contracts
    expiration_date = date(2025, 3, 21)
    
    logger.info("Checking contracts before deletion")
    
    # Connect to database
    conn = duckdb.connect(DB_PATH, read_only=False)
    
    try:
        # Check contracts before deletion
        for symbol in contracts:
            exists, row_count, first_date, last_date = check_contract(conn, symbol)
            
            if exists and last_date:
                # Check if contract data extends beyond expiration
                if last_date.date() > expiration_date:
                    logger.info(f"{symbol} has data past expiration ({last_date.date()} > {expiration_date})")
                    logger.info(f"Will delete {symbol}")
                else:
                    logger.info(f"{symbol} already has correct expiration date")
        
        # Confirm deletion
        confirm = input("\nDelete contracts with incorrect expiration dates? (y/n): ")
        
        if confirm.lower() == 'y':
            # Delete contracts
            for symbol in contracts:
                exists, row_count, first_date, last_date = check_contract(conn, symbol)
                
                if exists and last_date and last_date.date() > expiration_date:
                    logger.info(f"Deleting {symbol}...")
                    delete_contract(conn, symbol)
            
            logger.info("\nContracts after deletion:")
            for symbol in contracts:
                check_contract(conn, symbol)
        else:
            logger.info("Deletion cancelled")
    
    finally:
        # Close connection
        conn.close()
        logger.info("Done")

if __name__ == "__main__":
    main() 