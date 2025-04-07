#!/usr/bin/env python
"""
Cleanup Market Data Script

This script cleans up the market data by:
1. Removing incorrect base symbol data
2. Regenerating continuous contracts
"""

import os
import sys
import logging
import duckdb
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def cleanup_market_data(db_path='./data/financial_data.duckdb'):
    """Clean up the market data."""
    try:
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Remove base symbol data
        conn.execute("DELETE FROM market_data WHERE symbol = 'ES'")
        logger.info("Removed base symbol data for ES")
        
        # Drop and recreate continuous contracts table
        conn.execute("DROP TABLE IF EXISTS continuous_contracts")
        logger.info("Dropped continuous_contracts table")
        
        # Create continuous contracts table
        conn.execute("""
        CREATE TABLE continuous_contracts (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT
        )
        """)
        logger.info("Created continuous_contracts table")
        
        conn.close()
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error cleaning up market data: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cleanup_market_data() 