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
            timestamp TIMESTAMP,      -- Date of the data point
            symbol VARCHAR,           -- Continuous contract symbol (e.g., VXc1)
            underlying_symbol VARCHAR,-- Specific contract used for this row (e.g., VXF10)
            open DOUBLE,              -- Opening price
            high DOUBLE,              -- Highest price
            low DOUBLE,               -- Lowest price
            close DOUBLE,             -- Closing price
            volume BIGINT,            -- Trading volume
            open_interest BIGINT,     -- Open Interest
            up_volume BIGINT,         -- Optional up volume
            down_volume BIGINT,       -- Optional down volume
            source VARCHAR,           -- Data source ('continuous')
            interval_value INTEGER,   -- Interval length (e.g., 1)
            interval_unit VARCHAR,    -- Interval unit (e.g., 'day')
            adjusted BOOLEAN,         -- Whether the price is adjusted (TRUE for continuous)
            quality INTEGER           -- Data quality indicator (e.g., 100)
            -- Define Primary Key if needed, e.g.:
            -- PRIMARY KEY (timestamp, symbol) 
        )
        """)
        logger.info("Created continuous_contracts table with updated schema (incl. underlying_symbol)")
        
        conn.close()
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error cleaning up market data: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cleanup_market_data() 