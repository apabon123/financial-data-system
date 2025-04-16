#!/usr/bin/env python
"""
Aggregate VX futures minute data into daily data.
"""

import logging
import duckdb
from pathlib import Path
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_vx_daily():
    """Aggregate VX futures minute data into daily data."""
    try:
        # Connect to database
        conn = duckdb.connect('./data/financial_data.duckdb')
        logger.info("Connected to database")

        # Create daily data from minute data
        conn.execute("""
            INSERT INTO market_data (
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                source,
                interval_value,
                interval_unit,
                adjusted,
                quality
            )
            SELECT 
                DATE_TRUNC('day', timestamp) as timestamp,
                symbol,
                FIRST(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close) as close,
                SUM(volume) as volume,
                'aggregated' as source,
                1 as interval_value,
                'day' as interval_unit,
                true as adjusted,
                100 as quality
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND interval_unit = 'minute'
            GROUP BY DATE_TRUNC('day', timestamp), symbol
            HAVING COUNT(*) > 0
        """)
        
        # Log the results
        result = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND interval_unit = 'day'
        """).fetchdf()
        
        logger.info(f"Aggregated VX daily data:")
        logger.info(f"Total rows: {result['total_rows'].iloc[0]}")
        logger.info(f"Unique symbols: {result['unique_symbols'].iloc[0]}")
        logger.info(f"Date range: {result['earliest_date'].iloc[0]} to {result['latest_date'].iloc[0]}")

    except Exception as e:
        logger.error(f"Error aggregating VX daily data: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    aggregate_vx_daily() 