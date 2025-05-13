#!/usr/bin/env python3
"""
Script to create and manage market data views.
This script ensures all market data views are properly created as views, not tables.
"""

import logging
import duckdb
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_market_views(db_path: str = "./data/financial_data.duckdb"):
    """Create all market data views from the market_data table."""
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")

        # List of views to create
        views = [
            'minute_bars',
            'five_minute_bars',
            'daily_bars',
            'weekly_bars',
            'monthly_bars',
            'latest_prices'
        ]

        # First, drop all existing objects
        for view in views:
            # Force drop everything
            try:
                conn.execute(f"DROP TABLE IF EXISTS {view} CASCADE")
                logger.info(f"Dropped table {view} if it existed")
            except Exception as e:
                logger.warning(f"Error dropping table {view}: {e}")

            try:
                conn.execute(f"DROP VIEW IF EXISTS {view} CASCADE")
                logger.info(f"Dropped view {view} if it existed")
            except Exception as e:
                logger.warning(f"Error dropping view {view}: {e}")

            # Additional cleanup
            try:
                conn.execute(f"DROP SCHEMA IF EXISTS {view} CASCADE")
                logger.info(f"Dropped schema {view} if it existed")
            except Exception as e:
                logger.warning(f"Error dropping schema {view}: {e}")

        # Create views with explicit schema
        view_definitions = {
            'minute_bars': """
                CREATE OR REPLACE VIEW minute_bars AS
                SELECT 
                    timestamp,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    up_volume,
                    down_volume,
                    source,
                    interval_value,
                    interval_unit,
                    adjusted,
                    quality
                FROM market_data
                WHERE interval_unit = 'minute' AND interval_value = 1
            """,
            'five_minute_bars': """
                CREATE OR REPLACE VIEW five_minute_bars AS
                SELECT 
                    TIME_BUCKET('5 minutes', timestamp) AS timestamp,
                    symbol,
                    FIRST(open) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close) AS close,
                    SUM(volume) AS volume,
                    SUM(up_volume) AS up_volume,
                    SUM(down_volume) AS down_volume,
                    source,
                    5 AS interval_value,
                    'minute' AS interval_unit
                FROM market_data
                WHERE interval_unit = 'minute' AND interval_value = 1
                GROUP BY TIME_BUCKET('5 minutes', timestamp), symbol, source
            """,
            'daily_bars': """
                CREATE OR REPLACE VIEW daily_bars AS
                SELECT 
                    DATE_TRUNC('day', timestamp) AS date,
                    symbol,
                    FIRST(open) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS open,
                    MAX(high) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS high,
                    MIN(low) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS low,
                    LAST(close) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS close,
                    SUM(volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS volume,
                    SUM(up_volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS up_volume,
                    SUM(down_volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS down_volume,
                    source,
                    'daily' AS interval_unit,
                    1 AS interval_value
                FROM market_data
                GROUP BY DATE_TRUNC('day', timestamp), symbol, source
            """,
            'weekly_bars': """
                CREATE OR REPLACE VIEW weekly_bars AS
                SELECT 
                    DATE_TRUNC('week', timestamp) AS week_start,
                    symbol,
                    FIRST(open) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close) AS close,
                    SUM(volume) AS volume,
                    SUM(up_volume) AS up_volume,
                    SUM(down_volume) AS down_volume,
                    source,
                    'weekly' AS interval_unit,
                    1 AS interval_value
                FROM market_data
                WHERE interval_unit = 'daily' OR interval_value = 1440
                GROUP BY DATE_TRUNC('week', timestamp), symbol, source
            """,
            'monthly_bars': """
                CREATE OR REPLACE VIEW monthly_bars AS
                SELECT 
                    DATE_TRUNC('month', timestamp) AS month_start,
                    symbol,
                    FIRST(open) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close) AS close,
                    SUM(volume) AS volume,
                    SUM(up_volume) AS up_volume,
                    SUM(down_volume) AS down_volume,
                    source,
                    'monthly' AS interval_unit,
                    1 AS interval_value
                FROM market_data
                WHERE interval_unit = 'daily' OR interval_value = 1440
                GROUP BY DATE_TRUNC('month', timestamp), symbol, source
            """,
            'latest_prices': """
                CREATE OR REPLACE VIEW latest_prices AS
                WITH ranked_prices AS (
                    SELECT 
                        symbol,
                        timestamp,
                        close,
                        ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM market_data
                )
                SELECT symbol, timestamp, close 
                FROM ranked_prices 
                WHERE rn = 1
            """
        }

        # Create each view
        for view_name, view_sql in view_definitions.items():
            try:
                # First ensure any existing object is dropped
                conn.execute(f"DROP TABLE IF EXISTS {view_name} CASCADE")
                conn.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE")
                
                # Then create the view
                conn.execute(view_sql)
                logger.info(f"Successfully created {view_name} view")
                
                # Verify the view was created
                result = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
                logger.info(f"Number of rows in {view_name} view: {result[0]}")
            except Exception as e:
                logger.error(f"Error creating {view_name} view: {e}")
                raise

    except Exception as e:
        logger.error(f"Error in create_market_views: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_market_views() 