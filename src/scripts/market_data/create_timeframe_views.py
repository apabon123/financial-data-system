"""Create views for different timeframes from market_data table."""

import logging
import duckdb
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_timeframe_views():
    """Create views for different timeframes from the market_data table."""
    try:
        # Connect to database
        conn = duckdb.connect('./data/financial_data.duckdb')
        logger.info("Connected to database at ./data/financial_data.duckdb")

        # Drop existing views and table
        views = ['minute_bars', 'five_minute_bars', 'daily_bars', 'weekly_bars', 'monthly_bars']
        for view in views:
            try:
                conn.execute(f"DROP VIEW IF EXISTS {view}")
                logger.info(f"Dropped existing {view} view if it existed")
            except Exception as e:
                if "is of type Table" in str(e):
                    conn.execute(f"DROP TABLE IF EXISTS {view}")
                    logger.info(f"Dropped existing {view} table")
                else:
                    raise

        # Create views for each timeframe
        view_sql = {
            'minute_bars': """
                CREATE VIEW minute_bars AS
                SELECT 
                    timestamp as date,
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
                WHERE interval_value = 1 AND interval_unit = 'minute'
            """,
            'five_minute_bars': """
                CREATE VIEW five_minute_bars AS
                SELECT 
                    timestamp as date,
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
                WHERE interval_value = 5 AND interval_unit = 'minute'
            """,
            'daily_bars': """
                CREATE VIEW daily_bars AS
                SELECT 
                    timestamp as date,
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
                WHERE interval_value = 1 AND interval_unit = 'day'
            """,
            'weekly_bars': """
                CREATE VIEW weekly_bars AS
                SELECT 
                    timestamp as date,
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
                WHERE interval_value = 1 AND interval_unit = 'week'
            """,
            'monthly_bars': """
                CREATE VIEW monthly_bars AS
                SELECT 
                    timestamp as date,
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
                WHERE interval_value = 1 AND interval_unit = 'month'
            """
        }

        # Create each view
        for view_name, sql in view_sql.items():
            conn.execute(sql)
            logger.info(f"Created {view_name} view")

        # Verify views were created
        for view in views:
            count = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            logger.info(f"View {view} has {count} rows")

    except Exception as e:
        logger.error(f"Error creating timeframe views: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    create_timeframe_views() 