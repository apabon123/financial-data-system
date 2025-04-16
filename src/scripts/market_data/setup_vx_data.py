import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_vx_data():
    """Set up VX futures data by creating necessary views and copying data."""
    db_path = Path('./data/financial_data.duckdb')
    conn = duckdb.connect(str(db_path))
    
    try:
        # Create daily_bars view if it doesn't exist
        logger.info("Creating daily_bars view...")
        conn.execute("""
        CREATE OR REPLACE VIEW daily_bars AS
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
        WHERE interval_value = 1 
        AND interval_unit = 'day'
        """)
        
        # Verify VX data exists in market_data
        logger.info("Checking VX data in market_data...")
        result = conn.execute("""
        SELECT COUNT(*) as count
        FROM market_data
        WHERE symbol LIKE 'VX%'
        AND interval_value = 1
        AND interval_unit = 'day'
        """).fetchdf()
        
        count = result['count'].iloc[0]
        logger.info(f"Found {count} VX futures records in market_data")
        
        if count > 0:
            # Get sample of VX symbols
            symbols = conn.execute("""
            SELECT DISTINCT symbol
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND interval_value = 1
            AND interval_unit = 'day'
            ORDER BY symbol
            LIMIT 5
            """).fetchdf()
            
            logger.info("Sample VX symbols:")
            for _, row in symbols.iterrows():
                logger.info(f"  {row['symbol']}")
                
            # Get date range
            date_range = conn.execute("""
            SELECT 
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND interval_value = 1
            AND interval_unit = 'day'
            """).fetchdf()
            
            logger.info(f"Date range: {date_range['earliest_date'].iloc[0]} to {date_range['latest_date'].iloc[0]}")
            
        else:
            logger.error("No VX futures data found in market_data table")
            
    except Exception as e:
        logger.error(f"Error setting up VX data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    setup_vx_data() 