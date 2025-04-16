import duckdb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_vx_data():
    """Analyze VX futures data in the database."""
    try:
        # Connect to database
        conn = duckdb.connect('./data/financial_data.duckdb')
        logger.info("Connected to database")

        # Check data distribution across intervals
        logger.info("\nData distribution across intervals:")
        result = conn.execute("""
            SELECT 
                interval_value,
                interval_unit,
                COUNT(*) as count,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date
            FROM market_data 
            WHERE symbol LIKE 'VX%'
            GROUP BY interval_value, interval_unit
            ORDER BY interval_value, interval_unit
        """).fetchdf()
        print(result)

        # Get sample of symbols
        logger.info("\nSample of VX symbols:")
        result = conn.execute("""
            SELECT DISTINCT symbol
            FROM market_data
            WHERE symbol LIKE 'VX%'
            ORDER BY symbol
            LIMIT 10
        """).fetchdf()
        print(result)

        # Check data quality
        logger.info("\nData quality metrics:")
        result = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(*) FILTER (WHERE open IS NULL) as null_open,
                COUNT(*) FILTER (WHERE high IS NULL) as null_high,
                COUNT(*) FILTER (WHERE low IS NULL) as null_low,
                COUNT(*) FILTER (WHERE close IS NULL) as null_close,
                COUNT(*) FILTER (WHERE volume IS NULL) as null_volume
            FROM market_data
            WHERE symbol LIKE 'VX%'
        """).fetchdf()
        print(result)

    except Exception as e:
        logger.error(f"Error analyzing VX data: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    analyze_vx_data() 