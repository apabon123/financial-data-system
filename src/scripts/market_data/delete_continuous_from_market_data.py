import duckdb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_continuous_from_market_data():
    """Delete all entries from market_data table that start with '@'."""
    try:
        # Connect to the database
        conn = duckdb.connect('data/financial_data.duckdb')
        
        # First count how many records we'll delete
        count_query = """
        SELECT COUNT(*) as count
        FROM market_data
        WHERE symbol LIKE '@%'
        """
        count_result = conn.execute(count_query).fetchone()
        logger.info(f"Found {count_result[0]} records to delete from market_data table")
        
        # Delete the records
        delete_query = """
        DELETE FROM market_data
        WHERE symbol LIKE '@%'
        """
        conn.execute(delete_query)
        
        # Verify deletion
        verify_query = """
        SELECT COUNT(*) as remaining_count
        FROM market_data
        WHERE symbol LIKE '@%'
        """
        verify_result = conn.execute(verify_query).fetchone()
        logger.info(f"Remaining records with '@' prefix: {verify_result[0]}")
        
        conn.close()
        logger.info("Database connection closed")
        
    except Exception as e:
        logger.error(f"Error deleting continuous contracts from market_data: {e}")
        raise

if __name__ == "__main__":
    delete_continuous_from_market_data() 