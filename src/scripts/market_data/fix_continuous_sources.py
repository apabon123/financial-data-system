import duckdb
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_continuous_contract_sources():
    """Fix the source field for continuous contracts with _d suffix."""
    try:
        # Connect to the database
        conn = duckdb.connect('data/financial_data.duckdb')
        
        # Update the source field
        update_query = """
        UPDATE continuous_contracts
        SET source = 'inhouse_built'
        WHERE symbol LIKE '%_d'
        AND source = 'tradestation'
        """
        conn.execute(update_query)
        
        # Get the number of records updated
        count_query = """
        SELECT COUNT(*) as records_updated
        FROM continuous_contracts
        WHERE symbol LIKE '%_d'
        AND source = 'inhouse_built'
        """
        result = conn.execute(count_query).fetchone()
        logger.info(f"Updated {result[0]} records to have source='inhouse_built'")
        
        conn.close()
        logger.info("Database connection closed")
        
    except Exception as e:
        logger.error(f"Error fixing continuous contract sources: {e}")
        raise

def delete_test_continuous_contract():
    """Delete the @ES=102XN_test daily tradestation record from the DB."""
    try:
        conn = duckdb.connect('data/financial_data.duckdb')
        delete_query = """
        DELETE FROM continuous_contracts
        WHERE symbol = '@ES=102XN_test'
          AND interval_value = 1
          AND interval_unit = 'daily'
          AND source = 'tradestation'
        """
        conn.execute(delete_query)
        logger.info("Deleted @ES=102XN_test daily tradestation record.")
        conn.close()
    except Exception as e:
        logger.error(f"Error deleting test continuous contract: {e}")
        raise

if __name__ == "__main__":
    fix_continuous_contract_sources()
    delete_test_continuous_contract() 