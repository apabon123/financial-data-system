"""
Database utility functions for the application.
"""

import os
import duckdb
import logging
from pathlib import Path

# Get the project root
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Get the logger
logger = logging.getLogger(__name__)

def get_db_engine(db_path=None):
    """
    Get a DuckDB connection.
    
    Args:
        db_path: Path to the database file, or None to use the default location
        
    Returns:
        DuckDB connection object
    """
    if db_path is None:
        # Default database location is in the data directory
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, 'market_data.duckdb')
    
    logger.debug(f"Connecting to database: {db_path}")
    
    # Create a connection to the database
    conn = duckdb.connect(db_path)
    
    return conn

def ensure_market_data_table(conn):
    """
    Ensure the market_data table exists with a unified schema for all data types.
    
    Args:
        conn: DuckDB connection object
    """
    try:
        # Create the unified market_data table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            date VARCHAR,            -- Date in YYYY-MM-DD format for compatibility
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            settle DOUBLE,           -- Settlement price (especially for futures)
            volume DOUBLE,
            open_interest DOUBLE,    -- For futures contracts
            up_volume DOUBLE,        -- TradeStation specific
            down_volume DOUBLE,      -- TradeStation specific
            interval_value INTEGER,
            interval_unit VARCHAR,
            source VARCHAR,
            changed BOOLEAN DEFAULT FALSE,  -- Flag for indicating changed/filled data
            adjusted BOOLEAN DEFAULT FALSE, -- Flag for adjusted prices
            quality INTEGER DEFAULT 100,    -- Quality score for data
            PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
        )
        """)
        
        # Create indices for common query patterns
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date)")
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring market_data table: {e}")
        return False

def migrate_date_based_to_timestamp(conn):
    """
    Migrate data from old schema (date-based) to new schema (timestamp-based).
    This is a one-time migration to unify schemas.
    
    Args:
        conn: DuckDB connection object
        
    Returns:
        bool: True if migration was successful, False otherwise
    """
    try:
        # Check if we have the old-schema table
        old_schema_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'market_data' 
        AND EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'market_data' AND column_name = 'date' AND column_name != 'timestamp'
        )
        """).fetchone()[0] > 0
        
        if not old_schema_exists:
            logger.info("No migration needed - data already in timestamp-based schema")
            return True
            
        # Create a temporary table with the new schema
        conn.execute("DROP TABLE IF EXISTS market_data_new")
        conn.execute("""
        CREATE TABLE market_data_new (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            date VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            settle DOUBLE,
            volume DOUBLE,
            open_interest DOUBLE,
            up_volume DOUBLE,
            down_volume DOUBLE,
            interval_value INTEGER,
            interval_unit VARCHAR,
            source VARCHAR,
            changed BOOLEAN DEFAULT FALSE,
            adjusted BOOLEAN DEFAULT FALSE,
            quality INTEGER DEFAULT 100,
            PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
        )
        """)
        
        # Migrate data from the old schema to the new schema
        conn.execute("""
        INSERT INTO market_data_new (
            timestamp, symbol, date, open, high, low, close, volume, 
            interval_value, interval_unit, source
        )
        SELECT 
            CAST(datetime AS TIMESTAMP), symbol, date, open, high, low, close, volume,
            interval_value, interval_unit, source
        FROM 
            market_data
        """)
        
        # Rename tables to complete the migration
        conn.execute("ALTER TABLE market_data RENAME TO market_data_old")
        conn.execute("ALTER TABLE market_data_new RENAME TO market_data")
        
        # Create indices on the new table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date)")
        
        logger.info("Successfully migrated market_data table to unified schema")
        return True
        
    except Exception as e:
        logger.error(f"Error during schema migration: {e}")
        return False 