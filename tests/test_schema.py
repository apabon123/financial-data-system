#!/usr/bin/env python3
"""
Tests for database schema initialization and management.
"""

import os
import sys
import pytest
import duckdb
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the initialization function
from init_database import initialize_database
from schema_management_agent import SchemaManagementAgent

# Test database path
TEST_DB_PATH = "./tests/test_financial_data.duckdb"

@pytest.fixture
def clean_test_db():
    """Fixture to ensure a clean test database."""
    # Remove the test database if it exists
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Create the tests directory if it doesn't exist
    os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)
    
    yield TEST_DB_PATH
    
    # Cleanup after the test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_initialize_database(clean_test_db):
    """Test database initialization."""
    # Initialize the database
    success = initialize_database(clean_test_db, verbose=True)
    
    # Check if initialization was successful
    assert success, "Database initialization failed"
    
    # Check if the database file was created
    assert os.path.exists(clean_test_db), "Database file was not created"
    
    # Connect to the database and check if tables were created
    conn = duckdb.connect(clean_test_db)
    
    # Check if market_data table exists
    result = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='market_data'").fetchone()
    assert result[0] == 1, "market_data table was not created"
    
    # Check if views were created
    result = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='daily_bars'").fetchone()
    assert result[0] == 1, "daily_bars view was not created"
    
    # Check if metadata was inserted
    result = conn.execute("SELECT COUNT(*) FROM metadata WHERE key='schema_version'").fetchone()
    assert result[0] == 1, "schema_version metadata was not inserted"
    
    # Close connection
    conn.close()

def test_schema_management_agent(clean_test_db):
    """Test the schema management agent."""
    # First initialize the database
    initialize_database(clean_test_db)
    
    # Create an instance of the schema management agent
    agent = SchemaManagementAgent(database_path=clean_test_db, verbose=True)
    
    # Test creating a new table
    result = agent.process_query("create table test_table with fields id as INTEGER, name as VARCHAR, value as DOUBLE")
    
    # Check if the operation was successful
    assert result["success"], f"Failed to create table: {result['results'].get('errors', [])}"
    
    # Check if the table was created
    conn = duckdb.connect(clean_test_db)
    result = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_table'").fetchone()
    assert result[0] == 1, "test_table was not created"
    
    # Check table schema
    columns = conn.execute("DESCRIBE test_table").fetchdf()["column_name"].tolist()
    assert "id" in columns, "id column not found in test_table"
    assert "name" in columns, "name column not found in test_table"
    assert "value" in columns, "value column not found in test_table"
    
    # Test adding a column
    result = agent.process_query("add column to test_table with fields timestamp as TIMESTAMP")
    
    # Check if the operation was successful
    assert result["success"], f"Failed to add column: {result['results'].get('errors', [])}"
    
    # Check if the column was added
    columns = conn.execute("DESCRIBE test_table").fetchdf()["column_name"].tolist()
    assert "timestamp" in columns, "timestamp column not added to test_table"
    
    # Close connections
    conn.close()
    agent.close()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])