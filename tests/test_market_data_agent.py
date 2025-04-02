#!/usr/bin/env python3
"""
Tests for the TradeStation Market Data Agent.
"""

import os
import sys
import json
import pytest
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import responses  # For mocking HTTP requests

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the agent
from tradestation_market_data_agent import TradeStationMarketDataAgent
from init_database import initialize_database

# Test database path
TEST_DB_PATH = "./tests/test_financial_data.duckdb"

@pytest.fixture
def initialized_db():
    """Fixture to ensure an initialized test database."""
    # Remove the test database if it exists
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Create the tests directory if it doesn't exist
    os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)
    
    # Initialize the database
    initialize_database(TEST_DB_PATH)
    
    yield TEST_DB_PATH
    
    # Cleanup after the test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_parse_query():
    """Test the query parsing functionality."""
    # Initialize the agent
    agent = TradeStationMarketDataAgent(database_path=":memory:")
    
    # Test daily data query
    query = "fetch daily data for AAPL, MSFT from 2023-01-01 to 2023-12-31"
    params = agent._parse_query(query)
    
    assert params["action"] == "fetch"
    assert params["timeframe"] == "daily"
    assert params["symbols"] == ["AAPL", "MSFT"]
    assert params["start_date"].isoformat() == "2023-01-01"
    assert params["end_date"].isoformat() == "2023-12-31"
    assert params["interval_unit"] == "day"
    assert params["interval_value"] == 1
    
    # Test minute data query
    query = "fetch 5-minute data for SPY from 2023-09-01 to 2023-09-30"
    params = agent._parse_query(query)
    
    assert params["action"] == "fetch"
    assert params["timeframe"] == "5-minute"
    assert params["symbols"] == ["SPY"]
    assert params["start_date"].isoformat() == "2023-09-01"
    assert params["end_date"].isoformat() == "2023-09-30"
    assert params["interval_unit"] == "minute"
    assert params["interval_value"] == 5
    
    # Test "from latest" query
    query = "fetch daily data for AAPL from latest"
    params = agent._parse_query(query)
    
    assert params["action"] == "fetch"
    assert params["timeframe"] == "daily"
    assert params["symbols"] == ["AAPL"]
    # start_date and end_date will be set dynamically
    
    # Close the agent
    agent.close()

@responses.activate
def test_fetch_market_data(initialized_db):
    """Test fetching market data with mocked API responses."""
    # Initialize the agent
    agent = TradeStationMarketDataAgent(database_path=initialized_db)
    
    # Mock the authentication endpoint
    responses.add(
        responses.POST,
        "https://signin.tradestation.com/oauth/token",
        json={
            "access_token": "mock_access_token",
            "expires_in": 3600,
            "refresh_token": "mock_refresh_token"
        },
        status=200
    )
    
    # Mock the market data endpoint for AAPL
    responses.add(
        responses.GET,
        "https://api.tradestation.com/v3/marketdata/barcharts/AAPL",
        json={
            "Bars": [
                {
                    "Open": 150.0,
                    "High": 152.0,
                    "Low": 149.0,
                    "Close": 151.0,
                    "Volume": 1000000,
                    "TimeStamp": "2023-01-01T00:00:00Z"
                },
                {
                    "Open": 151.0,
                    "High": 153.0,
                    "Low": 150.0,
                    "Close": 152.0,
                    "Volume": 1100000,
                    "TimeStamp": "2023-01-02T00:00:00Z"
                }
            ]
        },
        status=200
    )
    
    # Set up parameters for fetching
    params = {
        "action": "fetch",
        "symbols": ["AAPL"],
        "timeframe": "daily",
        "interval_unit": "day",
        "interval_value": 1,
        "start_date": datetime.now().date() - timedelta(days=30),
        "end_date": datetime.now().date(),
        "adjusted": True
    }
    
    # Fetch the data
    data = agent.fetch_market_data(params)
    
    # Check the results
    assert not data.empty, "No data returned from fetch_market_data"
    assert len(data) == 2, f"Expected 2 records, got {len(data)}"
    assert data["symbol"].iloc[0] == "AAPL", "Symbol mismatch"
    assert data["open"].iloc[0] == 150.0, "Open price mismatch"
    assert data["high"].iloc[0] == 152.0, "High price mismatch"
    assert data["low"].iloc[0] == 149.0, "Low price mismatch"
    assert data["close"].iloc[0] == 151.0, "Close price mismatch"
    assert data["volume"].iloc[0] == 1000000, "Volume mismatch"
    
    # Close the agent
    agent.close()

@responses.activate
def test_save_market_data(initialized_db):
    """Test saving market data to the database."""
    # Initialize the agent
    agent = TradeStationMarketDataAgent(database_path=initialized_db)
    
    # Create sample data
    data = {
        "timestamp": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        ],
        "symbol": ["AAPL", "AAPL"],
        "open": [150.0, 151.0],
        "high": [152.0, 153.0],
        "low": [149.0, 150.0],
        "close": [151.0, 152.0],
        "volume": [1000000, 1100000],
        "up_volume": [600000, 700000],
        "down_volume": [400000, 400000],
        "source": ["TradeStation API", "TradeStation API"],
        "interval_value": [1, 1],
        "interval_unit": ["day", "day"],
        "adjusted": [True, True],
        "quality": [100, 100]
    }
    df = pd.DataFrame(data)
    
    # Save the data
    total_records, new_records = agent.save_market_data(df)
    
    # Check the results
    assert total_records == 2, f"Expected 2 total records, got {total_records}"
    assert new_records == 2, f"Expected 2 new records, got {new_records}"
    
    # Check if the data was saved to the database
    conn = duckdb.connect(initialized_db)
    result = conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'AAPL'").fetchone()
    assert result[0] == 2, f"Expected 2 records in the database, got {result[0]}"
    
    # Test saving the same data again (should not insert duplicates)
    total_records, new_records = agent.save_market_data(df)
    assert total_records == 2, f"Expected 2 total records, got {total_records}"
    assert new_records == 0, f"Expected 0 new records (duplicates), got {new_records}"
    
    # Close connections
    conn.close()
    agent.close()

@responses.activate
def test_process_query(initialized_db):
    """Test the full query processing pipeline."""
    # Initialize the agent
    agent = TradeStationMarketDataAgent(database_path=initialized_db)
    
    # Mock the authentication endpoint
    responses.add(
        responses.POST,
        "https://signin.tradestation.com/oauth/token",
        json={
            "access_token": "mock_access_token",
            "expires_in": 3600,
            "refresh_token": "mock_refresh_token"
        },
        status=200
    )
    
    # Mock the market data endpoint for AAPL
    responses.add(
        responses.GET,
        "https://api.tradestation.com/v3/marketdata/barcharts/AAPL",
        json={
            "Bars": [
                {
                    "Open": 150.0,
                    "High": 152.0,
                    "Low": 149.0,
                    "Close": 151.0,
                    "Volume": 1000000,
                    "TimeStamp": "2023-01-01T00:00:00Z"
                },
                {
                    "Open": 151.0,
                    "High": 153.0,
                    "Low": 150.0,
                    "Close": 152.0,
                    "Volume": 1100000,
                    "TimeStamp": "2023-01-02T00:00:00Z"
                }
            ]
        },
        status=200
    )
    
    # Process a natural language query
    result = agent.process_query("fetch daily data for AAPL from 2023-01-01 to 2023-01-02")
    
    # Check the results
    assert result["success"], f"Query processing failed: {result['results'].get('errors', [])}"
    assert result["results"]["data_fetched"] == 2, f"Expected 2 records fetched, got {result['results']['data_fetched']}"
    assert result["results"]["data_saved"] == 2, f"Expected 2 records saved, got {result['results']['data_saved']}"
    
    # Close the agent
    agent.close()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])