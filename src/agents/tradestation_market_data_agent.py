#!/usr/bin/env python3
"""
Agent Name: TradeStation Market Data Agent
Purpose: Retrieve market data from TradeStation API and store in DuckDB
Author: Claude
Date: 2025-04-02

Description:
    This agent retrieves market data (OHLCV) from the TradeStation API and
    prepares it for storage in DuckDB. It handles authentication, rate limiting,
    and data normalization.

Usage:
    uv run tradestation_market_data_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run tradestation_market_data_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL from 2023-01-01 to 2023-12-31"
    uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch 5-minute data for SPY from 2023-09-01 to 2023-09-02" -v
"""

import os
import sys
import json
import logging
import argparse
import re
import time
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, date
from urllib.parse import urlencode

import typer
import duckdb
import pandas as pd
import requests
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)
logger = logging.getLogger("TradeStation Market Data Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "TradeStation Market Data Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Retrieves market data from TradeStation API and prepares it for storage in DuckDB"

# TradeStation API configuration
TRADESTATION_API_URL = "https://api.tradestation.com/v3"
TRADESTATION_AUTH_URL = "https://signin.tradestation.com/oauth/token"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class TradeStationMarketDataAgent:
    """Agent for retrieving market data from TradeStation API.
    
    This agent handles authentication with the TradeStation API,
    retrieves market data (OHLCV) for specified symbols and timeframes,
    and normalizes the data for storage in DuckDB.
    
    Attributes:
        database_path (str): Path to the DuckDB database file
        verbose (bool): Whether to enable verbose logging
        compute_loops (int): Number of reasoning iterations to perform
        conn (duckdb.DuckDBPyConnection): Database connection
        access_token (str): TradeStation API access token
        token_expiry (datetime): Expiry time of the access token
    """
    
    def __init__(
        self, 
        database_path: str,
        verbose: bool = False,
        compute_loops: int = 3
    ):
        """Initialize the agent.
        
        Args:
            database_path: Path to DuckDB database
            verbose: Enable verbose output
            compute_loops: Number of reasoning iterations
        """
        self.database_path = database_path
        self.verbose = verbose
        self.compute_loops = compute_loops
        self.conn = None
        
        # TradeStation API configuration
        self.base_url = TRADESTATION_API_URL
        self.access_token = None
        self.token_expiry = None
        self.refresh_token = None
        
        # Rate limiting and retry settings
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._connect_database()
    
    def _connect_database(self) -> None:
        """Connect to DuckDB database.
        
        Establishes a connection to the DuckDB database and validates
        that the connection is successful. If the database file doesn't
        exist, it will be created.
        
        Raises:
            SystemExit: If the database connection fails
        """
        try:
            self.conn = duckdb.connect(self.database_path)
            logger.debug(f"Connected to database: {self.database_path}")
            
            # Check if the database connection is valid
            test_query = "SELECT 1"
            result = self.conn.execute(test_query).fetchone()
            
            if result and result[0] == 1:
                logger.debug("Database connection validated")
            else:
                logger.error("Database connection validation failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query to extract parameters.
        
        This method extracts structured parameters related to market data
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "fetch daily data for AAPL from 2023-01-01 to 2023-12-31"
            Output: {
                "action": "fetch",
                "timeframe": "daily",
                "symbols": ["AAPL"],
                "start_date": datetime.date(2023, 1, 1),
                "end_date": datetime.date(2023, 12, 31)
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "fetch",
            "timeframe": "daily",
            "symbols": [],
            "start_date": None,
            "end_date": None,
            "interval_value": 1,
            "interval_unit": "day",
            "adjusted": True
        }
        
        # Extract timeframe
        minute_match = re.search(r'(\d+)[- ]minute', query)
        if minute_match:
            minute_value = int(minute_match.group(1))
            params["timeframe"] = f"{minute_value}-minute"
            params["interval_value"] = minute_value
            params["interval_unit"] = "minute"
        elif 'daily' in query:
            params["timeframe"] = "daily"
            params["interval_value"] = 1
            params["interval_unit"] = "day"
        elif 'weekly' in query:
            params["timeframe"] = "weekly"
            params["interval_value"] = 1
            params["interval_unit"] = "week"
        elif 'monthly' in query:
            params["timeframe"] = "monthly"
            params["interval_value"] = 1
            params["interval_unit"] = "month"
        
        # Extract symbols
        symbols_match = re.search(r'for\s+([A-Za-z0-9,\s]+)(?:\s+from|\s+with|\s*$)', query)
        if symbols_match:
            symbols_str = symbols_match.group(1)
            params["symbols"] = [s.strip() for s in symbols_str.split(',')]
        
        # Extract date range
        start_date_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})', query)
        if start_date_match:
            start_date_str = start_date_match.group(1)
            params["start_date"] = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        
        end_date_match = re.search(r'to\s+(\d{4}-\d{2}-\d{2})', query)
        if end_date_match:
            end_date_str = end_date_match.group(1)
            params["end_date"] = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # Handle "from latest" pattern
        if 'from latest' in query:
            # Use the most recent date in the database as the start date
            try:
                result = self.conn.execute("""
                    SELECT MAX(timestamp)::DATE 
                    FROM market_data 
                    WHERE interval_unit = ? AND interval_value = ?
                """, [params["interval_unit"], params["interval_value"]]).fetchone()
                
                if result and result[0]:
                    latest_date = result[0]
                    params["start_date"] = latest_date
                    params["end_date"] = datetime.now().date()
                else:
                    # If no data exists, use last 30 days
                    params["end_date"] = datetime.now().date()
                    params["start_date"] = params["end_date"] - timedelta(days=30)
            except Exception as e:
                logger.error(f"Error determining latest date: {e}")
                params["end_date"] = datetime.now().date()
                params["start_date"] = params["end_date"] - timedelta(days=30)
        
        # Extract adjusted flag
        if 'adjusted' in query:
            params["adjusted"] = True
        elif 'unadjusted' in query:
            params["adjusted"] = False
        
        logger.debug(f"Parsed parameters: {params}")
        return params
    
    def authenticate(self) -> bool:
        """Authenticate with TradeStation API using refresh token."""
        try:
            client_id = os.getenv('CLIENT_ID')
            client_secret = os.getenv('CLIENT_SECRET')
            refresh_token = os.getenv('TRADESTATION_REFRESH_TOKEN')
            
            if not all([client_id, client_secret, refresh_token]):
                logger.error("Missing TradeStation API credentials in environment variables")
                logger.error("Please set CLIENT_ID, CLIENT_SECRET, and TRADESTATION_REFRESH_TOKEN")
                return False
            
            # API credentials authentication using refresh token
            url = "https://signin.tradestation.com/oauth/token"
            payload = f"grant_type=refresh_token&client_id={client_id}&client_secret={client_secret}&refresh_token={refresh_token}"
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            logger.debug("Requesting access token")
            response = requests.post(url, headers=headers, data=payload)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                expires_in = data.get('expires_in', 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # Buffer of 60 seconds
                self.refresh_token = data.get('refresh_token')
                logger.debug("Successfully obtained access token")
                return True
            else:
                logger.error(f"Authentication failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def fetch_market_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Fetch market data from TradeStation API.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            DataFrame containing the fetched market data
        """
        if not self.authenticate():
            logger.error("Authentication failed. Cannot fetch market data.")
            return pd.DataFrame()
        
        all_data = []
        
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Fetching {params['timeframe']} data for {len(params['symbols'])} symbols...", 
                total=len(params['symbols'])
            )
            
            for symbol in params['symbols']:
                try:
                    logger.debug(f"Fetching data for {symbol}")
                    
                    # Determine API endpoint and parameters based on timeframe
                    if params["interval_unit"] in ["minute", "hour"]:
                        endpoint = f"{TRADESTATION_API_URL}/marketdata/barcharts/{symbol}"
                        api_params = {
                            "interval": params["interval_value"],
                            "unit": params["interval_unit"],
                            "barsback": 1000  # Default if no date range specified
                        }
                        
                        # Add date range if specified
                        if params["start_date"] and params["end_date"]:
                            # Convert to format expected by API
                            start_str = params["start_date"].strftime("%Y-%m-%dT00:00:00Z")
                            end_str = params["end_date"].strftime("%Y-%m-%dT23:59:59Z")
                            api_params["startdate"] = start_str
                            api_params["enddate"] = end_str
                            del api_params["barsback"]  # Remove barsback when using date range
                            
                    else:  # daily, weekly, monthly
                        endpoint = f"{TRADESTATION_API_URL}/marketdata/barcharts/{symbol}"
                        
                        # Calculate required bars based on timeframe and date range
                        if params["start_date"] and params["end_date"]:
                            days_diff = (params["end_date"] - params["start_date"]).days
                            
                            # Calculate base bars needed based on interval unit and value
                            if params["interval_unit"] == "minute":
                                # For intraday data, calculate based on trading hours
                                # Assuming 6.5 hours of trading per day (9:30 AM - 4:00 PM ET)
                                trading_minutes_per_day = 390  # 6.5 hours * 60 minutes
                                bars_per_day = trading_minutes_per_day // params["interval_value"]
                                base_bars = bars_per_day * days_diff
                            elif params["interval_unit"] == "hour":
                                # For hourly data, calculate based on trading hours
                                trading_hours_per_day = 6.5  # 9:30 AM - 4:00 PM ET
                                bars_per_day = trading_hours_per_day // params["interval_value"]
                                base_bars = bars_per_day * days_diff
                            elif params["interval_unit"] == "day":
                                base_bars = days_diff
                            elif params["interval_unit"] == "week":
                                base_bars = days_diff // 7
                            elif params["interval_unit"] == "month":
                                base_bars = (days_diff // 30)  # Approximate
                            else:
                                base_bars = days_diff
                            
                            # Add buffer for holidays/weekends
                            # For intraday data, we need more buffer due to market hours
                            if params["interval_unit"] in ["minute", "hour"]:
                                buffer_multiplier = 1.2  # 20% buffer for intraday
                            else:
                                buffer_multiplier = 1.1  # 10% buffer for daily and above
                            
                            bars_back = int(base_bars * buffer_multiplier) + 1
                            logger.debug(f"Calculated bars_back: {bars_back} (base: {base_bars}, days: {days_diff}, interval: {params['interval_value']} {params['interval_unit']})")
                        else:
                            # Default to 30 bars if no date range specified
                            bars_back = 30
                        
                        api_params = {
                            "interval": params["interval_value"],
                            "unit": "daily" if params["interval_unit"] == "day" else params["interval_unit"],
                            "barsback": bars_back,
                            "lastdate": params["end_date"].strftime("%Y-%m-%dT23:59:59Z")  # Use end date as lastdate
                        }
                        
                        # Remove startdate and enddate parameters as we'll use lastdate instead
                        if "startdate" in api_params:
                            del api_params["startdate"]
                        if "enddate" in api_params:
                            del api_params["enddate"]
                    
                    # Add session template for intraday data
                    if params["interval_unit"] == "minute":
                        api_params["sessiontemplate"] = "USEQPre"  # US Equities with pre/post market
                    
                    headers = {
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    logger.debug(f"API request: {endpoint} with params {api_params}")
                    response = requests.get(endpoint, params=api_params, headers=headers)
                    
                    # Check for rate limiting headers
                    if 'X-RateLimit-Remaining' in response.headers:
                        self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                        logger.debug(f"Rate limit remaining: {self.rate_limit_remaining}")
                    
                    if 'X-RateLimit-Reset' in response.headers:
                        self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
                        logger.debug(f"Rate limit resets in: {self.rate_limit_reset} seconds")
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        wait_time = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                        time.sleep(wait_time)
                        
                        # Retry the request
                        response = requests.get(endpoint, params=api_params, headers=headers)
                    
                    # Process the response
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'Bars' in data and data['Bars']:
                            # Log the columns we received
                            logger.debug(f"Received columns: {list(data['Bars'][0].keys())}")
                            # Log the full response for debugging
                            logger.debug(f"API Response: {json.dumps(data, indent=2)}")
                            df = self._process_market_data(data['Bars'], symbol, params)
                            all_data.append(df)
                            logger.debug(f"Fetched {len(df)} bars for {symbol}")
                        else:
                            logger.warning(f"No data returned for {symbol}")
                            # Log the full response even when no bars
                            logger.debug(f"API Response: {json.dumps(data, indent=2)}")
                    else:
                        logger.error(f"API error for {symbol}: {response.status_code} - {response.text}")
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    if self.verbose:
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Sleep briefly to avoid hammering the API
                time.sleep(0.2)
                progress.update(task, advance=1)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def _process_market_data(self, data: List[Dict], symbol: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Process raw market data from the API into DataFrame format.
        
        Args:
            data: List of bar data from the API
            symbol: The symbol the data is for
            params: Query parameters
            
        Returns:
            DataFrame with processed market data
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime for filtering
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Sort by timestamp in descending order (newest first)
        df = df.sort_values('TimeStamp', ascending=False)
        
        # Filter data within the requested date range
        if params.get('start_date') and params.get('end_date'):
            # Convert start and end dates to UTC timestamps
            start_date = pd.Timestamp(params['start_date']).tz_localize('UTC')
            end_date = pd.Timestamp(params['end_date']).tz_localize('UTC') + pd.Timedelta(days=1)  # Include the end date
            
            # Log the date range we're filtering for
            logger.debug(f"Filtering data between {start_date} and {end_date}")
            logger.debug(f"Data range before filtering: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
            
            # Filter the data
            df = df[(df['TimeStamp'] >= start_date) & (df['TimeStamp'] < end_date)]
            
            # Log the filtered data range
            if not df.empty:
                logger.debug(f"Data range after filtering: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
            else:
                logger.warning(f"No data found within the specified date range for {symbol}")
                return pd.DataFrame()
        
        # Rename columns to match our schema
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'TotalVolume': 'volume',
            'TimeStamp': 'timestamp',
            'DownVolume': 'down_volume',
            'UpVolume': 'up_volume'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add missing columns
        if 'down_volume' not in df.columns:
            df['down_volume'] = None
        
        if 'up_volume' not in df.columns:
            df['up_volume'] = None
        
        # Add symbol and other metadata
        df['symbol'] = symbol
        df['interval_value'] = params['interval_value']
        df['interval_unit'] = params['interval_unit']
        df['adjusted'] = params['adjusted']
        df['source'] = 'TradeStation API'
        df['quality'] = 100  # Default quality score
        
        # Keep only the columns we need
        columns = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'up_volume', 'down_volume', 'source', 'interval_value', 'interval_unit',
            'adjusted', 'quality'
        ]
        
        return df[columns]
    
    def save_market_data(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Save market data to the database.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Tuple of (total records, new records)
        """
        if df.empty:
            logger.warning("No data to save")
            return (0, 0)
        
        logger.debug(f"Saving {len(df)} records to database")
        
        try:
            # Create a temporary table for the new data
            self.conn.execute("""
                CREATE TEMPORARY TABLE temp_market_data AS 
                SELECT * FROM market_data LIMIT 0
            """)
            
            # Insert data into the temporary table
            self.conn.execute(
                "INSERT INTO temp_market_data SELECT * FROM df"
            )
            
            # Count total records
            total_records = len(df)
            
            # Insert new records into the main table
            result = self.conn.execute("""
                INSERT INTO market_data
                SELECT t.*
                FROM temp_market_data t
                LEFT JOIN market_data m ON 
                    t.timestamp = m.timestamp AND 
                    t.symbol = m.symbol AND 
                    t.interval_value = m.interval_value AND 
                    t.interval_unit = m.interval_unit
                WHERE m.timestamp IS NULL
            """)
            
            # Get count of new records
            new_records = result.fetchall()[0][0]
            
            # Drop the temporary table
            self.conn.execute("DROP TABLE temp_market_data")
            
            logger.debug(f"Saved {new_records} new records to database")
            return (total_records, new_records)
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            return (0, 0)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        This is the main entry point for agent functionality. It parses
        the query, retrieves market data from the API, and saves it to
        the database.
        
        Args:
            query: Natural language query to process
            
        Returns:
            Dict containing the results and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Parse query to extract parameters
        params = self._parse_query(query)
        
        # Execute query processing based on extracted parameters
        results = self._execute_compute_loops(params)
        
        # Compile and return results
        return {
            "query": query,
            "parameters": params,
            "results": results,
            "success": results.get("success", False),
            "timestamp": datetime.now().isoformat()
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a query from a JSON file.
        
        Args:
            file_path: Path to JSON file containing query parameters
            
        Returns:
            Dict containing the results and metadata
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            # Execute query processing based on extracted parameters
            results = self._execute_compute_loops(params)
            
            # Compile and return results
            return {
                "file": file_path,
                "parameters": params,
                "results": results,
                "success": results.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {e}")
            raise
    
    def _execute_compute_loops(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning iterations.
        
        This method implements the multi-step reasoning process for the agent.
        Each loop builds on the results of the previous loop to iteratively
        refine the results.
        
        Args:
            params: Query parameters extracted from natural language query
            
        Returns:
            Processed results as a dictionary
        """
        # Default result structure
        result = {
            "action": params.get("action", "fetch"),
            "success": False,
            "errors": [],
            "warnings": [],
            "data_fetched": 0,
            "data_saved": 0,
            "symbols_processed": 0,
            "metadata": {
                "compute_loops": self.compute_loops,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "execution_time_ms": 0
            }
        }
        
        start_time = datetime.now()
        
        # Execute compute loops
        for i in range(self.compute_loops):
            loop_start = datetime.now()
            logger.debug(f"Compute loop {i+1}/{self.compute_loops}")
            
            try:
                if i == 0:
                    # Loop 1: Parameter validation
                    self._loop_validate_parameters(params, result)
                elif i == 1:
                    # Loop 2: Data retrieval from API
                    self._loop_fetch_data(params, result)
                elif i == 2:
                    # Loop 3: Save data to database
                    self._loop_save_data(params, result)
                else:
                    # Additional loops if compute_loops > 3
                    pass
            except Exception as e:
                logger.error(f"Error in compute loop {i+1}: {e}")
                result["errors"].append(f"Error in compute loop {i+1}: {str(e)}")
                result["success"] = False
                break
            
            loop_end = datetime.now()
            loop_duration = (loop_end - loop_start).total_seconds() * 1000
            logger.debug(f"Loop {i+1} completed in {loop_duration:.2f}ms")
        
        # Calculate total execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Update metadata
        result["metadata"]["end_time"] = end_time.isoformat()
        result["metadata"]["execution_time_ms"] = execution_time
        
        # Set success flag if no errors occurred
        if not result["errors"]:
            result["success"] = True
        
        return result
    
    def _loop_validate_parameters(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """First compute loop: Validate input parameters.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
            
        Raises:
            ValueError: If parameters are invalid
        """
        logger.debug("Validating parameters")
        
        # Validate action
        if params.get("action") != "fetch":
            result["errors"].append(f"Invalid action: {params.get('action')}. Only 'fetch' is supported.")
            return
        
        # Validate symbols
        if not params.get("symbols"):
            result["errors"].append("No symbols specified")
            return
        
        # Validate timeframe
        valid_interval_units = ["minute", "hour", "day", "week", "month"]
        if params.get("interval_unit") not in valid_interval_units:
            result["errors"].append(f"Invalid interval unit: {params.get('interval_unit')}. Must be one of {valid_interval_units}")
            return
        
        # Validate interval value
        if params.get("interval_value") <= 0:
            result["errors"].append(f"Invalid interval value: {params.get('interval_value')}. Must be positive.")
            return
        
        # Validate date range
        if params.get("start_date") and params.get("end_date"):
            if params["start_date"] > params["end_date"]:
                result["errors"].append(f"Invalid date range: start date {params['start_date']} is after end date {params['end_date']}")
                return
            
            # Check if date range is too large
            if params["interval_unit"] == "minute":
                delta_days = (params["end_date"] - params["start_date"]).days
                if delta_days > 30:
                    result["warnings"].append(f"Date range is {delta_days} days which may be too large for minute data. Consider using a smaller range.")
        
        # Set default date range if not specified
        if not params.get("start_date") or not params.get("end_date"):
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)  # Default to 30 days
            
            if not params.get("start_date"):
                params["start_date"] = start_date
                result["warnings"].append(f"No start date specified. Using default: {start_date}")
            
            if not params.get("end_date"):
                params["end_date"] = end_date
                result["warnings"].append(f"No end date specified. Using default: {end_date}")
        
        logger.debug("Parameters validated successfully")
    
    def _loop_fetch_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Fetch data from TradeStation API.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Fetching data from TradeStation API")
        
        # Store the raw data in the result for the next loop
        data = self.fetch_market_data(params)
        
        if data.empty:
            result["errors"].append("No data returned from API")
            return
        
        # Store the data temporarily
        result["raw_data"] = data
        result["data_fetched"] = len(data)
        result["symbols_processed"] = len(params["symbols"])
        
        logger.debug(f"Fetched {len(data)} records for {len(params['symbols'])} symbols")
    
    def _loop_save_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Save data to database.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Saving data to database")
        
        # Check if we have data to save
        if "raw_data" not in result:
            result["errors"].append("No data to save")
            return
        
        data = result["raw_data"]
        total_records, new_records = self.save_market_data(data)
        
        # Update result with save metrics
        result["data_saved"] = new_records
        
        # Remove raw data from result to save space
        del result["raw_data"]
        
        logger.debug(f"Saved {new_records} new records out of {total_records} total records")
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display request summary
            console.print(f"[cyan]Request:[/] {results['query']}")
            console.print(f"[cyan]Symbols:[/] {', '.join(results['parameters']['symbols'])}")
            console.print(f"[cyan]Timeframe:[/] {results['parameters']['timeframe']}")
            console.print(f"[cyan]Date Range:[/] {results['parameters']['start_date']} to {results['parameters']['end_date']}")
            
            # Display results summary
            console.print(f"\n[bold]Results Summary:[/]")
            console.print(f"Symbols processed: {results['results']['symbols_processed']}")
            console.print(f"Records fetched: {results['results']['data_fetched']}")
            console.print(f"New records saved: {results['results']['data_saved']}")
            
            # Display warnings if any
            if results['results']['warnings']:
                console.print("\n[bold yellow]Warnings:[/]")
                for warning in results['results']['warnings']:
                    console.print(f"[yellow]- {warning}[/]")
            
            # Display execution time
            execution_time = results['results']['metadata']['execution_time_ms'] / 1000
            console.print(f"\nExecution time: {execution_time:.2f} seconds")
            
        else:
            console.print(Panel(f"[bold red]Error![/]", title=AGENT_NAME))
            for error in results['results'].get("errors", []):
                console.print(f"[red]- {error}[/]")
    
    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

@app.command()
def query(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    query_str: str = typer.Option(None, "--query", "-q", help="Natural language query"),
    file: str = typer.Option(None, "--file", "-f", help="Path to JSON query file"),
    compute_loops: int = typer.Option(3, "--compute_loops", "-c", help="Number of reasoning iterations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save results (JSON format)"),
):
    """
    Process a query using natural language or a JSON file.
    
    Examples:
        uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL from 2023-01-01 to 2023-12-31"
        uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -f ./queries/fetch_aapl.json -v
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = TradeStationMarketDataAgent(
            database_path=database,
            verbose=verbose,
            compute_loops=compute_loops
        )
        
        # Process query or file
        if query_str:
            console.print(f"Processing query: [italic]{query_str}[/]")
            result = agent.process_query(query_str)
        else:
            console.print(f"Processing file: [italic]{file}[/]")
            result = agent.process_file(file)
        
        # Display results
        agent.display_results(result)
        
        # Save results to file if specified
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)  # default=str to handle date objects
            console.print(f"Results saved to [bold]{output}[/]")
        
        # Clean up
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

@app.command()
def version():
    """Display version information."""
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")

@app.command()
def test_connection(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Test database connection and API authentication."""
    try:
        agent = TradeStationMarketDataAgent(database_path=database, verbose=verbose)
        console.print("[bold green]Database connection successful![/]")
        
        # Test API authentication
        if agent.authenticate():
            console.print("[bold green]TradeStation API authentication successful![/]")
        else:
            console.print("[bold red]TradeStation API authentication failed![/]")
        
        agent.close()
    except Exception as e:
        console.print(f"[bold red]Connection failed:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()