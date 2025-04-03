#!/usr/bin/env python3
"""
Agent Name: Economic Data API Agent
Purpose: Retrieve economic data from sources like FRED and store in DuckDB
Author: Claude
Date: 2025-04-02

Description:
    This agent retrieves economic data from APIs like FRED (Federal Reserve Economic Data)
    and prepares it for storage in DuckDB. It handles authentication, data normalization,
    and tracking of data revisions.

Usage:
    uv run economic_data_api_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run economic_data_api_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run economic_data_api_agent.py -d ./financial_data.duckdb -q "fetch economic indicators GDP, CPI, UNEMPLOYMENT_RATE from 2022-01-01 to 2023-12-31"
    uv run economic_data_api_agent.py -d ./financial_data.duckdb -q "update economic indicators"
"""

import os
import sys
import json
import logging
import argparse
import re
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, date

import typer
import duckdb
import pandas as pd
import numpy as np
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
logger = logging.getLogger("Economic Data API Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Economic Data API Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Retrieve economic data from sources like FRED and store in DuckDB"

# FRED API configuration
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"

# Known economic indicators
KNOWN_INDICATORS = {
    "GDP": {
        "series_id": "GDP",
        "display_name": "Gross Domestic Product",
        "description": "Gross Domestic Product, quarterly, seasonally adjusted annual rate",
        "unit": "Billions of Dollars",
        "frequency": "quarterly"
    },
    "GDPC1": {
        "series_id": "GDPC1",
        "display_name": "Real Gross Domestic Product",
        "description": "Real Gross Domestic Product, quarterly, seasonally adjusted annual rate",
        "unit": "Billions of Chained 2017 Dollars",
        "frequency": "quarterly"
    },
    "UNRATE": {
        "series_id": "UNRATE",
        "display_name": "Unemployment Rate",
        "description": "Civilian Unemployment Rate, monthly, seasonally adjusted",
        "unit": "Percent",
        "frequency": "monthly"
    },
    "CPIAUCSL": {
        "series_id": "CPIAUCSL",
        "display_name": "Consumer Price Index",
        "description": "Consumer Price Index for All Urban Consumers: All Items, monthly, seasonally adjusted",
        "unit": "Index 1982-1984=100",
        "frequency": "monthly"
    },
    "CPI": {
        "series_id": "CPIAUCSL",
        "display_name": "Consumer Price Index",
        "description": "Consumer Price Index for All Urban Consumers: All Items, monthly, seasonally adjusted",
        "unit": "Index 1982-1984=100",
        "frequency": "monthly"
    },
    "INFLATION": {
        "series_id": "CPIAUCSL",
        "display_name": "Consumer Price Index",
        "description": "Consumer Price Index for All Urban Consumers: All Items, monthly, seasonally adjusted",
        "unit": "Index 1982-1984=100",
        "frequency": "monthly"
    },
    "FEDFUNDS": {
        "series_id": "FEDFUNDS",
        "display_name": "Federal Funds Effective Rate",
        "description": "Federal Funds Effective Rate, monthly, not seasonally adjusted",
        "unit": "Percent",
        "frequency": "monthly"
    },
    "FEDERAL_FUNDS_RATE": {
        "series_id": "FEDFUNDS",
        "display_name": "Federal Funds Effective Rate",
        "description": "Federal Funds Effective Rate, monthly, not seasonally adjusted",
        "unit": "Percent",
        "frequency": "monthly"
    },
    "UNEMPLOYMENT_RATE": {
        "series_id": "UNRATE",
        "display_name": "Unemployment Rate",
        "description": "Civilian Unemployment Rate, monthly, seasonally adjusted",
        "unit": "Percent",
        "frequency": "monthly"
    }
}

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class EconomicDataAPIAgent:
    """Agent for retrieving economic data from APIs like FRED.
    
    This agent handles authentication with economic data APIs like FRED,
    retrieves economic indicator data, and normalizes it for storage in DuckDB.
    
    Attributes:
        database_path (str): Path to the DuckDB database file
        verbose (bool): Whether to enable verbose logging
        compute_loops (int): Number of reasoning iterations to perform
        conn (duckdb.DuckDBPyConnection): Database connection
        fred_api_key (str): FRED API key from environment variables
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
        
        # API authentication
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
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
        
        This method extracts structured parameters related to economic data
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "fetch economic indicators GDP, CPI, UNEMPLOYMENT_RATE from 2022-01-01 to 2023-12-31"
            Output: {
                "action": "fetch",
                "indicators": ["GDP", "CPI", "UNEMPLOYMENT_RATE"],
                "start_date": datetime.date(2022, 1, 1),
                "end_date": datetime.date(2023, 12, 31),
                "frequency": "monthly"
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "fetch",
            "indicators": [],
            "start_date": None,
            "end_date": None,
            "frequency": "monthly"
        }
        
        # Extract action
        if "update" in query.lower():
            params["action"] = "update"
        elif "fetch" in query.lower() or "get" in query.lower() or "retrieve" in query.lower():
            params["action"] = "fetch"
        
        # Extract indicators
        indicators_match = re.search(r'indicators\s+([A-Za-z0-9_,\s]+)(?:\s+from|\s+between|\s+for|\s*$)', query)
        if indicators_match:
            indicators_str = indicators_match.group(1)
            params["indicators"] = [ind.strip().upper() for ind in indicators_str.split(',')]
        
        # Extract date range
        date_range_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', query)
        if date_range_match:
            start_date_str = date_range_match.group(1)
            end_date_str = date_range_match.group(2)
            params["start_date"] = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            params["end_date"] = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # Extract "past X days/weeks/months/years" pattern
        past_period_match = re.search(r'past\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', query)
        if past_period_match:
            amount = int(past_period_match.group(1))
            unit = past_period_match.group(2)
            
            end_date = datetime.now().date()
            if unit in ['day', 'days']:
                start_date = end_date - timedelta(days=amount)
            elif unit in ['week', 'weeks']:
                start_date = end_date - timedelta(days=amount * 7)
            elif unit in ['month', 'months']:
                # Approximation for months
                start_date = end_date - timedelta(days=amount * 30)
            elif unit in ['year', 'years']:
                # Approximation for years
                start_date = end_date - timedelta(days=amount * 365)
            
            params["start_date"] = start_date
            params["end_date"] = end_date
        
        # Extract frequency
        if "daily" in query.lower():
            params["frequency"] = "daily"
        elif "weekly" in query.lower():
            params["frequency"] = "weekly"
        elif "monthly" in query.lower():
            params["frequency"] = "monthly"
        elif "quarterly" in query.lower():
            params["frequency"] = "quarterly"
        elif "annual" in query.lower() or "yearly" in query.lower():
            params["frequency"] = "annual"
        
        logger.debug(f"Parsed parameters: {params}")
        return params
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Pandas DataFrame containing query results
            
        Raises:
            Exception: If query execution fails
        """
        try:
            logger.debug(f"Executing SQL: {sql}")
            result = self.conn.execute(sql).fetchdf()
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        This is the main entry point for agent functionality. It parses
        the query, retrieves economic data from APIs, and saves it to the database.
        
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
            "indicators_processed": 0,
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
        valid_actions = ["fetch", "update"]
        if params.get("action") not in valid_actions:
            result["errors"].append(f"Invalid action: {params.get('action')}. Must be one of {valid_actions}")
            return
        
        # For "update" action, fetch the most recent data
        if params["action"] == "update":
            try:
                # Get latest timestamp for each indicator from the database
                latest_data = self.conn.execute("""
                    SELECT indicator, MAX(timestamp) as latest_date
                    FROM economic_data
                    GROUP BY indicator
                """).fetchdf()
                
                if not latest_data.empty:
                    # Set update parameters
                    params["_update_info"] = {}
                    
                    for _, row in latest_data.iterrows():
                        indicator = row['indicator']
                        latest_date = row['latest_date'].date() if isinstance(row['latest_date'], datetime) else row['latest_date']
                        
                        # Add a small buffer to avoid duplicates (e.g., if data was updated same day)
                        start_date = latest_date - timedelta(days=7)
                        
                        params["_update_info"][indicator] = {
                            "latest_date": latest_date,
                            "start_date": start_date
                        }
                    
                    # Set defaults for any indicators not in the database
                    params["start_date"] = datetime.now().date() - timedelta(days=365)  # 1 year ago
                    params["end_date"] = datetime.now().date()
                    
                    # If no indicators specified, update all in the database
                    if not params.get("indicators"):
                        params["indicators"] = latest_data['indicator'].tolist()
                        
                else:
                    # No existing data, treat as new fetch
                    params["action"] = "fetch"
                    if not params.get("start_date"):
                        params["start_date"] = datetime.now().date() - timedelta(days=365)  # 1 year ago
                    if not params.get("end_date"):
                        params["end_date"] = datetime.now().date()
                    
                    result["warnings"].append("No existing economic data found. Treating as initial fetch.")
                
            except Exception as e:
                result["errors"].append(f"Error determining update parameters: {str(e)}")
                return
        
        # Check that FRED API key is available
        if not self.fred_api_key:
            result["errors"].append("FRED API key not found in environment variables. Please set FRED_API_KEY.")
            return
        
        # For fetch action, validate date range
        if params["action"] == "fetch":
            if not params.get("start_date") or not params.get("end_date"):
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
                if not params.get("start_date"):
                    params["start_date"] = start_date
                    result["warnings"].append(f"No start date specified. Using default: {start_date}")
                
                if not params.get("end_date"):
                    params["end_date"] = end_date
                    result["warnings"].append(f"No end date specified. Using default: {end_date}")
            
            # Ensure start_date is before end_date
            if params["start_date"] > params["end_date"]:
                result["errors"].append(f"Invalid date range: start date {params['start_date']} is after end date {params['end_date']}")
                return
        
        # If no indicators specified, use defaults
        if not params.get("indicators"):
            params["indicators"] = ["GDP", "CPI", "UNRATE", "FEDFUNDS"]
            result["warnings"].append(f"No indicators specified. Using defaults: {', '.join(params['indicators'])}")
        
        # Check if indicators are known
        unknown_indicators = [ind for ind in params["indicators"] if ind not in KNOWN_INDICATORS]
        if unknown_indicators:
            result["warnings"].append(f"Unknown indicators: {', '.join(unknown_indicators)}. They will be skipped.")
            # Remove unknown indicators
            params["indicators"] = [ind for ind in params["indicators"] if ind in KNOWN_INDICATORS]
            
            if not params["indicators"]:
                result["errors"].append("No valid indicators to fetch")
                return
        
        logger.debug("Parameters validated successfully")
    
    def _loop_fetch_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Fetch data from economic data APIs.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Fetching data from economic data APIs")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        # Prepare to store fetched data
        all_data = []
        total_records = 0
        
        # Process indicators
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Fetching economic indicators...", 
                total=len(params["indicators"])
            )
            
            for indicator_name in params["indicators"]:
                logger.debug(f"Processing indicator: {indicator_name}")
                
                try:
                    # Get the FRED series ID
                    if indicator_name in KNOWN_INDICATORS:
                        series_info = KNOWN_INDICATORS[indicator_name]
                        series_id = series_info["series_id"]
                        
                        # Determine date range for this indicator
                        if params["action"] == "update" and indicator_name in params.get("_update_info", {}):
                            # Use the update information for this indicator
                            update_info = params["_update_info"][indicator_name]
                            start_date = update_info["start_date"]
                            end_date = params["end_date"]
                        else:
                            # Use the global date range
                            start_date = params["start_date"]
                            end_date = params["end_date"]
                        
                        # Fetch data from FRED
                        indicator_data = self._fetch_fred_data(
                            series_id=series_id,
                            start_date=start_date,
                            end_date=end_date,
                            frequency=params["frequency"]
                        )
                        
                        if not indicator_data.empty:
                            # Add indicator name and other metadata
                            indicator_data["indicator"] = indicator_name
                            indicator_data["source"] = "FRED"
                            indicator_data["frequency"] = series_info["frequency"]
                            indicator_data["revision_number"] = 0  # Initial import
                            
                            # Append to the all_data list
                            all_data.append(indicator_data)
                            total_records += len(indicator_data)
                            
                            logger.debug(f"Fetched {len(indicator_data)} records for {indicator_name}")
                        else:
                            logger.warning(f"No data returned for indicator {indicator_name}")
                    else:
                        logger.warning(f"Skipping unknown indicator: {indicator_name}")
                
                except Exception as e:
                    logger.error(f"Error fetching data for {indicator_name}: {e}")
                    result["errors"].append(f"Error fetching data for {indicator_name}: {str(e)}")
                
                # Update progress
                progress.update(task, advance=1)
        
        # Combine all data if any was fetched
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            result["fetched_data"] = combined_data
            result["data_fetched"] = total_records
            result["indicators_processed"] = len(params["indicators"])
            
            logger.debug(f"Fetched {total_records} records for {len(params['indicators'])} indicators")
        else:
            result["warnings"].append("No data was fetched from the APIs")
            result["data_fetched"] = 0
            result["indicators_processed"] = 0
    
    def _fetch_fred_data(self, series_id: str, start_date: date, end_date: date, frequency: str) -> pd.DataFrame:
        """Fetch data from FRED API for a specific series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency (daily, weekly, monthly, quarterly, annual)
            
        Returns:
            DataFrame containing the fetched data
        """
        # Map frequency to FRED frequency codes
        fred_frequency = {
            "daily": "d",
            "weekly": "w",
            "monthly": "m",
            "quarterly": "q",
            "annual": "a"
        }.get(frequency, "m")  # Default to monthly
        
        # Construct the API URL
        url = f"{FRED_API_BASE_URL}/series/observations"
        
        # Format dates for the API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Set up the API parameters
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date_str,
            "observation_end": end_date_str,
            "frequency": fred_frequency,
            "sort_order": "asc",
            "units": "lin"  # Use linear units (not percent change)
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        
        # Handle response
        if response.status_code == 200:
            data = response.json()
            
            if "observations" in data and data["observations"]:
                # Convert the observations to a DataFrame
                df = pd.DataFrame(data["observations"])
                
                # Convert date string to datetime
                df["date"] = pd.to_datetime(df["date"])
                
                # Rename columns to match our schema
                df = df.rename(columns={"date": "timestamp", "value": "value"})
                
                # Convert value to float, handling 'null' or '.' values
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                
                # Drop rows with NA values in the value column
                df = df.dropna(subset=["value"])
                
                # Keep only the columns we need
                df = df[["timestamp", "value"]]
                
                return df
            else:
                logger.warning(f"No observations returned for series {series_id}")
                return pd.DataFrame()
        else:
            logger.error(f"FRED API error: {response.status_code} - {response.text}")
            raise Exception(f"FRED API error: {response.status_code} - {response.text}")
    
    def _loop_save_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Save data to database.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Saving data to database")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Skip if no data was fetched
        if "fetched_data" not in result or result["data_fetched"] == 0:
            logger.debug("No data to save")
            return
        
        # Get the fetched data
        data = result["fetched_data"]
        
        try:
            # First, ensure indicator metadata is saved
            self._save_indicator_metadata(params["indicators"])
            
            # Create a temporary table for the new data
            self.conn.execute("""
                CREATE TEMPORARY TABLE temp_economic_data AS 
                SELECT * FROM economic_data LIMIT 0
            """)
            
            # Insert data into the temporary table
            self.conn.execute(
                "INSERT INTO temp_economic_data SELECT * FROM data"
            )
            
            # Count total records
            total_records = len(data)
            
            # Insert new records into the main table
            result_set = self.conn.execute("""
                INSERT INTO economic_data
                SELECT t.*
                FROM temp_economic_data t
                LEFT JOIN economic_data e ON 
                    t.timestamp = e.timestamp AND 
                    t.indicator = e.indicator
                WHERE e.timestamp IS NULL
            """)
            
            # Get count of new records
            new_records = result_set.fetchone()[0]
            
            # Check if we need to handle revisions
            if new_records < total_records:
                revised_records = self._handle_data_revisions(data)
                new_records += revised_records
            
            # Drop the temporary table
            self.conn.execute("DROP TABLE temp_economic_data")
            
            # Update result with save metrics
            result["data_saved"] = new_records
            
            logger.debug(f"Saved {new_records} records to database")
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            result["errors"].append(f"Error saving data to database: {str(e)}")
    
    def _save_indicator_metadata(self, indicators: List[str]) -> None:
        """Save indicator metadata to the database.
        
        Args:
            indicators: List of indicators to save metadata for
        """
        # Process each indicator
        for indicator in indicators:
            if indicator in KNOWN_INDICATORS:
                indicator_info = KNOWN_INDICATORS[indicator]
                
                try:
                    # Check if metadata already exists for this indicator
                    exists = self.conn.execute(
                        f"SELECT COUNT(*) FROM indicator_metadata WHERE indicator_name = '{indicator}'"
                    ).fetchone()[0]
                    
                    if not exists:
                        # Insert new metadata
                        self.conn.execute("""
                            INSERT INTO indicator_metadata (
                                indicator_name, display_name, description, unit, source
                            ) VALUES (?, ?, ?, ?, ?)
                        """, [
                            indicator,
                            indicator_info.get("display_name", indicator),
                            indicator_info.get("description", ""),
                            indicator_info.get("unit", ""),
                            "FRED"
                        ])
                        
                        logger.debug(f"Saved metadata for indicator {indicator}")
                    
                except Exception as e:
                    logger.error(f"Error saving metadata for indicator {indicator}: {e}")
    
    def _handle_data_revisions(self, new_data: pd.DataFrame) -> int:
        """Handle revisions in economic data.
        
        This method checks for revisions in the data and updates
        the database with the revised values, incrementing the revision_number.
        
        Args:
            new_data: DataFrame containing the new data
            
        Returns:
            Number of revised records
        """
        try:
            # Create temporary tables for the analysis
            self.conn.execute("""
                CREATE TEMPORARY TABLE new_data AS
                SELECT * FROM new_data
            """)
            
            # Find records that already exist but with different values
            revision_query = """
                SELECT n.timestamp, n.indicator, n.value, e.value as old_value, e.revision_number
                FROM new_data n
                JOIN economic_data e ON
                    n.timestamp = e.timestamp AND
                    n.indicator = e.indicator
                WHERE n.value != e.value
            """
            
            revisions_df = self.conn.execute(revision_query).fetchdf()
            
            if not revisions_df.empty:
                # Create a DataFrame for the revised records
                revised_data = []
                
                for _, row in revisions_df.iterrows():
                    # Create a new record with incremented revision number
                    revised_record = {
                        "timestamp": row["timestamp"],
                        "indicator": row["indicator"],
                        "value": row["value"],
                        "source": "FRED",
                        "frequency": KNOWN_INDICATORS.get(row["indicator"], {}).get("frequency", "monthly"),
                        "revision_number": row["revision_number"] + 1
                    }
                    
                    revised_data.append(revised_record)
                
                # Convert to DataFrame
                revised_df = pd.DataFrame(revised_data)
                
                # Register the DataFrame with DuckDB
                self.conn.register("revised_df", revised_df)
                
                # Update existing records with revised values
                update_query = """
                    UPDATE economic_data
                    SET 
                        value = r.value,
                        revision_number = r.revision_number
                    FROM revised_df r
                    WHERE
                        economic_data.timestamp = r.timestamp AND
                        economic_data.indicator = r.indicator
                """
                
                self.conn.execute(update_query)
                
                logger.debug(f"Updated {len(revised_df)} records with revised values")
                
                return len(revised_df)
            else:
                logger.debug("No revisions found")
                return 0
                
        except Exception as e:
            logger.error(f"Error handling data revisions: {e}")
            return 0
        finally:
            # Clean up temporary tables
            try:
                self.conn.execute("DROP TABLE IF EXISTS new_data")
            except:
                pass
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display request summary
            if results["parameters"]["action"] == "fetch":
                console.print(f"[cyan]Action:[/] Fetch economic data")
            else:
                console.print(f"[cyan]Action:[/] Update economic data")
                
            console.print(f"[cyan]Indicators:[/] {', '.join(results['parameters']['indicators'])}")
            
            if results["parameters"].get("start_date") and results["parameters"].get("end_date"):
                console.print(f"[cyan]Date Range:[/] {results['parameters']['start_date']} to {results['parameters']['end_date']}")
            
            # Display results summary
            console.print(f"\n[bold]Results Summary:[/]")
            console.print(f"Indicators processed: {results['results']['indicators_processed']}")
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
        uv run economic_data_api_agent.py -d ./financial_data.duckdb -q "fetch economic indicators GDP, CPI, UNEMPLOYMENT_RATE from 2022-01-01 to 2023-12-31"
        uv run economic_data_api_agent.py -d ./financial_data.duckdb -q "update economic indicators"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = EconomicDataAPIAgent(
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
def list_indicators(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
):
    """List available economic indicators."""
    try:
        agent = EconomicDataAPIAgent(database_path=database)
        
        # First check what's in the database
        indicators_df = agent.conn.execute("""
            SELECT indicator, COUNT(*) as data_points, MIN(timestamp) as first_date, MAX(timestamp) as last_date
            FROM economic_data
            GROUP BY indicator
            ORDER BY indicator
        """).fetchdf()
        
        if not indicators_df.empty:
            console.print(Panel("[bold]Economic Indicators in Database[/]", border_style="cyan"))
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Indicator")
            table.add_column("Data Points")
            table.add_column("First Date")
            table.add_column("Last Date")
            
            for _, row in indicators_df.iterrows():
                table.add_row(
                    row["indicator"],
                    str(row["data_points"]),
                    row["first_date"].strftime("%Y-%m-%d") if pd.notna(row["first_date"]) else "N/A",
                    row["last_date"].strftime("%Y-%m-%d") if pd.notna(row["last_date"]) else "N/A"
                )
            
            console.print(table)
        else:
            console.print("[yellow]No economic indicators found in the database.[/]")
        
        # Display known indicators
        console.print(Panel("[bold]Available Indicators for Fetching[/]", border_style="green"))
        
        known_table = Table(show_header=True, header_style="bold green")
        known_table.add_column("ID")
        known_table.add_column("Display Name")
        known_table.add_column("Frequency")
        known_table.add_column("Unit")
        
        for indicator, info in KNOWN_INDICATORS.items():
            known_table.add_row(
                indicator,
                info.get("display_name", indicator),
                info.get("frequency", "monthly"),
                info.get("unit", "")
            )
        
        console.print(known_table)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()