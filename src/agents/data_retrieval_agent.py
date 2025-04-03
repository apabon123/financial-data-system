#!/usr/bin/env python3
"""
Agent Name: Data Retrieval Agent
Purpose: Retrieve and format data from DuckDB
Author: Claude
Date: 2025-04-02

Description:
    This agent retrieves data from DuckDB based on natural language queries,
    formatting the results as needed. It handles complex query construction,
    optimized data retrieval, and result formatting.

Usage:
    uv run data_retrieval_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run data_retrieval_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run data_retrieval_agent.py -d ./financial_data.duckdb -q "get daily close prices for AAPL, MSFT from 2023-01-01 to 2023-12-31"
    uv run data_retrieval_agent.py -d ./financial_data.duckdb -q "show monthly average price for SPY for the past year"
"""

import os
import sys
import json
import logging
import argparse
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, date

import typer
import duckdb
import pandas as pd
import numpy as np
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
logger = logging.getLogger("Data Retrieval Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Data Retrieval Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Retrieve and format data from DuckDB"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class DataRetrievalAgent:
    """Agent for retrieving and formatting data from DuckDB.
    
    This agent constructs SQL queries from natural language requests,
    executes optimized queries against DuckDB, and formats the results
    as needed for the user.
    
    Attributes:
        database_path (str): Path to the DuckDB database file
        verbose (bool): Whether to enable verbose logging
        compute_loops (int): Number of reasoning iterations to perform
        conn (duckdb.DuckDBPyConnection): Database connection
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
        
        # Mapping from natural language to SQL constructs
        self.time_period_mapping = {
            "daily": "daily",
            "day": "daily",
            "minute": "minute",
            "5-minute": "five_minute_bars",
            "weekly": "weekly",
            "week": "weekly",
            "monthly": "monthly",
            "month": "monthly"
        }
        
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
        
        This method extracts structured parameters related to data retrieval
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "get daily close prices for AAPL, MSFT from 2023-01-01 to 2023-12-31"
            Output: {
                "action": "get",
                "data_type": "market_data",
                "timeframe": "daily",
                "fields": ["close"],
                "symbols": ["AAPL", "MSFT"],
                "start_date": datetime.date(2023, 1, 1),
                "end_date": datetime.date(2023, 12, 31)
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "get",
            "data_type": "market_data",
            "timeframe": "daily",
            "fields": ["*"],
            "symbols": [],
            "indicators": [],
            "start_date": None,
            "end_date": None,
            "limit": None,
            "aggregation": None,
            "order_by": None,
            "group_by": None,
            "format": "dataframe"
        }
        
        # Extract action (get, show, display, find, etc.)
        if query.lower().startswith(("get", "show", "display", "find", "retrieve", "fetch")):
            params["action"] = "get"
        elif query.lower().startswith(("list", "count")):
            params["action"] = "list"
        elif query.lower().startswith(("calculate", "compute")):
            params["action"] = "calculate"
        
        # Extract timeframe
        for period, sql_name in self.time_period_mapping.items():
            if period in query.lower():
                params["timeframe"] = sql_name
                break
        
        # Extract data type
        if "price" in query.lower() or any(sym in query.lower() for sym in ["aapl", "msft", "spy", "stock", "etf"]):
            params["data_type"] = "market_data"
        elif "economic" in query.lower() or any(ind in query.lower() for ind in ["gdp", "cpi", "unemployment", "indicator"]):
            params["data_type"] = "economic_data"
        elif "position" in query.lower() or "holding" in query.lower():
            params["data_type"] = "positions"
        elif "account" in query.lower() or "balance" in query.lower():
            params["data_type"] = "account_balances"
        elif "order" in query.lower():
            params["data_type"] = "orders"
        elif "trade" in query.lower():
            params["data_type"] = "trades"
        elif "indicator" in query.lower() or any(ind in query.lower() for ind in ["rsi", "macd", "sma", "ema"]):
            params["data_type"] = "derived_indicators"
        
        # Extract symbols
        symbols_match = re.search(r'for\s+([A-Za-z0-9,\s]+)(?:\s+from|\s+between|\s+in|\s+during|\s*$)', query)
        if symbols_match:
            symbols_str = symbols_match.group(1)
            # Check if this might be an indicator instead of symbols
            if params["data_type"] == "economic_data" and any(ind in symbols_str.lower() for ind in ["gdp", "cpi", "unemployment", "rate", "inflation"]):
                params["indicators"] = [s.strip() for s in symbols_str.split(',')]
            else:
                params["symbols"] = [s.strip() for s in symbols_str.split(',')]
        
        # Extract fields
        fields_list = []
        
        # Look for specific price components
        if "open" in query.lower():
            fields_list.append("open")
        if "high" in query.lower():
            fields_list.append("high")
        if "low" in query.lower():
            fields_list.append("low")
        if "close" in query.lower():
            fields_list.append("close")
        if "volume" in query.lower():
            fields_list.append("volume")
        
        # Look for OHLC or OHLCV patterns
        if "ohlcv" in query.lower():
            fields_list = ["open", "high", "low", "close", "volume"]
        elif "ohlc" in query.lower():
            fields_list = ["open", "high", "low", "close"]
        
        # If fields were specified, update the params
        if fields_list:
            params["fields"] = fields_list
        
        # Extract date range
        # Standard date format
        date_range_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', query)
        if date_range_match:
            start_date_str = date_range_match.group(1)
            end_date_str = date_range_match.group(2)
            params["start_date"] = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            params["end_date"] = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # "past X days/weeks/months/years" pattern
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
        
        # Extract aggregation
        if "average" in query.lower() or "avg" in query.lower():
            params["aggregation"] = "AVG"
        elif "sum" in query.lower() or "total" in query.lower():
            params["aggregation"] = "SUM"
        elif "max" in query.lower() or "maximum" in query.lower() or "highest" in query.lower():
            params["aggregation"] = "MAX"
        elif "min" in query.lower() or "minimum" in query.lower() or "lowest" in query.lower():
            params["aggregation"] = "MIN"
        
        # Extract group by
        if "by symbol" in query.lower():
            params["group_by"] = "symbol"
        elif "by date" in query.lower() or "by day" in query.lower():
            params["group_by"] = "DATE(timestamp)"
        elif "by month" in query.lower():
            params["group_by"] = "DATE_TRUNC('month', timestamp)"
        elif "by week" in query.lower():
            params["group_by"] = "DATE_TRUNC('week', timestamp)"
        
        # Extract order by
        if "order by date" in query.lower() or "sorted by date" in query.lower():
            params["order_by"] = "timestamp"
        elif "order by symbol" in query.lower() or "sorted by symbol" in query.lower():
            params["order_by"] = "symbol"
        elif "order by price" in query.lower() or "sorted by price" in query.lower():
            params["order_by"] = "close"
        elif "order by volume" in query.lower() or "sorted by volume" in query.lower():
            params["order_by"] = "volume"
        
        # Extract limit
        limit_match = re.search(r'limit\s+(\d+)', query)
        if limit_match:
            params["limit"] = int(limit_match.group(1))
        elif "top" in query.lower():
            top_match = re.search(r'top\s+(\d+)', query.lower())
            if top_match:
                params["limit"] = int(top_match.group(1))
                if "order by" not in query.lower() and "sorted by" not in query.lower():
                    # Default ordering for "top N" queries
                    if "volume" in query.lower():
                        params["order_by"] = "volume DESC"
                    else:
                        params["order_by"] = "close DESC"
        
        # Extract output format
        if "as csv" in query.lower() or "in csv" in query.lower():
            params["format"] = "csv"
        elif "as json" in query.lower() or "in json" in query.lower():
            params["format"] = "json"
        
        logger.debug(f"Parsed parameters: {params}")
        return params
    
    def execute_query(self, sql: str) -> Any:
        """Execute a SQL query and return results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Query results as a DataFrame or specified format
            
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
    
    def get_active_symbols(self) -> List[str]:
        """Retrieve the list of active symbols from the database.
        
        Returns:
            List of active symbol strings
        """
        try:
            query = "SELECT DISTINCT symbol FROM symbols WHERE active = true ORDER BY symbol"
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                # Fallback to market_data table if symbols table is empty
                query = "SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
                result = self.conn.execute(query).fetchdf()
            
            return result['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error retrieving active symbols: {e}")
            return []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        This is the main entry point for agent functionality. It parses
        the query, constructs and executes SQL, and returns formatted results.
        
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
            "action": params.get("action", "get"),
            "success": False,
            "errors": [],
            "warnings": [],
            "data": None,
            "sql": None,
            "record_count": 0,
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
                    # Loop 2: Construct SQL query
                    self._loop_construct_sql_query(params, result)
                elif i == 2:
                    # Loop 3: Execute query and format results
                    self._loop_execute_and_format(params, result)
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
        valid_actions = ["get", "list", "calculate"]
        if params.get("action") not in valid_actions:
            result["errors"].append(f"Invalid action: {params.get('action')}. Must be one of {valid_actions}")
            return
        
        # Validate data_type and check if the corresponding table exists
        try:
            data_type = params.get("data_type")
            table_name = self._get_table_name_for_data_type(data_type, params.get("timeframe"))
            
            table_exists = self.conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' OR type='view' AND name='{table_name}'").fetchone()[0]
            if not table_exists:
                result["errors"].append(f"Table or view '{table_name}' does not exist in the database")
                return
            
            # Store the actual table/view name to use
            params["table_name"] = table_name
            
        except Exception as e:
            result["errors"].append(f"Error validating data type: {str(e)}")
            return
        
        # If symbols are specified, validate they exist
        if params.get("symbols"):
            try:
                symbols_str = "'" + "','".join(params["symbols"]) + "'"
                query = f"SELECT DISTINCT symbol FROM {table_name} WHERE symbol IN ({symbols_str})"
                result_df = self.conn.execute(query).fetchdf()
                
                found_symbols = result_df['symbol'].tolist()
                missing_symbols = [s for s in params["symbols"] if s not in found_symbols]
                
                if missing_symbols:
                    result["warnings"].append(f"Symbols not found in database: {', '.join(missing_symbols)}")
                    # Remove missing symbols from the list
                    params["symbols"] = found_symbols
                    
                if not found_symbols:
                    result["errors"].append(f"None of the specified symbols {', '.join(params['symbols'])} found in the database")
                    return
                
            except Exception as e:
                result["warnings"].append(f"Error validating symbols: {str(e)}")
        
        # If indicators are specified, validate they exist (for economic data)
        if params.get("indicators") and data_type == "economic_data":
            try:
                indicators_str = "'" + "','".join(params["indicators"]) + "'"
                query = f"SELECT DISTINCT indicator FROM economic_data WHERE indicator IN ({indicators_str})"
                result_df = self.conn.execute(query).fetchdf()
                
                found_indicators = result_df['indicator'].tolist()
                missing_indicators = [i for i in params["indicators"] if i not in found_indicators]
                
                if missing_indicators:
                    result["warnings"].append(f"Indicators not found in database: {', '.join(missing_indicators)}")
                    # Remove missing indicators from the list
                    params["indicators"] = found_indicators
                    
                if not found_indicators:
                    result["errors"].append(f"None of the specified indicators {', '.join(params['indicators'])} found in the database")
                    return
                
            except Exception as e:
                result["warnings"].append(f"Error validating indicators: {str(e)}")
        
        # If no date range was specified, set reasonable defaults
        if not params.get("start_date") or not params.get("end_date"):
            end_date = datetime.now().date()
            
            # Default time ranges based on timeframe
            if params.get("timeframe") in ["minute", "five_minute_bars"]:
                # For minute data, default to last 3 days
                start_date = end_date - timedelta(days=3)
            else:
                # For daily and other data, default to last 30 days
                start_date = end_date - timedelta(days=30)
            
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
        
        logger.debug("Parameters validated successfully")
    
    def _get_table_name_for_data_type(self, data_type: str, timeframe: str) -> str:
        """Get the appropriate table or view name for a given data type and timeframe.
        
        Args:
            data_type: Type of data (market_data, economic_data, etc.)
            timeframe: Timeframe (daily, minute, etc.)
            
        Returns:
            Table or view name to query
        """
        # Market data has different views based on timeframe
        if data_type == "market_data":
            if timeframe == "daily":
                return "daily_bars"
            elif timeframe == "minute":
                return "minute_bars"
            elif timeframe == "five_minute_bars":
                return "five_minute_bars"
            elif timeframe == "weekly":
                return "weekly_bars"
            elif timeframe == "monthly":
                return "monthly_bars"
            else:
                return "market_data"
        
        # Other data types map directly to their tables
        elif data_type in ["economic_data", "derived_indicators", "positions", "account_balances", "orders", "trades"]:
            return data_type
        
        # Default to the data_type as the table name
        return data_type
    
    def _loop_construct_sql_query(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Construct SQL query from parameters.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Constructing SQL query")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        table_name = params["table_name"]
        
        # Start building the SQL query
        select_clause = self._build_select_clause(params)
        from_clause = f"FROM {table_name}"
        where_clause = self._build_where_clause(params)
        group_by_clause = self._build_group_by_clause(params)
        order_by_clause = self._build_order_by_clause(params)
        limit_clause = f"LIMIT {params['limit']}" if params.get("limit") else ""
        
        # Combine the clauses into a complete SQL query
        sql_query = f"""
            {select_clause}
            {from_clause}
            {where_clause}
            {group_by_clause}
            {order_by_clause}
            {limit_clause}
        """
        
        # Store the SQL query in the result
        result["sql"] = sql_query.strip()
        
        logger.debug(f"Constructed SQL query: {result['sql']}")
    
    def _build_select_clause(self, params: Dict[str, Any]) -> str:
        """Build the SELECT clause of the SQL query.
        
        Args:
            params: Query parameters
            
        Returns:
            SELECT clause as a string
        """
        # Handle different data types
        if params["data_type"] == "market_data":
            # For market data, we need to handle fields and aggregations
            if params.get("aggregation") and params.get("group_by"):
                # If we're aggregating, select the group by field and aggregated values
                if "symbol" in params.get("group_by", ""):
                    group_field = "symbol"
                else:
                    group_field = "DATE_TRUNC('day', timestamp) as date"
                
                select_fields = []
                
                # Add the grouping field
                select_fields.append(group_field)
                
                # Add aggregations for selected fields
                agg_func = params["aggregation"]
                if params["fields"] == ["*"] or params["fields"] == []:
                    # Default to aggregating close price if no fields specified
                    select_fields.append(f"{agg_func}(close) as avg_close")
                else:
                    for field in params["fields"]:
                        if field in ["open", "high", "low", "close", "volume"]:
                            select_fields.append(f"{agg_func}({field}) as {agg_func.lower()}_{field}")
                
                return f"SELECT {', '.join(select_fields)}"
            else:
                # No aggregation, select specific fields
                if params["fields"] == ["*"] or params["fields"] == []:
                    return "SELECT timestamp, symbol, open, high, low, close, volume"
                else:
                    fields = ["timestamp", "symbol"] + params["fields"]
                    return f"SELECT {', '.join(fields)}"
        
        elif params["data_type"] == "economic_data":
            # For economic data, always include timestamp and indicator
            if params.get("aggregation") and params.get("group_by"):
                # Aggregating economic data
                agg_func = params["aggregation"]
                return f"SELECT indicator, {agg_func}(value) as {agg_func.lower()}_value"
            else:
                return "SELECT timestamp, indicator, value, source, frequency, revision_number"
        
        elif params["data_type"] == "derived_indicators":
            # For derived indicators, include indicator details
            if params["fields"] == ["*"] or params["fields"] == []:
                return "SELECT timestamp, symbol, indicator_name, value, parameters"
            else:
                fields = ["timestamp", "symbol", "indicator_name"] + params["fields"]
                return f"SELECT {', '.join(fields)}"
        
        elif params["data_type"] in ["positions", "account_balances", "orders", "trades"]:
            # For account data, just select all fields
            return "SELECT *"
        
        # Default to selecting all fields
        return "SELECT *"
    
    def _build_where_clause(self, params: Dict[str, Any]) -> str:
        """Build the WHERE clause of the SQL query.
        
        Args:
            params: Query parameters
            
        Returns:
            WHERE clause as a string
        """
        conditions = []
        
        # Add date range condition
        if params.get("start_date") and params.get("end_date"):
            # Format dates for SQL query
            start_date_str = params["start_date"].strftime("%Y-%m-%d")
            end_date_str = params["end_date"].strftime("%Y-%m-%d")
            conditions.append(f"timestamp BETWEEN '{start_date_str}' AND '{end_date_str}'")
        
        # Add symbols condition for market data
        if params["data_type"] in ["market_data", "derived_indicators"] and params.get("symbols"):
            symbols_str = "'" + "','".join(params["symbols"]) + "'"
            conditions.append(f"symbol IN ({symbols_str})")
        
        # Add indicators condition for economic data
        if params["data_type"] == "economic_data" and params.get("indicators"):
            indicators_str = "'" + "','".join(params["indicators"]) + "'"
            conditions.append(f"indicator IN ({indicators_str})")
        
        # Combine conditions with AND
        if conditions:
            return "WHERE " + " AND ".join(conditions)
        else:
            return ""
    
    def _build_group_by_clause(self, params: Dict[str, Any]) -> str:
        """Build the GROUP BY clause of the SQL query.
        
        Args:
            params: Query parameters
            
        Returns:
            GROUP BY clause as a string
        """
        if params.get("aggregation") and params.get("group_by"):
            group_by_field = params["group_by"]
            
            if params["data_type"] == "economic_data":
                return "GROUP BY indicator"
            elif "DATE_TRUNC" in group_by_field or "DATE" in group_by_field:
                return f"GROUP BY {group_by_field}"
            else:
                return f"GROUP BY {group_by_field}"
        else:
            return ""
    
    def _build_order_by_clause(self, params: Dict[str, Any]) -> str:
        """Build the ORDER BY clause of the SQL query.
        
        Args:
            params: Query parameters
            
        Returns:
            ORDER BY clause as a string
        """
        if params.get("order_by"):
            order_by_field = params["order_by"]
            
            # Check if ascending or descending order is specified
            if "DESC" in order_by_field or "ASC" in order_by_field:
                return f"ORDER BY {order_by_field}"
            else:
                return f"ORDER BY {order_by_field} ASC"
        elif params.get("aggregation") and "symbol" in params.get("group_by", ""):
            # When grouping by symbol with aggregation, order by the aggregated value
            agg_func = params["aggregation"].lower()
            if params["fields"] == ["*"] or params["fields"] == []:
                return f"ORDER BY {agg_func}_close DESC"
            else:
                return f"ORDER BY {agg_func}_{params['fields'][0]} DESC"
        elif params.get("aggregation") and params.get("group_by"):
            # When grouping by date with aggregation, order by the grouping field
            return "ORDER BY 1"
        elif params["data_type"] in ["market_data", "derived_indicators", "economic_data"]:
            # Default ordering for time series data
            if params.get("symbols") and len(params["symbols"]) > 1:
                return "ORDER BY symbol ASC, timestamp ASC"
            else:
                return "ORDER BY timestamp ASC"
        else:
            return ""
    
    def _loop_execute_and_format(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Execute the SQL query and format results.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Executing query and formatting results")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Execute the SQL query
        try:
            sql_query = result["sql"]
            data = self.execute_query(sql_query)
            
            if data.empty:
                result["warnings"].append("Query returned no data")
                result["data"] = data
                result["record_count"] = 0
                return
            
            # Store the data and record count
            result["data"] = data
            result["record_count"] = len(data)
            
            # Format the data according to the requested format
            if params.get("format") == "csv":
                # Convert to CSV string
                result["formatted_data"] = data.to_csv(index=False)
                result["format"] = "csv"
            elif params.get("format") == "json":
                # Convert to JSON string
                result["formatted_data"] = data.to_json(orient="records", date_format="iso")
                result["format"] = "json"
            else:
                # Default to keeping as DataFrame
                result["formatted_data"] = data
                result["format"] = "dataframe"
            
            logger.debug(f"Query executed successfully, returned {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            result["errors"].append(f"Error executing query: {str(e)}")
            return
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Query Successful![/]", title=AGENT_NAME))
            
            # Get the data
            data = results["results"].get("data")
            record_count = results["results"].get("record_count", 0)
            
            # Display summary
            console.print(f"\n[bold]Query Results:[/]")
            console.print(f"Records returned: {record_count}")
            
            # Display the data
            if data is not None and not data.empty:
                # Limit displayed rows for large results
                display_rows = min(len(data), 10)
                
                # Create Rich table
                table = Table(show_header=True, header_style="bold cyan")
                
                # Add columns
                for column in data.columns:
                    table.add_column(str(column))
                
                # Add rows
                for i in range(display_rows):
                    row = data.iloc[i]
                    table.add_row(*[str(val) for val in row])
                
                console.print(table)
                
                if len(data) > display_rows:
                    console.print(f"\n[dim]Showing {display_rows} of {len(data)} records[/]")
                
                # Provide hint about saving data
                console.print("\n[dim]To save these results to a file, use the --output option:[/]")
                console.print(f"[dim]uv run data_retrieval_agent.py -d {self.database_path} -q \"{results['query']}\" -o results.csv[/]")
            
            # Display warnings if any
            if results['results'].get('warnings'):
                console.print("\n[bold yellow]Warnings:[/]")
                for warning in results['results']['warnings']:
                    console.print(f"[yellow]- {warning}[/]")
            
            # Display execution time
            execution_time = results['results']['metadata']['execution_time_ms'] / 1000
            console.print(f"\nExecution time: {execution_time:.2f} seconds")
            
        else:
            console.print(Panel(f"[bold red]Query Failed![/]", title=AGENT_NAME))
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
    output: str = typer.Option(None, "--output", "-o", help="Path to save results (CSV format)"),
):
    """
    Process a query using natural language or a JSON file.
    
    Examples:
        uv run data_retrieval_agent.py -d ./financial_data.duckdb -q "get daily close prices for AAPL, MSFT from 2023-01-01 to 2023-12-31"
        uv run data_retrieval_agent.py -d ./financial_data.duckdb -q "show monthly average price for SPY for the past year"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = DataRetrievalAgent(
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
        if output and result["success"] and "data" in result["results"]:
            data = result["results"]["data"]
            
            # Determine output format from extension
            file_ext = os.path.splitext(output)[1].lower()
            
            if file_ext == '.csv':
                data.to_csv(output, index=False)
            elif file_ext == '.json':
                data.to_json(output, orient="records", date_format="iso")
            elif file_ext == '.parquet':
                data.to_parquet(output, index=False)
            else:
                # Default to CSV
                data.to_csv(output, index=False)
                
            console.print(f"\nResults saved to [bold]{output}[/]")
        
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
def symbols(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """List available symbols in the database."""
    try:
        agent = DataRetrievalAgent(database_path=database, verbose=verbose)
        symbols = agent.get_active_symbols()
        
        if symbols:
            console.print(Panel(f"[bold green]Found {len(symbols)} symbols[/]", title=AGENT_NAME))
            
            # Display symbols in a grid
            table = Table(show_header=False, border_style="dim")
            
            # Calculate number of columns based on symbol length
            max_symbol_len = max(len(s) for s in symbols)
            col_width = max_symbol_len + 2
            term_width = os.get_terminal_size().columns
            num_cols = max(1, term_width // col_width)
            
            # Create rows
            rows = []
            for i in range(0, len(symbols), num_cols):
                rows.append(symbols[i:i+num_cols])
            
            # Add columns
            for i in range(num_cols):
                table.add_column()
            
            # Add rows
            for row in rows:
                table.add_row(*row + [''] * (num_cols - len(row)))
            
            console.print(table)
        else:
            console.print("[yellow]No symbols found in the database[/]")
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()