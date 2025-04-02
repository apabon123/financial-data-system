#!/usr/bin/env python3
"""
Agent Name: Derived Indicators Agent
Purpose: Calculate technical indicators from market data
Author: Claude
Date: 2025-04-02

Description:
    This agent calculates technical indicators from market data stored in the database.
    It supports various indicators like moving averages, RSI, MACD, Bollinger Bands, etc.
    and stores the calculated values in the derived_indicators table.

Usage:
    uv run derived_indicators_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run derived_indicators_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run derived_indicators_agent.py -d ./financial_data.duckdb -q "calculate RSI for AAPL using daily data from 2023-01-01 to 2023-12-31 with parameters: period=14"
    uv run derived_indicators_agent.py -d ./financial_data.duckdb -q "calculate SMA, EMA, MACD for MSFT using daily data from 2023-01-01"
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
logger = logging.getLogger("Derived Indicators Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Derived Indicators Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Calculate technical indicators from market data"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class DerivedIndicatorsAgent:
    """Agent for calculating technical indicators from market data.
    
    This agent calculates various technical indicators from market data
    stored in the database and saves the results to the derived_indicators table.
    
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
        
        # Define supported indicators with default parameters
        self.supported_indicators = {
            "SMA": {
                "description": "Simple Moving Average",
                "default_parameters": {"period": 20},
                "required_fields": ["close"],
                "function": self._calculate_sma
            },
            "EMA": {
                "description": "Exponential Moving Average",
                "default_parameters": {"period": 20},
                "required_fields": ["close"],
                "function": self._calculate_ema
            },
            "RSI": {
                "description": "Relative Strength Index",
                "default_parameters": {"period": 14},
                "required_fields": ["close"],
                "function": self._calculate_rsi
            },
            "MACD": {
                "description": "Moving Average Convergence Divergence",
                "default_parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "required_fields": ["close"],
                "function": self._calculate_macd
            },
            "BBANDS": {
                "description": "Bollinger Bands",
                "default_parameters": {"period": 20, "std_dev": 2.0},
                "required_fields": ["close"],
                "function": self._calculate_bbands
            },
            "ATR": {
                "description": "Average True Range",
                "default_parameters": {"period": 14},
                "required_fields": ["high", "low", "close"],
                "function": self._calculate_atr
            },
            "STOCH": {
                "description": "Stochastic Oscillator",
                "default_parameters": {"k_period": 14, "k_slowing_period": 3, "d_period": 3},
                "required_fields": ["high", "low", "close"],
                "function": self._calculate_stochastic
            }
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
        
        This method extracts structured parameters related to indicator calculation
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "calculate RSI for AAPL using daily data from 2023-01-01 to 2023-12-31 with parameters: period=14"
            Output: {
                "action": "calculate",
                "indicators": ["RSI"],
                "symbols": ["AAPL"],
                "timeframe": "daily",
                "start_date": datetime.date(2023, 1, 1),
                "end_date": datetime.date(2023, 12, 31),
                "parameters": {"RSI": {"period": 14}}
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "calculate",
            "indicators": [],
            "symbols": [],
            "timeframe": "daily",
            "start_date": None,
            "end_date": None,
            "parameters": {},
            "update_existing": False
        }
        
        # Extract indicators
        indicators_pattern = r'calculate\s+([\w\s,]+)\s+for'
        indicators_match = re.search(indicators_pattern, query, re.IGNORECASE)
        if indicators_match:
            indicators_str = indicators_match.group(1)
            params["indicators"] = [ind.strip().upper() for ind in indicators_str.split(',')]
        
        # Extract symbols
        symbols_pattern = r'for\s+([\w\s,]+)\s+using'
        symbols_match = re.search(symbols_pattern, query, re.IGNORECASE)
        if symbols_match:
            symbols_str = symbols_match.group(1)
            params["symbols"] = [s.strip().upper() for s in symbols_str.split(',')]
        
        # Extract timeframe
        timeframe_pattern = r'using\s+([\w\-]+)\s+data'
        timeframe_match = re.search(timeframe_pattern, query, re.IGNORECASE)
        if timeframe_match:
            timeframe = timeframe_match.group(1).lower()
            
            # Map to standard timeframes
            if timeframe in ['daily', 'day']:
                params["timeframe"] = "daily"
            elif "minute" in timeframe:
                # Handle cases like "5-minute", "1-minute", etc.
                minute_match = re.search(r'(\d+)-minute', timeframe)
                if minute_match and minute_match.group(1) == "5":
                    params["timeframe"] = "five_minute_bars"
                else:
                    params["timeframe"] = "minute"
            elif timeframe in ['weekly', 'week']:
                params["timeframe"] = "weekly"
            elif timeframe in ['monthly', 'month']:
                params["timeframe"] = "monthly"
        
        # Extract date range
        date_pattern = r'from\s+(\d{4}-\d{2}-\d{2})(?:\s+to\s+(\d{4}-\d{2}-\d{2}))?'
        date_match = re.search(date_pattern, query)
        if date_match:
            start_date_str = date_match.group(1)
            params["start_date"] = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            
            # End date might not be specified
            if date_match.group(2):
                end_date_str = date_match.group(2)
                params["end_date"] = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            else:
                params["end_date"] = datetime.now().date()
        
        # Extract parameters
        params_pattern = r'with\s+parameters:\s+([\w=,\s\.]+)(?:\s|$)'
        params_match = re.search(params_pattern, query, re.IGNORECASE)
        if params_match:
            params_str = params_match.group(1)
            
            # Split by comma for multiple parameters
            params_list = [p.strip() for p in params_str.split(',')]
            
            # Parse each parameter
            for p in params_list:
                # Split by equals sign
                if '=' in p:
                    key, value = p.split('=')
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert value to appropriate type
                    try:
                        # Try as int first
                        value = int(value)
                    except ValueError:
                        try:
                            # Then try as float
                            value = float(value)
                        except ValueError:
                            # Keep as string
                            pass
                    
                    # Determine which indicator this parameter belongs to
                    # For simplicity, assign to all indicators if not specified
                    for indicator in params["indicators"]:
                        if indicator not in params["parameters"]:
                            params["parameters"][indicator] = {}
                        
                        params["parameters"][indicator][key] = value
        
        # Extract update_existing flag
        if "update existing" in query.lower() or "overwrite" in query.lower():
            params["update_existing"] = True
        
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
        the query, calculates indicators, and saves results to the database.
        
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
            "action": params.get("action", "calculate"),
            "success": False,
            "errors": [],
            "warnings": [],
            "indicators_calculated": 0,
            "symbols_processed": 0,
            "calculations_performed": 0,
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
                    # Loop 2: Retrieve market data
                    self._loop_retrieve_market_data(params, result)
                elif i == 2:
                    # Loop 3: Calculate and save indicators
                    self._loop_calculate_indicators(params, result)
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
        if params.get("action") != "calculate":
            result["errors"].append(f"Invalid action: {params.get('action')}. Only 'calculate' is supported.")
            return
        
        # Validate indicators
        if not params.get("indicators"):
            result["errors"].append("No indicators specified")
            return
        
        # Check for unsupported indicators
        unsupported = [ind for ind in params["indicators"] if ind not in self.supported_indicators]
        if unsupported:
            result["warnings"].append(f"Unsupported indicators: {', '.join(unsupported)}. They will be skipped.")
            # Remove unsupported indicators
            params["indicators"] = [ind for ind in params["indicators"] if ind in self.supported_indicators]
            
            if not params["indicators"]:
                result["errors"].append("No supported indicators to calculate")
                return
        
        # Validate symbols
        if not params.get("symbols"):
            result["errors"].append("No symbols specified")
            return
        
        # Validate timeframe
        valid_timeframes = ["daily", "minute", "five_minute_bars", "weekly", "monthly"]
        if params.get("timeframe") not in valid_timeframes:
            result["errors"].append(f"Invalid timeframe: {params.get('timeframe')}. Must be one of {valid_timeframes}")
            return
        
        # Resolve timeframe to appropriate table/view
        if params["timeframe"] == "daily":
            params["table_name"] = "daily_bars"
        elif params["timeframe"] == "minute":
            params["table_name"] = "minute_bars"
        elif params["timeframe"] == "five_minute_bars":
            params["table_name"] = "five_minute_bars"
        elif params["timeframe"] == "weekly":
            params["table_name"] = "weekly_bars"
        elif params["timeframe"] == "monthly":
            params["table_name"] = "monthly_bars"
        else:
            params["table_name"] = "market_data"
        
        # Validate date range
        if not params.get("start_date"):
            # Default to 1 year ago
            params["start_date"] = datetime.now().date() - timedelta(days=365)
            result["warnings"].append(f"No start date specified. Using default: {params['start_date']}")
        
        if not params.get("end_date"):
            params["end_date"] = datetime.now().date()
            result["warnings"].append(f"No end date specified. Using default: {params['end_date']}")
        
        # Ensure start_date is before end_date
        if params["start_date"] > params["end_date"]:
            result["errors"].append(f"Invalid date range: start date {params['start_date']} is after end date {params['end_date']}")
            return
        
        # Set default parameters for indicators that don't have them
        for indicator in params["indicators"]:
            if indicator not in params["parameters"]:
                params["parameters"][indicator] = {}
            
            # Apply default parameters where missing
            defaults = self.supported_indicators[indicator]["default_parameters"]
            for key, value in defaults.items():
                if key not in params["parameters"][indicator]:
                    params["parameters"][indicator][key] = value
        
        # Map interval unit and value based on timeframe
        if params["timeframe"] == "daily":
            params["interval_unit"] = "day"
            params["interval_value"] = 1
        elif params["timeframe"] == "minute":
            params["interval_unit"] = "minute"
            params["interval_value"] = 1
        elif params["timeframe"] == "five_minute_bars":
            params["interval_unit"] = "minute"
            params["interval_value"] = 5
        elif params["timeframe"] == "weekly":
            params["interval_unit"] = "week"
            params["interval_value"] = 1
        elif params["timeframe"] == "monthly":
            params["interval_unit"] = "month"
            params["interval_value"] = 1
        
        logger.debug("Parameters validated successfully")
    
    def _loop_retrieve_market_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Retrieve market data from database.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Retrieving market data from database")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        # Construct the query to retrieve market data
        table_name = params["table_name"]
        symbols = params["symbols"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        
        # Format symbols for SQL query
        symbols_str = "'" + "','".join(symbols) + "'"
        
        # Determine required fields for all indicators
        required_fields = set(['timestamp', 'symbol'])
        for indicator in params["indicators"]:
            indicator_fields = self.supported_indicators[indicator]["required_fields"]
            required_fields.update(indicator_fields)
        
        # Convert to comma-separated string
        fields_str = ", ".join(required_fields)
        
        # Construct SQL query
        sql = f"""
            SELECT {fields_str}
            FROM {table_name}
            WHERE symbol IN ({symbols_str})
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY symbol, timestamp
        """
        
        try:
            # Execute the query
            data = self.execute_query(sql)
            
            if data.empty:
                result["errors"].append(f"No market data found for the specified symbols and time range")
                return
            
            # Store data in the result for the next loop
            result["market_data"] = data
            
            # Log the data retrieval
            symbol_counts = data.groupby('symbol').size().to_dict()
            logger.debug(f"Retrieved {len(data)} records for {len(symbol_counts)} symbols")
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            result["errors"].append(f"Error retrieving market data: {str(e)}")
            return
    
    def _loop_calculate_indicators(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Calculate and save technical indicators.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Calculating and saving technical indicators")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Get the market data
        market_data = result["market_data"]
        
        # Track calculation metrics
        total_calculations = 0
        indicators_calculated = len(params["indicators"])
        symbols_processed = len(params["symbols"])
        
        # Create a list to store all calculated indicators
        all_indicators = []
        
        # Process each symbol
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Calculating indicators for {len(params['symbols'])} symbols...", 
                total=len(params["symbols"])
            )
            
            for symbol in params["symbols"]:
                # Filter data for this symbol
                symbol_data = market_data[market_data['symbol'] == symbol].copy()
                
                if symbol_data.empty:
                    logger.warning(f"No data found for symbol {symbol}")
                    continue
                
                # Sort by timestamp
                symbol_data = symbol_data.sort_values('timestamp')
                
                # Calculate each indicator
                for indicator_name in params["indicators"]:
                    logger.debug(f"Calculating {indicator_name} for {symbol}")
                    
                    try:
                        # Get the indicator function
                        indicator_function = self.supported_indicators[indicator_name]["function"]
                        
                        # Get the parameters for this indicator
                        indicator_params = params["parameters"].get(indicator_name, {})
                        
                        # Calculate the indicator
                        indicator_result = indicator_function(symbol_data, **indicator_params)
                        
                        if not indicator_result.empty:
                            # Add to the list of calculated indicators
                            all_indicators.append(indicator_result)
                            
                            # Update calculation count
                            total_calculations += len(indicator_result)
                            
                            logger.debug(f"Calculated {len(indicator_result)} {indicator_name} values for {symbol}")
                        else:
                            logger.warning(f"No {indicator_name} values calculated for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
                        result["warnings"].append(f"Error calculating {indicator_name} for {symbol}: {str(e)}")
                
                # Update progress
                progress.update(task, advance=1)
        
        # Combine all indicator results
        if all_indicators:
            combined_indicators = pd.concat(all_indicators, ignore_index=True)
            
            # Save to database
            rows_saved = self._save_indicators_to_database(combined_indicators, params.get("update_existing", False))
            
            # Update result with metrics
            result["calculations_performed"] = total_calculations
            result["indicators_calculated"] = indicators_calculated
            result["symbols_processed"] = symbols_processed
            result["rows_saved"] = rows_saved
            
            logger.debug(f"Calculated {total_calculations} indicator values for {symbols_processed} symbols")
        else:
            result["warnings"].append("No indicator values were calculated")
            result["calculations_performed"] = 0
            result["rows_saved"] = 0
    
    def _save_indicators_to_database(self, indicators_df: pd.DataFrame, update_existing: bool = False) -> int:
        """Save calculated indicators to the database.
        
        Args:
            indicators_df: DataFrame containing calculated indicators
            update_existing: Whether to update existing indicator values
            
        Returns:
            Number of rows saved to the database
        """
        try:
            # Create a temporary table
            self.conn.execute("""
                CREATE TEMPORARY TABLE temp_indicators AS
                SELECT * FROM derived_indicators LIMIT 0
            """)
            
            # Insert data into the temporary table
            self.conn.register("indicators_df", indicators_df)
            self.conn.execute("INSERT INTO temp_indicators SELECT * FROM indicators_df")
            
            # Count total records
            total_records = len(indicators_df)
            
            if update_existing:
                # Delete existing records that would conflict
                delete_sql = """
                    DELETE FROM derived_indicators
                    WHERE (timestamp, symbol, indicator_name, interval_value, interval_unit) IN (
                        SELECT timestamp, symbol, indicator_name, interval_value, interval_unit
                        FROM temp_indicators
                    )
                """
                self.conn.execute(delete_sql)
                
                # Insert all records
                insert_sql = "INSERT INTO derived_indicators SELECT * FROM temp_indicators"
                self.conn.execute(insert_sql)
                
                rows_saved = total_records
            else:
                # Insert only new records
                insert_sql = """
                    INSERT INTO derived_indicators
                    SELECT t.*
                    FROM temp_indicators t
                    LEFT JOIN derived_indicators d ON
                        t.timestamp = d.timestamp AND
                        t.symbol = d.symbol AND
                        t.indicator_name = d.indicator_name AND
                        t.interval_value = d.interval_value AND
                        t.interval_unit = d.interval_unit
                    WHERE d.indicator_name IS NULL
                """
                result = self.conn.execute(insert_sql)
                rows_saved = result.fetchone()[0]
            
            # Drop the temporary table
            self.conn.execute("DROP TABLE temp_indicators")
            
            logger.debug(f"Saved {rows_saved} indicator values to database")
            return rows_saved
            
        except Exception as e:
            logger.error(f"Error saving indicators to database: {e}")
            raise
    
    # ---------- Indicator calculation functions ----------
    
    def _calculate_sma(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Simple Moving Average.
        
        Args:
            data: Market data DataFrame
            period: SMA period
            
        Returns:
            DataFrame with calculated SMA values
        """
        # Calculate SMA
        sma = data['close'].rolling(window=period).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'SMA',
            'value': sma,
            'parameters': json.dumps({'period': period}),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        })
        
        # Drop rows with NaN values
        result = result.dropna(subset=['value'])
        
        return result
    
    def _calculate_ema(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Market data DataFrame
            period: EMA period
            
        Returns:
            DataFrame with calculated EMA values
        """
        # Calculate EMA
        ema = data['close'].ewm(span=period, adjust=False).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'EMA',
            'value': ema,
            'parameters': json.dumps({'period': period}),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        })
        
        # Drop rows with NaN values
        result = result.dropna(subset=['value'])
        
        return result
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index.
        
        Args:
            data: Market data DataFrame
            period: RSI period
            
        Returns:
            DataFrame with calculated RSI values
        """
        # Calculate price changes
        delta = data['close'].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'RSI',
            'value': rsi,
            'parameters': json.dumps({'period': period}),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        })
        
        # Drop rows with NaN values
        result = result.dropna(subset=['value'])
        
        return result
    
    def _calculate_macd(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence.
        
        Args:
            data: Market data DataFrame
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with calculated MACD values
        """
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Create DataFrames for the three components
        macd_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'MACD_LINE',
            'value': macd_line,
            'parameters': json.dumps({
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        signal_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'MACD_SIGNAL',
            'value': signal_line,
            'parameters': json.dumps({
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        hist_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'MACD_HIST',
            'value': histogram,
            'parameters': json.dumps({
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        # Combine all components
        result = pd.concat([macd_df, signal_df, hist_df], ignore_index=True)
        
        return result
    
    def _calculate_bbands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands.
        
        Args:
            data: Market data DataFrame
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with calculated Bollinger Bands values
        """
        # Calculate middle band (SMA)
        middle_band = data['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Create DataFrames for the three components
        middle_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'BBANDS_MIDDLE',
            'value': middle_band,
            'parameters': json.dumps({
                'period': period,
                'std_dev': std_dev
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        upper_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'BBANDS_UPPER',
            'value': upper_band,
            'parameters': json.dumps({
                'period': period,
                'std_dev': std_dev
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        lower_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'BBANDS_LOWER',
            'value': lower_band,
            'parameters': json.dumps({
                'period': period,
                'std_dev': std_dev
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        # Combine all components
        result = pd.concat([middle_df, upper_df, lower_df], ignore_index=True)
        
        return result
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range.
        
        Args:
            data: Market data DataFrame
            period: ATR period
            
        Returns:
            DataFrame with calculated ATR values
        """
        # Calculate true range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR (simple moving average of true range)
        atr = true_range.rolling(window=period).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'ATR',
            'value': atr,
            'parameters': json.dumps({'period': period}),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        })
        
        # Drop rows with NaN values
        result = result.dropna(subset=['value'])
        
        return result
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, k_slowing_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.
        
        Args:
            data: Market data DataFrame
            k_period: %K period
            k_slowing_period: %K slowing period
            d_period: %D period
            
        Returns:
            DataFrame with calculated Stochastic Oscillator values
        """
        # Calculate %K
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        # Avoid division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        
        fast_k = 100 * (data['close'] - lowest_low) / denom
        
        # Apply slowing if specified
        if k_slowing_period > 1:
            slow_k = fast_k.rolling(window=k_slowing_period).mean()
        else:
            slow_k = fast_k
        
        # Calculate %D (SMA of %K)
        slow_d = slow_k.rolling(window=d_period).mean()
        
        # Create DataFrames for %K and %D
        k_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'STOCH_K',
            'value': slow_k,
            'parameters': json.dumps({
                'k_period': k_period,
                'k_slowing_period': k_slowing_period,
                'd_period': d_period
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        d_df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'indicator_name': 'STOCH_D',
            'value': slow_d,
            'parameters': json.dumps({
                'k_period': k_period,
                'k_slowing_period': k_slowing_period,
                'd_period': d_period
            }),
            'interval_value': data['interval_value'].iloc[0] if 'interval_value' in data.columns else None,
            'interval_unit': data['interval_unit'].iloc[0] if 'interval_unit' in data.columns else None,
            'created_at': datetime.now()
        }).dropna(subset=['value'])
        
        # Combine both components
        result = pd.concat([k_df, d_df], ignore_index=True)
        
        return result
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display request summary
            console.print(f"[cyan]Indicators:[/] {', '.join(results['parameters']['indicators'])}")
            console.print(f"[cyan]Symbols:[/] {', '.join(results['parameters']['symbols'])}")
            console.print(f"[cyan]Timeframe:[/] {results['parameters']['timeframe']}")
            console.print(f"[cyan]Date Range:[/] {results['parameters']['start_date']} to {results['parameters']['end_date']}")
            
            # Display parameters for each indicator
            console.print("\n[bold]Parameters Used:[/]")
            for indicator, params in results['parameters']['parameters'].items():
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                console.print(f"{indicator}: {params_str}")
            
            # Display results summary
            console.print(f"\n[bold]Results Summary:[/]")
            console.print(f"Symbols processed: {results['results']['symbols_processed']}")
            console.print(f"Indicators calculated: {results['results']['indicators_calculated']}")
            console.print(f"Total calculations: {results['results']['calculations_performed']}")
            console.print(f"Data points saved: {results['results'].get('rows_saved', 0)}")
            
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
        uv run derived_indicators_agent.py -d ./financial_data.duckdb -q "calculate RSI for AAPL using daily data from 2023-01-01 to 2023-12-31 with parameters: period=14"
        uv run derived_indicators_agent.py -d ./financial_data.duckdb -q "calculate SMA, EMA, MACD for MSFT using daily data from 2023-01-01"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = DerivedIndicatorsAgent(
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
    """List supported technical indicators with default parameters."""
    try:
        agent = DerivedIndicatorsAgent(database_path=database)
        
        # Display supported indicators
        console.print(Panel("[bold]Supported Technical Indicators[/]", border_style="cyan"))
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Indicator")
        table.add_column("Description")
        table.add_column("Default Parameters")
        table.add_column("Required Fields")
        
        for indicator, info in agent.supported_indicators.items():
            table.add_row(
                indicator,
                info["description"],
                ", ".join([f"{k}={v}" for k, v in info["default_parameters"].items()]),
                ", ".join(info["required_fields"])
            )
        
        console.print(table)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()