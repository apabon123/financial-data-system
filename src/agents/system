#!/usr/bin/env python3
"""
Agent Name: Data Normalization Agent
Purpose: Transform and standardize data from various sources
Author: Claude
Date: 2025-04-02

Description:
    This agent normalizes data from different sources into a standardized format 
    for storage in DuckDB. It handles data transformations, missing value handling,
    outlier detection, and data quality metrics.

Usage:
    uv run data_normalization_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run data_normalization_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run data_normalization_agent.py -d ./financial_data.duckdb -q "normalize data.csv for market_data"
    uv run data_normalization_agent.py -d ./financial_data.duckdb -q "normalize earnings calendar data"
"""

import os
import sys
import json
import logging
import argparse
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

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
logger = logging.getLogger("Data Normalization Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Data Normalization Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Transform and standardize data from various sources"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class DataNormalizationAgent:
    """Agent for normalizing and transforming data.
    
    This agent handles data standardization, transformation, missing value
    handling, and outlier detection for data from various sources.
    
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
        
        # Define normalization patterns for different data types
        self.normalization_patterns = {
            "market_data": {
                "timestamp": {"required": True, "type": "datetime"},
                "symbol": {"required": True, "type": "str", "upper": True},
                "open": {"required": True, "type": "float", "min": 0},
                "high": {"required": True, "type": "float", "min": 0},
                "low": {"required": True, "type": "float", "min": 0},
                "close": {"required": True, "type": "float", "min": 0},
                "volume": {"required": True, "type": "int", "min": 0},
                "up_volume": {"required": False, "type": "int", "min": 0},
                "down_volume": {"required": False, "type": "int", "min": 0},
                "source": {"required": True, "type": "str"},
                "interval_value": {"required": True, "type": "int", "min": 1},
                "interval_unit": {"required": True, "type": "str", "allowed": ["minute", "hour", "day", "week", "month"]},
                "adjusted": {"required": True, "type": "bool"},
                "quality": {"required": True, "type": "int", "min": 0, "max": 100}
            },
            "economic_data": {
                "timestamp": {"required": True, "type": "datetime"},
                "indicator": {"required": True, "type": "str", "upper": True},
                "value": {"required": True, "type": "float"},
                "source": {"required": True, "type": "str"},
                "frequency": {"required": True, "type": "str", "allowed": ["daily", "weekly", "monthly", "quarterly", "annual"]},
                "revision_number": {"required": True, "type": "int", "min": 0, "default": 0}
            },
            "earnings_calendar": {
                "symbol": {"required": True, "type": "str", "upper": True},
                "company_name": {"required": False, "type": "str"},
                "report_date": {"required": True, "type": "date"},
                "time": {"required": False, "type": "str", "allowed": ["before_market", "after_market", "during_market", "unspecified"]},
                "estimated_eps": {"required": False, "type": "float"},
                "actual_eps": {"required": False, "type": "float"},
                "surprise_percent": {"required": False, "type": "float"}
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
        
        This method extracts structured parameters related to data normalization
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "normalize data.csv for market_data"
            Output: {
                "action": "normalize",
                "source": "data.csv",
                "target_schema": "market_data",
                "mapping": {}
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "normalize",
            "source": None,
            "target_schema": None,
            "mapping": {},
            "create_temp_table": True
        }
        
        # Extract source file
        source_match = re.search(r'normalize\s+([^\s]+\.(?:csv|parquet|json))', query.lower())
        if source_match:
            params["source"] = source_match.group(1)
        else:
            # Look for data source references
            if "market data" in query.lower():
                params["source"] = "market_data"
            elif "economic data" in query.lower():
                params["source"] = "economic_data"
            elif "earnings calendar data" in query.lower():
                params["source"] = "earnings_calendar"
            elif "scraped data" in query.lower() or "web scraping data" in query.lower():
                params["source"] = "scraped_data"
        
        # Extract target schema
        target_match = re.search(r'for\s+(\w+)', query.lower())
        if target_match:
            params["target_schema"] = target_match.group(1)
        else:
            # Try to infer from query
            if "market data" in query.lower():
                params["target_schema"] = "market_data"
            elif "economic data" in query.lower():
                params["target_schema"] = "economic_data"
            elif "earnings calendar" in query.lower():
                params["target_schema"] = "earnings_calendar"
        
        # Extract mapping if specified
        mapping_match = re.search(r'using\s+format:\s+([\w\s\-\.,=:\(\)]+)(?:\s|$)', query)
        if mapping_match:
            mapping_str = mapping_match.group(1)
            
            # Parse the mapping string
            mapping_items = [item.strip() for item in mapping_str.split('-')]
            
            for item in mapping_items:
                if ':' in item:
                    field, field_type = item.split(':', 1)
                    field = field.strip()
                    field_type = field_type.strip()
                    
                    params["mapping"][field] = {"type": field_type}
        
        # Check for option to create temporary table
        if "no temp table" in query.lower() or "without temp table" in query.lower():
            params["create_temp_table"] = False
        
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
        the query, normalizes data, and returns results.
        
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
            "action": params.get("action", "normalize"),
            "success": False,
            "errors": [],
            "warnings": [],
            "normalized_data": None,
            "metrics": {
                "original_rows": 0,
                "normalized_rows": 0,
                "missing_values_filled": 0,
                "outliers_corrected": 0
            },
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
                    # Loop 1: Parameter validation and data loading
                    self._loop_validate_and_load(params, result)
                elif i == 1:
                    # Loop 2: Normalize data
                    self._loop_normalize_data(params, result)
                elif i == 2:
                    # Loop 3: Create temporary table and generate metrics
                    self._loop_create_temp_table(params, result)
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
    
    def _loop_validate_and_load(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """First compute loop: Validate parameters and load data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
            
        Raises:
            ValueError: If parameters are invalid
        """
        logger.debug("Validating parameters and loading data")
        
        # Validate action
        if params.get("action") != "normalize":
            result["errors"].append(f"Invalid action: {params.get('action')}. Only 'normalize' is supported.")
            return
        
        # Validate source
        if not params.get("source"):
            result["errors"].append("No data source specified")
            return
        
        # Validate target schema
        if not params.get("target_schema"):
            result["errors"].append("No target schema specified")
            return
        
        # Check if target schema is supported
        if params["target_schema"] not in self.normalization_patterns:
            result["errors"].append(f"Unsupported target schema: {params['target_schema']}. Supported schemas: {', '.join(self.normalization_patterns.keys())}")
            return
        
        # Load the data from the source
        source = params["source"]
        try:
            if source.endswith(".csv"):
                # Load from CSV file
                df = pd.read_csv(source)
            elif source.endswith(".parquet"):
                # Load from Parquet file
                df = pd.read_parquet(source)
            elif source.endswith(".json"):
                # Load from JSON file
                df = pd.read_json(source)
            elif source == "scraped_data":
                # Check if there's a temporary table for scraped data
                has_temp_table = self.conn.execute("SELECT 'scraped_data' IN (SELECT table_name FROM information_schema.tables WHERE table_schema = 'temp')").fetchone()[0]
                
                if not has_temp_table:
                    result["errors"].append("No scraped_data temporary table found. Run the web scraping agent first.")
                    return
                
                # Load from the temporary table
                df = self.conn.execute("SELECT * FROM scraped_data").fetchdf()
            else:
                # Check if it's a table in the database
                table_exists = self.conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{source}'").fetchone()[0]
                
                if table_exists:
                    # Load from existing table
                    df = self.conn.execute(f"SELECT * FROM {source}").fetchdf()
                else:
                    result["errors"].append(f"Source '{source}' is not a file, temporary table, or database table")
                    return
            
            # Store the original data in the result
            result["original_data"] = df
            result["metrics"]["original_rows"] = len(df)
            
            logger.debug(f"Loaded {len(df)} rows from {source}")
            
        except Exception as e:
            result["errors"].append(f"Error loading data from {source}: {str(e)}")
            return
        
        logger.debug("Parameters validated and data loaded successfully")
    
    def _loop_normalize_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Normalize the data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Normalizing data")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        # Get the data and normalization pattern
        data = result["original_data"]
        target_schema = params["target_schema"]
        pattern = self.normalization_patterns[target_schema]
        
        # Create a new DataFrame for the normalized data
        normalized_data = pd.DataFrame()
        missing_values_filled = 0
        outliers_corrected = 0
        
        # Apply the mapping if specified
        column_mapping = {}
        if params.get("mapping"):
            # Process custom mapping
            for field, field_info in params["mapping"].items():
                if "source_column" in field_info:
                    column_mapping[field_info["source_column"]] = field
        
        # Apply column mapping if any
        if column_mapping:
            data = data.rename(columns=column_mapping)
        
        # Process each field in the target schema
        for field, requirements in pattern.items():
            # Check if the field exists in the data
            if field in data.columns:
                # Get the field data
                field_data = data[field]
                
                # Normalize based on the field type
                if requirements["type"] == "datetime":
                    # Convert to datetime
                    try:
                        normalized_data[field] = pd.to_datetime(field_data)
                    except:
                        # Try common date formats
                        try:
                            normalized_data[field] = pd.to_datetime(field_data, errors='coerce')
                            # Count missing values filled
                            missing_values_filled += normalized_data[field].isna().sum()
                        except:
                            result["warnings"].append(f"Could not convert field '{field}' to datetime")
                            normalized_data[field] = None
                
                elif requirements["type"] == "date":
                    # Convert to date
                    try:
                        normalized_data[field] = pd.to_datetime(field_data).dt.date
                    except:
                        # Try common date formats
                        try:
                            normalized_data[field] = pd.to_datetime(field_data, errors='coerce').dt.date
                            # Count missing values filled
                            missing_values_filled += normalized_data[field].isna().sum()
                        except:
                            result["warnings"].append(f"Could not convert field '{field}' to date")
                            normalized_data[field] = None
                
                elif requirements["type"] == "float":
                    # Convert to float
                    normalized_data[field] = pd.to_numeric(field_data, errors='coerce')
                    
                    # Count missing values filled
                    missing_values_filled += normalized_data[field].isna().sum()
                    
                    # Apply min/max constraints
                    if "min" in requirements and not pd.isna(requirements["min"]):
                        # Count and fix outliers
                        outliers = (normalized_data[field] < requirements["min"]).sum()
                        if outliers > 0:
                            normalized_data.loc[normalized_data[field] < requirements["min"], field] = requirements["min"]
                            outliers_corrected += outliers
                            result["warnings"].append(f"Fixed {outliers} values below minimum for field '{field}'")
                    
                    if "max" in requirements and not pd.isna(requirements["max"]):
                        # Count and fix outliers
                        outliers = (normalized_data[field] > requirements["max"]).sum()
                        if outliers > 0:
                            normalized_data.loc[normalized_data[field] > requirements["max"], field] = requirements["max"]
                            outliers_corrected += outliers
                            result["warnings"].append(f"Fixed {outliers} values above maximum for field '{field}'")
                
                elif requirements["type"] == "int":
                    # Convert to integer
                    normalized_data[field] = pd.to_numeric(field_data, errors='coerce').astype('Int64')
                    
                    # Count missing values filled
                    missing_values_filled += normalized_data[field].isna().sum()
                    
                    # Apply min/max constraints
                    if "min" in requirements and not pd.isna(requirements["min"]):
                        # Count and fix outliers
                        outliers = (normalized_data[field] < requirements["min"]).sum()
                        if outliers > 0:
                            normalized_data.loc[normalized_data[field] < requirements["min"], field] = requirements["min"]
                            outliers_corrected += outliers
                            result["warnings"].append(f"Fixed {outliers} values below minimum for field '{field}'")
                    
                    if "max" in requirements and not pd.isna(requirements["max"]):
                        # Count and fix outliers
                        outliers = (normalized_data[field] > requirements["max"]).sum()
                        if outliers > 0:
                            normalized_data.loc[normalized_data[field] > requirements["max"], field] = requirements["max"]
                            outliers_corrected += outliers
                            result["warnings"].append(f"Fixed {outliers} values above maximum for field '{field}'")
                
                elif requirements["type"] == "bool":
                    # Convert to boolean
                    if field_data.dtype == bool:
                        normalized_data[field] = field_data
                    else:
                        # Try to convert various formats to boolean
                        bool_map = {
                            'true': True, 'yes': True, 'y': True, '1': True, 1: True, 'on': True,
                            'false': False, 'no': False, 'n': False, '0': False, 0: False, 'off': False
                        }
                        
                        normalized_data[field] = field_data.map(
                            lambda x: bool_map.get(str(x).lower(), None) if pd.notna(x) else None
                        )
                        
                        # Count missing values filled
                        missing_values_filled += normalized_data[field].isna().sum()
                
                elif requirements["type"] == "str":
                    # Convert to string
                    normalized_data[field] = field_data.astype(str)
                    
                    # Apply string transformations
                    if "upper" in requirements and requirements["upper"]:
                        normalized_data[field] = normalized_data[field].str.upper()
                    
                    if "lower" in requirements and requirements["lower"]:
                        normalized_data[field] = normalized_data[field].str.lower()
                    
                    # Validate against allowed values
                    if "allowed" in requirements:
                        invalid_values = ~normalized_data[field].isin(requirements["allowed"])
                        if invalid_values.sum() > 0:
                            result["warnings"].append(f"Field '{field}' contains {invalid_values.sum()} invalid values")
                            
                            # Replace with default if specified
                            if "default" in requirements:
                                normalized_data.loc[invalid_values, field] = requirements["default"]
                                missing_values_filled += invalid_values.sum()
            else:
                # Field doesn't exist in the source data
                if requirements["required"]:
                    # Add a warning
                    result["warnings"].append(f"Required field '{field}' is missing in the source data")
                    
                    # Add empty column
                    normalized_data[field] = None
                    
                    # Use default value if specified
                    if "default" in requirements:
                        normalized_data[field] = requirements["default"]
                        missing_values_filled += len(normalized_data)
        
        # Update result with normalized data and metrics
        result["normalized_data"] = normalized_data
        result["metrics"]["normalized_rows"] = len(normalized_data)
        result["metrics"]["missing_values_filled"] = missing_values_filled
        result["metrics"]["outliers_corrected"] = outliers_corrected
        
        logger.debug(f"Normalized {len(normalized_data)} rows")
    
    def _loop_create_temp_table(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Create a temporary table for the normalized data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Creating temporary table")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Skip if creating a temporary table is not requested
        if not params.get("create_temp_table", True):
            logger.debug("Skipping temporary table creation")
            return
        
        # Get the normalized data
        normalized_data = result["normalized_data"]
        
        try:
            # Create/replace the temporary table
            self.conn.execute("DROP TABLE IF EXISTS normalized_data")
            
            # Register the DataFrame with DuckDB
            self.conn.register("normalized_df", normalized_data)
            
            # Create the temporary table
            self.conn.execute("CREATE TABLE normalized_data AS SELECT * FROM normalized_df")
            
            # Check if the table was created successfully
            count = self.conn.execute("SELECT COUNT(*) FROM normalized_data").fetchone()[0]
            
            logger.debug(f"Created temporary table 'normalized_data' with {count} rows")
            
            # Add information to the result
            result["temporary_table"] = "normalized_data"
            result["temporary_table_rows"] = count
            
        except Exception as e:
            result["errors"].append(f"Error creating temporary table: {str(e)}")
            return
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display normalization summary
            console.print(f"[cyan]Source:[/] {results['parameters']['source']}")
            console.print(f"[cyan]Target Schema:[/] {results['parameters']['target_schema']}")
            
            # Display data sample
            if "normalized_data" in results["results"] and not results["results"]["normalized_data"].empty:
                data = results["results"]["normalized_data"]
                
                # Limit displayed rows for large results
                display_rows = min(len(data), 5)
                
                console.print("\n[bold]Normalized Data Sample:[/]")
                
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
            
            # Display metrics
            console.print("\n[bold]Normalization Metrics:[/]")
            console.print(f"Original rows: {results['results']['metrics']['original_rows']}")
            console.print(f"Normalized rows: {results['results']['metrics']['normalized_rows']}")
            console.print(f"Missing values filled: {results['results']['metrics']['missing_values_filled']}")
            console.print(f"Outliers corrected: {results['results']['metrics']['outliers_corrected']}")
            
            # Display temporary table information
            if "temporary_table" in results["results"]:
                console.print(f"\n[bold]Temporary Table:[/] {results['results']['temporary_table']}")
                console.print(f"Rows: {results['results']['temporary_table_rows']}")
                console.print(f"\nTo use this data, refer to the temporary table in your next query.")
                console.print(f"Example: use the DuckDB Write Agent to save this data permanently.")
            
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
        uv run data_normalization_agent.py -d ./financial_data.duckdb -q "normalize data.csv for market_data"
        uv run data_normalization_agent.py -d ./financial_data.duckdb -q "normalize earnings calendar data"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = DataNormalizationAgent(
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
def list_schemas(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
):
    """List supported normalization schemas."""
    try:
        agent = DataNormalizationAgent(database_path=database)
        
        console.print(Panel("[bold]Supported Normalization Schemas[/]", border_style="cyan"))
        
        for schema_name, schema in agent.normalization_patterns.items():
            console.print(f"\n[bold cyan]{schema_name}[/]")
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("Field")
            table.add_column("Type")
            table.add_column("Required")
            table.add_column("Constraints")
            
            for field, field_info in schema.items():
                constraints = []
                
                if "min" in field_info:
                    constraints.append(f"min: {field_info['min']}")
                if "max" in field_info:
                    constraints.append(f"max: {field_info['max']}")
                if "allowed" in field_info:
                    constraints.append(f"allowed: {field_info['allowed']}")
                if "upper" in field_info and field_info["upper"]:
                    constraints.append("uppercase")
                if "lower" in field_info and field_info["lower"]:
                    constraints.append("lowercase")
                if "default" in field_info:
                    constraints.append(f"default: {field_info['default']}")
                
                table.add_row(
                    field,
                    field_info["type"],
                    "Yes" if field_info["required"] else "No",
                    ", ".join(constraints) if constraints else ""
                )
            
            console.print(table)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()