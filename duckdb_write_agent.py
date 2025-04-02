#!/usr/bin/env python3
"""
Agent Name: DuckDB Write Agent
Purpose: Write data to DuckDB efficiently
Author: Claude
Date: 2025-04-02

Description:
    This agent efficiently writes data to DuckDB, handling various write modes
    (insert, upsert, replace), batching, and conflict resolution. It provides
    optimized performance for writing large datasets to the database.

Usage:
    uv run duckdb_write_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run duckdb_write_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run duckdb_write_agent.py -d ./financial_data.duckdb -q "insert data.csv into market_data"
    uv run duckdb_write_agent.py -d ./financial_data.duckdb -q "upsert data.parquet into market_data using timestamp, symbol as keys"
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
logger = logging.getLogger("DuckDB Write Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "DuckDB Write Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Write data to DuckDB efficiently"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class DuckDBWriteAgent:
    """Agent for efficiently writing data to DuckDB.
    
    This agent handles writing data to DuckDB with various modes,
    batch sizes, and conflict resolution strategies.
    
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
        
        # Default batch size for writing large datasets
        self.default_batch_size = 50000
        
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
        
        This method extracts structured parameters related to data writing
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "insert data.csv into market_data"
            Output: {
                "action": "insert",
                "source": "data.csv",
                "table": "market_data",
                "batch_size": 50000,
                "keys": []
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "insert",
            "source": None,
            "table": None,
            "batch_size": self.default_batch_size,
            "keys": [],
            "data": None
        }
        
        # Extract action (insert, upsert, replace)
        if "insert" in query.lower():
            params["action"] = "insert"
        elif "upsert" in query.lower():
            params["action"] = "upsert"
        elif "replace" in query.lower():
            params["action"] = "replace"
        
        # Extract source file
        source_match = re.search(r'(?:insert|upsert|replace)\s+([^\s]+\.(?:csv|parquet|json))', query.lower())
        if source_match:
            params["source"] = source_match.group(1)
        
        # Or check for data reference
        elif "normalized_data" in query.lower():
            params["data"] = "normalized_data"
        
        # Extract target table
        table_match = re.search(r'into\s+(\w+)', query.lower())
        if table_match:
            params["table"] = table_match.group(1)
        
        # Extract keys for upsert
        keys_match = re.search(r'using\s+([\w\s,]+)\s+as\s+keys', query.lower())
        if keys_match:
            keys_str = keys_match.group(1)
            params["keys"] = [k.strip() for k in keys_str.split(',')]
        
        # Look for batch size
        batch_match = re.search(r'batch\s+size\s+(\d+)', query.lower())
        if batch_match:
            params["batch_size"] = int(batch_match.group(1))
        
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
        the query, performs the data writing operation, and returns results.
        
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
            "action": params.get("action", "insert"),
            "success": False,
            "errors": [],
            "warnings": [],
            "rows_processed": 0,
            "rows_written": 0,
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
                    # Loop 2: Prepare data for writing
                    self._loop_prepare_data(params, result)
                elif i == 2:
                    # Loop 3: Write data to database
                    self._loop_write_data(params, result)
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
        valid_actions = ["insert", "upsert", "replace"]
        if params.get("action") not in valid_actions:
            result["errors"].append(f"Invalid action: {params.get('action')}. Must be one of {valid_actions}")
            return
        
        # Validate table
        if not params.get("table"):
            result["errors"].append("No target table specified")
            return
        
        # Check if the table exists
        try:
            table_exists = self.conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{params['table']}'").fetchone()[0]
            if not table_exists:
                result["errors"].append(f"Table '{params['table']}' does not exist in the database")
                return
        except Exception as e:
            result["errors"].append(f"Error checking if table exists: {e}")
            return
        
        # Get table schema to validate column names later
        try:
            schema_df = self.conn.execute(f"DESCRIBE {params['table']}").fetchdf()
            params["table_columns"] = schema_df["column_name"].tolist()
            logger.debug(f"Table columns: {params['table_columns']}")
        except Exception as e:
            result["errors"].append(f"Error retrieving table schema: {e}")
            return
        
        # For upsert action, validate keys
        if params["action"] == "upsert" and not params.get("keys"):
            # Try to determine primary key from table schema
            try:
                pk_query = f"""
                    SELECT column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu 
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_name = '{params['table']}'
                """
                pk_df = self.conn.execute(pk_query).fetchdf()
                
                if not pk_df.empty:
                    params["keys"] = pk_df["column_name"].tolist()
                    result["warnings"].append(f"No keys specified for upsert. Using primary key: {', '.join(params['keys'])}")
                else:
                    result["errors"].append("No keys specified for upsert and no primary key found in table")
                    return
            except Exception as e:
                result["errors"].append(f"Error determining primary key: {e}")
                return
        
        # Validate source or data exists
        if not params.get("source") and not params.get("data"):
            result["errors"].append("No data source specified")
            return
        
        # Load data if source file is provided
        if params.get("source"):
            source_path = params["source"]
            
            # Check if file exists
            if not os.path.exists(source_path):
                result["errors"].append(f"Source file '{source_path}' does not exist")
                return
                
            try:
                # Load data based on file extension
                file_ext = os.path.splitext(source_path)[1].lower()
                
                if file_ext == '.csv':
                    data = pd.read_csv(source_path)
                elif file_ext == '.parquet':
                    data = pd.read_parquet(source_path)
                elif file_ext == '.json':
                    data = pd.read_json(source_path)
                else:
                    result["errors"].append(f"Unsupported file format: {file_ext}. Supported formats: .csv, .parquet, .json")
                    return
                
                # Store data for next loop
                result["data"] = data
                result["rows_processed"] = len(data)
                
                logger.debug(f"Loaded {len(data)} rows from {source_path}")
                
            except Exception as e:
                result["errors"].append(f"Error loading data from {source_path}: {str(e)}")
                return
        
        # For normalized_data reference, we'll handle it in the next loop
        elif params.get("data") == "normalized_data":
            try:
                # Check if normalized_data exists in the connection's context
                has_normalized_data = self.conn.execute("SELECT 'normalized_data' IN (SELECT table_name FROM information_schema.tables WHERE table_schema = 'temp')").fetchone()[0]
                
                if has_normalized_data:
                    # Count rows in normalized_data
                    count = self.conn.execute("SELECT COUNT(*) FROM normalized_data").fetchone()[0]
                    result["rows_processed"] = count
                    logger.debug(f"Using existing normalized_data with {count} rows")
                else:
                    result["errors"].append("Referenced normalized_data, but no such temporary table exists")
                    return
            except Exception as e:
                result["errors"].append(f"Error checking for normalized_data: {str(e)}")
                return
        
        logger.debug("Parameters validated successfully")
    
    def _loop_prepare_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Prepare data for writing.
        
        This includes data validation, column mapping, and type conversion.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Preparing data for writing")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
            
        # Skip preparation if using normalized_data
        if params.get("data") == "normalized_data":
            logger.debug("Using normalized_data directly, no preparation needed")
            return
        
        # Get the data from the previous loop
        if "data" not in result:
            result["errors"].append("No data available for preparation")
            return
        
        data = result["data"]
        table_columns = params["table_columns"]
        
        # Validate and prepare the data
        try:
            # Check if columns match the table schema
            data_columns = data.columns.tolist()
            
            # Find missing required columns
            missing_columns = [col for col in table_columns if col not in data_columns]
            
            if missing_columns:
                # If timestamp is the only missing column and it's required, add it
                if missing_columns == ["timestamp"] and "timestamp" in table_columns:
                    data["timestamp"] = datetime.now()
                    result["warnings"].append("Added current timestamp to data")
                else:
                    result["errors"].append(f"Missing required columns in data: {', '.join(missing_columns)}")
                    return
            
            # Find extra columns in the data that don't exist in the table
            extra_columns = [col for col in data_columns if col not in table_columns]
            
            if extra_columns:
                result["warnings"].append(f"Dropping extra columns from data: {', '.join(extra_columns)}")
                data = data.drop(columns=extra_columns)
            
            # Reorder columns to match the table schema
            common_columns = [col for col in table_columns if col in data.columns]
            data = data[common_columns]
            
            # Update data in the result
            result["prepared_data"] = data
            
            logger.debug(f"Prepared {len(data)} rows for writing")
            
        except Exception as e:
            result["errors"].append(f"Error preparing data: {str(e)}")
            return
    
    def _loop_write_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Write data to the database.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Writing data to database")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        table_name = params["table"]
        action = params["action"]
        batch_size = params["batch_size"]
        
        try:
            # Different handling based on data source
            if params.get("data") == "normalized_data":
                # Write from the normalized_data temporary table
                rows_written = self._write_from_temp_table(params, result)
            elif "prepared_data" in result:
                # Write from the prepared data DataFrame
                rows_written = self._write_from_dataframe(params, result)
            else:
                result["errors"].append("No prepared data available for writing")
                return
            
            # Update result with rows written
            result["rows_written"] = rows_written
            
            logger.debug(f"Successfully wrote {rows_written} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Error writing data to database: {e}")
            result["errors"].append(f"Error writing data to database: {str(e)}")
            return
    
    def _write_from_temp_table(self, params: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Write data from a temporary table to the target table.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
            
        Returns:
            Number of rows written
        """
        table_name = params["table"]
        action = params["action"]
        
        # Different SQL based on action
        if action == "insert":
            sql = f"INSERT INTO {table_name} SELECT * FROM normalized_data"
        elif action == "replace":
            # Delete all data first
            self.conn.execute(f"DELETE FROM {table_name}")
            sql = f"INSERT INTO {table_name} SELECT * FROM normalized_data"
        elif action == "upsert":
            keys = params["keys"]
            
            if not keys:
                result["errors"].append("No keys specified for upsert")
                return 0
            
            # Get column names from the temp table
            columns = self.conn.execute("DESCRIBE normalized_data").fetchdf()["column_name"].tolist()
            
            # Build column assignments for update
            update_cols = [f"{col} = excluded.{col}" for col in columns if col not in keys]
            
            if not update_cols:
                result["warnings"].append("No columns to update in upsert operation. Performing insert instead.")
                sql = f"INSERT INTO {table_name} SELECT * FROM normalized_data"
            else:
                # Build upsert SQL with "ON CONFLICT DO UPDATE"
                sql = f"""
                    INSERT INTO {table_name} 
                    SELECT * FROM normalized_data
                    ON CONFLICT ({', '.join(keys)}) 
                    DO UPDATE SET {', '.join(update_cols)}
                """
        
        # Execute the write operation
        result_set = self.conn.execute(sql)
        rows_written = result_set.fetchone()[0]
        
        return rows_written
    
    def _write_from_dataframe(self, params: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Write data from a DataFrame to the target table.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
            
        Returns:
            Number of rows written
        """
        table_name = params["table"]
        action = params["action"]
        batch_size = params["batch_size"]
        data = result["prepared_data"]
        total_rows = len(data)
        
        # Check if we need to batch the writes
        if total_rows > batch_size:
            # Write in batches
            rows_written = 0
            batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Writing data to {table_name} in {batches} batches...", 
                    total=batches
                )
                
                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    batch_data = data.iloc[i:batch_end]
                    
                    # Write the batch
                    if action == "insert":
                        self.conn.register("batch_data", batch_data)
                        batch_result = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_data")
                        batch_rows = batch_result.fetchone()[0]
                    elif action == "replace" and i == 0:
                        # Only truncate once, for the first batch
                        self.conn.execute(f"DELETE FROM {table_name}")
                        self.conn.register("batch_data", batch_data)
                        batch_result = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_data")
                        batch_rows = batch_result.fetchone()[0]
                    elif action == "replace":
                        # Subsequent batches just insert
                        self.conn.register("batch_data", batch_data)
                        batch_result = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_data")
                        batch_rows = batch_result.fetchone()[0]
                    elif action == "upsert":
                        # Create a temporary table for this batch
                        self.conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS batch_data AS SELECT * FROM batch_data LIMIT 0")
                        self.conn.execute("DELETE FROM batch_data")
                        self.conn.register("batch_data_df", batch_data)
                        self.conn.execute("INSERT INTO batch_data SELECT * FROM batch_data_df")
                        
                        # Do the upsert from the temporary table
                        keys = params["keys"]
                        columns = batch_data.columns.tolist()
                        update_cols = [f"{col} = excluded.{col}" for col in columns if col not in keys]
                        
                        if not update_cols:
                            batch_result = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_data")
                            batch_rows = batch_result.fetchone()[0]
                        else:
                            batch_sql = f"""
                                INSERT INTO {table_name} 
                                SELECT * FROM batch_data
                                ON CONFLICT ({', '.join(keys)}) 
                                DO UPDATE SET {', '.join(update_cols)}
                            """
                            batch_result = self.conn.execute(batch_sql)
                            batch_rows = batch_result.fetchone()[0]
                    
                    rows_written += batch_rows
                    logger.debug(f"Batch {i//batch_size + 1}/{batches}: Wrote {batch_rows} rows")
                    
                    progress.update(task, advance=1)
            
            return rows_written
            
        else:
            # Write all at once
            if action == "insert":
                self.conn.register("data", data)
                result_set = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")
                rows_written = result_set.fetchone()[0]
            elif action == "replace":
                self.conn.execute(f"DELETE FROM {table_name}")
                self.conn.register("data", data)
                result_set = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")
                rows_written = result_set.fetchone()[0]
            elif action == "upsert":
                keys = params["keys"]
                
                if not keys:
                    result["errors"].append("No keys specified for upsert")
                    return 0
                
                # Create a temporary table
                self.conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS batch_data AS SELECT * FROM data LIMIT 0")
                self.conn.execute("DELETE FROM batch_data")
                self.conn.register("data", data)
                self.conn.execute("INSERT INTO batch_data SELECT * FROM data")
                
                # Build column assignments for update
                columns = data.columns.tolist()
                update_cols = [f"{col} = excluded.{col}" for col in columns if col not in keys]
                
                if not update_cols:
                    result["warnings"].append("No columns to update in upsert operation. Performing insert instead.")
                    result_set = self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_data")
                    rows_written = result_set.fetchone()[0]
                else:
                    # Build upsert SQL
                    sql = f"""
                        INSERT INTO {table_name} 
                        SELECT * FROM batch_data
                        ON CONFLICT ({', '.join(keys)}) 
                        DO UPDATE SET {', '.join(update_cols)}
                    """
                    result_set = self.conn.execute(sql)
                    rows_written = result_set.fetchone()[0]
            
            return rows_written
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display write operation summary
            console.print(f"\n[bold]Write Operation Summary:[/]")
            console.print(f"Action: {results['parameters']['action']}")
            console.print(f"Table: {results['parameters']['table']}")
            console.print(f"Rows processed: {results['results']['rows_processed']}")
            console.print(f"Rows written: {results['results']['rows_written']}")
            
            # Display source information
            if results['parameters'].get('source'):
                console.print(f"Source: {results['parameters']['source']}")
            elif results['parameters'].get('data'):
                console.print(f"Source: {results['parameters']['data']}")
            
            # Display batch information if relevant
            if results['results']['rows_processed'] > results['parameters']['batch_size']:
                batches = (results['results']['rows_processed'] + results['parameters']['batch_size'] - 1) // results['parameters']['batch_size']
                console.print(f"Batch size: {results['parameters']['batch_size']}")
                console.print(f"Number of batches: {batches}")
            
            # Display warnings if any
            if results['results'].get('warnings'):
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
        uv run duckdb_write_agent.py -d ./financial_data.duckdb -q "insert data.csv into market_data"
        uv run duckdb_write_agent.py -d ./financial_data.duckdb -q "upsert data.parquet into market_data using timestamp, symbol as keys"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = DuckDBWriteAgent(
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
def test_write(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    table: str = typer.Option("market_data", "--table", "-t", help="Table to test write on"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Test write performance with sample data."""
    try:
        agent = DuckDBWriteAgent(database_path=database, verbose=verbose)
        console.print(f"[bold]Testing write performance for {table}...[/]")
        
        # Generate sample data
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data based on table
        if table == "market_data":
            # Generate OHLCV data for a few symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            dates = [datetime.now().date() - timedelta(days=i) for i in range(10)]
            
            data = []
            for symbol in symbols:
                for date in dates:
                    base_price = np.random.uniform(100, 1000)
                    row = {
                        "timestamp": date,
                        "symbol": symbol,
                        "open": base_price,
                        "high": base_price * np.random.uniform(1.0, 1.05),
                        "low": base_price * np.random.uniform(0.95, 1.0),
                        "close": base_price * np.random.uniform(0.97, 1.03),
                        "volume": int(np.random.uniform(1000000, 10000000)),
                        "source": "Test Data",
                        "interval_value": 1,
                        "interval_unit": "day",
                        "adjusted": True,
                        "quality": 100
                    }
                    data.append(row)
            
            df = pd.DataFrame(data)
            
        elif table == "economic_data":
            # Generate economic indicator data
            indicators = ["GDP", "CPI", "UNEMPLOYMENT_RATE", "FEDERAL_FUNDS_RATE"]
            dates = [datetime.now().date() - timedelta(days=i*30) for i in range(12)]
            
            data = []
            for indicator in indicators:
                for date in dates:
                    row = {
                        "timestamp": date,
                        "indicator": indicator,
                        "value": np.random.uniform(0, 10),
                        "source": "Test Data",
                        "frequency": "monthly",
                        "revision_number": 0
                    }
                    data.append(row)
            
            df = pd.DataFrame(data)
            
        else:
            # Generic test data
            console.print(f"[yellow]No specific test data generator for {table}. Using generic test data.[/]")
            
            # Get table schema
            schema_df = agent.conn.execute(f"DESCRIBE {table}").fetchdf()
            columns = schema_df["column_name"].tolist()
            
            # Generate 10 rows of random data
            data = []
            for i in range(10):
                row = {}
                for col in columns:
                    if "id" in col.lower():
                        row[col] = f"test_{i}"
                    elif "date" in col.lower() or "time" in col.lower():
                        row[col] = datetime.now() - timedelta(days=i)
                    elif "number" in col.lower() or "value" in col.lower():
                        row[col] = np.random.uniform(0, 100)
                    else:
                        row[col] = f"test_value_{i}"
                data.append(row)
            
            df = pd.DataFrame(data)
        
        # Save to temp CSV
        temp_file = "temp_test_data.csv"
        df.to_csv(temp_file, index=False)
        
        # Run write test
        result = agent.process_query(f"insert {temp_file} into {table}")
        agent.display_results(result)
        
        # Clean up
        os.remove(temp_file)
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Test failed:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    app()