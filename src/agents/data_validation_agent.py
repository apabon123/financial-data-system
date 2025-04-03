#!/usr/bin/env python3
"""
Agent Name: Data Validation Agent
Purpose: Ensure data quality and integrity for financial data
Author: Claude
Date: 2025-04-02

Description:
    This agent validates financial data against defined rules and patterns,
    detecting anomalies, outliers, and data quality issues. It can apply
    validation rules to various data types and generate quality metrics.

Usage:
    uv run data_validation_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run data_validation_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run data_validation_agent.py -d ./financial_data.duckdb -q "validate market_data from last 7 days"
    uv run data_validation_agent.py -d ./financial_data.duckdb -q "validate economic_data where indicator = 'GDP'"
    uv run data_validation_agent.py -d ./financial_data.duckdb -q "auto_fix market_data where high < low"
"""

import os
import sys
import json
import logging
import argparse
import re
import uuid
import time
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
logger = logging.getLogger("Data Validation Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Data Validation Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Ensures data quality and integrity for financial data"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class DataValidationAgent:
    """Agent for validating financial data.
    
    This agent applies validation rules to financial data, detects anomalies,
    outliers, and data quality issues, and generates validation reports.
    
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
        
        # Standard validation rules for different data types
        self.validation_rules = {
            "market_data": {
                "timestamp_check": "timestamp IS NOT NULL",
                "symbol_check": "symbol IS NOT NULL AND symbol != ''",
                "price_range_check": "open > 0 AND high > 0 AND low > 0 AND close > 0",
                "high_low_check": "high >= low",
                "high_open_close_check": "high >= open AND high >= close",
                "low_open_close_check": "low <= open AND low <= close",
                "volume_check": "volume >= 0"
            },
            "economic_data": {
                "timestamp_check": "timestamp IS NOT NULL",
                "indicator_check": "indicator IS NOT NULL AND indicator != ''",
                "value_check": "value IS NOT NULL"
            },
            "derived_indicators": {
                "timestamp_check": "timestamp IS NOT NULL",
                "symbol_check": "symbol IS NOT NULL AND symbol != ''",
                "indicator_check": "indicator_name IS NOT NULL AND indicator_name != ''",
                "value_check": "value IS NOT NULL"
            }
        }
        
        # Auto-fix rules for common issues
        self.fix_rules = {
            "swap_high_low": {
                "description": "Swap high and low values when high < low",
                "detection": "high < low",
                "fix_sql": """
                    UPDATE market_data 
                    SET 
                        high = low,
                        low = high
                    WHERE high < low AND {additional_where}
                """
            },
            "cap_outlier_prices": {
                "description": "Cap extreme outlier prices (4+ standard deviations)",
                "detection": """
                    ABS(close - AVG(close) OVER(
                        PARTITION BY symbol 
                        ORDER BY timestamp
                        ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING
                    )) > 4 * STDDEV(close) OVER(
                        PARTITION BY symbol 
                        ORDER BY timestamp
                        ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING
                    )
                """,
                "fix_sql": """
                    UPDATE market_data
                    SET 
                        close = CASE
                            WHEN close > avg_close + 4*stddev_close THEN avg_close + 4*stddev_close
                            WHEN close < avg_close - 4*stddev_close THEN avg_close - 4*stddev_close
                            ELSE close
                        END,
                        high = CASE
                            WHEN high > avg_high + 4*stddev_high THEN avg_high + 4*stddev_high
                            ELSE high
                        END,
                        low = CASE
                            WHEN low < avg_low - 4*stddev_low THEN avg_low - 4*stddev_low
                            ELSE low
                        END,
                        open = CASE
                            WHEN open > avg_open + 4*stddev_open THEN avg_open + 4*stddev_open
                            WHEN open < avg_open - 4*stddev_open THEN avg_open - 4*stddev_open
                            ELSE open
                        END
                    FROM (
                        SELECT 
                            timestamp, 
                            symbol,
                            AVG(close) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as avg_close,
                            STDDEV(close) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as stddev_close,
                            AVG(high) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as avg_high,
                            STDDEV(high) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as stddev_high,
                            AVG(low) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as avg_low,
                            STDDEV(low) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as stddev_low,
                            AVG(open) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as avg_open,
                            STDDEV(open) OVER(PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) as stddev_open
                        FROM market_data
                        WHERE {additional_where}
                    ) stats
                    WHERE market_data.timestamp = stats.timestamp
                    AND market_data.symbol = stats.symbol
                    AND (
                        market_data.close > stats.avg_close + 4*stats.stddev_close
                        OR market_data.close < stats.avg_close - 4*stats.stddev_close
                        OR market_data.high > stats.avg_high + 4*stats.stddev_high
                        OR market_data.low < stats.avg_low - 4*stats.stddev_low
                        OR market_data.open > stats.avg_open + 4*stats.stddev_open
                        OR market_data.open < stats.avg_open - 4*stats.stddev_open
                    )
                """
            },
            "fill_missing_values": {
                "description": "Fill missing values with interpolated values",
                "detection": "open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL",
                "fix_sql": """
                    UPDATE market_data
                    SET 
                        open = COALESCE(open, close, 
                            (LAG(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp) + 
                             LEAD(open, 1) OVER(PARTITION BY symbol ORDER BY timestamp)) / 2),
                        high = COALESCE(high, GREATEST(open, close), 
                            (LAG(high, 1) OVER(PARTITION BY symbol ORDER BY timestamp) + 
                             LEAD(high, 1) OVER(PARTITION BY symbol ORDER BY timestamp)) / 2),
                        low = COALESCE(low, LEAST(open, close),
                            (LAG(low, 1) OVER(PARTITION BY symbol ORDER BY timestamp) + 
                             LEAD(low, 1) OVER(PARTITION BY symbol ORDER BY timestamp)) / 2),
                        close = COALESCE(close, open,
                            (LAG(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp) + 
                             LEAD(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp)) / 2)
                    FROM (
                        SELECT 
                            timestamp, 
                            symbol,
                            open,
                            high,
                            low,
                            close,
                            LAG(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as prev_close,
                            LEAD(open, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as next_open,
                            LAG(high, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as prev_high,
                            LEAD(high, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as next_high,
                            LAG(low, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as prev_low,
                            LEAD(low, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as next_low,
                            LAG(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as prev_close,
                            LEAD(close, 1) OVER(PARTITION BY symbol ORDER BY timestamp) as next_close
                        FROM market_data
                        WHERE {additional_where}
                    ) fill_data
                    WHERE market_data.timestamp = fill_data.timestamp
                    AND market_data.symbol = fill_data.symbol
                    AND (
                        market_data.open IS NULL
                        OR market_data.high IS NULL
                        OR market_data.low IS NULL
                        OR market_data.close IS NULL
                    )
                """
            }
        }
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._connect_database()
        self._ensure_validation_tables()
    
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
    
    def _ensure_validation_tables(self) -> None:
        """Ensure validation tables exist in the database.
        
        Creates the necessary tables for storing validation results
        if they don't already exist.
        """
        try:
            # Create validation_batches table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_batches (
                    batch_id VARCHAR PRIMARY KEY,
                    table_name VARCHAR,
                    validation_time TIMESTAMP,
                    records_checked INTEGER,
                    records_valid INTEGER,
                    records_invalid INTEGER,
                    query_filter VARCHAR
                )
            """)
            
            # Create validation_failures table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_failures (
                    failure_id VARCHAR PRIMARY KEY,
                    validation_batch_id VARCHAR,
                    rule_name VARCHAR,
                    table_name VARCHAR,
                    record_id VARCHAR,
                    column_name VARCHAR,
                    failure_message VARCHAR,
                    severity VARCHAR,
                    additional_info VARCHAR,
                    FOREIGN KEY (validation_batch_id) REFERENCES validation_batches(batch_id)
                )
            """)
            
            # Create validation_metrics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    validation_batch_id VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE,
                    metric_description VARCHAR,
                    FOREIGN KEY (validation_batch_id) REFERENCES validation_batches(batch_id)
                )
            """)
            
            # Create auto_fix_history table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS auto_fix_history (
                    fix_id VARCHAR PRIMARY KEY,
                    table_name VARCHAR,
                    fix_rule VARCHAR,
                    fix_time TIMESTAMP,
                    records_fixed INTEGER,
                    query_filter VARCHAR,
                    batch_id VARCHAR
                )
            """)
            
            logger.debug("Validation tables created or already exist")
            
        except Exception as e:
            logger.error(f"Failed to create validation tables: {e}")
            sys.exit(1)
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query to extract parameters.
        
        This method extracts structured parameters related to data validation
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "validate market_data from last 7 days"
            Output: {
                "action": "validate",
                "table_name": "market_data",
                "filter": "timestamp >= CURRENT_DATE - INTERVAL 7 DAY",
                "rules": ["standard"]
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "validate",
            "table_name": None,
            "filter": "",
            "rules": ["standard"],
            "batch_id": None,
            "fix_rule": None
        }
        
        # Extract action
        if query.startswith("validate"):
            params["action"] = "validate"
        elif query.startswith("auto_fix") or query.startswith("fix"):
            params["action"] = "auto_fix"
        
        # Extract table name
        table_match = re.search(r'(validate|auto_fix|fix)\s+(\w+)', query)
        if table_match:
            params["table_name"] = table_match.group(2)
        
        # Extract time-based filters
        if "last validation batch" in query or "validation batch" in query:
            # Get the most recent validation batch ID
            try:
                result = self.conn.execute("""
                    SELECT batch_id FROM validation_batches 
                    ORDER BY validation_time DESC LIMIT 1
                """).fetchone()
                
                if result:
                    params["batch_id"] = result[0]
                    logger.debug(f"Using latest validation batch: {params['batch_id']}")
            except Exception as e:
                logger.error(f"Error fetching latest validation batch: {e}")
        
        elif "batch" in query:
            # Extract specific batch ID
            batch_match = re.search(r'batch\s+([a-f0-9-]+)', query)
            if batch_match:
                params["batch_id"] = batch_match.group(1)
        
        elif "last" in query and "days" in query:
            # Extract "last X days" pattern
            days_match = re.search(r'last\s+(\d+)\s+days', query)
            if days_match:
                days = int(days_match.group(1))
                params["filter"] = f"timestamp >= CURRENT_DATE - INTERVAL '{days} days'"
        
        elif "from" in query and "to" in query:
            # Extract date range
            date_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', query)
            if date_match:
                start_date = date_match.group(1)
                end_date = date_match.group(2)
                params["filter"] = f"timestamp BETWEEN '{start_date}' AND '{end_date}'"
        
        elif "where" in query:
            # Extract custom where clause
            where_match = re.search(r'where\s+(.*?)(?:\s+using|\s*$)', query)
            if where_match:
                params["filter"] = where_match.group(1)
        
        # Extract validation rules
        rules_match = re.search(r'using\s+(.*?)(?:\s+|$)', query)
        if rules_match:
            rules_str = rules_match.group(1)
            
            if "standard" in rules_str or "standard price and volume" in rules_str:
                params["rules"] = ["standard"]
            else:
                params["rules"] = [r.strip() for r in rules_str.split(',')]
        
        # Extract fix rule for auto_fix action
        if params["action"] == "auto_fix":
            rule_match = re.search(r'using\s+rule\s+(\w+)', query)
            if rule_match:
                params["fix_rule"] = rule_match.group(1)
            elif "standard_fixes" in query:
                params["fix_rule"] = "standard_fixes"
        
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
        the query, validates data or applies fixes, and returns results.
        
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
            "action": params.get("action", "validate"),
            "success": False,
            "errors": [],
            "warnings": [],
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
                    # Loop 2: Perform validation or auto-fix
                    if params["action"] == "validate":
                        self._loop_validate_data(params, result)
                    elif params["action"] == "auto_fix":
                        self._loop_auto_fix(params, result)
                elif i == 2:
                    # Loop 3: Generate metrics and quality score
                    if params["action"] == "validate":
                        self._loop_generate_metrics(params, result)
                    else:
                        # Skip for auto_fix
                        pass
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
        valid_actions = ["validate", "auto_fix"]
        if params.get("action") not in valid_actions:
            result["errors"].append(f"Invalid action: {params.get('action')}. Must be one of {valid_actions}")
            return
        
        # Validate table name if not using batch_id
        if not params.get("batch_id") and not params.get("table_name"):
            result["errors"].append("No table name specified for validation")
            return
        
        # Check if the table exists
        if params.get("table_name"):
            try:
                table_exists = self.conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{params['table_name']}'").fetchone()[0]
                if not table_exists:
                    result["errors"].append(f"Table '{params['table_name']}' does not exist in the database")
                    return
            except Exception as e:
                result["errors"].append(f"Error checking if table exists: {e}")
                return
        
        # For auto_fix action, validate fix_rule
        if params["action"] == "auto_fix" and not params.get("fix_rule"):
            if params.get("rules") and "standard_fixes" in params["rules"]:
                params["fix_rule"] = "standard_fixes"
            else:
                result["errors"].append("No fix rule specified for auto_fix action")
                return
        
        # Validate rules
        if params["action"] == "validate" and not params.get("rules"):
            params["rules"] = ["standard"]
            result["warnings"].append("No validation rules specified. Using standard rules.")
        
        # Validate batch_id if specified
        if params.get("batch_id"):
            try:
                batch_exists = self.conn.execute(f"SELECT COUNT(*) FROM validation_batches WHERE batch_id='{params['batch_id']}'").fetchone()[0]
                if not batch_exists:
                    result["errors"].append(f"Validation batch '{params['batch_id']}' does not exist")
                    return
            except Exception as e:
                result["errors"].append(f"Error checking if validation batch exists: {e}")
                return
        
        logger.debug("Parameters validated successfully")
    
    def _loop_validate_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Validate data against rules.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Validating data")
        
        table_name = params["table_name"]
        filter_clause = params["filter"]
        
        # Generate a unique batch ID
        batch_id = str(uuid.uuid4())
        result["batch_id"] = batch_id
        
        # Build the WHERE clause
        where_clause = f"WHERE {filter_clause}" if filter_clause else ""
        
        # Count total records to validate
        count_query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
        total_records = self.conn.execute(count_query).fetchone()[0]
        
        if total_records == 0:
            result["warnings"].append(f"No records found in {table_name} {where_clause}")
            result["records_checked"] = 0
            result["records_valid"] = 0
            result["records_invalid"] = 0
            return
        
        logger.debug(f"Found {total_records} records to validate")
        
        # Get the validation rules for this table
        if table_name in self.validation_rules:
            rules = self.validation_rules[table_name]
        else:
            result["warnings"].append(f"No predefined validation rules for table {table_name}. Using basic existence checks.")
            rules = {
                "existence_check": "1=1"  # Placeholder rule that always passes
            }
        
        # Apply each validation rule
        invalid_records = 0
        failures = []
        
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Validating {total_records} records in {table_name}...", 
                total=len(rules)
            )
            
            for rule_name, rule_condition in rules.items():
                logger.debug(f"Applying rule: {rule_name}")
                
                # Find records that fail the rule
                invalid_query = f"""
                    SELECT *
                    FROM {table_name}
                    {where_clause} 
                    {' AND ' if where_clause else ' WHERE '}
                    NOT ({rule_condition})
                """
                
                try:
                    invalid_df = self.execute_query(invalid_query)
                    rule_failures = len(invalid_df)
                    
                    if rule_failures > 0:
                        logger.debug(f"Found {rule_failures} failures for rule {rule_name}")
                        invalid_records += rule_failures
                        
                        # Record failures
                        for _, row in invalid_df.iterrows():
                            # Determine the primary key field(s)
                            if "symbol" in row and "timestamp" in row:
                                record_id = f"{row['symbol']}_{row['timestamp']}"
                            else:
                                # Try to use the first field as an identifier
                                record_id = str(row.iloc[0])
                            
                            failure = {
                                "failure_id": str(uuid.uuid4()),
                                "validation_batch_id": batch_id,
                                "rule_name": rule_name,
                                "table_name": table_name,
                                "record_id": record_id,
                                "column_name": rule_name.split('_')[0] if '_' in rule_name else None,
                                "failure_message": f"Failed validation rule: {rule_name}",
                                "severity": "ERROR",
                                "additional_info": json.dumps(row.to_dict(), default=str)
                            }
                            
                            failures.append(failure)
                    
                except Exception as e:
                    logger.error(f"Error applying rule {rule_name}: {e}")
                    result["errors"].append(f"Error applying rule {rule_name}: {str(e)}")
                
                progress.update(task, advance=1)
        
        # Record validation batch in the database
        try:
            self.conn.execute("""
                INSERT INTO validation_batches (
                    batch_id, table_name, validation_time, 
                    records_checked, records_valid, records_invalid,
                    query_filter
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                batch_id, 
                table_name, 
                datetime.now().isoformat(), 
                total_records, 
                total_records - invalid_records, 
                invalid_records,
                filter_clause
            ])
            
            # Record failures if any
            if failures:
                # Convert failures to DataFrame and insert
                failures_df = pd.DataFrame(failures)
                self.conn.execute("INSERT INTO validation_failures SELECT * FROM failures_df")
            
            logger.debug(f"Recorded validation batch {batch_id}")
            
        except Exception as e:
            logger.error(f"Error recording validation batch: {e}")
            result["errors"].append(f"Error recording validation batch: {str(e)}")
        
        # Update result
        result["records_checked"] = total_records
        result["records_valid"] = total_records - invalid_records
        result["records_invalid"] = invalid_records
        
        logger.debug(f"Validation complete: {invalid_records} invalid records found out of {total_records}")
    
    def _loop_auto_fix(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Apply auto-fixes to data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Applying auto-fixes to data")
        
        table_name = params["table_name"]
        filter_clause = params["filter"]
        fix_rule = params["fix_rule"]
        batch_id = params.get("batch_id")
        
        # Generate a unique fix ID
        fix_id = str(uuid.uuid4())
        result["fix_id"] = fix_id
        
        # If using a batch ID, get the filter from the batch
        if batch_id:
            try:
                batch_query = f"SELECT table_name, query_filter FROM validation_batches WHERE batch_id='{batch_id}'"
                batch_info = self.conn.execute(batch_query).fetchone()
                
                if batch_info:
                    table_name = batch_info[0]
                    filter_clause = batch_info[1]
                else:
                    result["errors"].append(f"Validation batch '{batch_id}' not found")
                    return
            except Exception as e:
                result["errors"].append(f"Error retrieving batch information: {str(e)}")
                return
        
        # Build the WHERE clause
        where_clause = f"{filter_clause}" if filter_clause else "1=1"
        
        # Determine which fix rule(s) to apply
        if fix_rule == "standard_fixes":
            # Apply all standard fixes
            fix_rules = self.fix_rules
        elif fix_rule in self.fix_rules:
            # Apply a specific fix rule
            fix_rules = {fix_rule: self.fix_rules[fix_rule]}
        else:
            result["errors"].append(f"Fix rule '{fix_rule}' not found")
            return
        
        # Apply each fix rule
        total_fixed = 0
        
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Applying fixes to {table_name}...", 
                total=len(fix_rules)
            )
            
            for rule_name, rule_info in fix_rules.items():
                logger.debug(f"Applying fix rule: {rule_name}")
                
                # Check if there are records that need fixing
                detection_query = f"""
                    SELECT COUNT(*) 
                    FROM {table_name}
                    WHERE {where_clause} 
                    AND ({rule_info['detection']})
                """
                
                try:
                    records_to_fix = self.conn.execute(detection_query).fetchone()[0]
                    
                    if records_to_fix > 0:
                        logger.debug(f"Found {records_to_fix} records to fix with rule {rule_name}")
                        
                        # Apply the fix
                        fix_sql = rule_info['fix_sql'].format(additional_where=where_clause)
                        
                        # Execute the fix
                        fix_result = self.conn.execute(fix_sql)
                        records_fixed = fix_result.fetchone()[0]
                        total_fixed += records_fixed
                        
                        logger.debug(f"Fixed {records_fixed} records with rule {rule_name}")
                        
                        # Record the fix in the history
                        self.conn.execute("""
                            INSERT INTO auto_fix_history (
                                fix_id, table_name, fix_rule, fix_time,
                                records_fixed, query_filter, batch_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [
                            fix_id,
                            table_name,
                            rule_name,
                            datetime.now().isoformat(),
                            records_fixed,
                            where_clause,
                            batch_id
                        ])
                    else:
                        logger.debug(f"No records found needing fix: {rule_name}")
                    
                except Exception as e:
                    logger.error(f"Error applying fix rule {rule_name}: {e}")
                    result["errors"].append(f"Error applying fix rule {rule_name}: {str(e)}")
                
                progress.update(task, advance=1)
        
        # Update result
        result["records_fixed"] = total_fixed
        result["rules_applied"] = len(fix_rules)
        
        logger.debug(f"Auto-fix complete: {total_fixed} records fixed")
    
    def _loop_generate_metrics(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Generate data quality metrics.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Generating data quality metrics")
        
        # Only if we have a batch_id (either from params or previous loop)
        batch_id = params.get("batch_id") or result.get("batch_id")
        
        if not batch_id:
            logger.debug("No batch_id available, skipping metrics generation")
            return
        
        # Get batch information
        try:
            batch_query = f"SELECT table_name, query_filter FROM validation_batches WHERE batch_id='{batch_id}'"
            batch_info = self.conn.execute(batch_query).fetchone()
            
            if not batch_info:
                result["warnings"].append(f"Validation batch '{batch_id}' not found for metrics generation")
                return
                
            table_name = batch_info[0]
            filter_clause = batch_info[1]
            
        except Exception as e:
            result["errors"].append(f"Error retrieving batch information for metrics: {str(e)}")
            return
        
        # Build the WHERE clause
        where_clause = f"WHERE {filter_clause}" if filter_clause else ""
        
        # Generate metrics based on table type
        metrics = []
        
        if table_name == "market_data":
            # Market data specific metrics
            try:
                # Calculate price volatility
                volatility_query = f"""
                    SELECT 
                        symbol, 
                        STDDEV(close) / AVG(close) * 100 as price_volatility
                    FROM {table_name}
                    {where_clause}
                    GROUP BY symbol
                """
                
                volatility_df = self.execute_query(volatility_query)
                
                if not volatility_df.empty:
                    avg_volatility = volatility_df['price_volatility'].mean()
                    
                    metrics.append({
                        "metric_id": str(uuid.uuid4()),
                        "validation_batch_id": batch_id,
                        "metric_name": "avg_price_volatility",
                        "metric_value": float(avg_volatility),
                        "metric_description": "Average price volatility as percentage"
                    })
                
                # Calculate data completeness
                completeness_query = f"""
                    SELECT 
                        COUNT(*) as total_records,
                        SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as null_open,
                        SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as null_high,
                        SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as null_low,
                        SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                        SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
                    FROM {table_name}
                    {where_clause}
                """
                
                completeness_df = self.execute_query(completeness_query)
                
                if not completeness_df.empty and completeness_df.iloc[0, 0] > 0:
                    total_records = completeness_df.iloc[0, 0]
                    
                    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
                        null_count = completeness_df.iloc[0, i+1]
                        completeness = 100 - (null_count / total_records * 100)
                        
                        metrics.append({
                            "metric_id": str(uuid.uuid4()),
                            "validation_batch_id": batch_id,
                            "metric_name": f"{col}_completeness",
                            "metric_value": float(completeness),
                            "metric_description": f"Percentage of non-null {col} values"
                        })
                
                # Calculate overall data quality score
                quality_score = 100 - (result["records_invalid"] / result["records_checked"] * 100) if result["records_checked"] > 0 else 0
                
                metrics.append({
                    "metric_id": str(uuid.uuid4()),
                    "validation_batch_id": batch_id,
                    "metric_name": "overall_quality_score",
                    "metric_value": float(quality_score),
                    "metric_description": "Overall data quality score (0-100)"
                })
                
            except Exception as e:
                logger.error(f"Error generating market data metrics: {e}")
                result["warnings"].append(f"Error generating market data metrics: {str(e)}")
        
        elif table_name == "economic_data":
            # Economic data specific metrics
            try:
                # Calculate revision frequency
                revision_query = f"""
                    SELECT 
                        indicator,
                        AVG(revision_number) as avg_revisions
                    FROM {table_name}
                    {where_clause}
                    GROUP BY indicator
                """
                
                revision_df = self.execute_query(revision_query)
                
                if not revision_df.empty:
                    avg_revisions = revision_df['avg_revisions'].mean()
                    
                    metrics.append({
                        "metric_id": str(uuid.uuid4()),
                        "validation_batch_id": batch_id,
                        "metric_name": "avg_revision_count",
                        "metric_value": float(avg_revisions),
                        "metric_description": "Average number of revisions per indicator"
                    })
                
                # Calculate overall data quality score
                quality_score = 100 - (result["records_invalid"] / result["records_checked"] * 100) if result["records_checked"] > 0 else 0
                
                metrics.append({
                    "metric_id": str(uuid.uuid4()),
                    "validation_batch_id": batch_id,
                    "metric_name": "overall_quality_score",
                    "metric_value": float(quality_score),
                    "metric_description": "Overall data quality score (0-100)"
                })
                
            except Exception as e:
                logger.error(f"Error generating economic data metrics: {e}")
                result["warnings"].append(f"Error generating economic data metrics: {str(e)}")
        
        # Store metrics in the database
        if metrics:
            try:
                # Convert metrics to DataFrame and insert
                metrics_df = pd.DataFrame(metrics)
                self.conn.execute("INSERT INTO validation_metrics SELECT * FROM metrics_df")
                
                logger.debug(f"Stored {len(metrics)} metrics for batch {batch_id}")
                
                # Update result with metrics
                result["metrics"] = {m["metric_name"]: m["metric_value"] for m in metrics}
                result["quality_score"] = next((m["metric_value"] for m in metrics if m["metric_name"] == "overall_quality_score"), None)
                
            except Exception as e:
                logger.error(f"Error storing metrics: {e}")
                result["warnings"].append(f"Error storing metrics: {str(e)}")
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            if results["parameters"]["action"] == "validate":
                console.print(Panel(f"[bold green]Validation Complete![/]", title=AGENT_NAME))
                
                # Display validation summary
                console.print(f"\n[bold]Validation Summary:[/]")
                console.print(f"Table: {results['parameters']['table_name']}")
                console.print(f"Records checked: {results['results']['records_checked']}")
                console.print(f"Records valid: {results['results']['records_valid']}")
                console.print(f"Records with issues: {results['results']['records_invalid']}")
                
                # Display quality score if available
                if "quality_score" in results["results"]:
                    quality = results["results"]["quality_score"]
                    color = "green" if quality >= 90 else "yellow" if quality >= 70 else "red"
                    console.print(f"Quality score: [bold {color}]{quality:.1f}%[/]")
                
                # Display batch ID for reference
                console.print(f"\nValidation batch ID: {results['results']['batch_id']}")
                console.print("Use this ID for auto-fixing or further analysis")
                
                # Show examples of failures if there are any
                if results['results']['records_invalid'] > 0:
                    try:
                        batch_id = results['results']['batch_id']
                        failures = self.conn.execute(f"""
                            SELECT rule_name, COUNT(*) as count
                            FROM validation_failures
                            WHERE validation_batch_id = '{batch_id}'
                            GROUP BY rule_name
                            ORDER BY count DESC
                            LIMIT 5
                        """).fetchdf()
                        
                        if not failures.empty:
                            console.print("\n[bold yellow]Top validation issues:[/]")
                            for _, row in failures.iterrows():
                                console.print(f"- {row['rule_name']}: {row['count']} failures")
                            
                            console.print(f"\nTo fix these issues, run:")
                            console.print(f"uv run data_validation_agent.py -d {self.database_path} -q \"auto_fix {results['parameters']['table_name']} from validation batch {batch_id} using standard_fixes\"")
                    except Exception as e:
                        logger.error(f"Error retrieving failure summary: {e}")
                
            elif results["parameters"]["action"] == "auto_fix":
                console.print(Panel(f"[bold green]Auto-Fix Complete![/]", title=AGENT_NAME))
                
                # Display auto-fix summary
                console.print(f"\n[bold]Auto-Fix Summary:[/]")
                console.print(f"Table: {results['parameters']['table_name']}")
                console.print(f"Records fixed: {results['results']['records_fixed']}")
                console.print(f"Rules applied: {results['results']['rules_applied']}")
                
                # Display fix ID for reference
                console.print(f"\nFix ID: {results['results']['fix_id']}")
                
                # Suggest validation after fixing
                console.print(f"\nTo validate the fixed data, run:")
                console.print(f"uv run data_validation_agent.py -d {self.database_path} -q \"validate {results['parameters']['table_name']} {results['parameters']['filter']}\"")
            
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
        uv run data_validation_agent.py -d ./financial_data.duckdb -q "validate market_data from last 7 days"
        uv run data_validation_agent.py -d ./financial_data.duckdb -q "validate economic_data where indicator = 'GDP'"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = DataValidationAgent(
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
def test_validation(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    table: str = typer.Option("market_data", "--table", "-t", help="Table to test validation on"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Run a quick validation test on a sample of data."""
    try:
        agent = DataValidationAgent(database_path=database, verbose=verbose)
        console.print(f"[bold]Running test validation on {table}...[/]")
        
        # Check if the table has data
        count = agent.conn.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1").fetchone()[0]
        
        if count == 0:
            console.print(f"[yellow]Warning: No data found in {table}[/]")
        else:
            # Run a quick validation on a small sample
            result = agent.process_query(f"validate {table} limit 100")
            agent.display_results(result)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Test failed:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()