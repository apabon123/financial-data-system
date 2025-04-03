#!/usr/bin/env python3
"""
Agent Name: Schema Management Agent
Purpose: Manage database schema evolution
Author: Claude
Date: 2025-04-02

Description:
    This agent manages the database schema, including creation, migrations,
    and updates. It ensures schema integrity and handles version tracking.

Usage:
    uv run schema_management_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run schema_management_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run schema_management_agent.py -d ./financial_data.duckdb -q "create table symbols with fields symbol, name, sector"
    uv run schema_management_agent.py -d ./financial_data.duckdb -q "initialize database with schema from sql/init_schema.sql"
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
logger = logging.getLogger("Schema Management Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Schema Management Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Manage database schema evolution"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class SchemaManagementAgent:
    """Agent for managing database schema evolution.
    
    This agent handles database schema creation, migrations, and updates,
    ensuring schema integrity and tracking versions.
    
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
        
        This method extracts structured parameters related to schema management
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "create table symbols with fields symbol, name, sector"
            Output: {
                "action": "create_table",
                "table_name": "symbols",
                "fields": [{"name": "symbol", "type": "VARCHAR"}, 
                          {"name": "name", "type": "VARCHAR"}, 
                          {"name": "sector", "type": "VARCHAR"}]
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": None,
            "table_name": None,
            "script_path": None,
            "fields": [],
            "version": None,
            "backup": True
        }
        
        # Extract action
        if "initialize" in query.lower() or "init" in query.lower():
            params["action"] = "initialize"
        elif "create table" in query.lower():
            params["action"] = "create_table"
        elif "add column" in query.lower() or "add field" in query.lower():
            params["action"] = "add_column"
        elif "create index" in query.lower():
            params["action"] = "create_index"
        elif "create view" in query.lower():
            params["action"] = "create_view"
        elif "backup" in query.lower():
            params["action"] = "backup"
        elif "restore" in query.lower():
            params["action"] = "restore"
        elif "migrate" in query.lower() or "upgrade" in query.lower():
            params["action"] = "migrate"
        
        # Extract script path for initialization
        if params["action"] == "initialize":
            script_match = re.search(r'from\s+(.+?\.sql)', query)
            if script_match:
                params["script_path"] = script_match.group(1)
        
        # Extract table name
        table_match = re.search(r'table\s+(\w+)', query)
        if table_match:
            params["table_name"] = table_match.group(1)
        
        # Extract fields for create_table or add_column
        if params["action"] in ["create_table", "add_column"]:
            fields_match = re.search(r'with\s+fields\s+([\w\s,]+)(?:\s|$)', query)
            if fields_match:
                fields_str = fields_match.group(1)
                field_list = [f.strip() for f in fields_str.split(',')]
                
                for field in field_list:
                    # Check if type is specified
                    if ' as ' in field.lower():
                        name, type_ = field.split(' as ', 1)
                        params["fields"].append({
                            "name": name.strip(),
                            "type": type_.strip().upper()
                        })
                    else:
                        # Default to VARCHAR
                        params["fields"].append({
                            "name": field.strip(),
                            "type": "VARCHAR"
                        })
        
        # Extract schema version for migrations
        if params["action"] == "migrate":
            version_match = re.search(r'to\s+version\s+([\d\.]+)', query)
            if version_match:
                params["version"] = version_match.group(1)
        
        # Extract backup option
        if "no backup" in query.lower() or "without backup" in query.lower():
            params["backup"] = False
        
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
        the query, performs schema operations, and returns results.
        
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
            "action": params.get("action"),
            "success": False,
            "errors": [],
            "warnings": [],
            "operations_performed": [],
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
                    # Loop 2: Prepare schema operation
                    self._loop_prepare_schema_operation(params, result)
                elif i == 2:
                    # Loop 3: Execute schema operation
                    self._loop_execute_schema_operation(params, result)
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
        valid_actions = ["initialize", "create_table", "add_column", "create_index", "create_view", "backup", "restore", "migrate"]
        if params.get("action") not in valid_actions:
            result["errors"].append(f"Invalid action: {params.get('action')}. Must be one of {valid_actions}")
            return
        
        # Action-specific validation
        if params["action"] == "initialize":
            # Validate script path
            if not params.get("script_path"):
                # Use default path
                params["script_path"] = "sql/init_schema.sql"
                result["warnings"].append(f"No script path specified. Using default: {params['script_path']}")
            
            # Check if script file exists
            if not os.path.exists(params["script_path"]):
                result["errors"].append(f"Script file not found: {params['script_path']}")
                return
                
        elif params["action"] in ["create_table", "add_column", "create_index", "create_view"]:
            # Validate table name
            if not params.get("table_name"):
                result["errors"].append(f"No table name specified for {params['action']}")
                return
            
            # For these actions, validate additional parameters
            if params["action"] == "create_table" and not params.get("fields"):
                result["errors"].append("No fields specified for create_table")
                return
                
            if params["action"] == "add_column" and not params.get("fields"):
                result["errors"].append("No fields specified for add_column")
                return
                
        elif params["action"] == "migrate":
            # Validate version
            if not params.get("version"):
                result["errors"].append("No target version specified for migrate")
                return
        
        logger.debug("Parameters validated successfully")
    
    def _loop_prepare_schema_operation(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Prepare the schema operation.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Preparing schema operation")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        # Prepare the operation based on the action
        if params["action"] == "initialize":
            # Read the initialization script
            try:
                with open(params["script_path"], 'r') as f:
                    script_content = f.read()
                
                # Store the script in the result
                result["init_script"] = script_content
                
                # Split the script into individual statements
                statements = self._split_sql_statements(script_content)
                result["init_statements"] = statements
                
                logger.debug(f"Prepared {len(statements)} SQL statements for initialization")
                
            except Exception as e:
                result["errors"].append(f"Error reading initialization script: {str(e)}")
                return
                
        elif params["action"] == "create_table":
            # Generate CREATE TABLE statement
            table_name = params["table_name"]
            fields = params["fields"]
            
            # Generate field definitions
            field_defs = []
            for field in fields:
                field_defs.append(f"{field['name']} {field['type']}")
            
            # Generate SQL statement
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    " + ",\n    ".join(field_defs) + "\n)"
            
            # Store the statement in the result
            result["sql_statement"] = sql
            
            logger.debug(f"Prepared CREATE TABLE statement: {sql}")
            
        elif params["action"] == "add_column":
            # Generate ALTER TABLE statements
            table_name = params["table_name"]
            fields = params["fields"]
            
            # Check if the table exists
            try:
                table_exists = self.conn.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()[0]
                if not table_exists:
                    result["errors"].append(f"Table '{table_name}' does not exist")
                    return
                
                # Get existing columns
                existing_columns = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()["column_name"].tolist()
                
                # Generate ALTER TABLE statements
                alter_statements = []
                for field in fields:
                    if field["name"] in existing_columns:
                        result["warnings"].append(f"Column '{field['name']}' already exists in table '{table_name}'")
                    else:
                        sql = f"ALTER TABLE {table_name} ADD COLUMN {field['name']} {field['type']}"
                        alter_statements.append(sql)
                
                # Store the statements in the result
                result["sql_statements"] = alter_statements
                
                logger.debug(f"Prepared {len(alter_statements)} ALTER TABLE statements")
                
            except Exception as e:
                result["errors"].append(f"Error preparing ALTER TABLE statements: {str(e)}")
                return
                
        elif params["action"] == "create_index":
            # We need additional parameters for this action
            result["errors"].append("create_index requires additional parameters not provided in the query")
            return
            
        elif params["action"] == "create_view":
            # We need additional parameters for this action
            result["errors"].append("create_view requires additional parameters not provided in the query")
            return
            
        elif params["action"] == "backup":
            # Generate backup file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{os.path.splitext(self.database_path)[0]}_{timestamp}.bak"
            
            # Store the backup path in the result
            result["backup_path"] = backup_path
            
            logger.debug(f"Prepared backup operation to: {backup_path}")
            
        elif params["action"] == "restore":
            # We need additional parameters for this action
            result["errors"].append("restore requires a backup file path not provided in the query")
            return
            
        elif params["action"] == "migrate":
            # Prepare migration operations
            # For now, this is just a placeholder
            result["warnings"].append("migrate action is not fully implemented yet")
            return
    
    def _loop_execute_schema_operation(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Execute the schema operation.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Executing schema operation")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Execute the operation based on the action
        if params["action"] == "initialize":
            # Create a backup first if requested
            if params.get("backup", True) and os.path.exists(self.database_path) and os.path.getsize(self.database_path) > 0:
                self._create_backup(result)
            
            # Execute the initialization statements
            statements = result["init_statements"]
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Initializing database schema...", 
                    total=len(statements)
                )
                
                for i, statement in enumerate(statements):
                    logger.debug(f"Executing statement {i+1}/{len(statements)}")
                    
                    # Skip empty statements
                    if not statement.strip():
                        progress.update(task, advance=1)
                        continue
                    
                    try:
                        self.conn.execute(statement)
                        result["operations_performed"].append(f"Executed: {statement[:50]}...")
                    except Exception as e:
                        result["errors"].append(f"Error executing statement {i+1}: {str(e)}")
                        result["errors"].append(f"Statement: {statement}")
                        return
                    
                    progress.update(task, advance=1)
            
            # Store schema metadata
            try:
                # Check if metadata table exists
                metadata_exists = self.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()[0]
                
                if metadata_exists:
                    # Update schema_version in metadata
                    self.conn.execute("""
                        INSERT OR REPLACE INTO metadata (key, value, updated_at, description)
                        VALUES ('schema_version', '1.0.0', CURRENT_TIMESTAMP, 'Database schema version')
                    """)
                    
                    # Log initialization
                    self.conn.execute("""
                        INSERT OR REPLACE INTO metadata (key, value, updated_at, description)
                        VALUES ('schema_initialized', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'When the schema was initialized')
                    """)
                
                logger.debug("Updated schema metadata")
                
            except Exception as e:
                result["warnings"].append(f"Error updating schema metadata: {str(e)}")
            
            logger.debug("Schema initialization completed successfully")
            
        elif params["action"] == "create_table":
            # Create a backup first if requested
            if params.get("backup", True):
                self._create_backup(result)
            
            # Execute the CREATE TABLE statement
            sql = result["sql_statement"]
            
            try:
                self.conn.execute(sql)
                result["operations_performed"].append(f"Created table: {params['table_name']}")
                logger.debug(f"Created table: {params['table_name']}")
                
            except Exception as e:
                result["errors"].append(f"Error creating table: {str(e)}")
                return
            
        elif params["action"] == "add_column":
            # Create a backup first if requested
            if params.get("backup", True):
                self._create_backup(result)
            
            # Execute the ALTER TABLE statements
            statements = result["sql_statements"]
            
            for i, sql in enumerate(statements):
                try:
                    self.conn.execute(sql)
                    field_name = params["fields"][i]["name"]
                    result["operations_performed"].append(f"Added column: {field_name} to {params['table_name']}")
                    logger.debug(f"Added column: {field_name} to {params['table_name']}")
                    
                except Exception as e:
                    result["errors"].append(f"Error adding column: {str(e)}")
                    result["errors"].append(f"Statement: {sql}")
                    return
            
        elif params["action"] == "backup":
            # Create a backup
            self._create_backup(result)
            
        elif params["action"] == "restore":
            # Not implemented yet
            result["errors"].append("restore action is not implemented yet")
            return
            
        elif params["action"] == "migrate":
            # Not implemented yet
            result["errors"].append("migrate action is not implemented yet")
            return
    
    def _create_backup(self, result: Dict[str, Any]) -> None:
        """Create a backup of the database.
        
        Args:
            result: Result dictionary to update
        """
        try:
            # Generate backup path if not already set
            if "backup_path" not in result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{os.path.splitext(self.database_path)[0]}_{timestamp}.bak"
                result["backup_path"] = backup_path
            else:
                backup_path = result["backup_path"]
            
            # Close the current connection
            self.conn.close()
            
            # Copy the database file
            import shutil
            shutil.copy2(self.database_path, backup_path)
            
            # Reconnect to the database
            self._connect_database()
            
            result["operations_performed"].append(f"Created backup: {backup_path}")
            logger.debug(f"Created backup: {backup_path}")
            
        except Exception as e:
            result["errors"].append(f"Error creating backup: {str(e)}")
    
    def _split_sql_statements(self, script: str) -> List[str]:
        """Split a SQL script into individual statements.
        
        Args:
            script: SQL script to split
            
        Returns:
            List of SQL statements
        """
        # Split by semicolons
        statements = []
        current_statement = []
        
        # Simple splitting by semicolons
        # This doesn't handle all edge cases (like semicolons in quotes)
        for line in script.split('\n'):
            # Skip comments
            if line.strip().startswith('--'):
                continue
                
            # Add the line to the current statement
            current_statement.append(line)
            
            # Check if the line ends with a semicolon
            if line.strip().endswith(';'):
                # Join the lines and remove the trailing semicolon
                statement = '\n'.join(current_statement)
                statements.append(statement)
                current_statement = []
        
        # Handle the last statement if it doesn't end with a semicolon
        if current_statement:
            statement = '\n'.join(current_statement)
            statements.append(statement)
        
        return statements
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display action
            action = results["parameters"]["action"]
            console.print(f"[cyan]Action:[/] {action}")
            
            # Display table name if applicable
            if results["parameters"].get("table_name"):
                console.print(f"[cyan]Table:[/] {results['parameters']['table_name']}")
            
            # Display operations performed
            if results["results"]["operations_performed"]:
                console.print("\n[bold]Operations Performed:[/]")
                for op in results["results"]["operations_performed"]:
                    console.print(f"- {op}")
            
            # Display backup path if created
            if "backup_path" in results["results"]:
                console.print(f"\n[bold]Backup Created:[/] {results['results']['backup_path']}")
            
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
        uv run schema_management_agent.py -d ./financial_data.duckdb -q "create table symbols with fields symbol, name, sector"
        uv run schema_management_agent.py -d ./financial_data.duckdb -q "initialize database with schema from sql/init_schema.sql"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = SchemaManagementAgent(
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
def init_schema(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    script: str = typer.Option("sql/init_schema.sql", "--script", "-s", help="Path to schema initialization script"),
    no_backup: bool = typer.Option(False, "--no-backup", help="Don't create a backup before initialization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Initialize the database schema from a SQL script."""
    try:
        agent = SchemaManagementAgent(database_path=database, verbose=verbose)
        
        # Process the initialization
        result = agent.process_query(f"initialize database from {script}" + (" without backup" if no_backup else ""))
        
        # Display results
        agent.display_results(result)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    app()