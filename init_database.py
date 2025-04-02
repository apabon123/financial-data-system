#!/usr/bin/env python3
"""
Agent Name: Database Initialization Script
Purpose: Initialize DuckDB database with the required schema
Author: Claude
Date: 2025-04-02

Description:
    This script initializes a new DuckDB database with the schema defined in sql/init_schema.sql.
    It creates all required tables, views, and indexes for the financial data system.

Usage:
    uv run init_database.py -d ./path/to/database.duckdb [-v]
    
Examples:
    uv run init_database.py -d ./financial_data.duckdb
    uv run init_database.py -d ./financial_data.duckdb -v
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import duckdb
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress
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
logger = logging.getLogger("Database Initialization")

# Setup console
console = Console()

def initialize_database(database_path, verbose=False):
    """Initialize the database with the required schema.
    
    Args:
        database_path: Path to the DuckDB database to initialize
        verbose: Enable verbose logging
        
    Returns:
        True if initialization was successful, False otherwise
    """
    start_time = datetime.now()
    
    # Set logging level based on verbosity
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Check if the database directory exists
        db_dir = os.path.dirname(database_path)
        if db_dir and not os.path.exists(db_dir):
            logger.info(f"Creating directory {db_dir}")
            os.makedirs(db_dir)
        
        # Connect to the database (it will be created if it doesn't exist)
        logger.info(f"Connecting to database: {database_path}")
        conn = duckdb.connect(database_path)
        
        # Read the schema initialization script
        schema_path = os.path.join("sql", "init_schema.sql")
        logger.debug(f"Reading schema from {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_script = f.read()
        
        # Split script into individual statements
        statements = schema_script.split(';')
        
        # Execute each statement
        with Progress() as progress:
            task = progress.add_task("[cyan]Initializing database...", total=len(statements))
            
            for i, statement in enumerate(statements):
                if statement.strip():
                    if verbose:
                        logger.debug(f"Executing statement {i+1}/{len(statements)}")
                        logger.debug(statement.strip())
                    
                    try:
                        conn.execute(statement)
                    except Exception as e:
                        logger.error(f"Error executing statement: {e}")
                        logger.error(f"Statement: {statement.strip()}")
                        return False
                
                progress.update(task, advance=1)
        
        # Insert default metadata
        conn.execute("""
            INSERT INTO metadata (key, value, updated_at, description)
            VALUES 
                ('schema_version', '1.0.0', CURRENT_TIMESTAMP, 'Database schema version'),
                ('created_date', ?, CURRENT_TIMESTAMP, 'Database creation date'),
                ('system_version', '1.0.0', CURRENT_TIMESTAMP, 'Financial Data System version')
        """, [datetime.now().isoformat()])
        
        # Close the connection
        conn.close()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Database initialization completed successfully in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Initialize DuckDB database with required schema")
    parser.add_argument("-d", "--database", required=True, help="Path to DuckDB database file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    console.print(Panel.fit("Financial Data System - Database Initialization", border_style="cyan"))
    
    # Check if the database already exists
    if os.path.exists(args.database):
        console.print(f"[yellow]Warning: Database file {args.database} already exists.[/]")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower().strip() == 'y'
        
        if not overwrite:
            console.print("[yellow]Initialization cancelled.[/]")
            return
        
        console.print("[yellow]Overwriting existing database...[/]")
    
    # Initialize the database
    success = initialize_database(args.database, args.verbose)
    
    if success:
        console.print(Panel.fit(f"[bold green]Database initialized successfully![/]\nLocation: {args.database}", border_style="green"))
        console.print("\nNext steps:")
        console.print("1. Set up your API credentials in the .env file")
        console.print("2. Run a test query with one of the agents")
        console.print(f"   Example: uv run tradestation_market_data_agent.py -d {args.database} -q \"fetch daily data for SPY from 2023-01-01 to 2023-01-31\"")
    else:
        console.print(Panel.fit("[bold red]Database initialization failed![/]", border_style="red"))
        sys.exit(1)

if __name__ == "__main__":
    main()