#!/usr/bin/env python3
"""
Agent Name: Web Scraping Agent
Purpose: Collect financial data from websites not available via APIs
Author: Claude
Date: 2025-04-02

Description:
    This agent scrapes financial data from websites where APIs are not available.
    It handles scraping patterns, validation, and structuring data for storage.

Usage:
    uv run web_scraping_agent.py -d ./path/to/database.duckdb -q "natural language query"
    uv run web_scraping_agent.py -d ./path/to/database.duckdb -f ./path/to/input.json
    
Examples:
    uv run web_scraping_agent.py -d ./financial_data.duckdb -q "scrape earnings calendar from https://finance.example.com/earnings"
    uv run web_scraping_agent.py -d ./financial_data.duckdb -q "scrape dividend data for AAPL, MSFT"
"""

import os
import sys
import json
import logging
import argparse
import re
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

import typer
import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup
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
logger = logging.getLogger("Web Scraping Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Web Scraping Agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Collect financial data from websites not available via APIs"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class WebScrapingAgent:
    """Agent for scraping financial data from websites.
    
    This agent handles scraping financial data from websites, validating
    the scraped data, and structuring it for storage in DuckDB.
    
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
        
        # Scraping configuration
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.request_delay = 1.0  # Seconds between requests
        
        # Define supported scraping patterns
        self.scraping_patterns = {
            "earnings_calendar": {
                "description": "Earnings announcement dates and estimates",
                "url_templates": [
                    "https://finance.yahoo.com/calendar/earnings",
                    "https://www.marketwatch.com/tools/earnings-calendar"
                ],
                "extraction_func": self._extract_earnings_calendar,
                "schema": {
                    "symbol": "VARCHAR",
                    "company_name": "VARCHAR",
                    "report_date": "DATE",
                    "time": "VARCHAR",
                    "estimated_eps": "DOUBLE",
                    "actual_eps": "DOUBLE",
                    "surprise_percent": "DOUBLE"
                }
            },
            "dividend_data": {
                "description": "Dividend announcements and ex-dividend dates",
                "url_templates": [
                    "https://finance.yahoo.com/calendar/dividend",
                    "https://www.marketwatch.com/tools/dividend-calendar"
                ],
                "extraction_func": self._extract_dividend_data,
                "schema": {
                    "symbol": "VARCHAR",
                    "company_name": "VARCHAR",
                    "ex_date": "DATE",
                    "payment_date": "DATE",
                    "amount": "DOUBLE",
                    "yield": "DOUBLE",
                    "frequency": "VARCHAR"
                }
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
        
        This method extracts structured parameters related to web scraping
        from a natural language query.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "scrape earnings calendar from https://finance.example.com/earnings"
            Output: {
                "action": "scrape",
                "data_type": "earnings_calendar",
                "url": "https://finance.example.com/earnings",
                "symbols": []
            }
        """
        logger.debug(f"Parsing query: {query}")
        
        # Default parameters
        params = {
            "action": "scrape",
            "data_type": None,
            "url": None,
            "symbols": [],
            "start_date": None,
            "end_date": None
        }
        
        # Extract data type
        if "earnings calendar" in query.lower():
            params["data_type"] = "earnings_calendar"
        elif "dividend" in query.lower():
            params["data_type"] = "dividend_data"
        
        # Extract URL if specified
        url_match = re.search(r'from\s+(https?://\S+)', query)
        if url_match:
            params["url"] = url_match.group(1)
        elif params["data_type"] and self.scraping_patterns.get(params["data_type"]):
            # Use the default URL for this data type
            params["url"] = self.scraping_patterns[params["data_type"]]["url_templates"][0]
        
        # Extract symbols if specified
        symbols_match = re.search(r'for\s+([A-Za-z0-9,\s]+)(?:\s+from|\s+between|\s+in|\s+during|\s*$)', query)
        if symbols_match:
            symbols_str = symbols_match.group(1)
            params["symbols"] = [s.strip() for s in symbols_str.split(',')]
        
        # Extract date range
        date_range_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', query)
        if date_range_match:
            start_date_str = date_range_match.group(1)
            end_date_str = date_range_match.group(2)
            params["start_date"] = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            params["end_date"] = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # Extract time-based keywords
        if "next week" in query.lower():
            today = datetime.now().date()
            params["start_date"] = today + timedelta(days=(7 - today.weekday()))  # Next Monday
            params["end_date"] = params["start_date"] + timedelta(days=6)  # Sunday
        elif "this week" in query.lower():
            today = datetime.now().date()
            params["start_date"] = today - timedelta(days=today.weekday())  # This Monday
            params["end_date"] = params["start_date"] + timedelta(days=6)  # Sunday
        
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
        the query, performs web scraping, and returns structured data.
        
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
            "action": params.get("action", "scrape"),
            "success": False,
            "errors": [],
            "warnings": [],
            "scraped_data": None,
            "items_scraped": 0,
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
                    # Loop 2: Perform web scraping
                    self._loop_scrape_data(params, result)
                elif i == 2:
                    # Loop 3: Validate and structure data
                    self._loop_validate_data(params, result)
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
        if params.get("action") != "scrape":
            result["errors"].append(f"Invalid action: {params.get('action')}. Only 'scrape' is supported.")
            return
        
        # Validate data type
        if not params.get("data_type"):
            result["errors"].append("No data type specified")
            return
        
        # Check if data type is supported
        if params["data_type"] not in self.scraping_patterns:
            result["errors"].append(f"Unsupported data type: {params['data_type']}. Supported types: {', '.join(self.scraping_patterns.keys())}")
            return
        
        # Check if URL is valid
        if params.get("url"):
            if not params["url"].startswith("http"):
                result["errors"].append(f"Invalid URL: {params['url']}. URL must start with http:// or https://")
                return
        else:
            # Use the default URL for this data type
            params["url"] = self.scraping_patterns[params["data_type"]]["url_templates"][0]
            result["warnings"].append(f"No URL specified. Using default: {params['url']}")
        
        # Validate date range if specified
        if params.get("start_date") and params.get("end_date"):
            if params["start_date"] > params["end_date"]:
                result["errors"].append(f"Invalid date range: start date {params['start_date']} is after end date {params['end_date']}")
                return
        
        logger.debug("Parameters validated successfully")
    
    def _loop_scrape_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Perform web scraping.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Performing web scraping")
        
        # Skip if there were errors in the previous loop
        if result.get("errors"):
            return
        
        data_type = params["data_type"]
        url = params["url"]
        
        # Get the extraction function for this data type
        extraction_func = self.scraping_patterns[data_type]["extraction_func"]
        
        try:
            # Fetch the webpage
            headers = {"User-Agent": self.user_agent}
            logger.debug(f"Fetching URL: {url}")
            
            response = requests.get(url, headers=headers, timeout=10)
            
            # Check if the request was successful
            if response.status_code != 200:
                result["errors"].append(f"Failed to fetch URL: {url}. Status code: {response.status_code}")
                return
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the data using the appropriate function
            data = extraction_func(soup, params)
            
            # Store the scraped data
            result["scraped_data"] = data
            result["items_scraped"] = len(data)
            
            logger.debug(f"Scraped {len(data)} items from {url}")
            
        except requests.exceptions.RequestException as e:
            result["errors"].append(f"Error fetching URL: {url}. {str(e)}")
            return
        except Exception as e:
            result["errors"].append(f"Error scraping data: {str(e)}")
            return
    
    def _loop_validate_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Validate and structure scraped data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Validating and structuring scraped data")
        
        # Skip if there were errors in the previous loops
        if result.get("errors"):
            return
        
        # Skip if no data was scraped
        if not result.get("scraped_data"):
            result["warnings"].append("No data was scraped")
            return
        
        data_type = params["data_type"]
        data = result["scraped_data"]
        
        # Get the schema for this data type
        schema = self.scraping_patterns[data_type]["schema"]
        
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame(data)
            
            # Check if the required columns are present
            missing_columns = [col for col in schema.keys() if col not in df.columns]
            if missing_columns:
                result["warnings"].append(f"Missing columns in scraped data: {', '.join(missing_columns)}")
                
                # Add missing columns with None values
                for col in missing_columns:
                    df[col] = None
            
            # Convert data types according to the schema
            for col, dtype in schema.items():
                if col in df.columns:
                    try:
                        if dtype == "DATE":
                            df[col] = pd.to_datetime(df[col]).dt.date
                        elif dtype == "DOUBLE":
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif dtype == "INTEGER":
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    except Exception as e:
                        result["warnings"].append(f"Error converting column {col} to {dtype}: {str(e)}")
            
            # Drop rows with all None values
            df = df.dropna(how='all')
            
            # Filter by symbols if specified
            if params.get("symbols"):
                df = df[df['symbol'].isin(params["symbols"])]
                if df.empty:
                    result["warnings"].append(f"No data found for symbols: {', '.join(params['symbols'])}")
                
            # Filter by date range if specified and data contains dates
            date_columns = [col for col, dtype in schema.items() if dtype == "DATE" and col in df.columns]
            if params.get("start_date") and params.get("end_date") and date_columns:
                # Use the first date column for filtering
                date_col = date_columns[0]
                mask = (df[date_col] >= params["start_date"]) & (df[date_col] <= params["end_date"])
                df = df[mask]
                if df.empty:
                    result["warnings"].append(f"No data found for date range: {params['start_date']} to {params['end_date']}")
            
            # Update the scraped data with the validated DataFrame
            result["scraped_data"] = df
            result["items_scraped"] = len(df)
            
            logger.debug(f"Validated {len(df)} items")
            
        except Exception as e:
            result["errors"].append(f"Error validating data: {str(e)}")
            return
    
    def _extract_earnings_calendar(self, soup: BeautifulSoup, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract earnings calendar data from a webpage.
        
        Args:
            soup: BeautifulSoup object of the webpage
            params: Query parameters
            
        Returns:
            List of dictionaries containing earnings data
        """
        # Note: This is a simple example extraction. In a real-world scenario,
        # you would need to adapt this to the specific website structure.
        
        # For demonstration purposes, we'll create some sample data
        # In a real implementation, this would parse the HTML and extract actual data
        
        # Sample data for testing
        today = datetime.now().date()
        
        sample_data = [
            {
                "symbol": "AAPL",
                "company_name": "Apple Inc.",
                "report_date": today + timedelta(days=5),
                "time": "after_market",
                "estimated_eps": 1.43,
                "actual_eps": None,
                "surprise_percent": None
            },
            {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "report_date": today + timedelta(days=7),
                "time": "after_market",
                "estimated_eps": 2.32,
                "actual_eps": None,
                "surprise_percent": None
            },
            {
                "symbol": "GOOGL",
                "company_name": "Alphabet Inc.",
                "report_date": today + timedelta(days=12),
                "time": "after_market",
                "estimated_eps": 1.98,
                "actual_eps": None,
                "surprise_percent": None
            }
        ]
        
        # In a real implementation, you would do something like:
        # table = soup.find('table', {'class': 'earnings-table'})
        # rows = table.find_all('tr')
        # for row in rows:
        #     cells = row.find_all('td')
        #     data.append({
        #         "symbol": cells[0].text.strip(),
        #         "company_name": cells[1].text.strip(),
        #         ...
        #     })
        
        logger.debug(f"Extracted {len(sample_data)} earnings calendar entries")
        return sample_data
    
    def _extract_dividend_data(self, soup: BeautifulSoup, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract dividend data from a webpage.
        
        Args:
            soup: BeautifulSoup object of the webpage
            params: Query parameters
            
        Returns:
            List of dictionaries containing dividend data
        """
        # Note: This is a simple example extraction. In a real-world scenario,
        # you would need to adapt this to the specific website structure.
        
        # For demonstration purposes, we'll create some sample data
        # In a real implementation, this would parse the HTML and extract actual data
        
        # Sample data for testing
        today = datetime.now().date()
        
        sample_data = [
            {
                "symbol": "AAPL",
                "company_name": "Apple Inc.",
                "ex_date": today + timedelta(days=15),
                "payment_date": today + timedelta(days=30),
                "amount": 0.24,
                "yield": 0.52,
                "frequency": "quarterly"
            },
            {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "ex_date": today + timedelta(days=10),
                "payment_date": today + timedelta(days=25),
                "amount": 0.68,
                "yield": 0.85,
                "frequency": "quarterly"
            },
            {
                "symbol": "JNJ",
                "company_name": "Johnson & Johnson",
                "ex_date": today + timedelta(days=5),
                "payment_date": today + timedelta(days=20),
                "amount": 1.19,
                "yield": 2.65,
                "frequency": "quarterly"
            }
        ]
        
        # In a real implementation, you would do something like:
        # table = soup.find('table', {'class': 'dividend-table'})
        # rows = table.find_all('tr')
        # for row in rows:
        #     cells = row.find_all('td')
        #     data.append({
        #         "symbol": cells[0].text.strip(),
        #         "company_name": cells[1].text.strip(),
        #         ...
        #     })
        
        logger.debug(f"Extracted {len(sample_data)} dividend data entries")
        return sample_data
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if results["success"]:
            console.print(Panel(f"[bold green]Success![/]", title=AGENT_NAME))
            
            # Display scraping summary
            console.print(f"[cyan]Data Type:[/] {results['parameters']['data_type']}")
            console.print(f"[cyan]URL:[/] {results['parameters']['url']}")
            
            if results['parameters'].get('symbols'):
                console.print(f"[cyan]Symbols:[/] {', '.join(results['parameters']['symbols'])}")
            
            if results['parameters'].get('start_date') and results['parameters'].get('end_date'):
                console.print(f"[cyan]Date Range:[/] {results['parameters']['start_date']} to {results['parameters']['end_date']}")
            
            # Display results summary
            console.print(f"\n[bold]Results Summary:[/]")
            console.print(f"Items scraped: {results['results']['items_scraped']}")
            
            # Display data sample
            if results['results']['scraped_data'] is not None and not isinstance(results['results']['scraped_data'], pd.DataFrame):
                data = pd.DataFrame(results['results']['scraped_data'])
            else:
                data = results['results']['scraped_data']
            
            if data is not None and not data.empty:
                # Limit displayed rows for large results
                display_rows = min(len(data), 5)
                
                console.print("\n[bold]Data Sample:[/]")
                
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
        uv run web_scraping_agent.py -d ./financial_data.duckdb -q "scrape earnings calendar from https://finance.example.com/earnings"
        uv run web_scraping_agent.py -d ./financial_data.duckdb -q "scrape dividend data for AAPL, MSFT"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = WebScrapingAgent(
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
def list_patterns(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
):
    """List available scraping patterns."""
    try:
        agent = WebScrapingAgent(database_path=database)
        
        console.print(Panel("[bold]Available Scraping Patterns[/]", border_style="cyan"))
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Pattern")
        table.add_column("Description")
        table.add_column("Example URL")
        table.add_column("Schema")
        
        for name, pattern in agent.scraping_patterns.items():
            table.add_row(
                name,
                pattern["description"],
                pattern["url_templates"][0],
                ", ".join(pattern["schema"].keys())
            )
        
        console.print(table)
        
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()