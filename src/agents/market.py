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
            "interval_unit": "daily",
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
            params["interval_unit"] = "daily"
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
            refresh_token = os.getenv('REFRESH_TOKEN')
            
            if not all([client_id, client_secret, refresh_token]):
                logger.error("Missing TradeStation API credentials in environment variables")
                logger.error("Please set CLIENT_ID, CLIENT_SECRET, and REFRESH_TOKEN")
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
                logger.debug(f"Fetching data for symbol: {symbol}")
                start_date_param = params["start_date"].strftime("%m-%d-%Y") if params["start_date"] else None
                end_date_param = params["end_date"].strftime("%m-%d-%Y") if params["end_date"] else datetime.now().strftime("%m-%d-%Y")
                
                # Construct endpoint and query parameters
                endpoint = f"{self.base_url}/marketdata/barcharts/{symbol}"
                query_params = {
                    "interval": params["interval_value"],
                    "unit": params["interval_unit"].capitalize(),
                    "startDate": start_date_param,
                    "endDate": end_date_param,
                    "sessiontemplate": "USEQPreAndPostMarket" # Use 24-hour session
                }
                
                # Remove None values from query_params
                query_params = {k: v for k, v in query_params.items() if v is not None}
                
                # API request
                try:
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    response = requests.get(endpoint, headers=headers, params=query_params)
                    response.raise_for_status() # Raise HTTPError for bad responses
                    
                    data = response.json()
                    bars = data.get("Bars", [])
                    
                    if bars:
                        df = pd.DataFrame(bars)
                        df = self._normalize_data(df, symbol, params)
                        all_data.append(df)
                        logger.debug(f"Fetched {len(df)} records for {symbol}")
                    else:
                        logger.warning(f"No data returned for symbol: {symbol}")
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"API request failed for {symbol}: {e}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                
                progress.update(task, advance=1)
                
        if not all_data:
            logger.warning("No data fetched from API")
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)
    
    def _normalize_data(self, df: pd.DataFrame, symbol: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Normalize fetched data to match database schema."""
        logger.debug(f"Normalizing data for symbol: {symbol}")
        
        # Ensure TimeStamp column exists
        if 'TimeStamp' not in df.columns:
            logger.error(f"'TimeStamp' column not found in data for {symbol}")
            return pd.DataFrame() # Return empty DataFrame if TimeStamp is missing

        # Convert TimeStamp to datetime and make timezone-naive
        try:
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            if df['TimeStamp'].dt.tz is not None:
                df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(None)
        except Exception as e:
            logger.error(f"Error converting 'TimeStamp' to datetime for {symbol}: {e}")
            return pd.DataFrame() # Return empty if conversion fails
            
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
        
        # Ensure all expected columns exist, add if missing
        for col in columns:
            if col not in df.columns:
                df[col] = None
                
        return df[columns]
    
    def save_market_data(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Save market data to DuckDB, handling conflicts by updating existing rows.
        
        Args:
            df: DataFrame containing market data to save
            
        Returns:
            Tuple containing (total_records_processed, new_records_inserted)
        """
        if df.empty:
            logger.warning("No data to save to database")
            return 0, 0
            
        try:
            table_name = "market_data"
            
            # Define primary key for conflict resolution
            primary_key = ["timestamp", "symbol", "interval_value", "interval_unit"]
            
            # Create a temporary table for staging data
            temp_table_name = f"temp_{table_name}_{int(time.time())}"
            self.conn.register(temp_table_name, df)
            
            # Construct columns for INSERT and UPDATE
            insert_columns = ", ".join([f'"{col}"' for col in df.columns])
            update_columns = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in df.columns if col not in primary_key])
            
            # Construct UPSERT query
            # Note: DuckDB syntax for UPSERT using ON CONFLICT DO UPDATE
            query = f"""
            INSERT INTO {table_name} ({insert_columns})
            SELECT {insert_columns} FROM {temp_table_name}
            ON CONFLICT (timestamp, symbol, interval_value, interval_unit) DO UPDATE SET {update_columns}
            """
            
            self.conn.execute(query)
            self.conn.commit()
            
            # Get the number of rows affected (this is an estimate for DuckDB)
            # DuckDB's execute does not directly return row count for INSERT/UPDATE
            # We can query the table or assume all rows in df were processed.
            total_records = len(df)
            # For simplicity, assume all processed records could be new if we don't have a better way to count upserts.
            # This may not be accurate for actual *new* records vs *updated* records.
            new_records = total_records 
            
            logger.info(f"Saved {total_records} records to {table_name} (upserted)")
            return total_records, new_records
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            self.conn.rollback()
            return 0, 0
        finally:
            # Clean up temporary table
            try:
                if 'temp_table_name' in locals():
                    self.conn.unregister(temp_table_name)
            except Exception as e_cleanup:
                logger.warning(f"Error cleaning up temp table {temp_table_name}: {e_cleanup}")

    def _reason(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """First compute loop: Reason about the query and prepare parameters."""
        logger.debug("Reasoning about the query")
        # In a real agent, this would involve more complex reasoning
        # For now, just return the parsed parameters directly
        return params

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
    
    def run(self, query: Optional[str] = None, input_file: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent to retrieve and store market data.
        
        This method orchestrates the agent's workflow:
        1. Parse query or load from file
        2. Reason about the query (Loop 1)
        3. Fetch data from API (Loop 2)
        4. Save data to database (Loop 3)
        
        Args:
            query: Natural language query
            input_file: Path to JSON input file
            
        Returns:
            Dictionary containing the result of the operation
        """
        logger.info("TradeStation Market Data Agent started")
        
        params: Dict[str, Any] = {}
        result: Dict[str, Any] = {
            "status": "success",
            "message": "Market data processing completed",
            "data_fetched": 0,
            "data_saved": 0,
            "symbols_processed": 0,
            "errors": []
        }
        
        if input_file:
            try:
                with open(input_file, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded parameters from file: {input_file}")
            except Exception as e:
                logger.error(f"Error loading input file: {e}")
                result["status"] = "error"
                result["message"] = f"Error loading input file: {e}"
                return result
        elif query:
            params = self._parse_query(query)
            logger.info(f"Parsed query: {query}")
        else:
            logger.error("No query or input file provided")
            result["status"] = "error"
            result["message"] = "No query or input file provided"
            return result
            
        # Start compute loops
        try:
            for i in range(self.compute_loops):
                logger.info(f"--- Starting Compute Loop {i + 1} ---")
                
                if i == 0:
                    params = self._reason(params)
                    # Update result with parsed parameters if needed (e.g., for logging/auditing)
                    result["parsed_params"] = params 
                elif i == 1:
                    self._loop_fetch_data(params, result)
                elif i == 2:
                    self._loop_save_data(params, result)
                
                logger.info(f"--- Finished Compute Loop {i + 1} ---")
                
                # Check for errors and stop if necessary
                if result["errors"]:
                    logger.warning(f"Errors encountered in loop {i + 1}: {result['errors']}")
                    # Depending on error severity, we might decide to stop early
                    # For now, continue all loops but log errors.
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            result["status"] = "error"
            result["message"] = str(e)
            result["errors"].append(str(e))
        
        # Clean up database connection
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")
            
        logger.info("TradeStation Market Data Agent finished")
        
        if result["errors"] and result["status"] == "success":
            # If there were non-fatal errors, reflect that in the final status
            result["status"] = "completed_with_warnings"
            result["message"] = f"Processing completed with warnings: {'; '.join(result['errors'])}"
            
        return result

@app.command()
def run_agent(
    query: Optional[str] = typer.Option(None, "-q", "--query", help="Natural language query for the agent"),
    input_file: Optional[str] = typer.Option(None, "-f", "--input-file", help="Path to JSON input file"),
    database_path: str = typer.Option("data/financial_data.duckdb", "-d", "--database", help="Path to DuckDB database file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose output"),
    compute_loops: int = typer.Option(3, "-c", "--compute-loops", help="Number of reasoning iterations to perform")
):
    """Run the TradeStation Market Data Agent."""
    
    # Display agent information
    console.print(Panel(
        f"[bold cyan]{AGENT_NAME} v{AGENT_VERSION}[/bold cyan]\n{AGENT_DESCRIPTION}", 
        title="Agent Information", 
        expand=False
    ))
    
    # Initialize and run the agent
    agent = TradeStationMarketDataAgent(database_path, verbose, compute_loops)
    result = agent.run(query, input_file)
    
    # Display results
    console.print("\n[bold green]Agent Run Summary:[/bold green]")
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("Status:", result.get("status", "unknown"))
    summary_table.add_row("Message:", result.get("message", ""))
    summary_table.add_row("Symbols Processed:", str(result.get("symbols_processed", 0)))
    summary_table.add_row("Data Fetched:", str(result.get("data_fetched", 0)))
    summary_table.add_row("Data Saved:", str(result.get("data_saved", 0)))
    
    if result.get("errors"):
        summary_table.add_row("[bold red]Errors:[/bold red]", "\n".join(result["errors"]))
        
    console.print(summary_table)

if __name__ == "__main__":
    app() 