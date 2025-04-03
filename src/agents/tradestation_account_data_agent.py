#!/usr/bin/env python3
"""
Agent Name: TradeStation Account Data Agent
Purpose: Retrieve account data from TradeStation API
Author: Claude
Date: 2025-04-02

Description:
    This agent retrieves account data from the TradeStation API including balances,
    positions, orders, and trades. It handles authentication, manages rate limits,
    processes errors, and transforms the data to match our database schema.

Usage:
    uv run tradestation_account_data_agent.py -d ./path/to/database.duckdb -q "get account balances"
    uv run tradestation_account_data_agent.py -d ./path/to/database.duckdb -q "retrieve positions for account ABC123"
    uv run tradestation_account_data_agent.py -d ./path/to/database.duckdb -f ./queries/get_account_data.json -v
"""

import os
import sys
import json
import logging
import time
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

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
logger = logging.getLogger("TradeStation Account Data Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "TradeStation Account Data Agent"
AGENT_VERSION = "0.1.0"
AGENT_DESCRIPTION = "Retrieve account data from TradeStation API including balances, positions, orders, and trades"

# API configuration
API_BASE_URL = "https://api.tradestation.com/v3"
TOKEN_URL = "https://signin.tradestation.com/oauth/token"
AUTH_ENDPOINT = "/brokerage/accounts"
ACCOUNTS_ENDPOINT = "/brokerage/accounts"
BALANCES_ENDPOINT = "/brokerage/accounts/{account_id}/balances"
POSITIONS_ENDPOINT = "/brokerage/accounts/{account_id}/positions"
ORDERS_ENDPOINT = "/brokerage/accounts/{account_id}/orders"
TRADES_ENDPOINT = "/brokerage/accounts/{account_id}/trades"

# Data types for query parsing
class DataType(str, Enum):
    BALANCES = "balances"
    POSITIONS = "positions"
    ORDERS = "orders"
    TRADES = "trades"
    ALL = "all"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class TradeStationAccountDataAgent:
    """Agent for retrieving account data from TradeStation API.
    
    This agent connects to the TradeStation API to retrieve account-related data
    including balances, positions, orders, and trades. It handles authentication,
    manages rate limits, and transforms the data to match our database schema.
    
    Attributes:
        database_path (str): Path to the DuckDB database file
        verbose (bool): Whether to enable verbose logging
        compute_loops (int): Number of reasoning iterations to perform
        conn (duckdb.DuckDBPyConnection): Database connection
        access_token (str): TradeStation API access token
        token_expiry (datetime): When the current token expires
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
        self.access_token = None
        self.token_expiry = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        # API credentials from environment variables
        self.client_id = os.getenv("TRADESTATION_CLIENT_ID")
        self.client_secret = os.getenv("TRADESTATION_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            logger.error("TradeStation API credentials not found in environment variables")
            logger.error("Please set TRADESTATION_CLIENT_ID and TRADESTATION_CLIENT_SECRET")
            sys.exit(1)
        
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
    
    def _authenticate(self) -> bool:
        """Authenticate with TradeStation API using OAuth 2.0.
        
        Acquires or refreshes the OAuth access token required for API calls.
        
        Returns:
            bool: True if authentication succeeded, False otherwise
            
        Raises:
            Exception: If authentication fails
        """
        # Check if we have a valid token already
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            logger.debug("Using existing access token")
            return True
        
        logger.debug("Authenticating with TradeStation API")
        
        # Prepare authentication data
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "ReadAccount"
        }
        
        try:
            response = requests.post(TOKEN_URL, data=auth_data)
            response.raise_for_status()
            
            auth_result = response.json()
            self.access_token = auth_result.get("access_token")
            expires_in = auth_result.get("expires_in", 1800)  # Default to 30 minutes
            
            # Set token expiry time (with a safety margin)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            
            logger.debug(f"Authentication successful, token expires at {self.token_expiry}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an authenticated request to the TradeStation API.
        
        Args:
            endpoint: API endpoint (starting with /)
            params: Query parameters for the request
            
        Returns:
            Dict containing the API response
            
        Raises:
            Exception: If the API request fails
        """
        # Ensure we're authenticated
        if not self._authenticate():
            raise Exception("Authentication failed")
        
        url = f"{API_BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                
                # Update rate limit info if available in headers
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                    logger.debug(f"Rate limit remaining: {self.rate_limit_remaining}")
                
                if 'X-RateLimit-Reset' in response.headers:
                    self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
                    logger.debug(f"Rate limit resets at: {datetime.fromtimestamp(self.rate_limit_reset)}")
                
                # Handle rate limiting
                if response.status_code == 429:  # Too Many Requests
                    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                    wait_time = max(reset_time - time.time(), 0) + 1  # Add 1 second buffer
                    
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                if response.status_code >= 400:
                    error_msg = f"API request failed with status {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f": {error_data.get('message', '')}"
                    except:
                        error_msg += f": {response.text}"
                    
                    # Token expired
                    if response.status_code == 401:
                        self.access_token = None
                        if attempt < max_retries - 1:
                            logger.warning("Token expired, reauthenticating...")
                            self._authenticate()
                            continue
                    
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                # Parse and return successful response
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query to extract parameters.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "get account balances for ABC123"
            Output: {
                "action": "get",
                "data_type": "balances",
                "account_id": "ABC123"
            }
        """
        query = query.lower()
        
        # Default parameters
        params = {
            "action": "get",
            "data_type": DataType.ALL,
            "account_id": None,
            "start_date": None,
            "end_date": None,
            "symbol": None,
            "status": None,
            "order_id": None
        }
        
        # Determine data type
        if "balance" in query:
            params["data_type"] = DataType.BALANCES
        elif "position" in query:
            params["data_type"] = DataType.POSITIONS
        elif "order" in query:
            params["data_type"] = DataType.ORDERS
        elif "trade" in query:
            params["data_type"] = DataType.TRADES
        
        # Extract account ID
        if "account" in query and "for" in query:
            account_parts = query.split("for")
            if len(account_parts) > 1:
                account_id = account_parts[1].strip().split()[0]
                if account_id.isalnum():  # Simple validation
                    params["account_id"] = account_id
        
        # Extract date range
        if "from" in query and "to" in query:
            try:
                from_index = query.index("from") + 5
                to_index = query.index("to")
                start_date_str = query[from_index:to_index].strip()
                
                # Assuming date format of YYYY-MM-DD
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                params["start_date"] = start_date
                
                end_date_str = query[to_index + 3:].strip().split()[0]
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                params["end_date"] = end_date
            except (ValueError, IndexError):
                # If date parsing fails, ignore and continue
                pass
        
        # Extract symbol for positions or orders
        if "symbol" in query or "ticker" in query:
            for term in ["symbol", "ticker"]:
                if term in query:
                    term_index = query.index(term) + len(term) + 1
                    symbol = query[term_index:].strip().split()[0].upper()
                    if symbol.isalpha():  # Simple validation
                        params["symbol"] = symbol
                    break
        
        # Extract order status for orders
        if "status" in query:
            status_index = query.index("status") + 7
            status = query[status_index:].strip().split()[0].title()
            if status in ["Open", "Filled", "Cancelled", "Rejected"]:
                params["status"] = status
        
        # Extract specific order ID
        if "order" in query and "id" in query:
            order_parts = query.split("id")
            if len(order_parts) > 1:
                order_id = order_parts[1].strip().split()[0]
                if order_id.isalnum():  # Simple validation
                    params["order_id"] = order_id
        
        logger.debug(f"Parsed parameters: {params}")
        return params
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Retrieve list of accounts from TradeStation API.
        
        Returns:
            List of account dictionaries
            
        Raises:
            Exception: If the API request fails
        """
        logger.debug("Retrieving accounts list")
        response = self._make_api_request(ACCOUNTS_ENDPOINT)
        accounts = response.get("Accounts", [])
        logger.debug(f"Retrieved {len(accounts)} accounts")
        return accounts
    
    def get_account_balances(self, account_id: str) -> Dict[str, Any]:
        """Retrieve account balances for a specific account.
        
        Args:
            account_id: TradeStation account ID
            
        Returns:
            Dictionary containing account balance information
            
        Raises:
            Exception: If the API request fails
        """
        logger.debug(f"Retrieving balances for account {account_id}")
        endpoint = BALANCES_ENDPOINT.format(account_id=account_id)
        response = self._make_api_request(endpoint)
        return response
    
    def get_positions(self, account_id: str, symbol: str = None) -> List[Dict[str, Any]]:
        """Retrieve positions for a specific account.
        
        Args:
            account_id: TradeStation account ID
            symbol: Optional symbol to filter positions
            
        Returns:
            List of position dictionaries
            
        Raises:
            Exception: If the API request fails
        """
        logger.debug(f"Retrieving positions for account {account_id}")
        endpoint = POSITIONS_ENDPOINT.format(account_id=account_id)
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        response = self._make_api_request(endpoint, params)
        positions = response.get("Positions", [])
        logger.debug(f"Retrieved {len(positions)} positions")
        return positions
    
    def get_orders(self, 
                  account_id: str, 
                  symbol: str = None, 
                  status: str = None,
                  start_date: datetime = None,
                  end_date: datetime = None) -> List[Dict[str, Any]]:
        """Retrieve orders for a specific account.
        
        Args:
            account_id: TradeStation account ID
            symbol: Optional symbol to filter orders
            status: Optional status to filter orders
            start_date: Optional start date for order history
            end_date: Optional end date for order history
            
        Returns:
            List of order dictionaries
            
        Raises:
            Exception: If the API request fails
        """
        logger.debug(f"Retrieving orders for account {account_id}")
        endpoint = ORDERS_ENDPOINT.format(account_id=account_id)
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        if status:
            params["status"] = status
        
        if start_date:
            params["since"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if end_date:
            params["until"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = self._make_api_request(endpoint, params)
        orders = response.get("Orders", [])
        logger.debug(f"Retrieved {len(orders)} orders")
        return orders
    
    def get_trades(self, 
                  account_id: str, 
                  symbol: str = None,
                  start_date: datetime = None,
                  end_date: datetime = None) -> List[Dict[str, Any]]:
        """Retrieve trades for a specific account.
        
        Args:
            account_id: TradeStation account ID
            symbol: Optional symbol to filter trades
            start_date: Optional start date for trade history
            end_date: Optional end date for trade history
            
        Returns:
            List of trade dictionaries
            
        Raises:
            Exception: If the API request fails
        """
        logger.debug(f"Retrieving trades for account {account_id}")
        endpoint = TRADES_ENDPOINT.format(account_id=account_id)
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        if start_date:
            params["since"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if end_date:
            params["until"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = self._make_api_request(endpoint, params)
        trades = response.get("Trades", [])
        logger.debug(f"Retrieved {len(trades)} trades")
        return trades
    
    def normalize_balances(self, account_id: str, balance_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize account balance data to match the database schema.
        
        Args:
            account_id: TradeStation account ID
            balance_data: Account balance data from API
            
        Returns:
            DataFrame containing normalized balance data
        """
        logger.debug(f"Normalizing balance data for account {account_id}")
        
        # Extract relevant fields
        balances = balance_data.get("Balances", {})
        
        timestamp = datetime.now()
        
        normalized_data = {
            "timestamp": timestamp,
            "account_id": account_id,
            "cash_balance": balances.get("CashBalance", 0.0),
            "buying_power": balances.get("BuyingPower", 0.0),
            "day_trading_buying_power": balances.get("DayTradingBuyingPower", 0.0),
            "equity": balances.get("Equity", 0.0),
            "margin_balance": balances.get("MarginBalance", 0.0),
            "real_time_buying_power": balances.get("RealTimeBuyingPower", 0.0),
            "real_time_equity": balances.get("RealTimeEquity", 0.0),
            "real_time_cost_of_positions": balances.get("RealTimeCostOfPositions", 0.0),
            "day_trades_count": balances.get("DayTradesCount", 0),
            "day_trading_qualified": balances.get("DayTradingQualified", False),
            "source": "TradeStation",
            "currency": "USD"  # Assuming USD as default
        }
        
        # Create DataFrame with a single row
        df = pd.DataFrame([normalized_data])
        
        logger.debug(f"Normalized balance data: {df.shape[0]} rows")
        return df
    
    def normalize_positions(self, account_id: str, positions_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize positions data to match the database schema.
        
        Args:
            account_id: TradeStation account ID
            positions_data: Positions data from API
            
        Returns:
            DataFrame containing normalized positions data
        """
        logger.debug(f"Normalizing positions data for account {account_id}")
        
        normalized_data = []
        timestamp = datetime.now()
        
        for position in positions_data:
            normalized_position = {
                "timestamp": timestamp,
                "account_id": account_id,
                "symbol": position.get("Symbol", ""),
                "quantity": position.get("Quantity", 0.0),
                "average_price": position.get("AveragePrice", 0.0),
                "market_value": position.get("MarketValue", 0.0),
                "cost_basis": position.get("CostBasis", 0.0),
                "open_pl": position.get("UnrealizedPL", 0.0),
                "open_pl_percent": position.get("UnrealizedPLPercent", 0.0),
                "day_pl": position.get("TodaysPL", 0.0),
                "initial_margin": position.get("InitialMargin", 0.0),
                "maintenance_margin": position.get("MaintenanceMargin", 0.0),
                "position_id": f"{account_id}_{position.get('Symbol', '')}",
                "source": "TradeStation"
            }
            
            normalized_data.append(normalized_position)
        
        if not normalized_data:
            logger.debug(f"No positions found for account {account_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(normalized_data)
        logger.debug(f"Normalized positions data: {df.shape[0]} rows")
        return df
    
    def normalize_orders(self, account_id: str, orders_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize orders data to match the database schema.
        
        Args:
            account_id: TradeStation account ID
            orders_data: Orders data from API
            
        Returns:
            DataFrame containing normalized orders data
        """
        logger.debug(f"Normalizing orders data for account {account_id}")
        
        normalized_data = []
        
        for order in orders_data:
            # Convert execution time strings to timestamps
            execution_time = None
            if "ExecutionTime" in order:
                try:
                    execution_time = datetime.strptime(order["ExecutionTime"], "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    pass
            
            cancellation_time = None
            if "CancellationTime" in order:
                try:
                    cancellation_time = datetime.strptime(order["CancellationTime"], "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    pass
            
            # Extract timestamp from order data or use current time
            timestamp = None
            if "Time" in order:
                try:
                    timestamp = datetime.strptime(order["Time"], "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            normalized_order = {
                "timestamp": timestamp,
                "account_id": account_id,
                "order_id": order.get("OrderID", ""),
                "symbol": order.get("Symbol", ""),
                "quantity": order.get("Quantity", 0.0),
                "order_type": order.get("OrderType", ""),
                "side": order.get("Side", ""),
                "status": order.get("Status", ""),
                "limit_price": order.get("LimitPrice", 0.0),
                "stop_price": order.get("StopPrice", 0.0),
                "filled_quantity": order.get("FilledQuantity", 0.0),
                "remaining_quantity": order.get("RemainingQuantity", 0.0),
                "average_fill_price": order.get("AverageFillPrice", 0.0),
                "duration": order.get("Duration", ""),
                "route": order.get("Route", ""),
                "execution_time": execution_time,
                "cancellation_time": cancellation_time,
                "source": "TradeStation"
            }
            
            normalized_data.append(normalized_order)
        
        if not normalized_data:
            logger.debug(f"No orders found for account {account_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(normalized_data)
        logger.debug(f"Normalized orders data: {df.shape[0]} rows")
        return df
    
    def normalize_trades(self, account_id: str, trades_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize trades data to match the database schema.
        
        Args:
            account_id: TradeStation account ID
            trades_data: Trades data from API
            
        Returns:
            DataFrame containing normalized trades data
        """
        logger.debug(f"Normalizing trades data for account {account_id}")
        
        normalized_data = []
        
        for trade in trades_data:
            # Convert trade time string to timestamp
            trade_time = None
            if "TradeTime" in trade:
                try:
                    trade_time = datetime.strptime(trade["TradeTime"], "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, TypeError):
                    trade_time = datetime.now()
            else:
                trade_time = datetime.now()
            
            normalized_trade = {
                "timestamp": datetime.now(),
                "account_id": account_id,
                "order_id": trade.get("OrderID", ""),
                "trade_id": trade.get("TradeID", ""),
                "symbol": trade.get("Symbol", ""),
                "quantity": trade.get("Quantity", 0.0),
                "price": trade.get("Price", 0.0),
                "side": trade.get("Side", ""),
                "commission": trade.get("Commission", 0.0),
                "fees": trade.get("Fees", 0.0),
                "trade_time": trade_time,
                "position_effect": trade.get("PositionEffect", ""),
                "source": "TradeStation"
            }
            
            normalized_data.append(normalized_trade)
        
        if not normalized_data:
            logger.debug(f"No trades found for account {account_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(normalized_data)
        logger.debug(f"Normalized trades data: {df.shape[0]} rows")
        return df
    
    def save_account_data(self, data_type: str, df: pd.DataFrame) -> bool:
        """Save account data to the database.
        
        Args:
            data_type: Type of data (balances, positions, orders, trades)
            df: DataFrame containing the data to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        if df.empty:
            logger.warning(f"No {data_type} data to save")
            return False
        
        table_name = data_type.lower()
        logger.debug(f"Saving {df.shape[0]} rows to {table_name} table")
        
        try:
            # Convert timestamp columns to proper format if needed
            for col in df.columns:
                if col.endswith('_time') or col == 'timestamp':
                    if df[col].dtype == 'object':
                        df[col] = pd.to_datetime(df[col])
            
            # Write to database
            self.conn.execute(f"BEGIN TRANSACTION")
            
            # For upsert operations, we need to handle different primary keys
            if table_name == "account_balances":
                # Delete existing records for this timestamp and account
                self.conn.execute(f"""
                    DELETE FROM {table_name} 
                    WHERE timestamp = ? AND account_id = ?
                """, (df['timestamp'].iloc[0], df['account_id'].iloc[0]))
            
            elif table_name == "positions":
                # Delete existing records for this timestamp and account
                self.conn.execute(f"""
                    DELETE FROM {table_name} 
                    WHERE timestamp = ? AND account_id = ?
                """, (df['timestamp'].iloc[0], df['account_id'].iloc[0]))
            
            elif table_name == "orders":
                # Delete existing records for these order IDs
                order_ids = df['order_id'].tolist()
                if order_ids:
                    placeholders = ", ".join(["?"] * len(order_ids))
                    self.conn.execute(f"""
                        DELETE FROM {table_name} 
                        WHERE order_id IN ({placeholders})
                    """, order_ids)
            
            elif table_name == "trades":
                # Delete existing records for these trade IDs
                trade_ids = df['trade_id'].tolist()
                if trade_ids:
                    placeholders = ", ".join(["?"] * len(trade_ids))
                    self.conn.execute(f"""
                        DELETE FROM {table_name} 
                        WHERE trade_id IN ({placeholders})
                    """, trade_ids)
            
            # Insert new records
            self.conn.append(table_name, df)
            
            self.conn.execute(f"COMMIT")
            logger.debug(f"Successfully saved {df.shape[0]} rows to {table_name} table")
            return True
            
        except Exception as e:
            self.conn.execute(f"ROLLBACK")
            logger.error(f"Error saving {data_type} data: {e}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        This is the main entry point for agent functionality. It parses
        the query, performs the requested operation, and returns results.
        
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
            "errors": results.get("errors", []),
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
                "errors": results.get("errors", []),
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
            
        Loop 1: Validate parameters
        Loop 2: Retrieve data
        Loop 3: Transform and save data
        """
        # Default result structure
        result = {
            "action": params.get("action", "get"),
            "data_type": params.get("data_type", DataType.ALL),
            "success": False,
            "errors": [],
            "warnings": [],
            "data": None,
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
                    # Loop 2: Data retrieval
                    self._loop_retrieve_data(params, result)
                elif i == 2:
                    # Loop 3: Result validation, transformation, and storage
                    self._loop_transform_and_save(params, result)
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
        
        # If no account ID is provided, get all accounts and use first one
        if not params.get("account_id"):
            try:
                accounts = self.get_accounts()
                if accounts:
                    params["account_id"] = accounts[0].get("AccountID")
                    result["warnings"].append(f"No account ID provided, using first account: {params['account_id']}")
                else:
                    raise ValueError("No accounts available")
            except Exception as e:
                raise ValueError(f"Failed to retrieve accounts: {e}")
        
        # Validate data type
        data_type = params.get("data_type")
        if data_type not in [DataType.BALANCES, DataType.POSITIONS, DataType.ORDERS, DataType.TRADES, DataType.ALL]:
            raise ValueError(f"Invalid data type: {data_type}")
        
        # Validate date range if provided
        if params.get("start_date") and params.get("end_date"):
            if params["start_date"] > params["end_date"]:
                raise ValueError("Start date cannot be after end date")
        
        # Set default date range if not provided
        if not params.get("start_date"):
            params["start_date"] = datetime.now() - timedelta(days=30)
        
        if not params.get("end_date"):
            params["end_date"] = datetime.now()
        
        logger.debug("Parameters validated successfully")
    
    def _loop_retrieve_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Retrieve data from API.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Retrieving data from API")
        
        account_id = params.get("account_id")
        data_type = params.get("data_type")
        
        # Initialize data container in result
        result["data"] = {}
        
        # Retrieve data based on type
        if data_type == DataType.BALANCES or data_type == DataType.ALL:
            try:
                balance_data = self.get_account_balances(account_id)
                result["data"]["balances"] = balance_data
            except Exception as e:
                result["errors"].append(f"Error retrieving balances: {e}")
        
        if data_type == DataType.POSITIONS or data_type == DataType.ALL:
            try:
                positions_data = self.get_positions(account_id, params.get("symbol"))
                result["data"]["positions"] = positions_data
            except Exception as e:
                result["errors"].append(f"Error retrieving positions: {e}")
        
        if data_type == DataType.ORDERS or data_type == DataType.ALL:
            try:
                orders_data = self.get_orders(
                    account_id, 
                    params.get("symbol"),
                    params.get("status"),
                    params.get("start_date"),
                    params.get("end_date")
                )
                result["data"]["orders"] = orders_data
            except Exception as e:
                result["errors"].append(f"Error retrieving orders: {e}")
        
        if data_type == DataType.TRADES or data_type == DataType.ALL:
            try:
                trades_data = self.get_trades(
                    account_id,
                    params.get("symbol"),
                    params.get("start_date"),
                    params.get("end_date")
                )
                result["data"]["trades"] = trades_data
            except Exception as e:
                result["errors"].append(f"Error retrieving trades: {e}")
        
        logger.debug(f"Data retrieval completed for account {account_id}")
    
    def _loop_transform_and_save(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Transform and save data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Transforming and saving data")
        
        account_id = params.get("account_id")
        data_type = params.get("data_type")
        
        # Transform and save data based on type
        if data_type == DataType.BALANCES or data_type == DataType.ALL:
            if "balances" in result["data"]:
                try:
                    balances_df = self.normalize_balances(account_id, result["data"]["balances"])
                    save_success = self.save_account_data("account_balances", balances_df)
                    result["data"]["normalized_balances"] = balances_df.to_dict(orient="records")
                    result["data"]["balances_saved"] = save_success
                except Exception as e:
                    result["errors"].append(f"Error processing balances: {e}")
        
        if data_type == DataType.POSITIONS or data_type == DataType.ALL:
            if "positions" in result["data"]:
                try:
                    positions_df = self.normalize_positions(account_id, result["data"]["positions"])
                    save_success = self.save_account_data("positions", positions_df)
                    result["data"]["normalized_positions"] = positions_df.to_dict(orient="records") if not positions_df.empty else []
                    result["data"]["positions_saved"] = save_success
                except Exception as e:
                    result["errors"].append(f"Error processing positions: {e}")
        
        if data_type == DataType.ORDERS or data_type == DataType.ALL:
            if "orders" in result["data"]:
                try:
                    orders_df = self.normalize_orders(account_id, result["data"]["orders"])
                    save_success = self.save_account_data("orders", orders_df)
                    result["data"]["normalized_orders"] = orders_df.to_dict(orient="records") if not orders_df.empty else []
                    result["data"]["orders_saved"] = save_success
                except Exception as e:
                    result["errors"].append(f"Error processing orders: {e}")
        
        if data_type == DataType.TRADES or data_type == DataType.ALL:
            if "trades" in result["data"]:
                try:
                    trades_df = self.normalize_trades(account_id, result["data"]["trades"])
                    save_success = self.save_account_data("trades", trades_df)
                    result["data"]["normalized_trades"] = trades_df.to_dict(orient="records") if not trades_df.empty else []
                    result["data"]["trades_saved"] = save_success
                except Exception as e:
                    result["errors"].append(f"Error processing trades: {e}")
        
        logger.debug("Data transformation and storage completed")
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        data_type = results.get("parameters", {}).get("data_type", "unknown")
        account_id = results.get("parameters", {}).get("account_id", "unknown")
        
        if results["success"]:
            console.print(Panel(f"[bold green]Successfully retrieved {data_type} data for account {account_id}[/]", title=AGENT_NAME))
            
            # Display summary based on data type
            data = results.get("results", {}).get("data", {})
            
            if "normalized_balances" in data:
                balances = data["normalized_balances"][0] if data["normalized_balances"] else {}
                table = Table(title=f"Account Balances for {account_id}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green", justify="right")
                
                for key, value in balances.items():
                    if key not in ["timestamp", "account_id", "source", "currency"]:
                        table.add_row(key.replace("_", " ").title(), f"${value:,.2f}" if isinstance(value, (int, float)) else str(value))
                
                console.print(table)
            
            if "normalized_positions" in data:
                positions = data["normalized_positions"]
                if positions:
                    table = Table(title=f"Positions for {account_id}")
                    table.add_column("Symbol", style="cyan")
                    table.add_column("Quantity", justify="right")
                    table.add_column("Avg Price", justify="right")
                    table.add_column("Market Value", justify="right")
                    table.add_column("Unrealized P/L", justify="right")
                    
                    for position in positions:
                        pl_color = "green" if position.get("open_pl", 0) > 0 else "red"
                        table.add_row(
                            position.get("symbol", ""),
                            f"{position.get('quantity', 0):,.2f}",
                            f"${position.get('average_price', 0):,.2f}",
                            f"${position.get('market_value', 0):,.2f}",
                            f"[{pl_color}]${position.get('open_pl', 0):,.2f}[/{pl_color}]"
                        )
                    
                    console.print(table)
                else:
                    console.print("[yellow]No positions found[/]")
            
            if "normalized_orders" in data:
                orders = data["normalized_orders"]
                if orders:
                    table = Table(title=f"Orders for {account_id}")
                    table.add_column("Order ID", style="dim")
                    table.add_column("Symbol", style="cyan")
                    table.add_column("Side", justify="center")
                    table.add_column("Type", justify="center")
                    table.add_column("Status", justify="center")
                    table.add_column("Quantity", justify="right")
                    table.add_column("Price", justify="right")
                    
                    for order in orders:
                        side_color = "green" if order.get("side", "") == "Buy" else "red"
                        price = order.get("limit_price", 0) if order.get("order_type", "") == "Limit" else order.get("stop_price", 0)
                        
                        table.add_row(
                            order.get("order_id", "")[:8] + "...",
                            order.get("symbol", ""),
                            f"[{side_color}]{order.get('side', '')}[/{side_color}]",
                            order.get("order_type", ""),
                            order.get("status", ""),
                            f"{order.get('quantity', 0):,.2f}",
                            f"${price:,.2f}" if price > 0 else "Market"
                        )
                    
                    console.print(table)
                else:
                    console.print("[yellow]No orders found[/]")
            
            if "normalized_trades" in data:
                trades = data["normalized_trades"]
                if trades:
                    table = Table(title=f"Trades for {account_id}")
                    table.add_column("Trade ID", style="dim")
                    table.add_column("Symbol", style="cyan")
                    table.add_column("Side", justify="center")
                    table.add_column("Quantity", justify="right")
                    table.add_column("Price", justify="right")
                    table.add_column("Trade Time", justify="right")
                    
                    for trade in trades:
                        side_color = "green" if trade.get("side", "") == "Buy" else "red"
                        trade_time = trade.get("trade_time", "")
                        if isinstance(trade_time, datetime):
                            trade_time = trade_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        table.add_row(
                            trade.get("trade_id", "")[:8] + "...",
                            trade.get("symbol", ""),
                            f"[{side_color}]{trade.get('side', '')}[/{side_color}]",
                            f"{trade.get('quantity', 0):,.2f}",
                            f"${trade.get('price', 0):,.2f}",
                            str(trade_time)
                        )
                    
                    console.print(table)
                else:
                    console.print("[yellow]No trades found[/]")
            
        else:
            console.print(Panel(f"[bold red]Error retrieving {data_type} data for account {account_id}[/]", title=AGENT_NAME))
            for error in results.get("errors", []):
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
        uv run tradestation_account_data_agent.py -d ./financial_data.duckdb -q "get account balances"
        uv run tradestation_account_data_agent.py -d ./financial_data.duckdb -q "retrieve positions for account ABC123"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = TradeStationAccountDataAgent(
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
                json.dump(result, f, indent=2)
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
def test_connection(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Test database connection and API authentication."""
    try:
        agent = TradeStationAccountDataAgent(database_path=database, verbose=verbose)
        console.print("[bold green]Database connection successful![/]")
        
        # Test API authentication
        if agent._authenticate():
            console.print("[bold green]API authentication successful![/]")
            
            # Test account retrieval
            accounts = agent.get_accounts()
            if accounts:
                console.print(f"[bold green]Retrieved {len(accounts)} accounts[/]")
                for account in accounts:
                    console.print(f"- Account: {account.get('AccountID')}, Type: {account.get('AccountType')}")
            else:
                console.print("[yellow]No accounts found[/]")
        else:
            console.print("[bold red]API authentication failed![/]")
        
        agent.close()
    except Exception as e:
        console.print(f"[bold red]Connection test failed:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()