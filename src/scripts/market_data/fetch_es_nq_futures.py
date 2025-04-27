#!/usr/bin/env python3
"""
Fetch ES and NQ Futures Historical Data

This script fetches historical daily data for ES and NQ futures from 2004 to the present.
- ES: E-mini S&P 500 Futures
- NQ: E-mini Nasdaq-100 Futures

The script handles:
1. Generating the correct futures symbols (quarterly contracts: H, M, U, Z)
2. Fetching data from TradeStation API
3. Saving data to the specified database
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import yaml
import pandas as pd
import requests
import time
import json
import duckdb
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.table import Table

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

# Import required modules
try:
    # Try direct import (when run as a module)
    from src.agents.tradestation_market_data_agent import TradeStationMarketDataAgent
    from src.utils.database import get_db_engine, ensure_market_data_table, migrate_date_based_to_timestamp
except ModuleNotFoundError:
    # Try relative import (when run as a script)
    from agents.tradestation_market_data_agent import TradeStationMarketDataAgent
    from utils.database import get_db_engine, ensure_market_data_table, migrate_date_based_to_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Add file handler after ensuring directory exists
log_file = os.path.join(logs_dir, 'fetch_es_nq_futures.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("Futures Data Fetcher")

# Setup console
console = Console()

class FuturesDataFetcher:
    """Fetches historical futures data from TradeStation."""

    def __init__(self, db_path: str, verbose: bool = False):
        """
        Initialize the futures data fetcher.
        
        Args:
            db_path: Path to the DuckDB database
            verbose: Whether to enable verbose logging
        """
        self.db_path = db_path
        self.verbose = verbose
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
            
        # Initialize TradeStation agent
        self.ts_agent = TradeStationMarketDataAgent(database_path=db_path, verbose=verbose)
        
        # Month codes mapping
        self.month_codes = {
            3: 'H',  # March
            6: 'M',  # June
            9: 'U',  # September
            12: 'Z'  # December
        }
        
        # Ensure database table exists
        self._create_market_data_table()
        
    def _create_market_data_table(self):
        """Create the market_data table if it doesn't exist."""
        try:
            conn = get_db_engine()
            
            # Use the utility function to ensure the market_data table exists with unified schema
            ensure_market_data_table(conn)
            
            # If we had previous data in the old format, migrate it
            migrate_date_based_to_timestamp(conn)
                
            logger.info("Market data table verified/created")
            
        except Exception as e:
            logger.error(f"Error creating market_data table: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
                
    def generate_futures_symbols(self, base_symbol: str, start_year: int, end_year: int) -> List[str]:
        """
        Generate futures contract symbols for the given base symbol and date range.
        
        Args:
            base_symbol: Base symbol (e.g., 'ES', 'NQ')
            start_year: Start year (e.g., 2004)
            end_year: End year (e.g., current year)
            
        Returns:
            List of futures contract symbols (e.g., ['ESH04', 'ESM04', ...])
        """
        symbols = []
        current_year = datetime.now().year
        
        # Ensure end_year is not in the future
        if end_year > current_year:
            end_year = current_year
            
        for year in range(start_year, end_year + 1):
            year_code = str(year)[-2:]  # Get last 2 digits
            
            for month in [3, 6, 9, 12]:  # March, June, September, December
                # Skip future contracts from current year
                if year == current_year and month > datetime.now().month:
                    continue
                    
                month_code = self.month_codes[month]
                symbol = f"{base_symbol}{month_code}{year_code}"
                symbols.append(symbol)
                
        return symbols

    def fetch_data_for_symbol(self, symbol: str, start_date: date, end_date: date) -> bool:
        """
        Fetch daily data for a single futures contract.
        
        Args:
            symbol: Futures contract symbol (e.g., 'ESH04')
            start_date: Start date
            end_date: End date
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare parameters for the API call
            params = {
                "action": "fetch",
                "symbols": [symbol],
                "interval_value": 1,
                "interval_unit": "day",
                "timeframe": "daily",
                "start_date": start_date,
                "end_date": end_date,
                "adjusted": True  # Adjust for splits and dividends
            }
            
            # Log the parameters
            logger.debug(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Authenticate before fetching
            if not self.ts_agent.authenticate():
                logger.error(f"Authentication failed before fetching {symbol}")
                return False
                
            # Directly make API request without using ts_agent.fetch_market_data
            # which uses its own Progress bar that causes conflicts
            try:
                df = self._fetch_data_directly(params)
            except Exception as e:
                logger.error(f"Error during API request for {symbol}: {e}")
                return False
                
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return False
                
            # Log data summary
            logger.info(f"Fetched {len(df)} rows of daily data for {symbol}")
            
            # Save data to database using our custom method
            inserted, updated = self._save_market_data(df)
            logger.info(f"Saved {inserted} new rows, updated {updated} existing rows for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return False
            
    def _fetch_data_directly(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Direct implementation of data fetching to avoid nested Progress bars.
        This is a simplified version of the ts_agent.fetch_market_data method.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            DataFrame containing the fetched market data
        """
        all_data = []
        
        # The TradeStation base URL from the agent
        base_url = self.ts_agent.base_url
        
        for symbol in params['symbols']:
            # Determine API endpoint and parameters based on timeframe
            if params["interval_unit"] in ["minute", "hour"]:
                endpoint = f"{base_url}/marketdata/barcharts/{symbol}"
                api_params = {
                    "interval": params["interval_value"],
                    "unit": params["interval_unit"],
                    "barsback": 1000  # Default if no date range specified
                }
                
                # Add date range if specified
                if params.get("start_date") and params.get("end_date"):
                    start_str = params["start_date"].strftime("%Y-%m-%dT00:00:00Z")
                    end_str = params["end_date"].strftime("%Y-%m-%dT23:59:59Z")
                    api_params["startdate"] = start_str
                    api_params["enddate"] = end_str
                    del api_params["barsback"]  # Remove barsback when using date range
                    
            else:  # daily, weekly, monthly
                endpoint = f"{base_url}/marketdata/barcharts/{symbol}"
                
                # Map our interval_unit to TradeStation's expected values
                unit_map = {
                    "day": "Daily",
                    "daily": "Daily",
                    "week": "Weekly",
                    "weekly": "Weekly",
                    "month": "Monthly",
                    "monthly": "Monthly"
                }
                
                # Use the correct unit format for TradeStation API
                ts_unit = unit_map.get(params["interval_unit"].lower(), "Daily")
                
                # For simplicity in this direct implementation, use a fixed approach
                # without complex calculations
                api_params = {
                    "interval": params["interval_value"],
                    "unit": ts_unit,
                }
                
                # Add date range
                if params.get("start_date") and params.get("end_date"):
                    start_str = params["start_date"].strftime("%Y-%m-%dT00:00:00Z")
                    end_str = params["end_date"].strftime("%Y-%m-%dT23:59:59Z")
                    api_params["startdate"] = start_str
                    api_params["enddate"] = end_str
                else:
                    # If no date range, use barsback
                    api_params["barsback"] = 1000
            
            headers = {
                "Authorization": f"Bearer {self.ts_agent.access_token}",
                "Content-Type": "application/json"
            }
            
            logger.debug(f"API request: {endpoint} with params {api_params}")
            response = requests.get(endpoint, params=api_params, headers=headers)
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                time.sleep(wait_time)
                
                # Retry the request
                response = requests.get(endpoint, params=api_params, headers=headers)
            
            # Process the response
            if response.status_code == 200:
                data = response.json()
                
                # Log full response for debugging
                logger.debug(f"API Response: {json.dumps(data, indent=2)}")
                
                if 'Bars' in data and data['Bars']:
                    # Log the data received for debugging
                    logger.debug(f"Received {len(data['Bars'])} bars for {symbol}")
                    
                    # Process data similar to TradeStationMarketDataAgent._process_market_data
                    df = self._process_data(symbol, data)
                    
                    if not df.empty:
                        logger.info(f"Processed {len(df)} bars for {symbol}")
                        all_data.append(df)
                    else:
                        logger.warning(f"Processing returned empty DataFrame for {symbol}")
                else:
                    logger.warning(f"No bars data returned for {symbol}")
                    if 'Message' in data:
                        logger.warning(f"API message: {data['Message']}")
            else:
                logger.error(f"API error for {symbol}: {response.status_code} - {response.text}")
            
            # Sleep briefly to avoid hammering the API
            time.sleep(0.2)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.debug(f"Combined data shape: {combined_data.shape}")
            return combined_data
        else:
            logger.warning("No data to combine, returning empty DataFrame")
            return pd.DataFrame()
            
    def _process_data(self, symbol: str, data: dict) -> pd.DataFrame:
        """
        Process raw data from TradeStation API to create a DataFrame.
        
        Args:
            symbol: Symbol for the data
            data: Raw data from API
            
        Returns:
            DataFrame with processed data
        """
        if not data or "Bars" not in data or not data["Bars"]:
            logger.warning(f"No data received for {symbol}")
            return pd.DataFrame()
            
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data["Bars"])
            
            # Clean up DataFrame
            # Convert TotalVolume to numeric
            if "TotalVolume" in df.columns:
                df["TotalVolume"] = pd.to_numeric(df["TotalVolume"], errors="coerce")
                df.rename(columns={"TotalVolume": "volume"}, inplace=True)
            elif "Volume" in df.columns:
                df["volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                df.drop(columns=["Volume"], inplace=True, errors="ignore")
            else:
                df["volume"] = 0
                
            # Handle DateTime and Date
            if "TimeStamp" in df.columns:
                # Update format to include 'Z' for timezone
                df["timestamp"] = pd.to_datetime(df["TimeStamp"], format="%Y-%m-%dT%H:%M:%SZ")
                df["date"] = df["timestamp"].dt.date.astype(str)
                df.drop(columns=["TimeStamp"], inplace=True, errors="ignore")
            elif "DateTime" in df.columns:
                # Try different format options with error handling
                df["timestamp"] = pd.to_datetime(df["DateTime"], errors="coerce")
                df["date"] = df["timestamp"].dt.date.astype(str)
                df.drop(columns=["DateTime"], inplace=True, errors="ignore")
            
            # Rename columns
            rename_dict = {
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
            }
            df.rename(columns=rename_dict, inplace=True)
            
            # Add symbol
            df["symbol"] = symbol
            
            # Add interval info
            df["interval_value"] = 1
            df["interval_unit"] = "day"
            
            # Add source
            df["source"] = "tradestation"
            
            # Add additional columns required by unified schema
            df["settle"] = df["close"]  # Use close price as settle price
            df["open_interest"] = df.get("OpenInterest", 0)
            df["up_volume"] = df.get("UpVolume", None)
            df["down_volume"] = df.get("DownVolume", None)
            df["changed"] = False
            df["adjusted"] = False
            df["quality"] = 100
            
            # Ensure all required columns exist
            required_columns = [
                "symbol", "date", "timestamp", "open", "high", "low", "close", "settle",
                "volume", "open_interest", "up_volume", "down_volume",
                "interval_value", "interval_unit", "source", 
                "changed", "adjusted", "quality"
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    if col == "date" and "timestamp" in df.columns:
                        df["date"] = df["timestamp"].dt.date.astype(str)
                    else:
                        df[col] = None
            
            # Log processed data size
            logger.debug(f"Processed {len(df)} rows for {symbol}")
            
            return df[required_columns]
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _save_market_data(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Save market data to database using a basic but reliable approach.
        
        Args:
            df: DataFrame with market data to save
            
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if df.empty:
            logger.info("No data to save")
            return 0, 0
        
        # Get database connection
        conn = get_db_engine()
        
        # Initialize counters
        inserted_count = 0
        updated_count = 0
        
        try:
            # Log the actual data types of the DataFrame
            logger.debug(f"DataFrame types: {df.dtypes}")
            
            # Convert DataFrame column names to lowercase for easier comparison
            df_copy = df.copy()
            df_copy.columns = [col.lower() for col in df_copy.columns]
            
            # Ensure date is in the proper format
            if 'date' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
            
            # If we have timestamp but not date, create date from timestamp
            if 'timestamp' in df_copy.columns and 'date' not in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.strftime('%Y-%m-%d')
            
            # First, determine the actual columns in the database table
            try:
                # Get schema using PRAGMA
                table_info = conn.execute("PRAGMA table_info(market_data)").fetchall()
                actual_columns = [row[1].lower() for row in table_info]
                logger.debug(f"Actual database columns: {actual_columns}")
            except Exception as e:
                logger.warning(f"Error getting table schema: {e}")
                # Fallback to a minimal set of columns that should exist
                actual_columns = ["symbol", "date", "open", "high", "low", "close", "volume", 
                                 "interval_value", "interval_unit", "source"]
                logger.debug(f"Using fallback columns: {actual_columns}")
            
            # Remove any columns not in the database schema
            cols_to_use = [col for col in df_copy.columns if col in actual_columns]
            
            if not cols_to_use:
                logger.error("No usable columns found in DataFrame that match the database schema")
                return 0, 0
            
            # Create a new DataFrame with only valid columns
            df_to_save = df_copy[cols_to_use]
            logger.debug(f"Using columns for database insert: {cols_to_use}")
            
            # Process each row individually for maximum reliability
            for idx, row in df_to_save.iterrows():
                try:
                    # Try to insert the row
                    column_list = ", ".join([f'"{col}"' for col in row.index])
                    placeholders = ", ".join(["?"] * len(row))
                    values = list(row.values)
                    
                    insert_sql = f"INSERT INTO market_data ({column_list}) VALUES ({placeholders})"
                    
                    try:
                        conn.execute(insert_sql, values)
                        inserted_count += 1
                    except Exception as e:
                        logger.debug(f"Row {idx} insert failed, attempting update: {e}")
                        
                        # Try to update based on symbol, date and interval
                        key_cols = ['symbol', 'date', 'interval_value', 'interval_unit']
                        available_keys = [k for k in key_cols if k in row.index]
                        
                        if len(available_keys) < 2:  # Need at least symbol and date
                            logger.debug(f"Row {idx} - not enough key columns for update")
                            continue
                        
                        # Get values for columns to update (exclude key columns)
                        update_cols = [col for col in row.index if col not in available_keys]
                        
                        if not update_cols:
                            logger.debug(f"Row {idx} - no columns to update")
                            continue
                        
                        # Build WHERE clause
                        where_conditions = " AND ".join([f'"{col}" = ?' for col in available_keys])
                        where_values = [row[col] for col in available_keys]
                        
                        # Build SET clause
                        set_clause = ", ".join([f'"{col}" = ?' for col in update_cols])
                        set_values = [row[col] for col in update_cols]
                        
                        update_sql = f"UPDATE market_data SET {set_clause} WHERE {where_conditions}"
                        
                        try:
                            conn.execute(update_sql, set_values + where_values)
                            updated_count += 1
                        except Exception as update_err:
                            logger.debug(f"Update also failed for row {idx}: {update_err}")
                
                except Exception as row_err:
                    logger.debug(f"Error processing row {idx}: {row_err}")
            
            logger.info(f"Saved {inserted_count} new records and updated {updated_count} existing records")
            return inserted_count, updated_count
        
        except Exception as e:
            logger.error(f"Error in save_market_data: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
            return 0, 0

    def fetch_all_futures(self, base_symbols: List[str], start_year: int, end_year: int) -> Dict[str, Dict]:
        """
        Fetch data for all futures contracts of the given base symbols.
        
        Args:
            base_symbols: List of base symbols (e.g., ['ES', 'NQ'])
            start_year: Start year
            end_year: End year
            
        Returns:
            Dictionary with statistics about the fetch operation
        """
        results = {
            "total_contracts": 0,
            "successful_contracts": 0,
            "failed_contracts": 0,
            "details": {}
        }
        
        # Authenticate with TradeStation API
        if not self.ts_agent.authenticate():
            logger.error("Failed to authenticate with TradeStation API")
            return results
            
        logger.info(f"Successfully authenticated with TradeStation API")
        
        # Set date range
        start_date = date(start_year, 1, 1)
        end_date = datetime.now().date()
        
        for base_symbol in base_symbols:
            logger.info(f"Generating symbols for {base_symbol} from {start_year} to {end_year}")
            symbols = self.generate_futures_symbols(base_symbol, start_year, end_year)
            
            results["details"][base_symbol] = {
                "total_contracts": len(symbols),
                "successful_contracts": 0,
                "failed_contracts": 0
            }
            
            results["total_contracts"] += len(symbols)
            
            logger.info(f"Generated {len(symbols)} contract symbols for {base_symbol}")
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Fetching {base_symbol} futures data...", 
                    total=len(symbols)
                )
                
                for symbol in symbols:
                    logger.info(f"Processing {symbol}...")
                    success = self.fetch_data_for_symbol(symbol, start_date, end_date)
                    
                    if success:
                        results["successful_contracts"] += 1
                        results["details"][base_symbol]["successful_contracts"] += 1
                    else:
                        results["failed_contracts"] += 1
                        results["details"][base_symbol]["failed_contracts"] += 1
                        
                    progress.update(task, advance=1)
                    
        return results

    def fetch_recent_contracts(self, base_symbols: List[str], days_back: int = 90, num_contracts: int = 2) -> Dict[str, Dict]:
        """
        Fetch data for only the most recent non-expired contracts.
        
        Args:
            base_symbols: List of base symbols (e.g., ['ES', 'NQ'])
            days_back: Number of days of data to fetch
            num_contracts: Number of recent contracts to fetch per symbol
            
        Returns:
            Dictionary with statistics about the fetch operation
        """
        results = {
            "total_contracts": 0,
            "successful_contracts": 0,
            "failed_contracts": 0,
            "details": {}
        }
        
        # Authenticate with TradeStation API
        if not self.ts_agent.authenticate():
            logger.error("Failed to authenticate with TradeStation API")
            return results
            
        logger.info(f"Successfully authenticated with TradeStation API")
        
        # Calculate date range for recent data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Add a buffer day to ensure we capture yesterday's data
        fetch_start_date = start_date - timedelta(days=1)
        
        for base_symbol in base_symbols:
            # Get active contracts
            active_contracts = self.get_active_contracts(base_symbol, num_contracts)
            
            results["details"][base_symbol] = {
                "total_contracts": len(active_contracts),
                "successful_contracts": 0,
                "failed_contracts": 0
            }
            
            results["total_contracts"] += len(active_contracts)
            
            logger.info(f"Fetching {len(active_contracts)} active contracts for {base_symbol}: {', '.join(active_contracts)}")
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Updating {base_symbol} recent futures data...", 
                    total=len(active_contracts)
                )
                
                for symbol in active_contracts:
                    logger.info(f"Processing recent data for {symbol}...")
                    # Use fetch_start_date (with buffer) for API request, but filter for start_date in processing
                    success = self.fetch_data_for_symbol(symbol, fetch_start_date, end_date)
                    
                    if success:
                        results["successful_contracts"] += 1
                        results["details"][base_symbol]["successful_contracts"] += 1
                    else:
                        results["failed_contracts"] += 1
                        results["details"][base_symbol]["failed_contracts"] += 1
                        
                    progress.update(task, advance=1)
                    
        return results

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display fetch operation results in a formatted table."""
        console.print(Panel.fit(
            "[bold green]Futures Data Fetch Operation Complete[/bold green]",
            title="Results Summary"
        ))
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric")
        summary_table.add_column("Value")
        
        summary_table.add_row("Total Contracts", str(results["total_contracts"]))
        summary_table.add_row("Successful", f"[green]{results['successful_contracts']}[/green]")
        summary_table.add_row("Failed", f"[red]{results['failed_contracts']}[/red]")
        summary_table.add_row("Success Rate", f"{results['successful_contracts'] / results['total_contracts'] * 100:.1f}%")
        
        console.print(summary_table)
        
        # Display details for each base symbol
        for base_symbol, details in results["details"].items():
            symbol_table = Table(show_header=True, header_style="bold")
            symbol_table.add_column("Metric")
            symbol_table.add_column("Value")
            
            symbol_table.add_row("Total Contracts", str(details["total_contracts"]))
            symbol_table.add_row("Successful", f"[green]{details['successful_contracts']}[/green]")
            symbol_table.add_row("Failed", f"[red]{details['failed_contracts']}[/red]")
            
            console.print(Panel.fit(
                symbol_table,
                title=f"[bold]{base_symbol} Details[/bold]"
            ))
            
    def close(self) -> None:
        """Close the database connection."""
        if self.ts_agent:
            self.ts_agent.close()

    def get_active_contracts(self, base_symbol: str, num_contracts: int) -> List[str]:
        """
        Get the most recent active contracts for a base symbol.
        
        Args:
            base_symbol: Base symbol (e.g., 'ES' for S&P 500 futures)
            num_contracts: Number of active contracts to return
            
        Returns:
            List of active contract symbols
        """
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Map months to month codes for futures contracts
        month_codes = [
            (3, 'H'),   # March (Q1)
            (6, 'M'),   # June (Q2)
            (9, 'U'),   # September (Q3)
            (12, 'Z')   # December (Q4)
        ]
        
        # Find the current quarter and next quarters
        contracts = []
        
        # Determine the next available quarterly contract
        # If we're past the current quarter's month, we need to look at the next quarter
        current_quarter_index = 0
        for i, (month, _) in enumerate(month_codes):
            if current_month <= month:
                current_quarter_index = i
                break
        # If we've passed all quarterly months, move to next year's first quarter
        if current_month > month_codes[-1][0]:
            current_quarter_index = 0
            current_year += 1
            
        logger.debug(f"Starting with quarter index {current_quarter_index} for month {current_month}")
        
        # Generate the next num_contracts worth of active contracts
        added_contracts = 0
        year = current_year
        quarter_index = current_quarter_index
        
        while added_contracts < num_contracts:
            month, code = month_codes[quarter_index]
            year_code = str(year)[-2:]
            symbol = f"{base_symbol}{code}{year_code}"
            contracts.append(symbol)
            
            # Move to next quarter
            quarter_index = (quarter_index + 1) % 4
            if quarter_index == 0:  # Wrapped around to Q1 (March)
                year += 1
                
            added_contracts += 1
            
        if self.verbose:
            logger.debug(f"Active contracts for {base_symbol}: {contracts}")
        
        return contracts

    def update_config_with_symbols(self, base_symbol: str, symbols: List[str], output_path: str, description: str = None, exchange: str = None, calendar: str = None, num_active_contracts: int = 4) -> None:
        """
        Update the configuration file with generated symbols.
        
        Args:
            base_symbol: Base symbol (e.g., 'ES' for S&P 500 futures)
            symbols: List of symbols to add
            output_path: Path to save the configuration file
            description: Optional description of the futures contract
            exchange: Optional exchange where the futures contract trades
            calendar: Optional holiday calendar to use
            num_active_contracts: Number of active contracts to track
        """
        # Create a default config structure
        config = {
            'futures': [],
            'equities': [],
            'settings': {
                'default_start_date': '2004-01-01',
                'data_frequencies': [
                    {'name': '1min', 'interval': 1, 'unit': 'minute'},
                    {'name': '15min', 'interval': 15, 'unit': 'minute'},
                    {'name': 'daily', 'interval': 1, 'unit': 'day'}
                ],
                'holiday_calendars': {}
            }
        }
        
        # If output file exists, load it
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    config = yaml.safe_load(f) or config
            except Exception as e:
                logger.error(f"Error loading existing config: {e}")
        
        # Get active contracts
        active_contracts = self.get_active_contracts(base_symbol, num_active_contracts)
        
        # Find the start year and month from the symbols
        start_year = int(symbols[0][-2:]) if symbols else datetime.now().year
        
        # Check if the base symbol already exists
        base_symbol_exists = False
        for i, future in enumerate(config.get('futures', [])):
            if future.get('base_symbol') == base_symbol:
                base_symbol_exists = True
                # Update existing entry
                config['futures'][i]['num_active_contracts'] = num_active_contracts
                break
        
        # If base symbol doesn't exist, add it
        if not base_symbol_exists:
            new_entry = {
                'base_symbol': base_symbol,
                'frequencies': ['1min', '15min', 'daily'],
                'description': description or f'{base_symbol} Futures',
                'exchange': exchange or 'CME',
                'calendar': calendar or 'US',
                'num_active_contracts': num_active_contracts,
                'historical_contracts': {
                    'start_year': start_year,
                    'start_month': 1,
                    'patterns': ['H', 'M', 'U', 'Z']  # March, June, September, December
                }
            }
            config['futures'].append(new_entry)
        
        # Save the updated config
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Updated configuration for {base_symbol} in {output_path}")
            logger.info(f"Active contracts: {', '.join(active_contracts)}")
        except Exception as e:
            logger.error(f"Error saving symbols to config: {e}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Fetch historical daily data for ES and NQ futures from 2004 to present',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch both ES and NQ futures from 2004 to present (full history)
  python fetch_es_nq_futures.py --db-path ./data/financial_data.duckdb
  
  # Daily update mode: fetch only the 2 most recent contracts with 90 days of data
  python fetch_es_nq_futures.py --db-path ./data/financial_data.duckdb --mode update
  
  # Fetch only ES futures from 2010 to present
  python fetch_es_nq_futures.py --db-path ./data/financial_data.duckdb --symbols ES --start-year 2010
  
  # Fetch with verbose logging
  python fetch_es_nq_futures.py --db-path ./data/financial_data.duckdb --verbose
  
  # Update market symbols config with the futures data
  python fetch_es_nq_futures.py --db-path ./data/financial_data.duckdb --update-config
        """
    )
    
    parser.add_argument(
        '--db-path', 
        type=str, 
        default=os.path.join(project_root, "data", "financial_data.duckdb"),
        help='Path to the DuckDB database'
    )
    parser.add_argument(
        '--symbols', 
        type=str, 
        default="ES,NQ",
        help='Comma-separated list of base symbols to fetch (default: ES,NQ)'
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        default=2004,
        help='Start year for data fetching (default: 2004)'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        default=datetime.now().year,
        help='End year for data fetching (default: current year)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['initial', 'update'],
        default='initial',
        help='Operation mode: initial for full history, update for recent contracts only (default: initial)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=90,
        help='Number of days to look back when in update mode (default: 90)'
    )
    parser.add_argument(
        '--num-contracts',
        type=int,
        default=2,
        help='Number of recent contracts to update in update mode (default: 2)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--update-config',
        action='store_true',
        help='Update market symbols config file with the futures symbols'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default=os.path.join(project_root, "config", "market_symbols.yaml"),
        help='Path to the market symbols config file (default: config/market_symbols.yaml)'
    )
    parser.add_argument(
        '--descriptions',
        type=str,
        default="ES:E-mini S&P 500,NQ:E-mini Nasdaq-100",
        help='Comma-separated list of symbol:description pairs (default: ES:E-mini S&P 500,NQ:E-mini Nasdaq-100)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    base_symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Parse descriptions
    symbol_descriptions = {}
    for pair in args.descriptions.split(','):
        if ':' in pair:
            symbol, desc = pair.split(':', 1)
            symbol_descriptions[symbol.strip()] = desc.strip()
    
    # Create fetcher
    fetcher = FuturesDataFetcher(args.db_path, args.verbose)
    
    try:
        # Generate symbols for base symbols (needed for config updates)
        all_symbols = {}
        
        if args.mode == 'initial' or args.update_config:
            for base_symbol in base_symbols:
                symbols = fetcher.generate_futures_symbols(base_symbol, args.start_year, args.end_year)
                all_symbols[base_symbol] = symbols
                logger.info(f"Generated {len(symbols)} contract symbols for {base_symbol}")
        
        # Update config if requested
        if args.update_config:
            for base_symbol in base_symbols:
                description = symbol_descriptions.get(base_symbol, f"{base_symbol} Futures")
                fetcher.update_config_with_symbols(
                    base_symbol,
                    all_symbols[base_symbol],
                    args.config_path,
                    description=description
                )
            logger.info(f"Updated market symbols config at {args.config_path}")
        
        # Fetch data based on mode
        if args.mode == 'initial':
            logger.info(f"Running in INITIAL mode - fetching full history from {args.start_year} to {args.end_year}")
            results = fetcher.fetch_all_futures(base_symbols, args.start_year, args.end_year)
        else:  # update mode
            logger.info(f"Running in UPDATE mode - fetching {args.num_contracts} recent contracts with {args.days_back} days of data")
            results = fetcher.fetch_recent_contracts(base_symbols, args.days_back, args.num_contracts)
        
        # Display results
        fetcher.display_results(results)
        
    except Exception as e:
        logger.error(f"Error during futures data fetch: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
    finally:
        # Close database connection
        fetcher.close()

if __name__ == "__main__":
    main() 