#!/usr/bin/env python

"""
Script to analyze futures contracts for a given root symbol.
Shows all contracts in chronological order, identifies gaps, and displays contract details.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import typer
import yaml
import pandas as pd
import duckdb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.padding import Padding
from rich.columns import Columns
from dotenv import load_dotenv
import argparse

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

# Import MarketDataFetcher
from .fetch_market_data import MarketDataFetcher

# Load environment variables from .env file
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")

class FuturesContractAnalyzer:
    """Analyzes futures contracts data to identify gaps and missing contracts."""
    
    def __init__(
        self,
        base_symbol: str,
        start_year: int = 2004,
        end_year: int = 2025,
        interval_value: int = 1,
        interval_unit: str = "daily"
    ):
        """Initialize the analyzer."""
        self.base_symbol = base_symbol
        self.start_year = start_year
        self.end_year = end_year
        self.interval_value = interval_value
        self.interval_unit = interval_unit
        self.contracts = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize fetcher with shared database path
        config_path = os.path.join(project_root, "config", "market_symbols.yaml")
        self.fetcher = MarketDataFetcher(config_path=config_path, db_path=DEFAULT_DB_PATH)
        
        self.console = Console()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the market symbols configuration."""
        config_path = os.path.join(project_root, "config", "market_symbols.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Find the futures config for our symbol
        for future in config['futures']:
            if future['base_symbol'] == self.base_symbol:
                return future
                
        raise ValueError(f"No configuration found for futures symbol {self.base_symbol}")

    def _generate_contract_symbols(self) -> List[str]:
        """Generate all possible contract symbols for the given base symbol and years."""
        symbols = []
        # Get the contract patterns from the config
        patterns = self.config['historical_contracts']['patterns']
        for year in range(self.start_year, self.end_year + 1):
            for month in patterns:
                symbols.append(f"{self.base_symbol}{month}{str(year)[-2:]}")
        return symbols

    def _get_contract_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get contract data from the database."""
        try:
            query = f"""
                SELECT *
                FROM market_data
                WHERE symbol = '{symbol}'
                AND interval_value = {self.interval_value}
                AND interval_unit = '{self.interval_unit}'
                ORDER BY timestamp ASC
            """
            df = self.fetcher.conn.execute(query).fetchdf()
            return df if not df.empty else None
        except Exception as e:
            logging.error(f"Error getting contract data for {symbol}: {e}")
            return None

    def _find_missing_contracts(self, all_symbols: List[str], existing_symbols: List[str]) -> List[str]:
        """Find missing contracts by comparing all possible symbols with existing ones."""
        return [symbol for symbol in all_symbols if symbol not in existing_symbols]

    def _get_month_order(self, symbol: str) -> int:
        """Get the order of the month code for sorting."""
        month_order = {
            'F': 1,   # January
            'G': 2,   # February
            'H': 3,   # March
            'J': 4,   # April
            'K': 5,   # May
            'M': 6,   # June
            'N': 7,   # July
            'Q': 8,   # August
            'U': 9,   # September
            'V': 10,  # October
            'X': 11,  # November
            'Z': 12   # December
        }
        month_code = symbol[2]  # Get the month code from the symbol (e.g., 'K' from 'VXK23')
        return month_order.get(month_code, 0)

    def _get_contract_year(self, symbol: str) -> int:
        """Get the contract year from the symbol."""
        year_str = symbol[-2:]  # Get the year from the symbol (e.g., '23' from 'VXK23')
        return 2000 + int(year_str)  # Convert to full year (e.g., 2023)

    def analyze(self) -> None:
        """Analyze futures contracts and display results."""
        logging.info(f"Analyzing {self.base_symbol} futures contracts from {self.start_year} to {self.end_year}")
        
        # Check what intervals are actually available in the database
        available_intervals = self._get_available_intervals()
        if not available_intervals:
            logging.warning(f"No data found for {self.base_symbol} with interval_value={self.interval_value}, interval_unit={self.interval_unit}")
            logging.info(f"Available intervals: {available_intervals}")
            return
            
        logging.info(f"Using interval: {self.interval_value} {self.interval_unit}")
        
        # Generate all possible contract symbols
        all_symbols = self._generate_contract_symbols()
        
        # Collect data for each contract
        self.contracts = []  # Reset contracts list
        for symbol in all_symbols:
            df = self._get_contract_data(symbol)
            if df is not None and not df.empty:
                start_date = df['timestamp'].min()
                end_date = df['timestamp'].max()
                contract_info = {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'days': (end_date - start_date).days + 1,
                    'records': len(df)
                }
                self.contracts.append(contract_info)

        # Sort contracts by year and then by month order
        self.contracts = sorted(
            self.contracts,
            key=lambda x: (
                self._get_contract_year(x['symbol']),
                self._get_month_order(x['symbol'])
            )
        )

        # Create results table
        table = Table(
            title=f"{self.base_symbol} Futures Contracts Analysis ({self.interval_value} {self.interval_unit})",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        table.add_column("Symbol", style="cyan")
        table.add_column("Start Date", style="green")
        table.add_column("End Date", style="green")
        table.add_column("Days", justify="right", style="yellow")
        table.add_column("Records", justify="right", style="yellow")

        # Display contracts in chronological order
        for data in self.contracts:
            table.add_row(
                data['symbol'],
                data['start_date'].strftime('%Y-%m-%d'),
                data['end_date'].strftime('%Y-%m-%d'),
                str(data['days']),
                f"{data['records']:,}"
            )

        # Calculate summary statistics
        total_contracts = len(self.contracts)
        total_records = sum(d['records'] for d in self.contracts)
        avg_days = sum(d['days'] for d in self.contracts) / total_contracts if total_contracts > 0 else 0
        
        summary = Table.grid()
        summary.add_column()
        summary.add_column()
        summary.add_row("Total Contracts:", f"{total_contracts:,}")
        summary.add_row("Total Records:", f"{total_records:,}")
        summary.add_row("Average Days per Contract:", f"{avg_days:.1f}")

        # Find missing contracts
        found_symbols = [d['symbol'] for d in self.contracts]
        missing_symbols = self._find_missing_contracts(all_symbols, found_symbols)
        
        missing_table = None
        if missing_symbols:
            missing_table = Table(
                title=f"Missing Contracts ({self.interval_value} {self.interval_unit})",
                show_header=True,
                header_style="bold red",
                border_style="red"
            )
            missing_table.add_column("Symbol", style="red")
            for symbol in missing_symbols:
                missing_table.add_row(symbol)

        # Display results
        self.console.print("\n")
        self.console.print(Panel(table, title="Contract Details", border_style="blue"))
        self.console.print("\n")
        self.console.print(Panel(summary, title="Summary Statistics", border_style="green"))
        
        if missing_table:
            self.console.print("\n")
            self.console.print(Panel(missing_table, title="Missing Contracts", border_style="red"))
            
    def fetch_missing_contracts(self, missing_symbols):
        """Fetch data for missing contracts."""
        if not missing_symbols:
            return
            
        print("\nFetching data for missing contracts...")
        
        # Authenticate with TradeStation before fetching
        if not self.fetcher.ts_agent.authenticate():
            print("Failed to authenticate with TradeStation API. Cannot fetch missing contracts.")
            return
            
        print("Successfully authenticated with TradeStation API")
        
        # Set up logging to capture errors
        log_capture = []
        class LogCaptureHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        log_handler = LogCaptureHandler()
        log_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        
        # Counters for summary
        success_count = 0
        failed_count = 0
        
        # Process each missing symbol
        for symbol in missing_symbols:
            # Clear previous log messages
            log_capture.clear()
            
            print(f"\nProcessing {symbol}...")
            
            try:
                # Process the symbol with the correct interval parameters
                df = self.fetcher.fetch_data_since(
                    symbol=symbol,
                    interval=self.interval_value,
                    unit=self.interval_unit,
                    start_date=None,  # Let it use the default start date from config
                    end_date=None     # Let it fetch up to current date
                )
                
                # Check if we got data and save it to the database
                if df is not None and not df.empty:
                    # Delete any existing data for this symbol and interval
                    self.fetcher.delete_existing_data(symbol, self.interval_value, self.interval_unit)
                    # Save the new data
                    self.fetcher.save_to_db(df)
                    print(f"Successfully fetched and saved data for {symbol}")
                    success_count += 1
                else:
                    print(f"No data returned for {symbol}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                failed_count += 1
        
        # Remove the handler
        logging.getLogger().removeHandler(log_handler)
        
        # Print summary
        print("\nFetch Summary:")
        print(f"Successfully fetched: {success_count} contracts")
        print(f"Failed to fetch: {failed_count} contracts")

    def _get_available_intervals(self) -> List[Dict[str, Any]]:
        """Get available intervals from the database."""
        try:
            query = f"""
                SELECT DISTINCT interval_value, interval_unit, COUNT(*) as count
                FROM market_data 
                WHERE symbol LIKE '{self.base_symbol}%'
                GROUP BY interval_value, interval_unit
                ORDER BY interval_value, interval_unit
            """
            df = self.fetcher.conn.execute(query).fetchdf()
            
            return [
                {
                    'value': row['interval_value'],
                    'unit': row['interval_unit'],
                    'count': row['count']
                }
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logging.error(f"Error getting available intervals: {e}")
            return []

def main():
    """Main function to analyze futures contracts."""
    parser = argparse.ArgumentParser(description='Analyze futures contracts data')
    parser.add_argument('base_symbol', help='Base symbol (e.g., VX for VIX futures)')
    parser.add_argument('--start-year', type=int, default=2004, help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2025, help='End year for analysis')
    parser.add_argument('--interval-value', type=int, default=1, help='Interval value (e.g., 1 for daily, 15 for 15-minute)')
    parser.add_argument('--interval-unit', default='daily', choices=['daily', 'minute', 'hour'], help='Interval unit (use "daily" for daily data)')
    parser.add_argument('--missing-only', action='store_true', help='Show only missing contracts')
    parser.add_argument('--fetch-missing', action='store_true', help='Fetch data for missing contracts')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FuturesContractAnalyzer(
        base_symbol=args.base_symbol,
        start_year=args.start_year,
        end_year=args.end_year,
        interval_value=args.interval_value,
        interval_unit=args.interval_unit
    )
    
    # Analyze contracts
    analyzer.analyze()
    
    # If fetch_missing is specified, fetch data for missing contracts
    if args.fetch_missing:
        # Get missing contracts by comparing with contracts that have data
        found_symbols = [contract['symbol'] for contract in analyzer.contracts]
        missing_symbols = analyzer._find_missing_contracts(
            analyzer._generate_contract_symbols(),
            found_symbols
        )
        
        if missing_symbols:
            # Fetch data for missing contracts
            analyzer.fetch_missing_contracts(missing_symbols)
        else:
            print("No missing contracts to fetch.")

if __name__ == "__main__":
    main() 