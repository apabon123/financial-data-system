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
import subprocess

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)

# Import MarketDataFetcher
from src.scripts.market_data.fetch_market_data import MarketDataFetcher

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
METADATA_TABLE_NAME = "symbol_metadata" # Define constant

class FuturesContractAnalyzer:
    """Analyzes futures contracts data to identify gaps and missing contracts."""
    
    def __init__(
        self,
        base_symbol: str,
        end_year: int = 2025,
        interval_value: int = 1,
        interval_unit: str = "daily"
    ):
        """Initialize the analyzer."""
        self.base_symbol = base_symbol
        self.end_year = end_year
        self.interval_value = interval_value
        self.interval_unit = interval_unit # Store interval unit
        self.contracts = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Determine start_year from config['start_date']
        config_start_date_str = self.config.get('start_date')
        default_start_year = 2004
        if config_start_date_str:
            try:
                config_start_date = pd.to_datetime(config_start_date_str)
                self.start_year = config_start_date.year
                logging.info(f"Using start year {self.start_year} from config file start_date: {config_start_date_str}")
            except Exception as e:
                logging.warning(f"Could not parse start_date '{config_start_date_str}' from config: {e}. Defaulting start year to {default_start_year}.")
                self.start_year = default_start_year
        else:
            logging.warning(f"No start_date found in config for {base_symbol}. Defaulting start year to {default_start_year}.")
            self.start_year = default_start_year
        
        # Initialize fetcher with shared database path
        config_path = os.path.join(project_root, "config", "market_symbols.yaml")
        # Ensure db_path is passed correctly if needed by fetcher init
        self.fetcher = MarketDataFetcher(config_path=config_path, db_path=DEFAULT_DB_PATH) 
        # --- ADDED: Set Fetcher's logger level to INFO to reduce noise --- #
        self.fetcher.logger.setLevel(logging.INFO)
        # ------------------------------------------------------------------ #
        
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
        """Generate all possible contract symbols using MarketDataFetcher's method."""
        # Use the fetcher's method which respects exclude_contracts from YAML
        # Need to convert start/end year to datetime objects for the fetcher method
        start_dt = pd.Timestamp(f'{self.start_year}-01-01')
        end_dt = pd.Timestamp(f'{self.end_year}-12-31')
        
        # Ensure self.fetcher is initialized correctly
        if not hasattr(self, 'fetcher') or self.fetcher is None:
             # Initialize fetcher if it wasn't (should be in __init__ though)
             logging.warning("_generate_contract_symbols: Fetcher not initialized, attempting fallback init.")
             config_path = os.path.join(project_root, "config", "market_symbols.yaml")
             self.fetcher = MarketDataFetcher(config_path=config_path, db_path=DEFAULT_DB_PATH)

        try:
             return self.fetcher.generate_futures_contracts(
                 self.base_symbol,
                 start_date=start_dt,
                 end_date=end_dt
             )
        except Exception as e:
            logging.error(f"Error calling MarketDataFetcher.generate_futures_contracts: {e}")
            return [] # Return empty list on error

    def _get_data_table_for_symbol(self, symbol: str, interval_unit: str, interval_value: int) -> Optional[str]: # Add interval params
        """Query the metadata table to get the data table for a symbol/base symbol and interval."""
        try:
            # Try matching the exact symbol first (e.g., base_symbol)
            # This helper is primarily called with base_symbol, so prioritize that.
            query = f""" 
                SELECT data_table FROM {METADATA_TABLE_NAME} 
                WHERE base_symbol = ? AND interval_unit = ? AND interval_value = ?
                LIMIT 1
            """
            params = [symbol, interval_unit, interval_value]
            result = self.fetcher.conn.execute(query, params).fetchone()
            
            if result:
                return result[0]
            else:
                logging.warning(f"Metadata not found for {symbol} ({interval_value} {interval_unit}). Defaulting to 'market_data'.")
                return 'market_data' # Fallback
        except Exception as e:
            logging.error(f"Error querying {METADATA_TABLE_NAME} for {symbol} ({interval_value} {interval_unit}): {e}")
            return 'market_data' # Fallback on error

    def _get_contract_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get contract data from the database for the analyzer's interval."""
        try:
            # Determine the correct table for the base symbol and the analyzer's interval
            target_table = self._get_data_table_for_symbol(self.base_symbol, self.interval_unit, self.interval_value) 
            if not target_table:
                logging.error(f"Could not determine data table for base symbol {self.base_symbol}. Aborting data fetch for {symbol}.")
                return None

            # Use parameterized query for safety
            query = f""" 
                SELECT *
                FROM {target_table}
                WHERE symbol = ? 
                AND interval_value = ?
                AND interval_unit = ?
                ORDER BY timestamp ASC
            """
            params = [symbol, self.interval_value, self.interval_unit]
            df = self.fetcher.conn.execute(query, params).fetchdf()
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
        
        # Check what intervals are actually available in the database for the *specific* interval we care about
        target_table_for_check = self._get_data_table_for_symbol(self.base_symbol, self.interval_unit, self.interval_value)
        if not self._check_interval_exists(target_table_for_check, self.base_symbol, self.interval_unit, self.interval_value):
             logging.warning(f"No data found in table '{target_table_for_check}' for {self.base_symbol} with interval_value={self.interval_value}, interval_unit={self.interval_unit}")
             # Optionally, list all available intervals for the base symbol across all tables?
             # available_intervals = self._get_all_available_intervals_for_base(self.base_symbol)
             # logging.info(f"All available intervals found in metadata: {available_intervals}")
             return
            
        logging.info(f"Using interval: {self.interval_value} {self.interval_unit} in table '{target_table_for_check}'")
        
        # Generate all possible contract symbols (this part is independent of interval)
        all_symbols = self._generate_contract_symbols()
        
        # Collect data for each contract (using the analyzer's interval)
        # _get_contract_data already uses self.interval_unit/value and finds table via metadata
        self.contracts = []  
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
        """Fetch data for missing contracts FOR THE CURRENT ANALYZER INTERVAL using fetch_market_data.py."""
        if not missing_symbols:
            return

        # --- Use fetch_market_data.py to get missing contracts --- 
        fetcher_script_path = os.path.join(project_root, 'src', 'scripts', 'market_data', 'fetch_market_data.py')
        config_path = os.path.join(project_root, "config", "market_symbols.yaml") # Get standard config path
        db_path_to_use = DEFAULT_DB_PATH # Use the standard DB path
        
        print(f"\nFetching {len(missing_symbols)} missing contracts using {fetcher_script_path}...")
        
        success_count = 0
        failed_count = 0
        
        # Process each missing symbol by calling the script
        for symbol in missing_symbols:
            print(f"--- Attempting to fetch: {symbol} ({self.interval_value} {self.interval_unit}) ---")
            command = [
                sys.executable, # Use the current Python interpreter
                fetcher_script_path,
                '--symbol', symbol,
                '--interval-value', str(self.interval_value),
                '--interval-unit', self.interval_unit,
                '--force', # Force fetch for missing contracts
                '--config', config_path,
                '--db-path', db_path_to_use
            ]
            
            try:
                # Run the command
                result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
                # Print relevant output from the script (optional)
                # print("Fetcher Output:")
                # print(result.stdout)
                # print(result.stderr)
                print(f"Successfully fetched and processed {symbol}.")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Script execution failed for {symbol} with return code {e.returncode}")
                print("Stderr:")
                print(e.stderr)
                print("Stdout:")
                print(e.stdout)
                failed_count += 1
            except Exception as e:
                print(f"ERROR: An unexpected error occurred running subprocess for {symbol}: {e}")
                failed_count += 1
        
        print("\nFetch Summary:")
        print(f"Successfully fetched: {success_count} contracts")
        print(f"Failed to fetch: {failed_count} contracts")

    def _check_interval_exists(self, target_table: Optional[str], base_symbol: str, interval_unit: str, interval_value: int) -> bool:
        """Check if any data exists for the specific symbol and interval in the target table."""
        if not target_table:
            return False
        try:
            query = f"""
                SELECT 1
                FROM {target_table}
                WHERE symbol LIKE ?
                AND interval_unit = ?
                AND interval_value = ?
                LIMIT 1
            """
            params = [f"{base_symbol}%", interval_unit, interval_value]
            result = self.fetcher.conn.execute(query, params).fetchone()
            return result is not None
        except Exception as e:
            logging.error(f"Error checking interval existence for {base_symbol} in {target_table}: {e}")
            return False

    def _get_all_available_intervals_for_base(self, base_symbol: str) -> List[Dict[str, Any]]:
        """Get all available intervals for a base symbol from the metadata table."""
        try:
            query = f"SELECT interval_unit, interval_value, data_table, data_source FROM {METADATA_TABLE_NAME} WHERE base_symbol = ? ORDER BY interval_unit, interval_value"
            results = self.fetcher.conn.execute(query, [base_symbol]).fetchall()
            return [
                {'unit': r[0], 'value': r[1], 'table': r[2], 'source': r[3]} for r in results
            ]
        except Exception as e:
            logging.error(f"Error getting all available intervals for {base_symbol} from metadata: {e}")
            return []

def main():
    """Main function to analyze futures contracts."""
    parser = argparse.ArgumentParser(description='Analyze futures contracts data')
    parser.add_argument('base_symbol', help='Base symbol (e.g., VX for VIX futures)')
    parser.add_argument('--end-year', type=int, default=datetime.now().year, help='End year for analysis (defaults to current year)')
    parser.add_argument('--interval-value', '--interval_value', type=int, default=1, help='Interval value (e.g., 1 for daily, 15 for 15-minute)')
    parser.add_argument('--interval-unit', '--interval_unit', default='daily', choices=['daily', 'minute', 'hour'], help='Interval unit (use "daily" for daily data)')
    parser.add_argument('--missing-only', action='store_true', help='Show only missing contracts')
    parser.add_argument('--fetch-missing', action='store_true', help='Fetch data for missing contracts')
    parser.add_argument('--fetch-short', action='store_true', help='Fetch data for contracts with records below threshold')
    parser.add_argument('--short-threshold', type=int, default=100, help='Record count threshold for --fetch-short')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FuturesContractAnalyzer(
        base_symbol=args.base_symbol,
        end_year=args.end_year,
        interval_value=args.interval_value,
        interval_unit=args.interval_unit
    )
    
    # Analyze contracts
    analyzer.analyze()
    
    # --- Close DB connection before potentially spawning subprocesses --- #
    if analyzer.fetcher.conn:
        try:
            analyzer.fetcher.conn.close()
            logging.info("Closed analyzer database connection before fetching missing data.")
            analyzer.fetcher.conn = None # Prevent accidental reuse
        except Exception as e:
            logging.error(f"Error closing analyzer DB connection: {e}")
    # ----------------------------------------------------------------- #

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

    # If fetch_short is specified, fetch data for contracts below the threshold
    if args.fetch_short:
        short_contracts = [
            contract['symbol'] for contract in analyzer.contracts 
            if contract['records'] < args.short_threshold
        ]
        
        if short_contracts:
            print(f"\nFound {len(short_contracts)} contracts with fewer than {args.short_threshold} records.")
            # Re-use the fetch_missing_contracts method to force fetch these
            analyzer.fetch_missing_contracts(short_contracts)
        else:
            print(f"\nNo contracts found with fewer than {args.short_threshold} records.")

if __name__ == "__main__":
    main() 