#!/usr/bin/env python
"""
List Market Symbols Script

This script lists all indices and equities from the configuration file, showing:
- Symbol
- Description
- Exchange
- Type (Index/Stock/ETF)
- Available frequencies
- Data availability
- Missing intervals
"""

import os
import sys
import yaml
import logging
import duckdb
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List, Dict, Any
import argparse

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get database path from environment variable
DATA_DIR = os.getenv('DATA_DIR', './data')
DEFAULT_DB_PATH = os.path.join(DATA_DIR, 'financial_data.duckdb')

class MarketSymbolLister:
    """Lists market symbols (indices, equities, futures) and their details."""
    
    def __init__(self, config_path=None, db_path=None):
        """Initialize the market symbol lister."""
        self.config = self._load_config(config_path) if config_path else {}
        self.db_path = db_path or DEFAULT_DB_PATH
        self.console = Console()
        self.conn = self._connect_database()
        
    def _connect_database(self):
        """Connect to the DuckDB database."""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to the database
            conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def _load_config(self, config_path):
        """Load the configuration from the YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _get_available_intervals(self, symbol: str) -> List[str]:
        """Get available intervals for a symbol from the database."""
        try:
            query = f"""
            SELECT DISTINCT interval_value, interval_unit
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchall()
            if not result:
                return []
            
            intervals = []
            for interval_value, interval_unit in result:
                if interval_unit == 'minute':
                    intervals.append(f"{interval_value}-minute")
                else:
                    intervals.append(interval_unit)
            return sorted(intervals)
        except Exception as e:
            logger.error(f"Error getting intervals for {symbol}: {e}")
            return []
    
    def _get_data_date_range(self, symbol: str) -> tuple:
        """Get the date range of available data for a symbol."""
        try:
            query = f"""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchone()
            if result and result[0] and result[1]:
                return result[0], result[1]
            return None, None
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None, None
    
    def _find_missing_intervals(self, symbol: str, configured_intervals: List[str]) -> List[str]:
        """Find intervals that are configured but missing from the database."""
        available_intervals = self._get_available_intervals(symbol)
        return [interval for interval in configured_intervals if interval not in available_intervals]
    
    def _get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a symbol."""
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT timestamp::DATE) as trading_days,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchone()
            if result:
                return {
                    'total_rows': result[0],
                    'trading_days': result[1],
                    'first_date': result[2],
                    'last_date': result[3]
                }
            return {
                'total_rows': 0,
                'trading_days': 0,
                'first_date': None,
                'last_date': None
            }
        except Exception as e:
            logger.error(f"Error getting statistics for {symbol}: {e}")
            return {
                'total_rows': 0,
                'trading_days': 0,
                'first_date': None,
                'last_date': None
            }
    
    def _display_symbol_details(self, symbol: dict, statistics: Dict[str, Any], missing_intervals: List[str], symbol_type: str):
        """Display details for a symbol."""
        # Determine title and style based on type
        if symbol_type == 'index':
            title_style = "bold blue"
            title_prefix = "Index"
        elif symbol_type == 'equity':
            title_style = "bold green"
            title_prefix = "Equity"
        elif symbol_type == 'future':
            title_style = "bold magenta"
            title_prefix = "Future Base"
        else:
            title_style = "bold white"
            title_prefix = "Symbol"

        symbol_key = symbol.get('symbol') or symbol.get('base_symbol') # Futures use base_symbol

        # Create a table for the symbol details
        table = Table(title=f"[{title_style}]{title_prefix}: {symbol_key}[/{title_style}]")

        # Add columns
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Add rows
        table.add_row("Description", symbol.get('description', 'N/A'))
        table.add_row("Exchange", symbol.get('exchange', 'N/A'))
        if symbol_type != 'future': # Type is implicit for futures base
             table.add_row("Type", symbol.get('type', 'N/A'))

        # Handle potential missing data or stats lookup failure
        if statistics and statistics['first_date']:
            table.add_row("Configured Frequencies", ", ".join(symbol.get('frequencies', [])))
            table.add_row("Available Frequencies", ", ".join(self._get_available_intervals(symbol_key)))
            table.add_row("Missing Frequencies", ", ".join(missing_intervals) if missing_intervals else "None")
            table.add_row("Total Rows (DB)", str(statistics['total_rows']))
            table.add_row("Trading Days (DB)", str(statistics['trading_days']))
            table.add_row("First Date (DB)", str(statistics['first_date']) if statistics['first_date'] else "N/A")
            table.add_row("Last Date (DB)", str(statistics['last_date']) if statistics['last_date'] else "N/A")
        else:
            table.add_row("Configured Frequencies", ", ".join(symbol.get('frequencies', [])))
            table.add_row("Database Info", "[yellow]No data found in market_data[/yellow]")

        if symbol_type == 'future':
             table.add_row("Expiry Rule", str(symbol.get('expiry_rule', 'N/A')))
             table.add_row("Active Contracts Config", str(symbol.get('num_active_contracts', 'N/A')))

        # Display the table
        self.console.print(table)
        self.console.print()

    def list_symbols(self, mode='overview', symbol_type_filter='all'):
        """List market symbols based on the type filter."""
        if not self.config:
            logger.error("No configuration loaded")
            return

        self.console.print(f"[bold]Listing Symbols (Type: {symbol_type_filter.capitalize()}, Mode: {mode})[/bold]")
        self.console.print("---")

        symbol_types_to_process = []
        if symbol_type_filter == 'all':
            symbol_types_to_process = ['index', 'equity', 'future']
        else:
            symbol_types_to_process = [symbol_type_filter]

        for symbol_type in symbol_types_to_process:
            config_key = f"{symbol_type}es" # e.g., indices, equities, futures
            if config_key in self.config:
                self.console.print(Panel(f"{symbol_type.capitalize()} Symbols", style=f"bold {self._get_type_color(symbol_type)}"))
                for symbol_data in self.config[config_key]:
                    symbol_key = symbol_data.get('symbol') or symbol_data.get('base_symbol')
                    if not symbol_key:
                        logger.warning(f"Skipping entry in '{config_key}' section due to missing symbol/base_symbol.")
                        continue
                    
                    # Note: Statistics might be less relevant for future base symbols
                    # We query based on the base symbol, which might not exist directly in market_data
                    statistics = self._get_symbol_statistics(symbol_key)
                    missing_intervals = self._find_missing_intervals(symbol_key, symbol_data.get('frequencies', []))
                    self._display_symbol_details(symbol_data, statistics, missing_intervals, symbol_type)

        self.console.print("--- End of Listing ---")
        
    def _get_type_color(self, symbol_type):
        """Helper to get color based on symbol type."""
        if symbol_type == 'index': return "blue"
        if symbol_type == 'equity': return "green"
        if symbol_type == 'future': return "magenta"
        return "white"

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='List market symbols and their details from configuration.')
    parser.add_argument('--config', type=str, default='config/market_symbols.yaml', help='Path to configuration file')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to database file')
    parser.add_argument('--mode', choices=['overview', 'detailed'], default='overview',
                      help='Display mode (currently affects title only)')
    parser.add_argument('--type', type=str, choices=['all', 'equity', 'index', 'future'], default='all',
                        help='Filter symbols by type (equity, index, future, or all)')
    args = parser.parse_args()

    # Validate config path
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Create the lister
    lister = MarketSymbolLister(config_path=args.config, db_path=args.db_path)

    # List the symbols based on type filter
    lister.list_symbols(mode=args.mode, symbol_type_filter=args.type)

if __name__ == '__main__':
    main() 