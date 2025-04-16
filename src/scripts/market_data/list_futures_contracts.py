#!/usr/bin/env python
"""
List Futures Contracts Script

This script lists all futures contracts from the configuration file, showing:
- Base symbol
- First contract
- Last contract
- Missing contracts
- Contract patterns
- Number of active contracts
- Exchange and description
- Available intervals
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
from typing import List
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

class FuturesContractLister:
    """Lists futures contracts and their details."""
    
    def __init__(self, config_path=None, db_path=None):
        """Initialize the futures contract lister."""
        self.config = self._load_config(config_path) if config_path else {}
        self.db_path = db_path or DEFAULT_DB_PATH
        self.configured_intervals = ['daily', '1-minute', '15-minute']
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
    
    def _get_month_order(self, month_code: str) -> int:
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
        return month_order.get(month_code, 0)
    
    def _sort_contracts_chronologically(self, contracts: List[str]) -> List[str]:
        """Sort contracts chronologically by year and month code."""
        # Separate standard contracts from special ones
        standard_contracts = []
        special_contracts = []
        
        for contract in contracts:
            # Check if contract follows the standard pattern (ends with 2 digits for year)
            if len(contract) >= 3 and contract[-2:].isdigit():
                standard_contracts.append(contract)
            else:
                special_contracts.append(contract)
        
        # Sort standard contracts chronologically
        sorted_standard = sorted(standard_contracts, key=lambda x: (
            int(x[-2:]),  # Year
            self._get_month_order(x[-3])  # Month
        ))
        
        # Return sorted standard contracts followed by special contracts
        return sorted_standard + special_contracts
    
    def _generate_contract_symbols(self, base_symbol: str, start_year: int, patterns: list) -> list:
        """Generate all possible contract symbols for the given base symbol and years."""
        symbols = []
        current_year = datetime.now().year
        
        # Get the start date from the configuration
        start_date = None
        for future in self.config['futures']:
            if future['base_symbol'] == base_symbol:
                start_date = datetime.strptime(future['start_date'], '%Y-%m-%d').date()
                break
        
        # If start date is in the future, use current year
        if start_date and start_date > datetime.now().date():
            current_year = start_date.year
        
        # Get start month and year from the configuration
        start_month = None
        start_year_from_config = None
        
        if start_date:
            start_month = start_date.month
            start_year_from_config = start_date.year
            
            # Map month number to month code
            month_to_code = {
                1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
            }
            start_month_code = month_to_code.get(start_month, None)
        
        # Generate contracts from start_year to current_year
        for year in range(start_year, current_year + 1):
            for month in patterns:
                # Skip contracts before the start month and year
                if start_month_code and start_year_from_config:
                    # If this is the start year, only include months on or after the start month
                    if year == start_year_from_config:
                        # Get the month code for the current pattern
                        current_month_code = month
                        # Only include if the current month is on or after the start month
                        if self._get_month_order(current_month_code) >= self._get_month_order(start_month_code):
                            symbols.append(f"{base_symbol}{month}{str(year)[-2:]}")
                    # For years after the start year, include all months
                    elif year > start_year_from_config:
                        symbols.append(f"{base_symbol}{month}{str(year)[-2:]}")
                else:
                    # If no start date is specified, include all contracts
                    symbols.append(f"{base_symbol}{month}{str(year)[-2:]}")
        
        return symbols
    
    def _get_existing_contracts(self, base_symbol: str) -> list:
        """Get list of contracts that exist in the database."""
        try:
            query = f"""
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE '{base_symbol}%'
                ORDER BY symbol
            """
            result = self.conn.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error getting existing contracts for {base_symbol}: {e}")
            return []
    
    def _get_available_intervals_for_symbol(self, base_symbol: str) -> List[str]:
        try:
            query = f"""
            SELECT DISTINCT interval_value, interval_unit
            FROM market_data
            WHERE symbol LIKE '{base_symbol}%'
            """
            result = self.conn.execute(query).fetchall()
            if not result:
                return []
            
            intervals = []
            for row in result:
                interval_value, interval_unit = row
                if interval_value == 1 and interval_unit == 'daily':
                    intervals.append('daily')
                else:
                    intervals.append(f"{interval_value}-{interval_unit}")
            return sorted(intervals)
        except Exception as e:
            print(f"Error getting intervals for {base_symbol}: {e}")
            return []
    
    def _find_missing_contracts(self, all_contracts: list, existing_contracts: list) -> list:
        """Find contracts that are missing from the database."""
        return [contract for contract in all_contracts if contract not in existing_contracts]
    
    def _find_missing_intervals(self, symbol: str, configured_intervals: List[str]) -> List[str]:
        """Find which intervals are missing for a given symbol."""
        missing_intervals = []
        
        # Convert configured intervals to the format used in the database
        db_intervals = []
        for interval in configured_intervals:
            if interval == '1min':
                db_intervals.append((1, 'minute'))
            elif interval == '15min':
                db_intervals.append((15, 'minute'))
            elif interval == 'daily':
                db_intervals.append((1, 'day'))
        
        for interval_value, interval_unit in db_intervals:
            # Check if data exists for this interval
            query = f"""
                SELECT COUNT(*) as count
                FROM market_data
                WHERE symbol = '{symbol}'
                AND interval_value = {interval_value}
                AND (
                    interval_unit = '{interval_unit}' 
                    OR (interval_unit = 'day' AND '{interval_unit}' = 'daily')
                    OR (interval_unit = 'daily' AND '{interval_unit}' = 'day')
                )
            """
            
            try:
                result = self.conn.execute(query).fetchone()
                if result[0] == 0:
                    # Convert back to display format
                    if interval_unit == 'minute':
                        missing_intervals.append(f"{interval_value}-minute")
                    elif interval_unit == 'day':
                        missing_intervals.append('daily')
            except Exception as e:
                self.console.print(f"[red]Error checking interval {interval_value}{interval_unit} for {symbol}: {str(e)}[/red]")
                continue
                
        return missing_intervals
    
    def _format_interval(self, interval_value: int, interval_unit: str) -> str:
        """Format interval for display."""
        if interval_unit == 'minute':
            return f"{interval_value}min"
        elif interval_unit == 'hour':
            return f"{interval_value}h"
        elif interval_unit == 'day':
            return "daily"
        else:
            return f"{interval_value}{interval_unit}"
    
    def _display_missing_intervals(self, base_symbol: str, missing_intervals_by_contract: dict):
        """Display a table of contracts with missing intervals."""
        interval_table = Table(
            title=f"Contracts with Missing Intervals for {base_symbol}",
            show_header=True,
            header_style="bold yellow",
            border_style="yellow"
        )
        interval_table.add_column("Contract", style="cyan")
        interval_table.add_column("Missing Intervals", style="yellow")
        
        # Sort contracts chronologically
        sorted_contracts = self._sort_contracts_chronologically(list(missing_intervals_by_contract.keys()))
        
        for contract in sorted_contracts:
            missing_intervals = missing_intervals_by_contract[contract]
            missing_intervals_str = ', '.join(missing_intervals)
            interval_table.add_row(contract, missing_intervals_str)
        
        self.console.print("\n")
        self.console.print(Panel(interval_table, title=f"Missing Intervals for {base_symbol}", border_style="yellow"))
    
    def _display_missing_contracts(self, base_symbol: str, missing_contracts: List[str]):
        """Display a table of missing contracts."""
        missing_table = Table(
            title=f"Missing Contracts for {base_symbol}",
            show_header=True,
            header_style="bold red",
            border_style="red"
        )
        missing_table.add_column("Contract", style="red")
        missing_table.add_column("Year", style="yellow")
        missing_table.add_column("Month", style="yellow")
        missing_table.add_column("Missing Intervals", style="red")
        
        for contract in missing_contracts:
            year = contract[-2:]
            month = contract[-3]
            month_name = {
                'F': 'January', 'G': 'February', 'H': 'March',
                'J': 'April', 'K': 'May', 'M': 'June',
                'N': 'July', 'Q': 'August', 'U': 'September',
                'V': 'October', 'X': 'November', 'Z': 'December'
            }.get(month, month)
            
            missing_table.add_row(contract, f"20{year}", month_name, "All intervals")
        
        self.console.print("\n")
        self.console.print(Panel(missing_table, title=f"Missing Contracts for {base_symbol}", border_style="red"))
    
    def _get_contract_date_range(self, symbol: str) -> tuple:
        """Get the first and last date for a contract from the database."""
        try:
            query = f"""
                SELECT MIN(timestamp) as first_date, MAX(timestamp) as last_date
                FROM market_data
                WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchone()
            return result[0], result[1] if result and result[0] else (None, None)
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None, None

    def _validate_contract_dates(self, base_symbol: str, config_start_date: str) -> dict:
        """Validate contract dates against configuration start date."""
        date_issues = {}
        config_start = datetime.strptime(config_start_date, '%Y-%m-%d').date()
        
        existing_contracts = self._get_existing_contracts(base_symbol)
        for contract in existing_contracts:
            first_date, last_date = self._get_contract_date_range(contract)
            if first_date and first_date.date() < config_start:
                date_issues[contract] = {
                    'first_date': first_date,
                    'config_start': config_start,
                    'issue': 'Data exists before configured start date'
                }
        
        return date_issues

    def _validate_frequencies(self, base_symbol: str, configured_frequencies: List[str]) -> dict:
        """Validate frequencies against configuration and check for gaps."""
        frequency_issues = {}
        
        # Convert configured frequencies to database format
        db_frequencies = []
        for freq in configured_frequencies:
            if freq == '1min':
                db_frequencies.append((1, 'minute'))
            elif freq == '15min':
                db_frequencies.append((15, 'minute'))
            elif freq == 'daily':
                db_frequencies.append((1, 'day'))
        
        existing_contracts = self._get_existing_contracts(base_symbol)
        for contract in existing_contracts:
            contract_issues = []
            
            # Check each configured frequency
            for interval_value, interval_unit in db_frequencies:
                query = f"""
                    SELECT COUNT(*) as count,
                           MIN(timestamp) as first_date,
                           MAX(timestamp) as last_date
                    FROM market_data
                    WHERE symbol = '{contract}'
                    AND interval_value = {interval_value}
                    AND interval_unit = '{interval_unit}'
                """
                try:
                    result = self.conn.execute(query).fetchone()
                    if result[0] == 0:
                        contract_issues.append(f"Missing {interval_value}{interval_unit} data")
                    else:
                        # Check for gaps in the data
                        gap_query = f"""
                            WITH dates AS (
                                SELECT timestamp,
                                       LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                                FROM market_data
                                WHERE symbol = '{contract}'
                                AND interval_value = {interval_value}
                                AND interval_unit = '{interval_unit}'
                            )
                            SELECT prev_timestamp, timestamp
                            FROM dates
                            WHERE prev_timestamp IS NOT NULL
                            AND timestamp - prev_timestamp > INTERVAL '{interval_value} {interval_unit}'
                        """
                        gaps = self.conn.execute(gap_query).fetchall()
                        if gaps:
                            contract_issues.append(f"Found {len(gaps)} gaps in {interval_value}{interval_unit} data")
                except Exception as e:
                    logger.error(f"Error checking frequency {interval_value}{interval_unit} for {contract}: {e}")
                    continue
            
            if contract_issues:
                frequency_issues[contract] = contract_issues
        
        return frequency_issues

    def _display_contract_details(
        self,
        base_symbol: str,
        future: dict,
        existing_contracts: List[str],
        missing_contracts: List[str],
        date_issues: dict,
        frequency_issues: dict
    ):
        """Display detailed analysis of contract data."""
        # Create main overview table
        overview_table = Table(
            title=f"Overview for {base_symbol}",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="green")
        
        overview_table.add_row("Description", future['description'])
        overview_table.add_row("Exchange", future['exchange'])
        overview_table.add_row("Start Date", future['start_date'])
        overview_table.add_row("Configured Frequencies", ", ".join(future['frequencies']))
        overview_table.add_row("Total Possible Contracts", str(len(existing_contracts) + len(missing_contracts)))
        overview_table.add_row("Existing Contracts", str(len(existing_contracts)))
        overview_table.add_row("Missing Contracts", str(len(missing_contracts)))
        
        self.console.print(overview_table)
        
        # Check for missing frequencies in existing contracts
        missing_freqs = {}
        for contract in existing_contracts:
            missing = self._find_missing_intervals(contract, future['frequencies'])
            if missing:
                missing_freqs[contract] = missing
        
        # Display missing frequencies if any
        if missing_freqs:
            freq_table = Table(
                title=f"Missing Frequencies for {base_symbol}",
                show_header=True,
                header_style="bold yellow",
                border_style="yellow"
            )
            
            freq_table.add_column("Contract", style="cyan")
            freq_table.add_column("Missing Frequencies", style="yellow")
            
            # Sort contracts chronologically
            sorted_contracts = self._sort_contracts_chronologically(list(missing_freqs.keys()))
            
            for contract in sorted_contracts:
                missing = missing_freqs[contract]
                freq_table.add_row(contract, ", ".join(missing))
            
            self.console.print(freq_table)
        
        # Display missing contracts if any
        if missing_contracts:
            missing_table = Table(
                title=f"Missing Contracts for {base_symbol}",
                show_header=True,
                header_style="bold red",
                border_style="red"
            )
            
            missing_table.add_column("Contract", style="cyan")
            
            # Sort missing contracts chronologically
            sorted_missing = self._sort_contracts_chronologically(missing_contracts)
            
            for contract in sorted_missing:
                missing_table.add_row(contract)
            
            self.console.print(missing_table)

    def _get_contract_statistics(self, symbol: str) -> dict:
        """Get detailed statistics for a contract."""
        try:
            # Get statistics using a compatible SQL query
            query = f"""
            SELECT 
                COUNT(DISTINCT strftime('%Y-%m-%d', timestamp)) as days,
                COUNT(*) as records,
                MAX(high) as high,
                MIN(low) as low,
                strftime('%Y-%m-%d', MIN(timestamp)) as first_date,
                strftime('%Y-%m-%d', MAX(timestamp)) as last_date
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchone()
            if result and result[0]:
                stats = result
                return {
                    'days': stats[0] or 0,
                    'records': stats[1] or 0,
                    'high': stats[2] or 'N/A',
                    'low': stats[3] or 'N/A',
                    'first_date': stats[4] or 'N/A',
                    'last_date': stats[5] or 'N/A'
                }
            return {
                'days': 0,
                'records': 0,
                'high': 'N/A',
                'low': 'N/A',
                'first_date': 'N/A',
                'last_date': 'N/A'
            }
        except Exception as e:
            logger.error(f"Error getting statistics for {symbol}: {str(e)}")
            return {
                'days': 0,
                'records': 0,
                'high': 'N/A',
                'low': 'N/A',
                'first_date': 'N/A',
                'last_date': 'N/A'
            }
    
    def _display_contract_statistics(self, base_symbol: str, existing_contracts: List[str]):
        """Display detailed statistics for each contract."""
        # Create statistics table
        stats_table = Table(
            title=f"Contract Statistics for {base_symbol}",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        stats_table.add_column("Contract", style="cyan")
        stats_table.add_column("Days", style="green", justify="right")
        stats_table.add_column("Records", style="green", justify="right")
        stats_table.add_column("High", style="yellow", justify="right")
        stats_table.add_column("Low", style="yellow", justify="right")
        stats_table.add_column("First Date", style="blue")
        stats_table.add_column("Last Date", style="blue")
        
        # Sort contracts chronologically
        sorted_contracts = self._sort_contracts_chronologically(existing_contracts)
        
        for contract in sorted_contracts:
            stats = self._get_contract_statistics(contract)
            
            # Format dates
            first_date = stats['first_date'] if stats['first_date'] != 'N/A' else 'N/A'
            last_date = stats['last_date'] if stats['last_date'] != 'N/A' else 'N/A'
            
            # Format high/low with 2 decimal places
            high = f"{stats['high']:.2f}" if stats['high'] != 'N/A' else 'N/A'
            low = f"{stats['low']:.2f}" if stats['low'] != 'N/A' else 'N/A'
            
            stats_table.add_row(
                contract,
                str(stats['days']),
                str(stats['records']),
                high,
                low,
                first_date,
                last_date
            )
        
        self.console.print(stats_table)
    
    def list_contracts(self, mode='overview'):
        """List all futures contracts and their details.
        
        Args:
            mode: Display mode, one of:
                - 'overview': Show overview and missing contracts (default)
                - 'statistics': Show detailed statistics for each contract
                - 'all': Show both overview and statistics
        """
        if not self.config or 'futures' not in self.config:
            self.console.print("[red]No futures configuration found[/red]")
            return
        
        for future in self.config['futures']:
            base_symbol = future['base_symbol']
            self.console.print(f"\n[bold cyan]Analyzing {base_symbol} - {future['description']}[/bold cyan]")
            
            # Generate all possible contracts
            all_contracts = self._generate_contract_symbols(
                base_symbol,
                future['historical_contracts']['start_year'],
                future['historical_contracts']['patterns']
            )
            
            # Get existing contracts
            existing_contracts = self._get_existing_contracts(base_symbol)
            
            # Find missing contracts
            missing_contracts = self._find_missing_contracts(all_contracts, existing_contracts)
            
            # Validate dates
            date_issues = self._validate_contract_dates(base_symbol, future['start_date'])
            
            # Display results based on mode
            if mode in ['overview', 'all']:
                self._display_contract_details(
                    base_symbol,
                    future,
                    existing_contracts,
                    missing_contracts,
                    date_issues,
                    {}  # Empty dict for frequency issues since we're not using it
                )
            
            if mode in ['statistics', 'all']:
                self._display_contract_statistics(base_symbol, existing_contracts)

def main():
    """Main function to list futures contracts."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='List futures contracts and their details')
    parser.add_argument('--mode', choices=['overview', 'statistics', 'all'], default='overview',
                        help='Display mode: overview (default), statistics, or all')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--db', type=str, help='Path to database file')
    args = parser.parse_args()
    
    # Get config path
    config_path = args.config or os.path.join(project_root, "config", "market_symbols.yaml")
    
    # Create lister
    lister = FuturesContractLister(config_path, args.db)
    
    # List contracts
    lister.list_contracts(mode=args.mode)

if __name__ == "__main__":
    main() 