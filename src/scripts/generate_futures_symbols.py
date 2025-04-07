#!/usr/bin/env python
"""
Generate Futures Symbols Script

This script generates futures contract symbols for quarterly contracts (March, June, September, December).
It creates symbols in the format like ESH03 (ES March 2003) through the current date.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/futures_symbols.log')
    ]
)
logger = logging.getLogger(__name__)

class FuturesSymbolGenerator:
    """Class to generate futures contract symbols."""
    
    # Month codes for quarterly contracts
    QUARTERLY_MONTHS = {
        3: 'H',  # March
        6: 'M',  # June
        9: 'U',  # September
        12: 'Z'  # December
    }
    
    def __init__(self, config_path=None):
        """Initialize the futures symbol generator."""
        self.config = self._load_config(config_path) if config_path else {}
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path):
        """Load the configuration from the YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def generate_symbols(self, base_symbol, start_year, start_month=None):
        """
        Generate futures contract symbols from start date to current date.
        
        Args:
            base_symbol: Base symbol (e.g., 'ES' for S&P 500 futures)
            start_year: Year to start generating symbols from
            start_month: Optional month to start from (1-12)
            
        Returns:
            List of generated symbols
        """
        symbols = []
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # If no start month specified, use January
        if start_month is None:
            start_month = 1
            
        # Generate symbols for each year and quarter
        for year in range(start_year, current_year + 1):
            # For the start year, only include months after start_month
            months = self.QUARTERLY_MONTHS.keys()
            if year == start_year:
                months = [m for m in months if m >= start_month]
            # For the current year, only include months up to current_month
            elif year == current_year:
                months = [m for m in months if m <= current_month]
                
            for month in months:
                month_code = self.QUARTERLY_MONTHS[month]
                # Format year as 2 digits
                year_code = str(year)[-2:]
                symbol = f"{base_symbol}{month_code}{year_code}"
                symbols.append(symbol)
                
        return symbols
    
    def get_active_contracts(self, base_symbol, num_contracts):
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
        
        # Get all possible contracts for the current year and next year
        contracts = []
        for year in [current_year, current_year + 1]:
            for month, code in self.QUARTERLY_MONTHS.items():
                # Skip past months in current year
                if year == current_year and month < current_month:
                    continue
                year_code = str(year)[-2:]
                symbol = f"{base_symbol}{code}{year_code}"
                contracts.append(symbol)
        
        # Sort contracts by expiration (assuming quarterly pattern)
        # This is a simple sort - in production you might want to use actual expiration dates
        contracts.sort()
        
        # Return the requested number of contracts
        return contracts[:num_contracts]
    
    def update_config_with_symbols(self, base_symbol, symbols, output_path, description=None, exchange=None, calendar=None, num_active_contracts=4):
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
        # Create a new config if it doesn't exist
        if not self.config:
            self.config = {
                'futures': [],
                'equities': [],
                'settings': {
                    'default_start_date': '2003-01-01',
                    'data_frequencies': [
                        {'name': '1min', 'interval': 1, 'unit': 'minute'},
                        {'name': '15min', 'interval': 15, 'unit': 'minute'},
                        {'name': 'daily', 'interval': 1, 'unit': 'day'}
                    ],
                    'holiday_calendars': {}
                }
            }
        
        # Ensure the futures section exists
        if 'futures' not in self.config:
            self.config['futures'] = []
        
        # Get active contracts
        active_contracts = self.get_active_contracts(base_symbol, num_active_contracts)
        
        # Check if the base symbol already exists in the config
        base_symbol_exists = False
        for future in self.config['futures']:
            if future.get('base_symbol') == base_symbol:
                base_symbol_exists = True
                # Update the existing entry
                future['num_active_contracts'] = num_active_contracts
                break
        
        # If the base symbol doesn't exist, add it
        if not base_symbol_exists:
            # Find the start year and month from the symbols
            start_year = int(symbols[0][-2:]) if symbols else datetime.now().year
            start_month = 1  # Default to January
            
            # Create a new entry for the base symbol
            new_entry = {
                'base_symbol': base_symbol,
                'frequencies': ['1min', '15min', 'daily'],
                'description': description or f'{base_symbol} Futures',
                'exchange': exchange or 'CME',
                'calendar': calendar or 'US',
                'num_active_contracts': num_active_contracts,
                'historical_contracts': {
                    'start_year': start_year,
                    'start_month': start_month,
                    'patterns': ['H', 'M', 'U', 'Z']  # March, June, September, December
                }
            }
            self.config['futures'].append(new_entry)
        
        # Ensure the settings section exists with data_frequencies
        if 'settings' not in self.config:
            self.config['settings'] = {}
        
        if 'data_frequencies' not in self.config['settings']:
            self.config['settings']['data_frequencies'] = [
                {'name': '1min', 'interval': 1, 'unit': 'minute'},
                {'name': '15min', 'interval': 15, 'unit': 'minute'},
                {'name': 'daily', 'interval': 1, 'unit': 'day'}
            ]
        
        # Save the updated config
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Updated configuration for {base_symbol} in {output_path}")
            logger.info(f"Active contracts: {', '.join(active_contracts)}")
        except Exception as e:
            logger.error(f"Error saving symbols to config: {e}")
            raise

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Generate futures contract symbols for configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate symbols for ES starting from 2020
  python generate_futures_symbols.py --base-symbol ES --start-year 2020
  
  # Generate symbols for NQ starting from March 2021
  python generate_futures_symbols.py --base-symbol NQ --start-year 2021 --start-month 3
  
  # Use an existing config file and save to a custom path
  python generate_futures_symbols.py --base-symbol CL --start-year 2020 --config existing_config.yaml --output custom_output.yaml
  
  # Add description and exchange information
  python generate_futures_symbols.py --base-symbol ES --start-year 2020 --description "E-mini S&P 500 Futures" --exchange CME
        """
    )
    parser.add_argument('--base-symbol', type=str, required=True, help='Base symbol for futures contracts (e.g., ES, NQ, CL)')
    parser.add_argument('--start-year', type=int, required=True, help='Year to start generating symbols from')
    parser.add_argument('--start-month', type=int, choices=range(1, 13), help='Month to start from (1-12)')
    parser.add_argument('--output', default='config/market_symbols.yaml', help='Output configuration file path (default: config/market_symbols.yaml)')
    parser.add_argument('--config', help='Path to existing configuration file to merge with')
    parser.add_argument('--description', help='Description of the futures contract')
    parser.add_argument('--exchange', help='Exchange where the futures contract trades')
    parser.add_argument('--calendar', help='Holiday calendar to use (US, EU, ASIA)')
    parser.add_argument('--num-active-contracts', type=int, default=4, help='Number of active contracts to track (default: 4)')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    generator = FuturesSymbolGenerator(args.config)
    symbols = generator.generate_symbols(args.base_symbol, args.start_year, args.start_month)
    
    generator.update_config_with_symbols(
        args.base_symbol, 
        symbols, 
        args.output, 
        args.description, 
        args.exchange, 
        args.calendar,
        args.num_active_contracts
    )

if __name__ == '__main__':
    main() 