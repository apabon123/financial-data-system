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
        logging.FileHandler('futures_symbols.log')
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
    
    def save_symbols_to_config(self, symbols, output_path):
        """
        Save generated symbols to a YAML configuration file.
        
        Args:
            symbols: List of symbols to save
            output_path: Path to save the configuration file
        """
        config = {
            'futures': [
                {
                    'symbol': symbol,
                    'base_symbol': symbol[:2],  # Extract base symbol (e.g., 'ES')
                    'update_frequency': 'daily'  # Default to daily updates
                }
                for symbol in symbols
            ],
            'settings': {
                'default_update_frequency': 'daily',
                'default_start_date': '2003-01-01'
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved {len(symbols)} symbols to {output_path}")
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
        """
    )
    parser.add_argument('--base-symbol', type=str, required=True, help='Base symbol for futures contracts (e.g., ES, NQ, CL)')
    parser.add_argument('--start-year', type=int, required=True, help='Year to start generating symbols from')
    parser.add_argument('--start-month', type=int, choices=range(1, 13), help='Month to start from (1-12)')
    parser.add_argument('--output', default='config/market_symbols.yaml', help='Output configuration file path (default: config/market_symbols.yaml)')
    parser.add_argument('--config', help='Path to existing configuration file to merge with')
    
    args = parser.parse_args()
    
    generator = FuturesSymbolGenerator(args.config)
    symbols = generator.generate_symbols(args.base_symbol, args.start_year, args.start_month)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generator.save_symbols_to_config(symbols, args.output)

if __name__ == '__main__':
    import argparse
    main() 