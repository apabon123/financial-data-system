#!/usr/bin/env python
"""
Generate Continuous Contract Script

This script creates a continuous contract for futures by backadjusting historical data.
It supports multiple rollover methods and includes robust error detection for price discrepancies.

Features:
- Multiple rollover methods:
  * Volume-based: Rolls over when the next contract's volume exceeds the current contract's volume
    within 5 days of the current contract's expiration
  * Fixed: Rolls over one day before the current contract's expiration
- Price discrepancy detection: Identifies and logs suspicious price differences at rollover points
- Configurable price jump thresholds: Uses symbol-specific thresholds from config or defaults
- Comprehensive logging: Provides detailed information about the rollover process
- Force mode: Allows overwriting existing continuous contracts

Example:
    # Generate continuous contract for ES using volume-based rollover
    python generate_continuous_contract.py --symbol ES --output ES_backadj --rollover-method volume

    # Generate continuous contract for ES using fixed rollover (one day before expiration)
    python generate_continuous_contract.py --symbol ES --output ES_backadj --rollover-method fixed

    # Force rebuild of an existing continuous contract
    python generate_continuous_contract.py --symbol ES --output ES_backadj --force
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContinuousContractGenerator:
    """Class to generate continuous contracts for futures.
    
    This class handles the generation of continuous contracts by:
    1. Finding appropriate rollover dates between contracts
    2. Applying price adjustments to maintain continuity
    3. Detecting and logging suspicious price discrepancies
    4. Saving the resulting continuous contract to the database
    
    The class supports multiple rollover methods:
    - Volume-based: Rolls over when the next contract's volume exceeds the current
    - Fixed: Rolls over one day before the current contract's expiration
    """
    
    # Month codes for quarterly contracts
    QUARTERLY_MONTHS = {
        3: 'H',  # March
        6: 'M',  # June
        9: 'U',  # September
        12: 'Z'  # December
    }
    
    # Expiry dates for ES futures (approximate, in days before the end of the month)
    EXPIRY_DAYS = {
        'H': 5,  # March
        'M': 5,  # June
        'U': 5,  # September
        'Z': 5   # December
    }
    
    def __init__(self, db_path='./data/financial_data.duckdb', config_path='config/market_symbols.yaml'):
        """Initialize the continuous contract generator.
        
        Args:
            db_path (str): Path to the DuckDB database
            config_path (str): Path to the configuration file
        """
        self.db_path = db_path
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _load_config(self, config_path):
        """Load the configuration from the YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def get_contract_symbols(self, base_symbol, start_date=None, end_date=None, interval='15minute'):
        """Get all contract symbols for a base symbol with a specific interval."""
        try:
            # Parse the interval value and unit
            if interval == '1daily':
                interval_value = 1
                interval_unit = 'daily'
            else:
                interval_parts = interval.split('minute') if 'minute' in interval else interval.split('daily')
                interval_value = int(interval_parts[0])
                interval_unit = 'minute' if 'minute' in interval else 'daily'
            
            # Build the query
            query = f"""
            SELECT DISTINCT symbol, MIN(timestamp) as start_date, MAX(timestamp) as end_date
            FROM market_data
            WHERE symbol LIKE '{base_symbol}%'
            AND interval_value = {interval_value}
            AND interval_unit = '{interval_unit}'
            """
            
            # Add date filters if provided
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            # Add ordering
            query += " GROUP BY symbol ORDER BY symbol"
            
            # Execute the query
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                logger.warning(f"No contracts found for {base_symbol} with interval {interval}")
                return pd.DataFrame()
            
            # Calculate expiry dates for each contract
            expiry_dates = []
            for symbol in result['symbol']:
                expiry_date = self.calculate_expiry_date(symbol)
                expiry_dates.append(expiry_date)
            
            # Add expiry dates to the result
            result['expiry_date'] = expiry_dates
            
            # Sort by expiry date
            result = result.sort_values('expiry_date')
            
            logger.info(f"Found {len(result)} contracts for {base_symbol} with interval {interval}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting contract symbols for {base_symbol} with interval {interval}: {e}")
            return pd.DataFrame()
    
    def get_contract_data(self, symbol, start_date=None, end_date=None, interval='15minute'):
        """Get data for a specific contract."""
        try:
            # Parse the interval value and unit
            if interval == '1daily':
                interval_value = 1
                interval_unit = 'daily'
            else:
                interval_parts = interval.split('minute') if 'minute' in interval else interval.split('daily')
                interval_value = int(interval_parts[0])
                interval_unit = 'minute' if 'minute' in interval else 'daily'
            
            # Build the query
            query = f"""
            SELECT 
                timestamp, 
                open, 
                high, 
                low, 
                close, 
                volume,
                interval_value,
                interval_unit
            FROM market_data
            WHERE symbol = '{symbol}'
            AND interval_value = {interval_value}
            AND interval_unit = '{interval_unit}'
            """
            
            # Add date filters if provided
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            # Add ordering
            query += " ORDER BY timestamp"
            
            # Execute the query
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                logger.warning(f"No data found for {symbol} with interval {interval}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(result)} records for {symbol} with interval {interval}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} with interval {interval}: {e}")
            return pd.DataFrame()
    
    def get_contract_date_range(self, symbol):
        """Get the start and end dates for a contract."""
        try:
            query = f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchdf()
            return result.iloc[0] if not result.empty else None
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None
    
    def calculate_expiry_date(self, symbol):
        """Calculate the expiry date for a futures contract."""
        try:
            # Extract the base symbol and month code
            if len(symbol) < 4:
                logger.warning(f"Invalid symbol format: {symbol}")
                return None
            
            base_symbol = symbol[:-3]  # e.g., ES from ESH25
            month_code = symbol[-3]    # e.g., H from ESH25
            year_code = symbol[-2:]    # e.g., 25 from ESH25
            
            # Map month codes to months
            month_map = {
                'F': 1,  # January
                'G': 2,  # February
                'H': 3,  # March
                'J': 4,  # April
                'K': 5,  # May
                'M': 6,  # June
                'N': 7,  # July
                'Q': 8,  # August
                'U': 9,  # September
                'V': 10, # October
                'X': 11, # November
                'Z': 12  # December
            }
            
            if month_code not in month_map:
                logger.warning(f"Invalid month code: {month_code} in symbol {symbol}")
                return None
            
            # Get the month number
            month = month_map[month_code]
            
            # Convert year code to full year
            current_year = datetime.now().year
            century = current_year // 100 * 100
            year = century + int(year_code)
            
            # Adjust for 20xx vs 21xx
            if year > current_year + 10:
                year -= 100
            
            # Get the expiry rule from the config
            expiry_rule = None
            try:
                with open('config/market_symbols.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                    
                for future in config.get('futures', []):
                    if future.get('base_symbol') == base_symbol:
                        expiry_rule = future.get('expiry_rule', {})
                        break
            except Exception as e:
                logger.warning(f"Error reading expiry rule from config: {e}")
            
            # Default expiry rule if not found in config
            if not expiry_rule:
                expiry_rule = {
                    'day_type': 'friday',
                    'day_number': 3,
                    'adjust_for_holiday': True
                }
            
            # Calculate the expiry date based on the rule
            if expiry_rule.get('day_type') == 'friday':
                # Find the nth Friday of the month
                day_number = expiry_rule.get('day_number', 3)
                
                # Start with the first day of the month
                date = datetime(year, month, 1)
                
                # Move to the first Friday
                while date.weekday() != 4:  # 4 is Friday
                    date += timedelta(days=1)
                
                # Move to the nth Friday
                for _ in range(day_number - 1):
                    date += timedelta(days=7)
                
                # Adjust for holidays if needed
                if expiry_rule.get('adjust_for_holiday', True):
                    # Simple adjustment: if it's a holiday, move to the next business day
                    # In a real implementation, you would check against a holiday calendar
                    while date.weekday() >= 5:  # Saturday or Sunday
                        date += timedelta(days=1)
                
                return date
                
            elif expiry_rule.get('day_type') == 'business_day':
                # Find the nth business day before the reference day
                days_before = expiry_rule.get('days_before', 3)
                reference_day = expiry_rule.get('reference_day', 25)
                
                # Start with the reference day
                date = datetime(year, month, reference_day)
                
                # Count backwards to find the nth business day
                count = 0
                while count < days_before:
                    date -= timedelta(days=1)
                    if date.weekday() < 5:  # Monday to Friday
                        count += 1
                
                return date
                
            else:
                logger.warning(f"Unsupported expiry rule day_type: {expiry_rule.get('day_type')}")
                return None
            
        except Exception as e:
            logger.error(f"Error calculating expiry date for {symbol}: {e}")
            return None
    
    def get_price_discrepancy_thresholds(self, symbol):
        """Get price discrepancy thresholds for a specific contract.
        
        This method determines the threshold for detecting suspicious price differences
        at rollover points. It first checks the configuration file for symbol-specific
        thresholds, then falls back to default values if not found.
        
        Args:
            symbol (str): The contract symbol (e.g., 'ES', 'NQ')
            
        Returns:
            float: The price jump threshold as a decimal (e.g., 0.05 for 5%)
        """
        # Default thresholds
        thresholds = {
            'ES': {'price_diff': 100, 'price_ratio': 2.0},
            'NQ': {'price_diff': 300, 'price_ratio': 1.02},
            'CL': {'price_diff': 5, 'price_ratio': 1.05},
            'GC': {'price_diff': 20, 'price_ratio': 1.02},
            'SI': {'price_diff': 1, 'price_ratio': 1.05},
            'ZB': {'price_diff': 2, 'price_ratio': 1.02},
            'ZN': {'price_diff': 1, 'price_ratio': 1.02},
            'ZF': {'price_diff': 0.5, 'price_ratio': 1.02},
            'ZT': {'price_diff': 0.25, 'price_ratio': 1.02}
        }
        
        # Try to get threshold from config
        try:
            with open('config/market_symbols.yaml', 'r') as f:
                config = yaml.safe_load(f)
                
            # Find the base symbol (remove any suffix like _backadj)
            base_symbol = symbol.split('_')[0]
            
            # Look for the symbol in the futures section
            for future in config.get('futures', []):
                if future.get('base_symbol') == base_symbol and 'price_jump_threshold' in future:
                    # Use the percentage threshold from the config
                    threshold = future['price_jump_threshold']
                    logger.info(f"Using price jump threshold of {threshold*100}% for {base_symbol} from config")
                    return threshold
        except Exception as e:
            logger.warning(f"Error reading price jump threshold from config: {e}")
        
        # Fall back to default thresholds
        if base_symbol in thresholds:
            logger.info(f"Using default price jump threshold for {base_symbol}")
            return thresholds[base_symbol]['price_ratio'] - 1  # Convert to percentage
        
        # Default to 5% if no specific threshold is found
        logger.warning(f"No price jump threshold found for {base_symbol}, using default of 5%")
        return 0.05

    def find_rollover_dates(self, symbol, start_date=None, end_date=None, interval='15minute', rollover_method='volume'):
        """Find rollover dates for a symbol.
        
        This method determines the appropriate rollover dates between contracts based on
        the specified rollover method. It supports two methods:
        
        1. Volume-based: Rolls over when the next contract's volume exceeds the current
           contract's volume within 5 days of the current contract's expiration.
        2. Fixed: Rolls over one day before the current contract's expiration.
        
        The method also detects suspicious price differences at rollover points using
        configurable thresholds.
        
        Args:
            symbol (str): Base symbol (e.g., 'ES', 'NQ')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval (e.g., '15minute', '1daily')
            rollover_method (str): Method to determine rollover dates
                - 'volume': Roll when next contract volume exceeds current within 5 days of expiry
                - 'fixed': Roll one day before expiration
        
        Returns:
            dict: Dictionary of rollover dates and associated information
        """
        try:
            # Parse the interval value and unit
            if interval == '1daily':
                interval_value = 1
                interval_unit = 'daily'
            else:
                interval_parts = interval.split('minute') if 'minute' in interval else interval.split('daily')
                interval_value = int(interval_parts[0])
                interval_unit = 'minute' if 'minute' in interval else 'daily'
            
            # Get all contracts for the symbol
            contracts = self.get_contract_symbols(symbol, start_date, end_date, interval)
            if contracts.empty:
                logger.warning(f"No contracts found for {symbol}")
                return {}
            
            # Sort contracts by expiry date
            contracts = contracts.sort_values('expiry_date')
            
            # Find rollover dates
            rollovers = {}
            for i in range(len(contracts) - 1):
                current_contract = contracts.iloc[i]['symbol']
                next_contract = contracts.iloc[i + 1]['symbol']
                expiry_date = contracts.iloc[i]['expiry_date']
                
                if rollover_method == 'volume':
                    # Get daily data for both contracts within 5 days of expiry
                    lookback_start = expiry_date - pd.Timedelta(days=5)
                    current_data = self.get_contract_data(current_contract, lookback_start, expiry_date, interval='1daily')
                    next_data = self.get_contract_data(next_contract, lookback_start, expiry_date, interval='1daily')
                    
                    if current_data.empty or next_data.empty:
                        logger.warning(f"Missing data for rollover between {current_contract} and {next_contract}")
                        continue
                    
                    # Find the first day where next contract volume exceeds current
                    for date in pd.date_range(lookback_start, expiry_date):
                        current_vol = current_data[current_data['timestamp'].dt.date == date.date()]['volume'].sum()
                        next_vol = next_data[next_data['timestamp'].dt.date == date.date()]['volume'].sum()
                        
                        if next_vol > current_vol:
                            rollover_date = date
                            break
                    else:
                        # If no volume crossover found, use expiry date
                        rollover_date = expiry_date
                else:  # fixed rollover
                    # Roll one day before expiration
                    rollover_date = expiry_date - pd.Timedelta(days=1)
                
                # Get closing prices for both contracts on rollover date
                current_close = current_data[current_data['timestamp'].dt.date == rollover_date.date()]['close'].iloc[-1]
                next_close = next_data[next_data['timestamp'].dt.date == rollover_date.date()]['close'].iloc[-1]
                
                # Calculate price ratio for suspicious price detection
                price_ratio = next_close / current_close if current_close > 0 else float('inf')
                price_jump_threshold = self.get_price_discrepancy_thresholds(symbol)
                is_suspicious = abs(price_ratio - 1) > price_jump_threshold
                
                # Store rollover information
                rollovers[rollover_date] = {
                    'current_contract': current_contract,
                    'next_contract': next_contract,
                    'current_close': current_close,
                    'next_close': next_close,
                    'is_suspicious': is_suspicious
                }
                
                logger.info(f"Found rollover from {current_contract} to {next_contract} on {rollover_date}")
            
            if not rollovers:
                logger.error(f"No rollover dates found for {symbol}")
                return {}
            
            return rollovers
            
        except Exception as e:
            logger.error(f"Error finding rollover dates for {symbol}: {e}")
            return {}

    def get_front_month_contract(self, base_symbol, interval='15minute'):
        """Get the current front month contract based on today's date."""
        try:
            # Get all contracts for the base symbol with the specified interval
            contract_symbols = self.get_contract_symbols(base_symbol, interval)
            if not contract_symbols.empty:
                # Get today's date
                today = datetime.now()
                
                # Calculate expiry dates for all contracts
                contracts_with_expiry = []
                for symbol in contract_symbols['symbol']:
                    expiry_date = self.calculate_expiry_date(symbol)
                    if expiry_date and expiry_date > today:
                        contracts_with_expiry.append((symbol, expiry_date))
                
                if not contracts_with_expiry:
                    logger.error(f"No active contracts found for {base_symbol} with interval {interval}")
                    return None
                
                # Sort by expiry date and get the first one (nearest expiry)
                contracts_with_expiry.sort(key=lambda x: x[1])
                return contracts_with_expiry[0][0]
            else:
                logger.error(f"No contracts found for {base_symbol} with interval {interval}")
                return None
            
        except Exception as e:
            logger.error(f"Error finding front month contract for {base_symbol} with interval {interval}: {e}")
            return None

    def generate_continuous_contract(self, base_symbol, output_symbol, start_date=None, end_date=None, force=False, interval='15minute', rollover_method='volume'):
        """Generate a continuous contract by backadjusting historical data.
        
        This method creates a continuous contract by:
        1. Finding appropriate rollover dates between contracts
        2. Applying price adjustments to maintain continuity
        3. Detecting and logging suspicious price discrepancies
        4. Returning the adjusted data for saving to the database
        
        The method supports multiple rollover methods:
        - Volume-based: Rolls over when the next contract's volume exceeds the current
        - Fixed: Rolls over one day before the current contract's expiration
        
        Args:
            base_symbol (str): Base symbol (e.g., 'ES', 'NQ')
            output_symbol (str): Output symbol for the continuous contract
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            force (bool): Whether to force regeneration of existing data
            interval (str): Data interval (e.g., '15minute', '1daily')
            rollover_method (str): Method to determine rollover dates
                - 'volume': Roll when next contract volume exceeds current within 5 days of expiry
                - 'fixed': Roll one day before expiration
        
        Returns:
            pandas.DataFrame: The continuous contract data, or None if generation failed
        """
        try:
            # Get price jump threshold for this symbol
            price_jump_threshold = self.get_price_discrepancy_thresholds(base_symbol)
            logger.info(f"Using price jump threshold of {price_jump_threshold*100}% for {base_symbol}")
            
            # Find rollover dates
            rollover_dates = self.find_rollover_dates(base_symbol, start_date, end_date, interval, rollover_method)
            if not rollover_dates:
                logger.error(f"No rollover dates found for {base_symbol}")
                return None
            
            # Sort rollover dates
            rollover_dates = sorted(rollover_dates.items())
            
            # Initialize adjustment factor
            adjustment = 0.0
            
            # Initialize continuous data
            continuous_data = pd.DataFrame()
            
            # Process each rollover date
            for i, (rollover_date, rollover_info) in enumerate(rollover_dates):
                current_contract = rollover_info['current_contract']
                next_contract = rollover_info['next_contract']
                current_close = rollover_info['current_close']
                next_close = rollover_info['next_close']
                
                # Calculate price ratio
                price_ratio = next_close / current_close if current_close > 0 else float('inf')
                
                # Check for suspicious price differences
                if rollover_info['is_suspicious']:
                    logger.error(f"Large price discrepancy detected at rollover from {current_contract} to {next_contract} on {rollover_date}:")
                    logger.error(f"  {current_contract} close: {current_close}")
                    logger.error(f"  {next_contract} close: {next_close}")
                    logger.error(f"  Price ratio: {price_ratio:.4f} (threshold: {1 + price_jump_threshold:.4f})")
                
                # Calculate adjustment
                adjustment += current_close - next_close
                
                # Get data for the period
                period_start = start_date if i == 0 else rollover_dates[i-1][0]
                period_data = self.get_contract_data(current_contract, period_start, rollover_date, interval=interval)
                
                if not period_data.empty:
                    # Apply adjustment to all price columns
                    for col in ['open', 'high', 'low', 'close']:
                        period_data[col] += adjustment
                    
                    # Store the adjusted data
                    continuous_data = pd.concat([continuous_data, period_data])
            
            # Add the most recent contract's data
            if rollover_dates:
                last_rollover = rollover_dates[-1]
                recent_data = self.get_contract_data(last_rollover[1]['next_contract'], last_rollover[0], end_date, interval=interval)
                if not recent_data.empty:
                    # Apply adjustment to all price columns
                    for col in ['open', 'high', 'low', 'close']:
                        recent_data[col] += adjustment
                    
                    # Store the adjusted data
                    continuous_data = pd.concat([continuous_data, recent_data])
            
            # Sort by timestamp and remove duplicates
            if not continuous_data.empty:
                continuous_data = continuous_data.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                
                # Set the output symbol
                continuous_data['symbol'] = output_symbol
                
                # Ensure interval information is preserved
                if 'interval_value' not in continuous_data.columns:
                    # Parse the interval value and unit
                    interval_parts = interval.split('minute') if 'minute' in interval else interval.split('daily')
                    interval_value = int(interval_parts[0])
                    interval_unit = 'minute' if 'minute' in interval else 'daily'
                    
                    continuous_data['interval_value'] = interval_value
                    continuous_data['interval_unit'] = interval_unit
                
                logger.info(f"Successfully generated continuous contract {output_symbol} with {len(continuous_data)} records")
                return continuous_data
            else:
                logger.error(f"No data available for continuous contract {output_symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating continuous contract for {base_symbol}: {e}")
            return None
    
    def save_continuous_contract(self, data, output_symbol, force=False):
        """Save the continuous contract data to the database."""
        try:
            # Check if data exists for this symbol
            existing_data = self.conn.execute(
                f"SELECT COUNT(*) as count FROM market_data WHERE symbol = '{output_symbol}'"
            ).fetchone()[0]
            
            if existing_data > 0:
                if force:
                    logger.info(f"Deleting existing data for {output_symbol} in force mode")
                    self.conn.execute(f"DELETE FROM market_data WHERE symbol = '{output_symbol}'")
                else:
                    logger.warning(f"Data already exists for {output_symbol}. Use --force to overwrite.")
                    return False
            
            # Ensure interval information is preserved
            if 'interval_value' not in data.columns:
                data['interval_value'] = 15  # Default to 15-minute
                data['interval_unit'] = 'minute'
            
            # Convert DataFrame to list of dictionaries for insertion
            records = data.to_dict('records')
            
            # Insert data in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                placeholders = ','.join(['?' for _ in range(len(chunk[0]))])
                columns = ','.join(chunk[0].keys())
                
                # Create the insert query
                query = f"INSERT INTO market_data ({columns}) VALUES ({placeholders})"
                
                # Execute the insert
                self.conn.executemany(query, [list(record.values()) for record in chunk])
            
            logger.info(f"Successfully saved {len(records)} records for {output_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving continuous contract {output_symbol}: {e}")
            return False

if __name__ == "__main__":
    """Generate continuous contracts for futures data.
    
    This script creates continuous contracts by backadjusting historical data based on
    specified rollover methods. It supports two rollover methods:
    
    1. Volume-based: Rolls over when the next contract's volume exceeds the current
       contract's volume within 5 days of the current contract's expiration.
    2. Fixed: Rolls over one day before the current contract's expiration.
    
    The script also includes features for:
    - Detecting suspicious price differences at rollover points
    - Configurable price jump thresholds per symbol
    - Comprehensive logging of rollover events and price discrepancies
    - Force mode to rebuild existing continuous contracts
    
    Usage:
        python generate_continuous_contract.py [options]
        
    Options:
        --symbol SYMBOL       Base symbol (e.g., 'ES', 'NQ')
        --output OUTPUT       Output symbol for the continuous contract
        --start-date DATE     Start date in YYYY-MM-DD format
        --end-date DATE       End date in YYYY-MM-DD format
        --interval INTERVAL   Data interval (e.g., '15minute', '1daily')
        --rollover-method METHOD
                             Method to determine rollover dates:
                             - 'volume': Roll when next contract volume exceeds current
                             - 'fixed': Roll one day before expiration
        --force              Force regeneration of existing data
        
    Examples:
        # Generate ES continuous contract using volume-based rollover
        python generate_continuous_contract.py --symbol ES --output ES_backadj --rollover-method volume
        
        # Generate NQ continuous contract using fixed rollover
        python generate_continuous_contract.py --symbol NQ --output NQ_backadj --rollover-method fixed
        
        # Force rebuild of existing ES continuous contract
        python generate_continuous_contract.py --symbol ES --output ES_backadj --force
    """
    parser = argparse.ArgumentParser(description='Generate continuous contracts for futures data')
    parser.add_argument('--symbol', required=True, help='Base symbol (e.g., ES, NQ)')
    parser.add_argument('--output', required=True, help='Output symbol for the continuous contract')
    parser.add_argument('--start-date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', default='15minute', help='Data interval (e.g., 15minute, 1daily)')
    parser.add_argument('--rollover-method', default='volume', choices=['volume', 'fixed'],
                      help='Method to determine rollover dates')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing data')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ContinuousContractGenerator()
    
    # Generate continuous contract
    data = generator.generate_continuous_contract(
        base_symbol=args.symbol,
        output_symbol=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
        interval=args.interval,
        rollover_method=args.rollover_method
    )
    
    if data is not None:
        # Save the continuous contract
        generator.save_continuous_contract(data)
        logger.info(f"Successfully saved continuous contract {args.output}")
    else:
        logger.error(f"Failed to generate continuous contract {args.output}")
        sys.exit(1) 