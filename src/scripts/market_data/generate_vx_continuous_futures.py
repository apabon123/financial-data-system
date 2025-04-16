#!/usr/bin/env python
"""
Generate VX Continuous Futures Script

This script generates VXc2 through VXc8 continuous futures contracts by combining individual futures contracts
and handling rollovers on expiry days. For VX futures:
- Expiry is on the third Wednesday of each month
- Rollover happens on the expiry day
- Morning expiry is handled by using the next contract's data on expiry day
"""

import os
import sys
import yaml
import logging
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Any, Tuple, Optional
import argparse

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = './data/financial_data.duckdb'

class VXContinuousFuturesGenerator:
    def __init__(self, config_path=None, db_path=None):
        """Initialize the VX continuous futures generator."""
        self.config = self._load_config(config_path) if config_path else {}
        self.db_path = db_path or DEFAULT_DB_PATH
        self.console = Console()
        self.conn = self._connect_database()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
            
    def _connect_database(self) -> duckdb.DuckDBPyConnection:
        """Connect to the DuckDB database."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logger.error(f"Error connecting to database at {self.db_path}: {e}")
            sys.exit(1)
            
    def _get_vx_config(self) -> Dict[str, Any]:
        """Get VX futures configuration."""
        if not self.config or 'futures' not in self.config:
            logger.error("No futures configuration found")
            return {}
            
        for future in self.config['futures']:
            if future['base_symbol'] == 'VX':
                return future
                
        logger.error("No VX configuration found")
        return {}
        
    def _get_available_contracts(self) -> List[str]:
        """Get all available VX contracts from the database."""
        try:
            # Try market_data table first
            query = """
            SELECT DISTINCT symbol
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND symbol NOT LIKE 'VXc%'
            ORDER BY symbol
            """
            result = self.conn.execute(query).fetchall()
            contracts = [row[0] for row in result]
            
            if not contracts:
                # Try daily_bars table
                query = """
                SELECT DISTINCT symbol
                FROM daily_bars
                WHERE symbol LIKE 'VX%'
                AND symbol NOT LIKE 'VXc%'
                ORDER BY symbol
                """
                result = self.conn.execute(query).fetchall()
                contracts = [row[0] for row in result]
            
            return contracts
        except Exception as e:
            logger.error(f"Error getting available contracts: {e}")
            return []
            
    def _get_contract_data(self, symbol: str) -> pd.DataFrame:
        """Get data for a specific contract."""
        try:
            # Try market_data table first
            query = f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                up_volume,
                down_volume,
                source,
                interval_value,
                interval_unit,
                adjusted,
                quality
            FROM market_data
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                # Try daily_bars table
                query = f"""
                SELECT 
                    date as timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    NULL as up_volume,
                    NULL as down_volume,
                    'daily_bars' as source,
                    1 as interval_value,
                    'day' as interval_unit,
                    false as adjusted,
                    100 as quality
                FROM daily_bars
                WHERE symbol = '{symbol}'
                ORDER BY date
                """
                result = self.conn.execute(query).fetchdf()
            
            return result
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _calculate_expiry_date(self, contract: str) -> datetime:
        """Calculate expiry date for a VX contract."""
        # Extract month and year from contract (e.g., VXJ25 -> J=October, 25=2025)
        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        
        month_code = contract[2]
        year = 2000 + int(contract[3:])
        month = month_map[month_code]
        
        # Get the third Wednesday of the month
        first_day = datetime(year, month, 1)
        wednesday_count = 0
        for day in range(1, 32):
            current_date = first_day + timedelta(days=day-1)
            if current_date.month != month:
                break
            if current_date.weekday() == 2:  # Wednesday
                wednesday_count += 1
                if wednesday_count == 3:
                    return current_date
                    
        return None
        
    def _sort_contracts_by_expiry(self, contracts: List[str]) -> List[str]:
        """Sort contracts by expiry date."""
        contract_expiry = [(c, self._calculate_expiry_date(c)) for c in contracts]
        contract_expiry = [(c, d) for c, d in contract_expiry if d is not None]
        return [c for c, _ in sorted(contract_expiry, key=lambda x: x[1])]
        
    def _get_all_contract_data(self) -> pd.DataFrame:
        """Get all VX futures data at once to avoid multiple database queries."""
        try:
            # Try market_data table first
            query = """
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                up_volume,
                down_volume,
                source,
                interval_value,
                interval_unit,
                adjusted,
                quality
            FROM market_data
            WHERE symbol LIKE 'VX%'
            AND symbol NOT LIKE 'VXc%'
            ORDER BY timestamp
            """
            result = self.conn.execute(query).fetchdf()
            
            if result.empty:
                # Try daily_bars table
                query = """
                SELECT 
                    date as timestamp,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    NULL as up_volume,
                    NULL as down_volume,
                    'daily_bars' as source,
                    1 as interval_value,
                    'day' as interval_unit,
                    false as adjusted,
                    100 as quality
                FROM daily_bars
                WHERE symbol LIKE 'VX%'
                AND symbol NOT LIKE 'VXc%'
                ORDER BY date
                """
                result = self.conn.execute(query).fetchdf()
            
            return result
        except Exception as e:
            logger.error(f"Error getting all contract data: {e}")
            return pd.DataFrame()
            
    def _generate_continuous_future(self, contract_number: int, all_data: pd.DataFrame) -> pd.DataFrame:
        """Generate a continuous future contract (VXc{contract_number})."""
        if all_data.empty:
            logger.error("No data provided")
            return pd.DataFrame()
            
        # Get unique contracts and sort by expiry date
        contracts = sorted(all_data['symbol'].unique())
        sorted_contracts = self._sort_contracts_by_expiry(contracts)
        if not sorted_contracts:
            logger.error("No valid contracts found after sorting")
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame()
        
        # Process each contract
        for i in range(len(sorted_contracts) - contract_number + 1):
            # Get the current contract and its data
            current_contract = sorted_contracts[i]
            current_data = all_data[all_data['symbol'] == current_contract].copy()
            if current_data.empty:
                continue
                
            # Get the next contract that will become the current contract
            next_contract = sorted_contracts[i + contract_number - 1]
            next_data = all_data[all_data['symbol'] == next_contract].copy()
            if next_data.empty:
                continue
                
            # Calculate expiry date of the current contract
            expiry_date = self._calculate_expiry_date(current_contract)
            if expiry_date is None:
                continue
                
            # If this is the first contract in the sequence
            if i == 0:
                # Use current contract's data up to (but not including) expiry
                mask = current_data['timestamp'].dt.date < expiry_date.date()
                result = current_data[mask].copy()
                result['symbol'] = f'VXc{contract_number}'
                
                # On expiry day, use the next contract's data
                expiry_mask = next_data['timestamp'].dt.date == expiry_date.date()
                expiry_data = next_data[expiry_mask].copy()
                expiry_data['symbol'] = f'VXc{contract_number}'
                result = pd.concat([result, expiry_data])
                continue
                
            # For subsequent contracts
            # Get data from the day after previous expiry up to (but not including) current expiry
            prev_expiry = self._calculate_expiry_date(sorted_contracts[i-1])
            if prev_expiry is None:
                continue
                
            # Use current contract's data after previous expiry but before current expiry
            mask = (current_data['timestamp'].dt.date > prev_expiry.date()) & \
                   (current_data['timestamp'].dt.date < expiry_date.date())
            new_data = current_data[mask].copy()
            new_data['symbol'] = f'VXc{contract_number}'
            
            # On expiry day, use the next contract's data
            expiry_mask = next_data['timestamp'].dt.date == expiry_date.date()
            expiry_data = next_data[expiry_mask].copy()
            expiry_data['symbol'] = f'VXc{contract_number}'
            
            # Append to result
            result = pd.concat([result, new_data, expiry_data])
            
        # Sort by timestamp and remove any duplicates
        result = result.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
        return result
        
    def generate_continuous_futures(self, start_number: int = 2, end_number: int = 8):
        """Generate continuous futures from VXc{start_number} to VXc{end_number}."""
        # Get all contract data at once
        all_data = self._get_all_contract_data()
        if all_data.empty:
            logger.error("No data found for VX futures")
            return
            
        # Generate each continuous future
        all_continuous_data = []
        for contract_number in range(start_number, end_number + 1):
            logger.info(f"Generating VXc{contract_number}...")
            continuous_data = self._generate_continuous_future(contract_number, all_data)
            if not continuous_data.empty:
                all_continuous_data.append(continuous_data)
                logger.info(f"Successfully generated VXc{contract_number}")
            else:
                logger.error(f"Failed to generate VXc{contract_number}")
                
        # Save all continuous futures at once
        if all_continuous_data:
            combined_data = pd.concat(all_continuous_data)
            if self._save_to_db(combined_data):
                logger.info("Successfully saved all continuous futures")
            else:
                logger.error("Failed to save continuous futures")
                
    def _save_to_db(self, continuous_data: pd.DataFrame) -> bool:
        """Save continuous future data to database."""
        try:
            # Check if tables exist and are base tables
            tables_info = self.conn.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            AND table_name IN ('market_data', 'daily_bars')
            """).fetchdf()
            
            # Only use tables that are BASE TABLEs (not views)
            base_tables = tables_info[tables_info['table_type'] == 'BASE TABLE']['table_name'].tolist()
            
            if not base_tables:
                logger.error("No base tables found to save data")
                return False
            
            # Delete existing continuous futures data from base tables
            for table in base_tables:
                logger.info(f"Deleting existing VXc2 and VXc3 data from {table}")
                self.conn.execute(f"""
                DELETE FROM {table}
                WHERE symbol IN ('VXc2', 'VXc3')
                """)
            
            # Save to both tables to maintain consistency
            for table in base_tables:
                logger.info(f"Inserting new VXc2 and VXc3 data into {table}")
                if table == 'market_data':
                    self.conn.execute("""
                    INSERT INTO market_data
                    SELECT 
                        timestamp,
                        symbol,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        up_volume,
                        down_volume,
                        source,
                        interval_value,
                        interval_unit,
                        adjusted,
                        quality
                    FROM continuous_data
                    WHERE symbol IN ('VXc2', 'VXc3')
                    """)
                else:
                    self.conn.execute("""
                    INSERT INTO daily_bars
                    SELECT 
                        timestamp as date,
                        symbol,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM continuous_data
                    WHERE symbol IN ('VXc2', 'VXc3')
                    """)
            
            return True
        except Exception as e:
            logger.error(f"Error saving continuous futures: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Generate VX continuous futures contracts')
    parser.add_argument('--config', type=str, default='config/market_symbols.yaml',
                      help='Path to market symbols configuration file')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                      help='Path to DuckDB database file')
    parser.add_argument('--start', type=int, default=2,
                      help='Starting contract number (default: 2)')
    parser.add_argument('--end', type=int, default=3,  # Changed to 3 to only generate VXc2 and VXc3
                      help='Ending contract number (default: 3)')
    
    args = parser.parse_args()
    
    generator = VXContinuousFuturesGenerator(args.config, args.db_path)
    generator.generate_continuous_futures(args.start, args.end)

if __name__ == '__main__':
    main() 