#!/usr/bin/env python
"""
Generate Continuous Futures Script

This script generates continuous futures contracts by combining individual futures contracts
and handling rollovers on expiry days. For example, for VX futures:
- VXJ25 (VXc1) expires on a Wednesday
- VXK25 (VXc2) becomes VXc1 on the expiry day of VXJ25
- The rollover happens on the expiry day, not the next day

Database Structure:
- market_data: Main table containing all raw market data with various intervals
- daily_bars: VIEW on market_data that filters for daily interval data
- continuous_contracts: Table storing generated continuous futures contracts

The script reads raw futures data from the market_data table (filtered for daily data)
and writes the generated continuous contracts to the continuous_contracts table.
"""

import os
import sys
import yaml
import logging
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import calendar
from dateutil.relativedelta import relativedelta, WE
import argparse
from rich.console import Console
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"
DEFAULT_DB_PATH = "./data/financial_data.duckdb"

class ContinuousFuturesGenerator:
    def __init__(self, config_path=None, db_path=None):
        """Initialize the continuous futures generator."""
        self.config = self._load_config(config_path) if config_path else {}
        self.db_path = db_path or DEFAULT_DB_PATH
        self.console = Console()
        self.conn = self._connect_database()
        self._init_database()

    def _init_database(self):
        """Initialize the database tables and views if they don't exist."""
        try:
            # Create continuous_contracts table if it doesn't exist
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS continuous_contracts (
                date TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                source VARCHAR,
                interval_value INTEGER,
                interval_unit VARCHAR,
                adjusted BOOLEAN,
                quality INTEGER
            )
            """)
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def _connect_database(self) -> duckdb.DuckDBPyConnection:
        """Connect to the database."""
        try:
            return duckdb.connect(self.db_path)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load the configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _get_futures_config(self, root_symbol: str) -> Dict[str, Any]:
        """Get the configuration for a specific futures contract."""
        futures = self.config.get('futures', [])
        for future in futures:
            if future.get('base_symbol') == root_symbol:
                return future
        return None

    def _get_contract_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get contract data for a symbol."""
        try:
            # Build query
            query = """
                SELECT 
                    timestamp as date,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM market_data
                WHERE symbol = ?
                AND interval_value = 1 
                AND interval_unit = 'day'
            """
            params = [symbol]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp"

            # Execute query
            df = self.conn.execute(query, params).fetchdf()
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error getting contract data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _get_available_contracts(self, root_symbol: str) -> List[str]:
        """Get all available contracts for a root symbol."""
        try:
            # Query for contracts
            query = """
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE ?
                AND interval_value = 1 
                AND interval_unit = 'day'
                AND symbol NOT LIKE '%c_'  -- Exclude continuous contracts
                ORDER BY symbol
            """
            df = self.conn.execute(query, [f"{root_symbol}%"]).fetchdf()
            
            if df.empty:
                logger.warning(f"No contracts found for {root_symbol}")
                return []

            return df['symbol'].tolist()

        except Exception as e:
            logger.error(f"Error getting available contracts: {str(e)}")
            return []

    def _get_expiry_date(self, contract: str, future_config: dict) -> datetime:
        """Get the expiry date for a contract."""
        try:
            # Skip continuous contracts
            if 'c' in contract[-3:]:
                logger.debug(f"Skipping continuous contract {contract}")
                return None
                
            # Extract month and year from contract symbol (e.g., VXM04 -> M and 04)
            month_code = contract[-3]
            year_str = contract[-2:]
            
            # Convert year string to number
            try:
                year = 2000 + int(year_str)  # Assuming all years are 2000+
            except ValueError:
                logger.error(f"Invalid year in contract {contract}")
                return None
            
            # Convert month code to month number
            month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
            month = month_map.get(month_code)
            if not month:
                logger.error(f"Invalid month code {month_code} in contract {contract}")
                return None
            
            # Get expiry rule from config
            expiry_rule = future_config.get('expiry_rule', {})
            if not expiry_rule:
                logger.error(f"No expiry rule found for {contract}")
                return None
            
            # Parse expiry rule
            day_type = expiry_rule.get('day_type', '').lower()
            day_number = expiry_rule.get('day_number', 0)
            
            if day_type == 'wednesday':
                # Get the third Wednesday
                c = calendar.monthcalendar(year, month)
                third_wednesday = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0][2]
                expiry_date = datetime(year, month, third_wednesday)
                return expiry_date
            else:
                logger.error(f"Unsupported expiry rule: {expiry_rule}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting expiry date for {contract}: {e}")
            return None

    def _get_active_contracts(self, root_symbol: str, start_date: datetime, end_date: datetime) -> List[str]:
        """Get list of active contracts for the given period."""
        query = f"""
            SELECT DISTINCT symbol
            FROM market_data
            WHERE symbol LIKE '{root_symbol}%'
            AND timestamp >= '{start_date}'
            AND timestamp <= '{end_date}'
            AND interval_value = 1 
            AND interval_unit = 'day'
            AND symbol NOT LIKE '%c%'  -- Exclude continuous contracts
            ORDER BY symbol
        """
        result = self.conn.execute(query).fetchdf()
        return result['symbol'].tolist()

    def _sort_contracts_by_expiry(self, contracts: List[str], future_config: dict) -> List[str]:
        """Sort contracts by their expiry dates."""
        contract_dates = []
        for contract in contracts:
            expiry_date = self._get_expiry_date(contract, future_config)
            if expiry_date:
                contract_dates.append((contract, expiry_date))
        
        # Sort by expiry date
        contract_dates.sort(key=lambda x: x[1])
        return [contract for contract, _ in contract_dates]

    def _generate_continuous_future(self, root_symbol: str, contract_number: int) -> pd.DataFrame:
        """Generate a continuous future contract for a root symbol using rollover logic."""
        continuous_symbol = f"{root_symbol}c{contract_number}"
        logger.info(f"Generating continuous contract {continuous_symbol} (Contract Number: {contract_number})")

        future_config = self._get_futures_config(root_symbol)
        if not future_config:
            logger.error(f"No configuration found for {root_symbol}")
            return pd.DataFrame()

        # 1. Get all relevant contracts and sort them by expiry
        available_contracts = self._get_available_contracts(root_symbol)
        if not available_contracts:
            logger.warning(f"No individual contracts found for {root_symbol}")
            return pd.DataFrame()

        sorted_contracts_with_expiry = []
        for contract in available_contracts:
            # Ensure we don't try to get expiry for already generated continuous contracts
            if 'c' not in contract[-3:]:
                expiry = self._get_expiry_date(contract, future_config)
                if expiry:
                    sorted_contracts_with_expiry.append((contract, expiry))
                else:
                    logger.warning(f"Could not determine expiry for {contract}. Skipping.")

        # Sort contracts chronologically by expiry date
        sorted_contracts_with_expiry.sort(key=lambda x: x[1])

        if not sorted_contracts_with_expiry:
            logger.error(f"No contracts with valid expiry dates found for {root_symbol}.")
            return pd.DataFrame()
            
        logger.info(f"Found {len(sorted_contracts_with_expiry)} contracts with expiry dates for {root_symbol}.")

        # 2. Get all unique trading dates for the root symbol from daily data
        query_dates = """
            SELECT DISTINCT timestamp::DATE as date
            FROM market_data
            WHERE symbol LIKE ?
              AND interval_value = 1 AND interval_unit = 'day'
              AND symbol NOT LIKE '%c%' -- Exclude continuous symbols themselves
            ORDER BY date
        """
        try:
            all_dates_df = self.conn.execute(query_dates, [f"{root_symbol}%"]).fetchdf()
            if all_dates_df.empty:
                logger.warning(f"No trading dates found for {root_symbol}")
                return pd.DataFrame()
            all_dates = all_dates_df['date'].tolist()
            logger.info(f"Found {len(all_dates)} unique trading dates for {root_symbol}.")
        except Exception as e:
            logger.error(f"Error fetching trading dates for {root_symbol}: {e}")
            return pd.DataFrame()

        # 3. Iterate through dates and determine the active contract
        continuous_rows = []
        
        # The target index in sorted_contracts_with_expiry for the *first* contract to use for this continuous_symbol
        # Example: For c1 (contract_number=1), start with index 0. For c2 (contract_number=2), start with index 1.
        current_base_index = contract_number - 1 

        if current_base_index >= len(sorted_contracts_with_expiry):
            logger.warning(f"Not enough contracts ({len(sorted_contracts_with_expiry)}) to generate {continuous_symbol} (needs base index {current_base_index}).")
            return pd.DataFrame()

        # Determine the actual active contract index based on rollovers
        active_contract_index = current_base_index
        
        # Find the first date the *initial* active contract starts trading
        initial_active_contract_symbol = sorted_contracts_with_expiry[active_contract_index][0]
        query_start_date = """
            SELECT MIN(timestamp::DATE) as min_date
            FROM market_data
            WHERE symbol = ? AND interval_value = 1 AND interval_unit = 'day'
        """
        try:
            start_date_result = self.conn.execute(query_start_date, [initial_active_contract_symbol]).fetchone()
            if start_date_result and start_date_result[0]:
                effective_start_date = start_date_result[0]
                logger.info(f"Starting {continuous_symbol} generation from {effective_start_date} (first data for initial contract {initial_active_contract_symbol})")
            else:
                logger.warning(f"No data found for the initial contract {initial_active_contract_symbol}. Cannot generate {continuous_symbol}.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error finding start date for {initial_active_contract_symbol}: {e}")
            return pd.DataFrame()


        for date in all_dates:
            # Skip dates before the first data point of the initial contract for this continuous series
            if date.date() < effective_start_date:
                continue

            # Ensure we have enough contracts to determine the active one for this date
            if active_contract_index >= len(sorted_contracts_with_expiry):
                 logger.warning(f"Ran out of contracts to determine active contract for {continuous_symbol} on date {date}. Stopping generation.")
                 break 

            current_active_info = sorted_contracts_with_expiry[active_contract_index]
            current_expiry_date = current_active_info[1].date() # Compare date part only

            # Determine if we need to roll over based on the *current* active contract's expiry
            # Rollover happens ON the expiry day. So if date >= expiry, we shift the index.
            if date.date() >= current_expiry_date:
                # Check if there's a next contract to roll into
                if active_contract_index + 1 < len(sorted_contracts_with_expiry):
                    active_contract_index += 1
                    new_active_info = sorted_contracts_with_expiry[active_contract_index]
                    logger.info(f"{continuous_symbol}: Rolled over on {date}. Previous contract {current_active_info[0]} expired {current_expiry_date}. New active contract: {new_active_info[0]} expiring {new_active_info[1].date()}")
                else:
                    # No more contracts to roll into for this continuous number
                    logger.info(f"Reached end of available contracts for {continuous_symbol} after expiry of {current_active_info[0]} on {current_expiry_date}. Stopping generation.")
                    break


            # 4. Fetch data for the *current* active contract on this specific date
            active_symbol_for_date = sorted_contracts_with_expiry[active_contract_index][0]
            
            query_data = """
                SELECT
                    timestamp::TIMESTAMP as date, ?::VARCHAR as symbol, open::DOUBLE, high::DOUBLE, low::DOUBLE, close::DOUBLE, volume::BIGINT,
                    'continuous'::VARCHAR as source, 1::INTEGER as interval_value, 'day'::VARCHAR as interval_unit,
                    TRUE::BOOLEAN as adjusted, 100::INTEGER as quality
                FROM market_data
                WHERE symbol = ?
                  AND timestamp::DATE = ?
                  AND interval_value = 1 AND interval_unit = 'day'
                LIMIT 1
            """
            try:
                row_data = self.conn.execute(query_data, [continuous_symbol, active_symbol_for_date, date]).fetchone()
                if row_data:
                    # Append tuple directly, DataFrame conversion happens later
                    continuous_rows.append(row_data) 
                else:
                    # Handle missing data (Point 2) - Log a warning and leave a gap
                    logger.warning(f"No data found for active contract {active_symbol_for_date} on {date}. Leaving gap in {continuous_symbol}.")
                    # No row is appended, creating the gap.
            except Exception as e:
                logger.error(f"Error fetching data for {active_symbol_for_date} on {date}: {e}")
                # Optionally decide whether to stop or continue with gaps on error


        if not continuous_rows:
            logger.warning(f"No data generated for {continuous_symbol} after processing all dates.")
            return pd.DataFrame()

        # 5. Create DataFrame and insert into DB
        result_df = pd.DataFrame(continuous_rows, columns=[
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'source', 'interval_value', 'interval_unit', 'adjusted', 'quality'
        ])
        
        # Ensure correct types before insertion
        result_df = result_df.astype({
            'date': 'datetime64[ns]', 'symbol': 'str', 'open': 'float64', 'high': 'float64',
            'low': 'float64', 'close': 'float64', 'volume': 'int64', 'source': 'str',
            'interval_value': 'int32', 'interval_unit': 'str', 'adjusted': 'bool', 'quality': 'int32'
        })

        try:
            # Delete existing data for this specific continuous symbol first
            logger.info(f"Deleting existing data for {continuous_symbol} before insertion.")
            delete_query = "DELETE FROM continuous_contracts WHERE symbol = ?"
            with self.conn.cursor() as cur:
                 cur.execute(delete_query, [continuous_symbol])
                 deleted_count = cur.rowcount
                 logger.info(f"Deleted {deleted_count} existing rows for {continuous_symbol}.")

            logger.info(f"Inserting {len(result_df)} rows for {continuous_symbol} into continuous_contracts table.")
            # Use DuckDB's efficient insertion from DataFrame
            self.conn.execute(f"INSERT INTO continuous_contracts SELECT * FROM result_df")
            
            logger.info(f"Successfully generated and inserted continuous contract {continuous_symbol}")

        except Exception as e:
            logger.error(f"Error inserting data for {continuous_symbol}: {e}")
            # Consider rollback logic if transactions are used
            return pd.DataFrame() # Return empty on insertion failure

        return result_df # Return the generated data

def main():
    """Main function to generate continuous futures."""
    parser = argparse.ArgumentParser(description='Generate continuous futures contracts')
    parser.add_argument('--root-symbol', type=str, required=True, help='Root symbol (e.g. VX for VIX futures)')
    parser.add_argument('--config', type=str, required=True, help='Path to market symbols config file')
    args = parser.parse_args()

    generator = ContinuousFuturesGenerator(config_path=args.config)
    
    # Generate continuous contracts c1 and c2
    for i in range(1, 3):
        continuous_symbol = f"{args.root_symbol}c{i}"
        logger.info(f"Generating continuous contract {continuous_symbol}")
        generator._generate_continuous_future(args.root_symbol, i)

if __name__ == '__main__':
    main() 