#!/usr/bin/env python
"""
Populates the futures_roll_calendar table for ES and NQ futures based on their roll dates
and contract specifications from the futures.yaml configuration.
"""

import os
import sys
import pandas as pd
import yaml
import logging
from datetime import datetime, date, timedelta
import duckdb
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "futures.yaml"

def load_config(config_path):
    """Load futures configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def get_contract_month_code(month):
    """Convert month number to futures contract month code."""
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    return month_codes.get(month)

def get_third_friday(year, month):
    """Calculate the third Friday of a given month."""
    # Start with the first day of the month
    first_day = date(year, month, 1)
    
    # Find the first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # Add two weeks to get the third Friday
    third_friday = first_friday + timedelta(days=14)
    return third_friday

def generate_contract_calendar(root_symbol, config, start_year, end_year):
    """Generate roll calendar entries for a futures contract."""
    futures_config = config['futures'].get(root_symbol) if 'futures' in config else config.get(root_symbol)
    if not futures_config:
        logger.error(f"No configuration found for {root_symbol}")
        return pd.DataFrame()
    
    # Get contract patterns and start year from config
    patterns = futures_config['contract_info']['patterns']
    config_start_year = futures_config['contract_info']['start_year']
    
    # Use the later of config start year or provided start year
    actual_start_year = max(start_year, config_start_year)
    
    calendar_entries = []
    
    for year in range(actual_start_year, end_year + 1):
        for month in range(1, 13):
            month_code = get_contract_month_code(month)
            if month_code in patterns:
                # Generate contract code (e.g., ESH24)
                contract_code = f"{root_symbol}{month_code}{str(year)[-2:]}"
                
                # Calculate last trading day (third Friday of the month)
                last_trading_day = get_third_friday(year, month)
                
                # Calculate roll date (typically 5 days before last trading day)
                roll_date = last_trading_day - timedelta(days=5)
                
                # Calculate expiration date (day after last trading day)
                expiration_date = last_trading_day + timedelta(days=1)
                
                # Calculate first notice day (typically 2 business days before last trading day)
                first_notice_day = last_trading_day - timedelta(days=2)
                
                # Calculate active trading start (typically 3 months before expiration)
                active_trading_start = last_trading_day - timedelta(days=90)
                
                calendar_entries.append({
                    'contract_code': contract_code,
                    'root_symbol': root_symbol,
                    'year': year,
                    'month': month,
                    'expiry_date': expiration_date,
                    'roll_date': roll_date,
                    'last_trading_day': last_trading_day,
                    'first_notice_day': first_notice_day,
                    'active_trading_start': active_trading_start,
                    'roll_method': 'volume'  # From config
                })
    
    return pd.DataFrame(calendar_entries)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Populate futures roll calendar for ES and NQ futures.')
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to DuckDB database')
    parser.add_argument('--config-path', default=DEFAULT_CONFIG_PATH, help='Path to futures configuration YAML')
    parser.add_argument('--start-year', type=int, default=2003, help='Start year for calendar generation')
    parser.add_argument('--end-year', type=int, default=date.today().year + 5, help='End year for calendar generation')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Connect to database
    try:
        conn = duckdb.connect(database=str(args.db_path))
        logger.info(f"Connected to database: {args.db_path}")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)
    
    try:
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS futures_roll_calendar (
                contract_code VARCHAR NOT NULL,
                root_symbol VARCHAR NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                expiry_date DATE NOT NULL,
                roll_date DATE NOT NULL,
                last_trading_day DATE NOT NULL,
                first_notice_day DATE NOT NULL,
                active_trading_start DATE NOT NULL,
                roll_method VARCHAR NOT NULL,
                PRIMARY KEY (contract_code)
            );
        """)
        
        # Process each symbol
        for symbol in ['ES', 'NQ']:
            logger.info(f"Generating roll calendar for {symbol}...")
            
            # Generate calendar entries
            calendar_df = generate_contract_calendar(symbol, config, args.start_year, args.end_year)
            
            if calendar_df.empty:
                logger.warning(f"No calendar entries generated for {symbol}")
                continue
            
            # Delete existing entries for this symbol
            conn.execute("DELETE FROM futures_roll_calendar WHERE root_symbol = ?", [symbol])
            
            # Insert new entries
            conn.register("temp_calendar", calendar_df)
            conn.execute("""
                INSERT INTO futures_roll_calendar
                SELECT * FROM temp_calendar
            """)
            conn.unregister("temp_calendar")
            
            logger.info(f"Successfully inserted {len(calendar_df)} entries for {symbol}")
        
        conn.commit()
        logger.info("Roll calendar population completed successfully")
        
    except Exception as e:
        logger.error(f"Error populating roll calendar: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main() 