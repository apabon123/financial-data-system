#!/usr/bin/env python
"""
Check Market Data Script

This script provides utilities to check the market data in the database:
1. List all available symbols
2. Show date ranges for each symbol
3. Check for gaps in the data

Examples:
    # List all available symbols
    python check_market_data.py --list-symbols
    
    # Analyze a specific symbol
    python check_market_data.py --symbol ES
    
    # Check for gaps with a custom threshold
    python check_market_data.py --symbol ES --max-gap-days 5
    
    # Analyze all symbols
    python check_market_data.py
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class MarketDataChecker:
    """Class to check market data in the database."""
    
    def __init__(self, db_path='./data/financial_data.duckdb'):
        """Initialize the market data checker."""
        self.db_path = db_path
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_all_symbols(self):
        """Get a list of all unique symbols in the database."""
        try:
            query = """
            SELECT DISTINCT symbol
            FROM market_data
            ORDER BY symbol
            """
            result = self.conn.execute(query).fetchdf()
            return result['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

    def get_symbol_date_range(self, symbol):
        """Get the start and end dates for a symbol."""
        try:
            query = f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as total_records
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchdf()
            return result.iloc[0] if not result.empty else None
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None

    def check_gaps(self, symbol, max_gap_days=3):
        """
        Check for gaps in the data for a symbol.
        A gap is defined as a period longer than max_gap_days where we should have data.
        """
        try:
            # Get all dates for the symbol
            query = f"""
            SELECT DISTINCT timestamp::date as date
            FROM market_data
            WHERE symbol = '{symbol}'
            ORDER BY date
            """
            dates_df = self.conn.execute(query).fetchdf()
            
            if dates_df.empty:
                return []

            # Convert to datetime if needed
            dates_df['date'] = pd.to_datetime(dates_df['date'])
            
            # Find gaps
            gaps = []
            for i in range(len(dates_df) - 1):
                current_date = dates_df.iloc[i]['date']
                next_date = dates_df.iloc[i + 1]['date']
                gap_days = (next_date - current_date).days
                
                if gap_days > max_gap_days:
                    gaps.append({
                        'start_date': current_date,
                        'end_date': next_date,
                        'gap_days': gap_days
                    })
            
            return gaps
        except Exception as e:
            logger.error(f"Error checking gaps for {symbol}: {e}")
            return []

    def analyze_symbol(self, symbol):
        """Analyze a symbol's data comprehensively."""
        date_range = self.get_symbol_date_range(symbol)
        if date_range is None:
            logger.error(f"No data found for symbol {symbol}")
            return

        logger.info(f"\nAnalysis for {symbol}:")
        logger.info(f"Start date: {date_range['start_date']}")
        logger.info(f"End date: {date_range['end_date']}")
        logger.info(f"Total records: {date_range['total_records']}")

        gaps = self.check_gaps(symbol)
        if gaps:
            logger.info("\nGaps found:")
            for gap in gaps:
                logger.info(f"Gap of {gap['gap_days']} days between {gap['start_date']} and {gap['end_date']}")
        else:
            logger.info("\nNo significant gaps found")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Check market data in the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available symbols
  python check_market_data.py --list-symbols
  
  # Analyze a specific symbol
  python check_market_data.py --symbol ES
  
  # Check for gaps with a custom threshold
  python check_market_data.py --symbol ES --max-gap-days 5
  
  # Analyze all symbols
  python check_market_data.py
        """
    )
    parser.add_argument('--symbol', type=str, help='Symbol to analyze (e.g., ES, NQ, CL)')
    parser.add_argument('--list-symbols', action='store_true', help='List all available symbols in the database')
    parser.add_argument('--check-gaps', action='store_true', help='Check for gaps in the data (this is enabled by default when analyzing a symbol)')
    parser.add_argument('--max-gap-days', type=int, default=3, help='Maximum gap in days to consider significant (default: 3)')
    parser.add_argument('--db-path', type=str, default='./data/financial_data.duckdb', help='Path to the database file (default: ./data/financial_data.duckdb)')
    
    args = parser.parse_args()
    
    try:
        checker = MarketDataChecker(db_path=args.db_path)
        
        if args.list_symbols:
            symbols = checker.get_all_symbols()
            logger.info("\nAvailable symbols:")
            for symbol in symbols:
                logger.info(symbol)
            return

        if args.symbol:
            checker.analyze_symbol(args.symbol)
        else:
            symbols = checker.get_all_symbols()
            for symbol in symbols:
                checker.analyze_symbol(symbol)
                
    except Exception as e:
        logger.error(f"Error running market data checker: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 