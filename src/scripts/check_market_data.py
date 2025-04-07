#!/usr/bin/env python
"""
Check Market Data Script

This script provides utilities to check the market data in the database:
1. List all available symbols
2. Show date ranges for each symbol
3. Check for gaps in the data, accounting for market holidays

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
import yaml
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from pathlib import Path
import calendar
from dateutil.relativedelta import relativedelta
from dateutil.easter import easter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class MarketDataChecker:
    """Class to check market data in the database."""
    
    def __init__(self, db_path='./data/financial_data.duckdb', config_path='config/market_symbols.yaml'):
        """Initialize the market data checker."""
        self.db_path = db_path
        self.config_path = config_path
        self.holiday_calendars = {}
        self.load_config()
        
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def load_config(self):
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'settings' in config and 'holiday_calendars' in config['settings']:
                    self.holiday_calendars = config['settings']['holiday_calendars']
                    logger.info(f"Loaded {len(self.holiday_calendars)} holiday calendars")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.holiday_calendars = {}

    def get_all_symbols(self):
        """Get a list of all unique symbols in the database."""
        try:
            # Get symbols from market_data table
            market_data_query = """
            SELECT DISTINCT symbol, 'market_data' as source
            FROM market_data
            ORDER BY symbol
            """
            market_data_symbols = self.conn.execute(market_data_query).fetchdf()
            
            # Get symbols from continuous_contracts table
            continuous_query = """
            SELECT DISTINCT symbol, 'continuous' as source
            FROM continuous_contracts
            ORDER BY symbol
            """
            continuous_symbols = self.conn.execute(continuous_query).fetchdf()
            
            # Combine the results
            all_symbols = pd.concat([market_data_symbols, continuous_symbols], ignore_index=True)
            return all_symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return pd.DataFrame(columns=['symbol', 'source'])

    def get_symbol_date_range(self, symbol):
        """Get the start and end dates for a symbol."""
        try:
            # Determine which table to query based on the symbol
            if symbol.endswith('_continuous'):
                query = f"""
                SELECT 
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    COUNT(*) as total_records
                FROM continuous_contracts
                WHERE symbol = '{symbol}'
                """
            else:
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
            
    def get_symbol_frequencies(self, symbol):
        """Get the frequencies available for a symbol."""
        try:
            query = f"""
            SELECT DISTINCT 
                interval_value,
                interval_unit,
                COUNT(*) as record_count
            FROM market_data
            WHERE symbol = '{symbol}'
            GROUP BY interval_value, interval_unit
            ORDER BY interval_value, interval_unit
            """
            result = self.conn.execute(query).fetchdf()
            return result if not result.empty else None
        except Exception as e:
            logger.error(f"Error getting frequencies for {symbol}: {e}")
            return None

    def is_holiday(self, date, calendar_name='US'):
        """
        Check if a date is a holiday in the specified calendar.
        
        Args:
            date: The date to check
            calendar_name: The name of the holiday calendar to use
            
        Returns:
            True if the date is a holiday, False otherwise
        """
        if calendar_name not in self.holiday_calendars:
            return False
            
        calendar_data = self.holiday_calendars[calendar_name]
        year = date.year
        month = date.month
        day = date.day
        
        for holiday in calendar_data:
            # Check fixed date holidays
            if 'date' in holiday and 'rule' not in holiday:
                holiday_date = datetime.strptime(f"{year}-{holiday['date']}", "%Y-%m-%d").date()
                if date.date() == holiday_date:
                    return True
                    
            # Check rule-based holidays
            if 'rule' in holiday:
                if holiday['rule'] == 'third_monday':
                    # Third Monday of the month
                    cal = calendar.monthcalendar(year, month)
                    third_monday = cal[2][calendar.MONDAY] if len(cal) > 2 else None
                    if third_monday and day == third_monday:
                        return True
                        
                elif holiday['rule'] == 'last_monday':
                    # Last Monday of the month
                    cal = calendar.monthcalendar(year, month)
                    last_monday = cal[-1][calendar.MONDAY] if cal[-1][calendar.MONDAY] != 0 else cal[-2][calendar.MONDAY]
                    if day == last_monday:
                        return True
                        
                elif holiday['rule'] == 'first_monday':
                    # First Monday of the month
                    cal = calendar.monthcalendar(year, month)
                    first_monday = cal[0][calendar.MONDAY] if cal[0][calendar.MONDAY] != 0 else cal[1][calendar.MONDAY]
                    if day == first_monday:
                        return True
                        
                elif holiday['rule'] == 'fourth_thursday':
                    # Fourth Thursday of the month
                    cal = calendar.monthcalendar(year, month)
                    fourth_thursday = cal[3][calendar.THURSDAY] if len(cal) > 3 else None
                    if fourth_thursday and day == fourth_thursday:
                        return True
                        
                elif holiday['rule'] == 'easter_friday':
                    # Good Friday (Friday before Easter)
                    easter_date = easter(year)
                    good_friday = easter_date - timedelta(days=2)
                    if date.date() == good_friday:
                        return True
                        
                elif holiday['rule'] == 'easter_monday':
                    # Easter Monday (Monday after Easter)
                    easter_date = easter(year)
                    easter_monday = easter_date + timedelta(days=1)
                    if date.date() == easter_monday:
                        return True
        
        return False

    def count_trading_days(self, start_date, end_date, calendar_name='US'):
        """
        Count the number of trading days between two dates, excluding holidays.
        
        Args:
            start_date: The start date
            end_date: The end date
            calendar_name: The name of the holiday calendar to use
            
        Returns:
            The number of trading days
        """
        trading_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday to Friday
                # Skip holidays
                if not self.is_holiday(current_date, calendar_name):
                    trading_days += 1
            current_date += timedelta(days=1)
            
        return trading_days

    def check_gaps(self, symbol, max_gap_days=3, calendar_name='US', interval_value=None, interval_unit=None):
        """
        Check for gaps in the data for a symbol.
        A gap is defined as a period longer than max_gap_days where we should have data,
        accounting for market holidays.
        """
        try:
            # Build the query based on whether we're filtering by frequency
            base_query = f"""
            SELECT DISTINCT timestamp::date as date
            FROM market_data
            WHERE symbol = '{symbol}'
            """
            
            if interval_value is not None and interval_unit is not None:
                base_query += f" AND interval_value = {interval_value} AND interval_unit = '{interval_unit}'"
                
            base_query += " ORDER BY date"
            
            dates_df = self.conn.execute(base_query).fetchdf()
            
            if dates_df.empty:
                return []

            # Convert to datetime if needed
            dates_df['date'] = pd.to_datetime(dates_df['date'])
            
            # Find gaps
            gaps = []
            for i in range(len(dates_df) - 1):
                current_date = dates_df.iloc[i]['date']
                next_date = dates_df.iloc[i + 1]['date']
                
                # Count trading days between dates
                trading_days = self.count_trading_days(current_date, next_date, calendar_name)
                
                # Adjust for the fact that we're counting the start and end dates
                trading_days -= 2
                
                if trading_days > max_gap_days:
                    gaps.append({
                        'start_date': current_date,
                        'end_date': next_date,
                        'calendar_days': (next_date - current_date).days,
                        'trading_days': trading_days
                    })
            
            return gaps
        except Exception as e:
            logger.error(f"Error checking gaps for {symbol}: {e}")
            return []

    def get_total_records(self, symbol):
        """Get total number of records for a symbol."""
        try:
            # Determine which table to use
            table_name = 'continuous_contracts' if '_continuous' in symbol else 'market_data'
            
            query = f"""
            SELECT COUNT(*) as count
            FROM {table_name}
            WHERE symbol = '{symbol}'
            """
            result = self.conn.execute(query).fetchdf()
            return result.iloc[0]['count'] if not result.empty else 0
        except Exception as e:
            logger.error(f"Error getting total records for {symbol}: {e}")
            return 0

    def check_price_data(self, symbol, start_date=None, end_date=None):
        """
        Validate price data for a symbol.
        
        Checks:
        1. Range checks for reasonable bounds
        2. OHLC consistency
        3. Zero/negative values
        """
        try:
            # Determine which table to use
            table_name = 'continuous_contracts' if '_continuous' in symbol else 'market_data'
            
            # Build query with date range if provided
            where_clause = f"WHERE symbol = '{symbol}'"
            if start_date:
                where_clause += f" AND timestamp >= '{start_date}'::DATE"
            if end_date:
                where_clause += f" AND timestamp <= '{end_date}'::DATE"
            
            query = f"""
            WITH price_stats AS (
                SELECT 
                    timestamp::DATE as date,
                    open, high, low, close,
                    -- OHLC consistency checks
                    CASE 
                        WHEN high < open OR high < close OR low > open OR low > close THEN 1
                        ELSE 0
                    END as ohlc_violation,
                    -- Zero/negative value checks
                    CASE 
                        WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 1
                        ELSE 0
                    END as zero_price_violation,
                    -- Extreme price checks (>50% move in a day)
                    CASE 
                        WHEN (high/NULLIF(low, 0) - 1) > 0.5 THEN 1
                        ELSE 0
                    END as extreme_price_violation
                FROM {table_name}
                {where_clause}
            )
            SELECT 
                COUNT(*) as total_records,
                SUM(ohlc_violation) as ohlc_violations,
                SUM(zero_price_violation) as zero_price_violations,
                SUM(extreme_price_violation) as extreme_price_violations,
                MIN(open) as min_open,
                MAX(open) as max_open,
                MIN(high) as min_high,
                MAX(high) as max_high,
                MIN(low) as min_low,
                MAX(low) as max_low,
                MIN(close) as min_close,
                MAX(close) as max_close
            FROM price_stats
            """
            
            result = self.conn.execute(query).fetchdf()
            if result.empty:
                logger.warning(f"No price data found for {symbol}")
                return None
            
            return result.iloc[0]
        except Exception as e:
            logger.error(f"Error checking price data for {symbol}: {e}")
            return None

    def check_volume_data(self, symbol, start_date=None, end_date=None, z_score_threshold=3):
        """
        Validate volume data for a symbol.
        
        Checks:
        1. Zero volume days
        2. Unusual volume spikes (using z-score)
        3. Volume consistency
        """
        try:
            # Determine which table to use
            table_name = 'continuous_contracts' if '_continuous' in symbol else 'market_data'
            
            # Build query with date range if provided
            where_clause = f"WHERE symbol = '{symbol}'"
            if start_date:
                where_clause += f" AND timestamp >= '{start_date}'::DATE"
            if end_date:
                where_clause += f" AND timestamp <= '{end_date}'::DATE"
            
            query = f"""
            WITH volume_stats AS (
                SELECT 
                    timestamp::DATE as date,
                    volume,
                    -- Calculate z-score for volume
                    (volume - AVG(volume) OVER ()) / NULLIF(STDDEV(volume) OVER (), 0) as volume_zscore
                FROM {table_name}
                {where_clause}
            )
            SELECT 
                COUNT(*) as total_records,
                SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_days,
                SUM(CASE WHEN volume_zscore > {z_score_threshold} THEN 1 ELSE 0 END) as high_volume_days,
                MIN(volume) as min_volume,
                MAX(volume) as max_volume,
                AVG(volume) as avg_volume,
                STDDEV(volume) as volume_std
            FROM volume_stats
            """
            
            result = self.conn.execute(query).fetchdf()
            if result.empty:
                logger.warning(f"No volume data found for {symbol}")
                return None
            
            return result.iloc[0]
        except Exception as e:
            logger.error(f"Error checking volume data for {symbol}: {e}")
            return None

    def check_statistical_validity(self, symbol, start_date=None, end_date=None, z_score_threshold=3):
        """
        Perform statistical validation checks.
        
        Checks:
        1. Price outliers using z-scores
        2. Large price gaps between sessions
        3. Volatility checks
        """
        try:
            # Determine which table to use
            table_name = 'continuous_contracts' if '_continuous' in symbol else 'market_data'
            
            # Build query with date range if provided
            where_clause = f"WHERE symbol = '{symbol}'"
            if start_date:
                where_clause += f" AND timestamp >= '{start_date}'::DATE"
            if end_date:
                where_clause += f" AND timestamp <= '{end_date}'::DATE"
            
            query = f"""
            WITH price_stats AS (
                SELECT 
                    timestamp::DATE as date,
                    open, high, low, close,
                    -- Daily returns
                    (close - LAG(close) OVER (ORDER BY timestamp)) / NULLIF(LAG(close) OVER (ORDER BY timestamp), 0) as daily_return,
                    -- Daily high-low range
                    (high - low) / NULLIF(low, 0) as daily_range,
                    -- Gap from previous close
                    (open - LAG(close) OVER (ORDER BY timestamp)) / NULLIF(LAG(close) OVER (ORDER BY timestamp), 0) as overnight_gap
                FROM {table_name}
                {where_clause}
            ),
            stats AS (
                SELECT 
                    *,
                    -- Z-scores
                    (daily_return - AVG(daily_return) OVER ()) / NULLIF(STDDEV(daily_return) OVER (), 0) as return_zscore,
                    (daily_range - AVG(daily_range) OVER ()) / NULLIF(STDDEV(daily_range) OVER (), 0) as range_zscore,
                    (overnight_gap - AVG(overnight_gap) OVER ()) / NULLIF(STDDEV(overnight_gap) OVER (), 0) as gap_zscore
                FROM price_stats
            )
            SELECT 
                COUNT(*) as total_records,
                -- Outlier counts
                SUM(CASE WHEN ABS(return_zscore) > {z_score_threshold} THEN 1 ELSE 0 END) as return_outliers,
                SUM(CASE WHEN ABS(range_zscore) > {z_score_threshold} THEN 1 ELSE 0 END) as range_outliers,
                SUM(CASE WHEN ABS(gap_zscore) > {z_score_threshold} THEN 1 ELSE 0 END) as gap_outliers,
                -- Summary statistics
                AVG(daily_return) as avg_daily_return,
                STDDEV(daily_return) as daily_return_std,
                AVG(daily_range) as avg_daily_range,
                STDDEV(daily_range) as daily_range_std,
                AVG(ABS(overnight_gap)) as avg_abs_gap,
                MAX(ABS(overnight_gap)) as max_abs_gap
            FROM stats
            """
            
            result = self.conn.execute(query).fetchdf()
            if result.empty:
                logger.warning(f"No data found for statistical analysis of {symbol}")
                return None
            
            return result.iloc[0]
        except Exception as e:
            logger.error(f"Error performing statistical checks for {symbol}: {e}")
            return None

    def analyze_symbol(self, symbol, calendar_name='US', frequency=None):
        """Perform a comprehensive analysis of a symbol's data."""
        # Get date range
        date_range = self.get_symbol_date_range(symbol)
        if date_range is None:
            logger.error(f"Could not determine date range for {symbol}")
            return
        
        logger.info(f"\nAnalysis for {symbol}:")
        logger.info(f"Start date: {date_range['start_date']}")
        logger.info(f"End date: {date_range['end_date']}")
        
        # Get total record count
        total_records = self.get_total_records(symbol)
        logger.info(f"Total records: {total_records}")
        
        # Price data validation
        logger.info("\nPrice Data Validation:")
        price_checks = self.check_price_data(symbol)
        if price_checks is not None:
            logger.info(f"  OHLC violations: {price_checks['ohlc_violations']} records")
            logger.info(f"  Zero/negative prices: {price_checks['zero_price_violations']} records")
            logger.info(f"  Extreme price moves: {price_checks['extreme_price_violations']} records")
            logger.info(f"  Price range: {price_checks['min_low']:.2f} - {price_checks['max_high']:.2f}")
        
        # Volume checks
        logger.info("\nVolume Analysis:")
        volume_checks = self.check_volume_data(symbol)
        if volume_checks is not None:
            logger.info(f"  Zero volume days: {volume_checks['zero_volume_days']} records")
            logger.info(f"  Unusual volume days: {volume_checks['high_volume_days']} records")
            logger.info(f"  Average daily volume: {volume_checks['avg_volume']:,.0f}")
            logger.info(f"  Volume range: {volume_checks['min_volume']:,.0f} - {volume_checks['max_volume']:,.0f}")
        
        # Statistical validation
        logger.info("\nStatistical Analysis:")
        stat_checks = self.check_statistical_validity(symbol)
        if stat_checks is not None:
            logger.info(f"  Return outliers: {stat_checks['return_outliers']} records")
            logger.info(f"  Range outliers: {stat_checks['range_outliers']} records")
            logger.info(f"  Gap outliers: {stat_checks['gap_outliers']} records")
            logger.info(f"  Average daily range: {(stat_checks['avg_daily_range'] * 100):.2f}%")
            logger.info(f"  Average daily volatility: {(stat_checks['daily_return_std'] * 100):.2f}%")
            logger.info(f"  Maximum gap between sessions: {(stat_checks['max_abs_gap'] * 100):.2f}%")
        
        # Get individual contract information
        try:
            query = f"""
            SELECT 
                symbol,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as record_count
            FROM market_data
            WHERE symbol LIKE '{symbol}%'
            GROUP BY symbol
            ORDER BY symbol
            """
            contracts_df = self.conn.execute(query).fetchdf()
            if not contracts_df.empty:
                logger.info("\nIndividual contracts:")
                for _, row in contracts_df.iterrows():
                    logger.info(f"  {row['symbol']}: {row['record_count']} records from {row['start_date']} to {row['end_date']}")
        except Exception as e:
            logger.error(f"Error getting contract information: {e}")
        
        # Check for gaps
        if frequency:
            # Parse frequency string (e.g., "1min", "15min", "daily")
            if frequency.endswith('min'):
                interval_value = int(frequency[:-3])
                interval_unit = 'minute'
            elif frequency == 'daily':
                interval_value = 1
                interval_unit = 'day'
            else:
                logger.warning(f"Unrecognized frequency format: {frequency}. Using all frequencies.")
                interval_value = None
                interval_unit = None
                
            gaps = self.check_gaps(symbol, calendar_name=calendar_name, 
                                 interval_value=interval_value, interval_unit=interval_unit)
        else:
            gaps = self.check_gaps(symbol, calendar_name=calendar_name)
            
        if gaps:
            logger.info("\nGaps found:")
            for gap in gaps:
                logger.info(f"Gap of {gap['trading_days']} trading days ({gap['calendar_days']} calendar days) between {gap['start_date']} and {gap['end_date']}")
        else:
            logger.info("\nNo significant gaps found")
            
    def get_continuous_contract_info(self, base_symbol):
        """Get information about the continuous contract for a base symbol."""
        try:
            # First check if there's a continuous contract with the base symbol name
            query = f"""
            SELECT 
                symbol,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as total_records
            FROM continuous_contracts
            WHERE symbol LIKE '{base_symbol}%'
            GROUP BY symbol
            """
            result = self.conn.execute(query).fetchdf()
            
            if not result.empty:
                row = result.iloc[0]
                return {
                    'symbol': row['symbol'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'total_records': row['total_records']
                }
            
            # If no direct match, check for any continuous contract that might be related
            query = f"""
            SELECT 
                symbol,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as total_records
            FROM continuous_contracts
            WHERE symbol LIKE '%{base_symbol}%'
            GROUP BY symbol
            """
            result = self.conn.execute(query).fetchdf()
            
            if not result.empty:
                row = result.iloc[0]
                return {
                    'symbol': row['symbol'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'total_records': row['total_records']
                }
                
            return None
        except Exception as e:
            logger.error(f"Error getting continuous contract info for {base_symbol}: {e}")
            return None

    def get_all_symbol_intervals(self):
        """Get a list of all unique symbol-interval combinations in the database."""
        try:
            # Get symbols and intervals from market_data table
            query = """
            SELECT DISTINCT 
                symbol,
                interval_value,
                interval_unit,
                COUNT(*) as record_count,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM market_data
            GROUP BY symbol, interval_value, interval_unit
            ORDER BY symbol, 
                CASE interval_unit 
                    WHEN 'daily' THEN 1 
                    WHEN 'minute' THEN 2 
                    ELSE 3 
                END,
                interval_value
            """
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting symbol-interval combinations: {e}")
            return pd.DataFrame()

    def format_interval(self, interval_value, interval_unit):
        """Format interval value and unit into a readable string."""
        if interval_unit == 'daily':
            return 'Daily'
        elif interval_unit == 'minute':
            if interval_value == 1:
                return '1 Minute'
            else:
                return f'{interval_value} Minutes'
        else:
            return f'{interval_value} {interval_unit}'

    def print_symbol_intervals(self):
        """Print all symbol-interval combinations in a formatted table."""
        df = self.get_all_symbol_intervals()
        if df.empty:
            logger.error("No data found in the database")
            return

        # Group by symbol for better organization
        symbols = df['symbol'].unique()
        
        print("\nSymbol-Interval Combinations:")
        print("=" * 100)
        print(f"{'Symbol':<15} {'Interval':<15} {'Records':<10} {'Date Range':<50}")
        print("-" * 100)
        
        for symbol in sorted(symbols):
            symbol_data = df[df['symbol'] == symbol]
            first_row = True
            
            for _, row in symbol_data.iterrows():
                interval_str = self.format_interval(row['interval_value'], row['interval_unit'])
                date_range = f"{row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}"
                
                if first_row:
                    print(f"{symbol:<15} {interval_str:<15} {row['record_count']:<10,d} {date_range:<50}")
                    first_row = False
                else:
                    print(f"{'↳':<15} {interval_str:<15} {row['record_count']:<10,d} {date_range:<50}")
            
            print("-" * 100)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Check market data in the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all symbol-interval combinations
    python check_market_data.py --list-symbols
    
    # Analyze a specific symbol
    python check_market_data.py --symbol ES
    
    # Check for gaps with a custom threshold
    python check_market_data.py --symbol ES --max-gap-days 5
    
    # Analyze all symbols
    python check_market_data.py
        """
    )
    parser.add_argument('--list-symbols', action='store_true', help='List all symbol-interval combinations')
    parser.add_argument('--symbol', help='Symbol to analyze')
    parser.add_argument('--max-gap-days', type=int, default=3, help='Maximum allowed gap in days')
    parser.add_argument('--calendar', default='US', help='Holiday calendar to use')
    args = parser.parse_args()
    
    try:
        checker = MarketDataChecker()
        
        if args.list_symbols:
            checker.print_symbol_intervals()
            return
            
        if args.symbol:
            checker.analyze_symbol(args.symbol, args.calendar)
        else:
            symbols = checker.get_all_symbols()
            for _, row in symbols.iterrows():
                checker.analyze_symbol(row['symbol'], args.calendar)
                
    except Exception as e:
        logger.error(f"Error running script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 