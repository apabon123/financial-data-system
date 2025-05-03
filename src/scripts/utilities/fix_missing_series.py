#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script to add the standard VX continuous contract series (101-501) for dates where only 
higher-numbered series (601-901) exist or where the pattern is incomplete.
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def connect_db(db_path, read_only=False):
    """Connect to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to database: {db_path} (Read-Only: {read_only})")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def find_dates_needing_standard_series(conn, start_date, end_date):
    """
    Find dates that need the standard series (101-501) fixed.
    This includes dates with higher series but missing standard series,
    as well as dates with incomplete standard series.
    """
    query = """
    WITH all_dates AS (
        SELECT DISTINCT CAST(timestamp AS DATE) as date
        FROM continuous_contracts
        WHERE symbol LIKE '@VX=%'
        AND timestamp BETWEEN ? AND ?
    ),
    complete_dates AS (
        SELECT date
        FROM (
            SELECT 
                CAST(timestamp AS DATE) as date, 
                COUNT(DISTINCT CASE WHEN symbol LIKE '@VX=1%' THEN symbol END) as has_101,
                COUNT(DISTINCT CASE WHEN symbol LIKE '@VX=2%' THEN symbol END) as has_201,
                COUNT(DISTINCT CASE WHEN symbol LIKE '@VX=3%' THEN symbol END) as has_301,
                COUNT(DISTINCT CASE WHEN symbol LIKE '@VX=4%' THEN symbol END) as has_401,
                COUNT(DISTINCT CASE WHEN symbol LIKE '@VX=5%' THEN symbol END) as has_501
            FROM continuous_contracts
            WHERE symbol LIKE '@VX=%'
            AND timestamp BETWEEN ? AND ?
            GROUP BY date
        )
        WHERE has_101 > 0 AND has_201 > 0 AND has_301 > 0 AND has_401 > 0 AND has_501 > 0
    )
    SELECT date FROM all_dates
    WHERE date NOT IN (SELECT date FROM complete_dates)
    ORDER BY date
    """
    
    try:
        result = conn.execute(query, [start_date, end_date, start_date, end_date]).fetchdf()
        dates = pd.to_datetime(result['date']).tolist() if not result.empty else []
        logger.info(f"Found {len(dates)} dates needing standard series fixes")
        return dates
    except Exception as e:
        logger.error(f"Error finding dates: {e}")
        return []

def load_underlying_data(conn, symbol, date):
    """Load market data for a specific VX futures contract on a given date."""
    query = """
    SELECT 
        timestamp, symbol, open, high, low, close, settle, volume, open_interest, source
    FROM market_data_cboe
    WHERE symbol = ?
    AND timestamp = ?
    AND interval_unit = 'daily'
    """
    
    try:
        df = conn.execute(query, [symbol, date.strftime('%Y-%m-%d')]).fetchdf()
        if df.empty:
            logger.debug(f"No data found for {symbol} on {date.strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {symbol} on {date.strftime('%Y-%m-%d')}: {e}")
        return pd.DataFrame()

def generate_standard_series_for_date(conn, date):
    """Generate standard series (101-501) continuous contract data for a specific date."""
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Generating standard series for: {date_str}")
    
    # Get contract mappings for this date
    try:
        # Query for all contract mappings
        query = """
        SELECT continuous_symbol, underlying_symbol
        FROM continuous_contract_mapping
        WHERE date = ?
        AND continuous_symbol LIKE '@VX=%'
        ORDER BY continuous_symbol
        """
        
        result = conn.execute(query, [date_str]).fetchdf()
        
        if result.empty:
            logger.warning(f"No contract mappings found for {date_str}")
            return pd.DataFrame()
            
        contracts_map = dict(zip(result['continuous_symbol'], result['underlying_symbol']))
        
        # Check which series already exist in the database
        existing_query = """
        SELECT symbol
        FROM continuous_contracts
        WHERE timestamp = ?
        AND symbol LIKE '@VX=%'
        """
        
        existing_symbols = conn.execute(existing_query, [date_str]).fetchdf()
        existing_set = set(existing_symbols['symbol'].tolist()) if not existing_symbols.empty else set()
        
        logger.debug(f"Existing symbols for {date_str}: {existing_set}")
    except Exception as e:
        logger.error(f"Error preparing for {date_str}: {e}")
        return pd.DataFrame()
    
    # Create rows for the standard 5 continuous contracts (101XN through 501XN)
    rows = []
    for c_num in range(1, 6):  # Series 1-5 (101-501)
        continuous_symbol = f"@VX={c_num}01XN"
        
        # Skip if this symbol already exists
        if continuous_symbol in existing_set:
            logger.debug(f"Skipping existing symbol {continuous_symbol} on {date_str}")
            continue
            
        # Check if we have a mapping for this symbol
        if continuous_symbol not in contracts_map:
            logger.debug(f"No mapping for {continuous_symbol} on {date_str}")
            continue
            
        underlying_symbol = contracts_map[continuous_symbol]
        
        # Load data for the underlying contract
        df_sym = load_underlying_data(conn, underlying_symbol, date)
        
        if df_sym.empty:
            logger.warning(f"No data found for underlying contract {underlying_symbol} on {date_str}")
            continue
            
        # Extract values (using first row if multiple exist)
        row_data = df_sym.iloc[0].to_dict()
        
        # Create a record for the continuous contract
        continuous_record = {
            'timestamp': date,
            'symbol': continuous_symbol,
            'interval_value': 1,
            'interval_unit': 'daily',
            'open': row_data.get('open'),
            'high': row_data.get('high'),
            'low': row_data.get('low'),
            'close': row_data.get('close'),
            'settle': row_data.get('settle', row_data.get('close')),  # Fallback to close
            'volume': row_data.get('volume'),
            'open_interest': row_data.get('open_interest'),
            'source': row_data.get('source', 'DERIVED'),
            'underlying_symbol': underlying_symbol,
            'built_by': 'fix_missing_series'  # Mark these records
        }
        
        rows.append(continuous_record)
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def insert_continuous_data(conn, df):
    """Insert the generated continuous contract data into the database."""
    if df.empty:
        return 0
        
    # Convert timestamp to string format for insertion
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Insert rows
    try:
        table_name = 'continuous_contracts'
        count = 0
        
        for _, row in df.iterrows():
            # Check if this record already exists
            check_query = f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE timestamp = ? AND symbol = ?
            """
            exists = conn.execute(check_query, [row['timestamp'], row['symbol']]).fetchone()[0] > 0
            
            if exists:
                # Skip existing record
                logger.debug(f"Record already exists for {row['symbol']} on {row['timestamp']}")
                continue
                
            # Insert new record
            columns = ', '.join([f'"{col}"' for col in row.index])
            placeholders = ', '.join(['?'] * len(row))
            
            insert_query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
            """
            
            conn.execute(insert_query, list(row))
            count += 1
        
        logger.info(f"Inserted {count} new continuous contract records")
        return count
    except Exception as e:
        logger.error(f"Error inserting continuous data: {e}")
        return 0

def fix_missing_standard_series(db_path, start_date, end_date, dry_run=False):
    """
    Fix dates with incomplete standard series.
    
    Args:
        db_path: Path to the DuckDB database
        start_date: Start date for the fix operation
        end_date: End date for the fix operation
        dry_run: If True, only show what would be fixed without making changes
    """
    # Connect to database
    conn = connect_db(db_path, read_only=dry_run)
    
    try:
        # Find dates to fix
        dates_to_fix = find_dates_needing_standard_series(conn, start_date, end_date)
        
        if not dates_to_fix:
            logger.info("No dates found needing standard series fixes")
            return
            
        logger.info(f"Found {len(dates_to_fix)} dates to fix from {dates_to_fix[0]} to {dates_to_fix[-1]}")
        
        # Process each date
        fix_count = 0
        for date in dates_to_fix:
            # Generate standard series for this date
            df = generate_standard_series_for_date(conn, date)
            
            if df.empty:
                logger.warning(f"Could not generate standard series for {date.strftime('%Y-%m-%d')}")
                continue
                
            if dry_run:
                logger.info(f"Would insert {len(df)} series records for {date.strftime('%Y-%m-%d')}")
            else:
                # Insert the generated data
                inserted = insert_continuous_data(conn, df)
                if inserted > 0:
                    fix_count += 1
        
        if dry_run:
            logger.info(f"Dry run completed. Would fix {len(dates_to_fix)} dates.")
        else:
            logger.info(f"Fixed standard series for {fix_count} dates")
            
    finally:
        # Close the database connection
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Fix missing standard VX continuous contract series')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', 
                        help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2024-11-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-02-28', 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be fixed without making changes')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting VX standard series fix from {args.start_date} to {args.end_date}")
    
    # Run the fix operation
    fix_missing_standard_series(
        args.db_path,
        args.start_date,
        args.end_date,
        args.dry_run
    )
    
    logger.info("VX standard series fix operation completed")

if __name__ == "__main__":
    main() 