#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixes missing VX continuous contracts data by regenerating dates where VIX data exists.
This script focuses on generating only the missing dates rather than regenerating all data.
It ensures that all standard series (101XN through 501XN) are generated for each date.
"""

import sys
import os
from pathlib import Path

# Add project root to the path for imports
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(project_root)

import argparse
import logging
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils.continuous_contracts import get_active_contract, get_all_active_contracts

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO):
    """Configure logging for this script."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def connect_db(db_path, read_only=False):
    """Connect to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to database: {db_path} (Read-Only: {read_only})")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def find_missing_dates(conn, start_date, end_date):
    """
    Find dates where VIX data exists but VX continuous data is missing.
    Returns a list of dates that need to be fixed.
    """
    query = """
    WITH vix_dates AS (
        SELECT DISTINCT timestamp::DATE AS date
        FROM market_data_cboe
        WHERE symbol = '$VIX.X'
        AND interval_unit = 'daily'
        AND timestamp BETWEEN ? AND ?
    ),
    vx_dates AS (
        SELECT DISTINCT timestamp::DATE AS date
        FROM continuous_contracts
        WHERE symbol LIKE '@VX=%'
        AND timestamp BETWEEN ? AND ?
    )
    SELECT date FROM vix_dates
    WHERE date NOT IN (SELECT date FROM vx_dates)
    ORDER BY date
    """
    
    try:
        result = conn.execute(query, [start_date, end_date, start_date, end_date]).fetchdf()
        missing_dates = pd.to_datetime(result['date']).tolist() if not result.empty else []
        logger.info(f"Found {len(missing_dates)} dates with VIX data but missing VX continuous data")
        return missing_dates
    except Exception as e:
        logger.error(f"Error finding missing dates: {e}")
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

def generate_continuous_for_date(conn, date):
    """
    Generate continuous contract data for a specific date.
    
    IMPORTANT: This function always generates all standard series (101XN through 501XN)
    for each date to ensure complete data coverage.
    """
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Generating continuous data for: {date_str}")
    
    # Get all contract mappings for this date
    try:
        # Get all mappings without restricting to specific series
        query = """
        SELECT continuous_symbol, underlying_symbol
        FROM continuous_contract_mapping
        WHERE date = ?
        AND continuous_symbol LIKE '@VX=%'
        ORDER BY continuous_symbol
        """
        
        result = conn.execute(query, [date_str]).fetchdf()
        
        if result.empty:
            logger.warning(f"No contract mapping found for {date_str}")
            return pd.DataFrame()
            
        contracts_map = dict(zip(result['continuous_symbol'], result['underlying_symbol']))
        logger.debug(f"Found {len(contracts_map)} contract mappings for {date_str}")
    except Exception as e:
        logger.error(f"Error getting contract mappings for {date_str}: {e}")
        return pd.DataFrame()
    
    # Create rows for the standard 5 continuous contracts (101XN through 501XN)
    rows = []
    standard_series_count = 0
    
    # Always process the standard series in order (101XN through 501XN)
    for c_num in range(1, 6):  # Standard series 1-5 (101-501)
        continuous_symbol = f"@VX={c_num}01XN"
        
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
            'built_by': 'vx_fix_script'  # Mark these records as fixed by our script
        }
        
        rows.append(continuous_record)
        standard_series_count += 1
    
    # Log summary of generated data
    if standard_series_count > 0:
        logger.info(f"Generated {standard_series_count} standard series contracts for {date_str}")
    else:
        logger.warning(f"Could not generate any standard series contracts for {date_str}")
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def insert_continuous_data(conn, df):
    """Insert the generated continuous contract data into the database."""
    if df.empty:
        return 0
        
    # Prepare for insertion
    df = df.copy()
    if 'underlying_symbol' in df.columns:
        # Keep underlying_symbol for continuous contracts
        pass
    
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

def check_missing_standard_series(conn, start_date, end_date):
    """
    Check for dates that have some VX continuous data but are missing standard series.
    This can happen if higher-numbered series exist but the standard 101-501 are missing.
    """
    query = """
    WITH vx_dates AS (
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
    SELECT date FROM vx_dates
    WHERE date NOT IN (SELECT date FROM complete_dates)
    ORDER BY date
    """
    
    try:
        result = conn.execute(query, [start_date, end_date, start_date, end_date]).fetchdf()
        incomplete_dates = pd.to_datetime(result['date']).tolist() if not result.empty else []
        
        if incomplete_dates:
            logger.info(f"Found {len(incomplete_dates)} dates with VX data but missing standard series (101-501)")
        else:
            logger.info("All dates with VX data have complete standard series (101-501)")
        
        return incomplete_dates
    except Exception as e:
        logger.error(f"Error checking for incomplete standard series: {e}")
        return []

def fix_missing_vx_data(db_path, start_date, end_date, check_standard_series=True, dry_run=False):
    """
    Fix missing VX continuous contract data for dates where VIX data exists.
    
    Args:
        db_path: Path to the DuckDB database
        start_date: Start date for the fix operation
        end_date: End date for the fix operation
        check_standard_series: If True, also check for dates with missing standard series
        dry_run: If True, only show what would be fixed without making changes
    """
    # Connect to database
    conn = connect_db(db_path, read_only=dry_run)
    
    try:
        # First find completely missing dates
        missing_dates = find_missing_dates(conn, start_date, end_date)
        
        if missing_dates:
            logger.info(f"Found {len(missing_dates)} dates completely missing VX continuous data")
            
            # Process each missing date
            fix_count = 0
            for date in missing_dates:
                # Generate continuous data for this date
                df = generate_continuous_for_date(conn, date)
                
                if df.empty:
                    logger.warning(f"Could not generate data for {date.strftime('%Y-%m-%d')}")
                    continue
                    
                if dry_run:
                    logger.info(f"Would insert {len(df)} records for {date.strftime('%Y-%m-%d')}")
                else:
                    # Insert the generated data
                    inserted = insert_continuous_data(conn, df)
                    if inserted > 0:
                        fix_count += 1
            
            if dry_run:
                logger.info(f"Dry run completed. Would fix {len(missing_dates)} completely missing dates.")
            else:
                logger.info(f"Fixed {fix_count} completely missing dates")
        else:
            logger.info("No completely missing dates found")
        
        # Now check for dates with incomplete standard series
        if check_standard_series and not dry_run:
            incomplete_dates = check_missing_standard_series(conn, start_date, end_date)
            
            if incomplete_dates:
                logger.info(f"Fixing {len(incomplete_dates)} dates with incomplete standard series")
                
                # Process each incomplete date
                fix_count = 0
                for date in incomplete_dates:
                    # Generate continuous data for this date
                    df = generate_continuous_for_date(conn, date)
                    
                    if df.empty:
                        logger.warning(f"Could not generate standard series for {date.strftime('%Y-%m-%d')}")
                        continue
                        
                    # Insert the generated data (will skip existing records)
                    inserted = insert_continuous_data(conn, df)
                    if inserted > 0:
                        fix_count += 1
                
                logger.info(f"Fixed {fix_count} dates with incomplete standard series")
    finally:
        # Close the database connection
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Fix missing VX continuous contract data')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb', 
                        help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2004-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'), 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--check-standard-series', action='store_true', 
                        help='Also check for dates with incomplete standard series (101-501)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be fixed without making changes')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger.info(f"Starting VX continuous data fix from {args.start_date} to {args.end_date}")
    
    # Run the fix operation
    fix_missing_vx_data(
        args.db_path,
        args.start_date,
        args.end_date,
        args.check_standard_series,
        args.dry_run
    )
    
    logger.info("VX continuous data fix operation completed")

if __name__ == "__main__":
    main() 