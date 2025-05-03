#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verify that all VX continuous series (101-501) are complete for
the specified date range.
"""

import argparse
import logging
import duckdb
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def connect_db(db_path, read_only=True):
    """Connect to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        return None

def check_series_completeness(conn, start_date, end_date):
    """
    Check if all VX continuous series are complete for the given date range.
    
    A complete series means each trading day has all standard series (101-501) present.
    """
    query = """
    WITH all_dates AS (
        SELECT DISTINCT CAST(timestamp AS DATE) as date
        FROM continuous_contracts
        WHERE symbol LIKE '@VX=%'
        AND timestamp BETWEEN ? AND ?
    ),
    series_counts AS (
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
    ),
    incomplete_dates AS (
        SELECT 
            s.date,
            s.has_101, s.has_201, s.has_301, s.has_401, s.has_501
        FROM series_counts s
        WHERE s.has_101 = 0 OR s.has_201 = 0 OR s.has_301 = 0 OR s.has_401 = 0 OR s.has_501 = 0
    )
    
    SELECT 
        (SELECT COUNT(*) FROM all_dates) as total_dates,
        (SELECT COUNT(*) FROM series_counts WHERE 
            has_101 > 0 AND has_201 > 0 AND has_301 > 0 AND has_401 > 0 AND has_501 > 0) as complete_dates,
        (SELECT COUNT(*) FROM incomplete_dates) as incomplete_dates
    """
    
    try:
        summary = conn.execute(query, [start_date, end_date, start_date, end_date]).fetchone()
        total_dates, complete_dates, incomplete_dates = summary
        
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Total dates with VX data: {total_dates}")
        logger.info(f"Complete dates (all series 101-501): {complete_dates}")
        logger.info(f"Incomplete dates: {incomplete_dates}")
        
        # If there are any incomplete dates, show them
        if incomplete_dates > 0:
            details_query = """
            WITH series_counts AS (
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
            SELECT 
                date, 
                has_101, has_201, has_301, has_401, has_501
            FROM series_counts
            WHERE has_101 = 0 OR has_201 = 0 OR has_301 = 0 OR has_401 = 0 OR has_501 = 0
            ORDER BY date
            """
            
            incomplete = conn.execute(details_query, [start_date, end_date]).fetchdf()
            logger.info("Incomplete dates details:")
            for _, row in incomplete.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                missing_series = []
                if row['has_101'] == 0: missing_series.append("101")
                if row['has_201'] == 0: missing_series.append("201")
                if row['has_301'] == 0: missing_series.append("301")
                if row['has_401'] == 0: missing_series.append("401")
                if row['has_501'] == 0: missing_series.append("501")
                
                logger.info(f"  {date_str}: Missing series {', '.join(missing_series)}")
            
            # Check if these days are holidays
            holidays_query = """
            SELECT CAST(timestamp AS DATE) as date, symbol 
            FROM market_data_cboe 
            WHERE symbol = '$VIX.X' 
            AND interval_unit = 'daily'
            AND timestamp BETWEEN ? AND ?
            ORDER BY date
            """
            vix_dates = conn.execute(holidays_query, [start_date, end_date]).fetchdf()
            vix_dates_set = set(vix_dates['date'].dt.strftime('%Y-%m-%d').tolist()) if not vix_dates.empty else set()
            
            missing_dates_set = set(incomplete['date'].dt.strftime('%Y-%m-%d').tolist())
            
            holidays = missing_dates_set - vix_dates_set
            if holidays:
                logger.info("These incomplete dates appear to be holidays (no VIX data):")
                for holiday in sorted(list(holidays)):
                    logger.info(f"  {holiday}")
        
        return complete_dates == total_dates
    except Exception as e:
        logger.error(f"Error checking series completeness: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify VX continuous series completeness')
    parser.add_argument('--db-path', type=str, default='data/financial_data.duckdb',
                        help='Path to the DuckDB database file')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-02-28',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting VX continuous series verification")
    
    # Connect to the database
    conn = connect_db(args.db_path)
    if not conn:
        return
    
    try:
        # Check completeness
        is_complete = check_series_completeness(conn, args.start_date, args.end_date)
        
        if is_complete:
            logger.info("✅ All VX continuous series are complete for the specified date range!")
        else:
            logger.warning("⚠️ There are incomplete VX continuous series in the specified date range.")
    finally:
        conn.close()
    
    logger.info("VX continuous series verification completed")

if __name__ == "__main__":
    main() 