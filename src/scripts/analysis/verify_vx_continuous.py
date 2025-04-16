#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verifies the generated VX continuous futures contracts for data quality issues.

Checks performed:
1.  Data points on Sundays.
2.  Large day-over-day price gaps (close-to-close).
3.  Missing trading days (date gaps, excluding weekends).
"""

import os
import sys
import logging
import argparse
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"
PRICE_GAP_THRESHOLD = 0.20 # Percentage (e.g., 0.20 for 20%)

def connect_db(db_file: Path = DB_PATH, read_only: bool = True) -> Optional[duckdb.DuckDBPyConnection]:
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=str(db_file), read_only=read_only)
        logger.info(f"Successfully connected to database: {db_file} (Read-Only)")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database {db_file}: {e}")
        return None

def get_continuous_contracts(conn: duckdb.DuckDBPyConnection, symbol_prefix: str = 'VXc') -> List[str]:
    """Gets a list of continuous contract symbols from the database."""
    query = "SELECT DISTINCT symbol FROM continuous_contracts WHERE symbol LIKE ? ORDER BY symbol"
    try:
        symbols = conn.execute(query, [f"{symbol_prefix}%"]).df()['symbol'].tolist()
        logger.info(f"Found continuous contracts: {symbols}")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching continuous contract symbols: {e}")
        return []

def check_sunday_data(conn: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    """Checks for data points falling on a Sunday."""
    logger.info(f"[{symbol}] Checking for data on Sundays...")
    query = f"""
    SELECT timestamp, symbol, close
    FROM continuous_contracts
    WHERE symbol = ?
      AND EXTRACT(DOW FROM timestamp) = 0 -- 0 = Sunday in DuckDB's EXTRACT(DOW)
    ORDER BY timestamp;
    """
    try:
        sunday_data = conn.execute(query, [symbol]).fetchdf()
        if not sunday_data.empty:
            logger.warning(f"[{symbol}] Found {len(sunday_data)} data points on Sundays:")
            # Log details of the first few issues
            for _, row in sunday_data.head().iterrows():
                logger.warning(f"  - {row['timestamp'].date()}")
        else:
            logger.info(f"[{symbol}] No data found on Sundays.")
        return sunday_data
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for Sunday data: {e}")
        return pd.DataFrame()

def check_price_gaps(conn: duckdb.DuckDBPyConnection, symbol: str, threshold: float) -> pd.DataFrame:
    """Checks for large day-over-day percentage changes in the closing price."""
    logger.info(f"[{symbol}] Checking for price gaps larger than {threshold:.0%}...")
    query = f"""
    WITH lagged_data AS (
        SELECT
            timestamp,
            close,
            LAG(close, 1) OVER (ORDER BY timestamp) as prev_close
        FROM continuous_contracts
        WHERE symbol = ?
    )
    SELECT
        timestamp,
        close,
        prev_close,
        (close - prev_close) / prev_close as pct_change
    FROM lagged_data
    WHERE prev_close IS NOT NULL
      AND ABS((close - prev_close) / prev_close) > ?
    ORDER BY timestamp;
    """
    try:
        gaps_df = conn.execute(query, [symbol, threshold]).fetchdf()
        if not gaps_df.empty:
            logger.warning(f"[{symbol}] Found {len(gaps_df)} large price gaps (threshold > {threshold:.0%}):")
            # Log details of the first few issues
            for _, row in gaps_df.head().iterrows():
                 logger.warning(f"  - {row['timestamp'].date()}: Close={row['close']:.2f}, PrevClose={row['prev_close']:.2f}, Change={row['pct_change']:.1%}")
        else:
            logger.info(f"[{symbol}] No large price gaps found.")
        return gaps_df
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for price gaps: {e}")
        return pd.DataFrame()

def check_date_gaps(conn: duckdb.DuckDBPyConnection, symbol: str) -> List[pd.Timestamp]:
    """Checks for missing trading days (excluding weekends)."""
    logger.info(f"[{symbol}] Checking for missing trading days (date gaps)...")
    query = f"""
    SELECT DISTINCT timestamp::DATE as date
    FROM continuous_contracts
    WHERE symbol = ?
    ORDER BY date;
    """
    try:
        dates_df = conn.execute(query, [symbol]).fetchdf()
        if dates_df.empty:
            logger.warning(f"[{symbol}] No data found to check for date gaps.")
            return []

        dates_df['date'] = pd.to_datetime(dates_df['date'])
        # Create a complete date range from min to max date
        min_date = dates_df['date'].min()
        max_date = dates_df['date'].max()
        all_days = pd.date_range(start=min_date, end=max_date, freq='B') # 'B' is business day frequency

        # Find missing dates
        missing_dates = all_days.difference(dates_df['date']).tolist()

        if missing_dates:
            logger.warning(f"[{symbol}] Found {len(missing_dates)} missing trading days (date gaps):")
            # Log the first few missing dates
            for missing_date in missing_dates[:5]:
                logger.warning(f"  - {missing_date.date()}")
            if len(missing_dates) > 5:
                logger.warning(f"  - ... and {len(missing_dates) - 5} more")
        else:
            logger.info(f"[{symbol}] No missing trading days found.")
        return missing_dates
    except Exception as e:
        logger.error(f"[{symbol}] Error checking for date gaps: {e}")
        return []

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Verify continuous futures contract data.")
    parser.add_argument("--symbol-prefix", type=str, default="VXc", help="Prefix for continuous contract symbols (e.g., 'VXc')")
    parser.add_argument("--gap-threshold", type=float, default=PRICE_GAP_THRESHOLD, help="Percentage threshold for price gap detection (e.g., 0.2 for 20%)")
    args = parser.parse_args()

    logger.info(f"Starting continuous contract verification for symbols starting with '{args.symbol_prefix}'")

    conn = connect_db()
    if not conn:
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)

    symbols_to_check = get_continuous_contracts(conn, args.symbol_prefix)
    if not symbols_to_check:
        logger.warning(f"No continuous contracts found with prefix '{args.symbol_prefix}'. Exiting.")
        conn.close()
        sys.exit(0)

    total_issues = 0
    try:
        for symbol in symbols_to_check:
            logger.info(f"--- Verifying {symbol} ---")
            sunday_issues = check_sunday_data(conn, symbol)
            price_gap_issues = check_price_gaps(conn, symbol, args.gap_threshold)
            date_gap_issues = check_date_gaps(conn, symbol)

            symbol_issues = len(sunday_issues) + len(price_gap_issues) + len(date_gap_issues)
            total_issues += symbol_issues
            if symbol_issues == 0:
                logger.info(f"[{symbol}] Verification complete. No issues found.")
            else:
                 logger.warning(f"[{symbol}] Verification complete. Found {symbol_issues} potential issues.")
            logger.info("------------------------")

    except Exception as e:
        logger.error(f"An unexpected error occurred during verification: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    logger.info(f"=== Verification Summary ===")
    if total_issues == 0:
        logger.info("All checked contracts passed verification.")
    else:
        logger.warning(f"Verification finished. Found a total of {total_issues} potential issues across all checked contracts.")
    logger.info("==========================")

if __name__ == "__main__":
    main() 