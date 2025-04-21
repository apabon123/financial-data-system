#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verifies generated continuous futures contracts against underlying data and roll calendar.

Compares the settlement price and underlying symbol used in a continuous contract
(e.g., VXc1) against the expected underlying contract derived from the
futures_roll_calendar table.
"""

import os
import sys
import duckdb
import pandas as pd
import argparse
import logging
from datetime import datetime
import re # For parsing continuous symbol

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"

# --- Database Operations ---
def connect_db(db_path):
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def load_roll_calendar(conn, root_symbol):
    """Loads the roll calendar for a root symbol into a dict {contract_code: last_trading_day}."""
    query = f"""
        SELECT contract_code, last_trading_day
        FROM futures_roll_calendar
        WHERE root_symbol = ?
        ORDER BY last_trading_day ASC
    """
    try:
        df_calendar = conn.execute(query, [root_symbol]).fetchdf()
        df_calendar['last_trading_day'] = pd.to_datetime(df_calendar['last_trading_day'])
        calendar_dict = pd.Series(df_calendar.last_trading_day.values, index=df_calendar.contract_code).to_dict()
        logger.info(f"Loaded {len(calendar_dict)} roll calendar entries for {root_symbol}")
        return calendar_dict
    except Exception as e:
        logger.error(f"Error loading roll calendar for {root_symbol}: {e}")
        return {}

def load_continuous_data(conn, continuous_symbol, start_date, end_date):
    """Loads data for the specified continuous contract."""
    query = f"""
        SELECT timestamp, settle, UnderlyingSymbol
        FROM market_data
        WHERE Symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    try:
        df = conn.execute(query, [continuous_symbol, start_date, end_date]).fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} data points for continuous contract {continuous_symbol}")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {continuous_symbol}: {e}")
        return pd.DataFrame()

def load_underlying_settles(conn, symbols: set, start_date, end_date):
    """Loads settlement prices for a set of underlying symbols into a dict {(symbol, date): settle}."""
    if not symbols:
        return {}
    # Convert set to list for parameter binding
    symbol_list = list(symbols)
    placeholders = ', '.join('?' * len(symbol_list))
    query = f"""
        SELECT timestamp, symbol, settle
        FROM market_data
        WHERE symbol IN ({placeholders}) AND timestamp BETWEEN ? AND ?
    """
    params = symbol_list + [start_date, end_date]
    settle_map = {}
    try:
        df = conn.execute(query, params).fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for _, row in df.iterrows():
            settle_map[(row['symbol'], row['timestamp'])] = row['settle']
        logger.info(f"Loaded {len(settle_map)} settlement prices for {len(symbols)} underlying contracts.")
        return settle_map
    except Exception as e:
        logger.error(f"Error loading underlying settlement data: {e}")
        return {}

# --- Verification Logic ---
def _get_expected_underlying_sequence(current_date, all_contract_codes, roll_calendar):
    """Determines the sequence of active contracts for a date based on the calendar."""
    current_date_naive = current_date.replace(tzinfo=None) if current_date.tzinfo else current_date

    # Filter and sort potential contracts by last trading day
    # Contract is active if its LTD is on or after the current date
    potential_contracts = sorted(
        [c for c in all_contract_codes if roll_calendar.get(c, datetime.min) >= current_date_naive],
        key=lambda c: roll_calendar.get(c, datetime.max)
    )

    return potential_contracts # The sequence is just the sorted list of contracts active on or after current_date

def verify_data(continuous_data, underlying_settles, roll_calendar, root_symbol, contract_num):
    """Performs the verification checks and returns results."""
    verification_results = []
    all_contract_codes = list(roll_calendar.keys()) # All codes known to the calendar

    for _, row in continuous_data.iterrows():
        current_dt = row['timestamp']
        continuous_settle = row['settle']
        generator_underlying = row['UnderlyingSymbol']

        expected_sequence = _get_expected_underlying_sequence(current_dt, all_contract_codes, roll_calendar)

        expected_underlying = None
        if len(expected_sequence) >= contract_num:
            expected_underlying = expected_sequence[contract_num - 1]
        else:
            logger.debug(f"Could not determine expected underlying contract #{contract_num} for {current_dt.date()}")

        # Look up settles using original values from continuous_data row first
        generator_underlying_settle_lookup = underlying_settles.get((generator_underlying, current_dt), pd.NA)
        expected_underlying_settle_lookup = underlying_settles.get((expected_underlying, current_dt), pd.NA) if expected_underlying else pd.NA

        # --- Determine Check Status (Revised Logic) ---
        check_status = "UNKNOWN" # Default

        if expected_underlying is None:
            check_status = "FAIL: No Expected Contract" # Cannot determine the Nth contract
        elif pd.isna(generator_underlying):
             check_status = "FAIL: Generator Missing Underlying" # Generator didn't store which contract it used
        elif pd.isna(continuous_settle):
            check_status = "FAIL: Continuous Settle is NA"
        else:
            # Both expected and generator underlying symbols are known, and continuous settle exists
            # Use the looked-up values for comparison logic
            generator_settle = generator_underlying_settle_lookup
            expected_settle = expected_underlying_settle_lookup

            if generator_underlying == expected_underlying:
                # --- Symbols Match ---
                if pd.isna(expected_settle):
                     check_status = "WARN: Missing Expected Settle" # Symbol OK, but expected price missing
                elif abs(continuous_settle - expected_settle) < 0.001: # Add tolerance for float comparison
                     check_status = "OK"
                else:
                     # Prices mismatch, even though symbols are correct
                     check_status = f"FAIL: Price Mismatch (Expected {expected_settle:.2f})"

            else:
                # --- Symbols Mismatch ---
                status_base = "FAIL: Underlying Mismatch"
                details = []
                if pd.isna(generator_settle):
                    details.append("GenSettle=NA")
                # Check if continuous settle matches the (wrong) generator settle
                elif not pd.isna(generator_settle) and abs(continuous_settle - generator_settle) > 0.001: # Use tolerance
                    details.append(f"Cont!=GenSettle({generator_settle:.2f})")

                if pd.isna(expected_settle):
                    details.append("ExpSettle=NA")

                if details:
                    check_status = f"{status_base} ({', '.join(details)})"
                else:
                    # Symbols mismatch, but Cont price *did* match Gen price (within tolerance), and Exp price exists
                    check_status = status_base
        # --- End Revised Logic ---

        verification_results.append({
            "Date": current_dt.date(),
            "ContinuousSettle": continuous_settle,
            "GeneratorUnderlying": generator_underlying,
            "ExpectedUnderlying": expected_underlying,
            "GeneratorSettle": generator_underlying_settle_lookup, # Report looked up value
            "ExpectedSettle": expected_underlying_settle_lookup,   # Report looked up value
            "CheckStatus": check_status
        })

    return pd.DataFrame(verification_results)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Verify Continuous Futures Contracts.')
    parser.add_argument('--symbol', type=str, required=True, help='Continuous contract symbol (e.g., VXc1).')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD).')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')

    args = parser.parse_args()

    # Parse continuous symbol
    match = re.match(r'^([A-Z]+)c(\d+)$', args.symbol, re.IGNORECASE)
    if not match:
        logger.error(f"Invalid continuous symbol format: {args.symbol}. Expected format like 'VXc1'.")
        sys.exit(1)
    root_symbol = match.group(1).upper()
    contract_num = int(match.group(2))
    logger.info(f"Verifying {args.symbol} (Root: {root_symbol}, Number: {contract_num})")

    conn = None
    try:
        conn = connect_db(args.db_path)

        # Load Roll Calendar
        roll_calendar = load_roll_calendar(conn, root_symbol)
        if not roll_calendar:
            logger.error("Failed to load roll calendar. Cannot proceed.")
            sys.exit(1)

        # Load Continuous Data
        continuous_data = load_continuous_data(conn, args.symbol, args.start_date, args.end_date)
        if continuous_data.empty:
            logger.warning(f"No data found for {args.symbol} in the specified date range.")
            sys.exit(0)

        # Identify and Load Underlying Data
        needed_underlying_symbols = set(continuous_data['UnderlyingSymbol'].dropna().unique())
        # Also add symbols from the calendar that *could* be expected
        if roll_calendar: # Ensure calendar was loaded
            for code, ltd in roll_calendar.items():
                try:
                    # Heuristic to load contracts relevant to the date range (+/- 1 year)
                    ltd_year = ltd.year
                    start_year = datetime.strptime(args.start_date, '%Y-%m-%d').year
                    end_year = datetime.strptime(args.end_date, '%Y-%m-%d').year
                    if start_year - 1 <= ltd_year <= end_year + 1:
                        needed_underlying_symbols.add(code)
                except Exception as e:
                     logger.warning(f"Could not process roll calendar entry {code}: {ltd} - {e}")

        underlying_settles = load_underlying_settles(conn, needed_underlying_symbols, args.start_date, args.end_date)

        # Perform Verification
        results_df = verify_data(continuous_data, underlying_settles, roll_calendar, root_symbol, contract_num)

        # Print Results
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n--- Verification Results ---")
        print(results_df.to_string(index=False))

        # Summary
        status_counts = results_df['CheckStatus'].value_counts()
        print("\n--- Summary ---")
        print(status_counts)

    except Exception as e:
        logger.error(f"An unexpected error occurred during verification: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 