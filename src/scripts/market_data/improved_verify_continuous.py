#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Verification Script for Continuous Futures

Compares continuous futures contracts with their underlying contracts and roll calendar data.
Provides detailed output with color-coded status indicators and comprehensive error reporting.
"""

import os
import sys
import duckdb
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime, date
import re
from typing import Dict, List, Set, Tuple, Optional, Any

# Setup logging with colors if available
try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
except ImportError:
    # Fallback to standard logging if colorlog is not available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"
COLOR_OUTPUT = True  # Set to False to disable color output formatting
PRICE_DIFF_THRESHOLD = 0.01  # 1% threshold for price difference warnings

# Color codes for terminal output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'

def c(text: str, color: str) -> str:
    """Conditionally apply color to text based on COLOR_OUTPUT setting."""
    if COLOR_OUTPUT:
        return f"{color}{text}{Colors.RESET}"
    return text

# --- Database Operations ---
def connect_db(db_path):
    """Connects to the DuckDB database."""
    try:
        # Verification should only read
        conn = duckdb.connect(database=db_path, read_only=True)
        logger.info(f"Connected to database: {db_path} (Read-Only)")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def load_roll_calendar(conn, root_symbol):
    """Loads the roll calendar for a root symbol into a dict {contract_code: last_trading_day} and {contract_code: final_settlement_date}."""
    query = f"""
        SELECT contract_code, last_trading_day, final_settlement_date
        FROM futures_roll_calendar
        WHERE root_symbol = ?
        ORDER BY last_trading_day ASC
    """
    try:
        df_calendar = conn.execute(query, [root_symbol]).fetchdf()
        df_calendar['last_trading_day'] = pd.to_datetime(df_calendar['last_trading_day'])
        df_calendar['final_settlement_date'] = pd.to_datetime(df_calendar['final_settlement_date'])
        
        # Create two dictionaries: one for last trading day and one for final settlement date
        ltd_dict = pd.Series(df_calendar.last_trading_day.values, index=df_calendar.contract_code).to_dict()
        fsd_dict = pd.Series(df_calendar.final_settlement_date.values, index=df_calendar.contract_code).to_dict()
        
        logger.info(f"Loaded {len(ltd_dict)} roll calendar entries for {root_symbol}")
        return ltd_dict, fsd_dict
    except Exception as e:
        logger.error(f"Error loading roll calendar for {root_symbol}: {e}")
        return {}, {}

def load_continuous_data(conn, continuous_symbol, start_date, end_date):
    """Loads data for the specified continuous contract."""
    query = f"""
        SELECT timestamp, settle, UnderlyingSymbol, source
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
        SELECT timestamp, symbol, settle, source
        FROM market_data
        WHERE symbol IN ({placeholders}) AND timestamp BETWEEN ? AND ?
    """
    params = symbol_list + [start_date, end_date]
    settle_map = {}
    source_map = {}
    try:
        df = conn.execute(query, params).fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for _, row in df.iterrows():
            settle_map[(row['symbol'], row['timestamp'])] = row['settle']
            source_map[(row['symbol'], row['timestamp'])] = row['source']
        logger.info(f"Loaded {len(settle_map)} settlement prices for {len(symbols)} underlying contracts.")
        return settle_map, source_map
    except Exception as e:
        logger.error(f"Error loading underlying settlement data: {e}")
        return {}, {}

# --- Verification Logic ---
def _get_expected_underlying_sequence(current_date, all_contract_codes, roll_calendar):
    """Determines the sequence of active contracts for a date based on the calendar."""
    current_date_naive = current_date.replace(tzinfo=None) if current_date.tzinfo else current_date
    
    last_trading_days = roll_calendar[0]  # First element is last_trading_day dict

    # Filter and sort potential contracts by last trading day
    # Contract is active if its LTD is on or after the current date
    potential_contracts = sorted(
        [c for c in all_contract_codes if last_trading_days.get(c, datetime.min) >= current_date_naive],
        key=lambda c: last_trading_days.get(c, datetime.max)
    )

    return potential_contracts  # The sequence is just the sorted list of contracts active on or after current_date

def verify_data(conn, continuous_symbol, start_date, end_date):
    """Performs the verification checks using the provided connection."""
    
    # Extract root symbol and contract number
    match = re.match(r'([A-Z0-9]+)c(\d+)', continuous_symbol)
    if not match:
        logger.error(f"Invalid continuous symbol format: {continuous_symbol}")
        return pd.DataFrame()
    root_symbol = match.group(1)
    contract_num = int(match.group(2))
    
    logger.info(f"Verifying {continuous_symbol} (Root: {root_symbol}, Number: {contract_num})")
    
    # Load data using the provided connection
    roll_calendar = load_roll_calendar(conn, root_symbol)
    if not roll_calendar or not roll_calendar[0]:
        logger.error("Failed to load roll calendar. Cannot verify.")
        return pd.DataFrame()
    
    continuous_data_df = load_continuous_data(conn, continuous_symbol, start_date, end_date)
    if continuous_data_df.empty:
        logger.warning(f"No data found for {continuous_symbol} in the specified date range.")
        return pd.DataFrame()
        
    # Identify all unique underlying symbols mentioned in the continuous data
    actual_underlying_symbols = set(continuous_data_df['UnderlyingSymbol'].dropna().unique())
    logger.debug(f"Actual underlying symbols found: {actual_underlying_symbols}")

    # Load settlement prices for these symbols
    underlying_settles, underlying_sources = load_underlying_settles(conn, actual_underlying_symbols, start_date, end_date)
    
    # --- The rest of the original verify_data logic starts here ---
    # (Assuming it now uses the pre-loaded data: continuous_data_df, 
    # underlying_settles, underlying_sources, roll_calendar)
    verification_results = []
    all_contract_codes = list(roll_calendar[0].keys())  # All codes known to the calendar
    
    # Unpack the roll calendar dictionaries
    last_trading_days, settlement_dates = roll_calendar
    
    # Pre-sort the continuous data to ensure we can detect changes correctly
    continuous_data_df = continuous_data_df.sort_values('timestamp')
    
    # To detect changes in underlying symbols, we need to track the previous value
    prev_generator_underlying = None
    
    # Create reverse lookups from dates to contracts for detecting expected rolls
    # Note: Use settlement_date (not last_trading_day) for expected rollover date
    settlement_date_to_contract = {}
    for contract, settlement_date in settlement_dates.items():
        settlement_date_only = settlement_date.date()  # Convert datetime to date
        if settlement_date_only not in settlement_date_to_contract:
            settlement_date_to_contract[settlement_date_only] = []
        settlement_date_to_contract[settlement_date_only].append(contract)

    for idx, row in continuous_data_df.iterrows():
        current_dt = row['timestamp']
        current_date = current_dt.date()  # Extract date component
        continuous_settle = row['settle']
        generator_underlying = row['UnderlyingSymbol']

        expected_sequence = _get_expected_underlying_sequence(current_dt, all_contract_codes, roll_calendar)

        expected_underlying = None
        if len(expected_sequence) >= contract_num:
            expected_underlying = expected_sequence[contract_num - 1]
        else:
            logger.debug(f"Could not determine expected underlying contract #{contract_num} for {current_dt.date()}")

        # Look up settles using original values from continuous_data row first
        generator_underlying_settle = underlying_settles.get((generator_underlying, current_dt), pd.NA)
        expected_underlying_settle = underlying_settles.get((expected_underlying, current_dt), pd.NA) if expected_underlying else pd.NA
        
        # Look up sources 
        generator_source = underlying_sources.get((generator_underlying, current_dt), pd.NA)
        expected_source = underlying_sources.get((expected_underlying, current_dt), pd.NA) if expected_underlying else pd.NA

        # Check if this date is a SETTLEMENT date according to the calendar
        # This is the day we should actually see the rollover to a new contract
        is_settlement_date = current_date in settlement_date_to_contract
        expiring_contracts = settlement_date_to_contract.get(current_date, [])
        
        # For a settlement date, we need to know which contract SHOULD have expired yesterday
        # This will be null for the first record
        prev_day_expired_contract = None
        if is_settlement_date and prev_generator_underlying is not None:
            # Check if the previous generator underlying is in our list of contracts expiring today
            # If so, we expect to see a rollover today
            prev_day_expired_contract = prev_generator_underlying if prev_generator_underlying in expiring_contracts else None

        # --- Determine Check Status (Enhanced Logic) ---
        check_status = "UNKNOWN" # Default
        details = ""
        is_rollover_day = False
        price_diff = None
        price_diff_percent = None
        
        # Check if this is a rollover day by comparing current underlying with previous
        # This is more reliable than comparing with expected underlying, which might already match
        if prev_generator_underlying is not None and generator_underlying != prev_generator_underlying:
            is_rollover_day = True
        
        # Update the prev_generator_underlying for the next iteration, but only if it's not None
        if pd.notna(generator_underlying):
            prev_generator_underlying = generator_underlying

        if expected_underlying is None:
            check_status = "MISSING_EXPECTED"
        elif pd.isna(generator_underlying):
            check_status = "MISSING_GENERATOR"
        elif pd.isna(continuous_settle):
            check_status = "MISSING_SETTLE"
        else:
            # Both expected and generator underlying symbols are known, and continuous settle exists
            
            # First handle rollover days
            if is_rollover_day:
                # Check if rollover occurs on a settlement date
                if is_settlement_date and prev_day_expired_contract is not None:
                    check_status = "ROLLOVER_ON_SETTLEMENT_DATE"
                else:
                    check_status = "ROLLOVER_NOT_ON_SETTLEMENT_DATE"
                
                # Additional checks for price integrity on rollover day
                if pd.isna(generator_underlying_settle):
                    check_status = "ROLLOVER_MISSING_GEN_DATA"
                elif pd.isna(expected_underlying_settle):
                    check_status = "ROLLOVER_MISSING_EXP_DATA"
                elif abs(continuous_settle - generator_underlying_settle) > 0.001:
                    # If the continuous price doesn't match the generator's price on rollover
                    price_diff = continuous_settle - generator_underlying_settle
                    price_diff_percent = price_diff / generator_underlying_settle if generator_underlying_settle != 0 else float('inf')
                    check_status = "ROLLOVER_PRICE_MISMATCH"
            else:
                # Check for expected roll date but no roll occurred
                if is_settlement_date and prev_day_expired_contract is not None:
                    check_status = "MISSED_SETTLEMENT_ROLLOVER"
                # Then check for mismatches between generator and expected
                elif generator_underlying != expected_underlying:
                    check_status = "UNDERLYING_MISMATCH"
                # Normal day (not a rollover day, symbols match)
                else:
                    if pd.isna(expected_underlying_settle):
                        check_status = "MISSING_DATA"
                    elif abs(continuous_settle - expected_underlying_settle) < 0.001:  # Add tolerance for float comparison
                        check_status = "OK"
                    else:
                        # Prices mismatch
                        price_diff = continuous_settle - expected_underlying_settle
                        price_diff_percent = price_diff / expected_underlying_settle if expected_underlying_settle != 0 else float('inf')
                        if abs(price_diff_percent) > PRICE_DIFF_THRESHOLD:
                            check_status = "LARGE_PRICE_DIFF"
                        else:
                            check_status = "SMALL_PRICE_DIFF"

        verification_results.append({
            "Date": current_dt.date(),
            "ContinuousSettle": continuous_settle,
            "GeneratorUnderlying": generator_underlying,
            "ExpectedUnderlying": expected_underlying,
            "GeneratorSettle": generator_underlying_settle,
            "ExpectedSettle": expected_underlying_settle,
            "GeneratorSource": generator_source,
            "ExpectedSource": expected_source,
            "CheckStatus": check_status,
            "IsRolloverDay": is_rollover_day,
            "IsSettlementDate": is_settlement_date,
            "PrevDayExpiredContract": prev_day_expired_contract,
            "PriceDiff": price_diff,
            "PriceDiffPercent": price_diff_percent,
            "PrevGeneratorUnderlying": prev_generator_underlying if idx > 0 else None
        })

    return pd.DataFrame(verification_results)

def format_verification_results(df):
    """Format verification results with color-coding and aligned columns."""
    if df.empty:
        return "No verification results to display."
    
    # Create formatted columns
    formatted_rows = []
    headers = ["Date", "Cont Settle", "Generator", "Expected", "Gen Settle", "Settlement Date", "Status"]
    
    # Calculate column widths for proper alignment
    col_widths = {
        "Date": max(len("Date"), max(len(str(x)) for x in df["Date"])),
        "Cont Settle": max(len("Cont Settle"), max(len(f"{x:.4f}" if pd.notna(x) else "N/A") for x in df["ContinuousSettle"])),
        "Generator": max(len("Generator"), max(len(str(x) if pd.notna(x) else "N/A") for x in df["GeneratorUnderlying"])),
        "Expected": max(len("Expected"), max(len(str(x) if pd.notna(x) else "N/A") for x in df["ExpectedUnderlying"])),
        "Gen Settle": max(len("Gen Settle"), max(len(f"{x:.4f}" if pd.notna(x) else "N/A") for x in df["GeneratorSettle"])),
        "Settlement Date": max(len("Settlement Date"), max(len("Yes" if x else "No") for x in df["IsSettlementDate"])),
        "Status": max(len("Status"), max(len(str(x)) for x in df["CheckStatus"]))
    }
    
    # Create header row
    header_row = "  ".join([h.ljust(col_widths[h]) for h in headers])
    formatted_rows.append(c(header_row, Colors.BOLD))
    
    # Format each data row
    for _, row in df.iterrows():
        date_str = str(row["Date"]).ljust(col_widths["Date"])
        
        cont_settle = f"{row['ContinuousSettle']:.4f}" if pd.notna(row['ContinuousSettle']) else "N/A"
        cont_settle_str = cont_settle.ljust(col_widths["Cont Settle"])
        
        gen_underlying = str(row["GeneratorUnderlying"]) if pd.notna(row["GeneratorUnderlying"]) else "N/A"
        gen_underlying_str = gen_underlying.ljust(col_widths["Generator"])
        
        exp_underlying = str(row["ExpectedUnderlying"]) if pd.notna(row["ExpectedUnderlying"]) else "N/A"
        exp_underlying_str = exp_underlying.ljust(col_widths["Expected"])
        
        gen_settle = f"{row['GeneratorSettle']:.4f}" if pd.notna(row['GeneratorSettle']) else "N/A"
        gen_settle_str = gen_settle.ljust(col_widths["Gen Settle"])
        
        # Settlement date info
        is_settlement_date = row["IsSettlementDate"]
        settlement_date_str = ("Yes" if is_settlement_date else "No").ljust(col_widths["Settlement Date"])
        if is_settlement_date:
            expired_contract = row["PrevDayExpiredContract"]
            if pd.notna(expired_contract):
                settlement_date_str = f"Yes ({expired_contract})".ljust(col_widths["Settlement Date"])
        
        status_str = str(row["CheckStatus"]).ljust(col_widths["Status"])
        
        # Apply color based on status
        if row["CheckStatus"] == "OK":
            status_str = c(status_str, Colors.GREEN)
        elif row["CheckStatus"] == "ROLLOVER_ON_SETTLEMENT_DATE":
            status_str = c(status_str, Colors.BLUE)
            settlement_date_str = c(settlement_date_str, Colors.BLUE)
        elif row["CheckStatus"] == "ROLLOVER_NOT_ON_SETTLEMENT_DATE":
            status_str = c(status_str, Colors.PURPLE)
        elif "ROLLOVER_" in row["CheckStatus"]:
            status_str = c(status_str, Colors.YELLOW)
        elif row["CheckStatus"] == "MISSED_SETTLEMENT_ROLLOVER":
            status_str = c(status_str, Colors.RED)
            settlement_date_str = c(settlement_date_str, Colors.RED)
        elif row["CheckStatus"] in ["LARGE_PRICE_DIFF", "MISSING_DATA", "MISSING_EXPECTED", "MISSING_GENERATOR", "MISSING_SETTLE"]:
            status_str = c(status_str, Colors.RED)
        elif row["CheckStatus"] == "SMALL_PRICE_DIFF":
            status_str = c(status_str, Colors.YELLOW)
        
        # Format row based on whether it's a rollover day
        row_elements = [date_str, cont_settle_str, gen_underlying_str, exp_underlying_str, 
                         gen_settle_str, settlement_date_str, status_str]
        
        formatted_row = "  ".join(row_elements)
        
        # Highlight entire row for rollover days
        if row["IsRolloverDay"]:
            formatted_row = c(formatted_row, Colors.BOLD)
            
        formatted_rows.append(formatted_row)
    
    return "\n".join(formatted_rows)

def summarize_verification_results(df):
    """Create a summary of verification result statuses."""
    if df.empty:
        return "No data to summarize."
    
    status_counts = df['CheckStatus'].value_counts().to_dict()
    
    # Group statuses into categories
    categories = {
        "OK": ["OK"],
        "Settlement Date Rollovers": ["ROLLOVER_ON_SETTLEMENT_DATE"],
        "Non-Settlement Date Rollovers": ["ROLLOVER_NOT_ON_SETTLEMENT_DATE"],
        "Missed Rollovers": ["MISSED_SETTLEMENT_ROLLOVER"],
        "Rollover Issues": [s for s in status_counts.keys() if "ROLLOVER_" in s and s not in ["ROLLOVER_ON_SETTLEMENT_DATE", "ROLLOVER_NOT_ON_SETTLEMENT_DATE"]],
        "Price Differences": ["LARGE_PRICE_DIFF", "SMALL_PRICE_DIFF"],
        "Missing Data": ["MISSING_DATA", "MISSING_EXPECTED", "MISSING_GENERATOR", "MISSING_SETTLE"],
        "Other Issues": ["UNDERLYING_MISMATCH", "UNKNOWN"]
    }
    
    # Calculate category totals
    category_totals = {}
    for category, statuses in categories.items():
        category_totals[category] = sum(status_counts.get(status, 0) for status in statuses)
    
    # Additional calendar stats
    settlement_dates = df["IsSettlementDate"].sum()
    actual_roll_days = df["IsRolloverDay"].sum()
    matching_roll_days = df[df["CheckStatus"] == "ROLLOVER_ON_SETTLEMENT_DATE"].shape[0]
    
    # Format summary
    lines = [
        c("=== Verification Summary ===", Colors.BOLD),
        f"Total rows checked: {len(df)}",
        f"Settlement dates in period: {settlement_dates}",
        f"Actual roll days: {actual_roll_days}",
        f"Rolls on settlement dates: {matching_roll_days} of {actual_roll_days} ({(matching_roll_days/actual_roll_days*100) if actual_roll_days else 0:.1f}%)",
        ""
    ]
    
    for category, total in category_totals.items():
        if total > 0:
            if category == "OK":
                color = Colors.GREEN
            elif category in ["Settlement Date Rollovers"]:
                color = Colors.BLUE
            elif category in ["Non-Settlement Date Rollovers", "Price Differences"]:
                color = Colors.YELLOW
            else:
                color = Colors.RED
                
            percentage = total / len(df) * 100
            lines.append(c(f"{category}: {total} ({percentage:.1f}%)", color))
            
            # For categories with multiple statuses, show breakdown
            statuses = categories[category]
            if len(statuses) > 1:
                for status in statuses:
                    count = status_counts.get(status, 0)
                    if count > 0:
                        lines.append(f"  - {status}: {count}")
    
    lines.append("")
    return "\n".join(lines)

def main(args_dict=None, existing_conn=None):
    """Main execution function."""
    if args_dict:
        args = argparse.Namespace(**args_dict)
        logger.info("Running improved_verify_continuous from direct call.")
    else:
        parser = argparse.ArgumentParser(description='Verify continuous futures contracts.')
        parser.add_argument('--symbol', required=True, help='Continuous contract symbol (e.g., VXc1)')
        parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
        parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to DuckDB database')
        parser.add_argument('--no-color', action='store_true', help='Disable color output')
        args = parser.parse_args()
        logger.info("Running improved_verify_continuous from command line.")

    # Disable color if requested
    global COLOR_OUTPUT
    if args.no_color:
        COLOR_OUTPUT = False
        
    conn = None
    close_conn_locally = False
    
    try:
        if existing_conn:
            conn = existing_conn
            logger.info("Using existing database connection.")
        else:
            conn = connect_db(args.db_path) # Connects read-only by default now
            close_conn_locally = True
            
        if not conn:
            logger.error("Failed to establish database connection.")
            sys.exit(1)

        # Perform verification using the connection
        results_df = verify_data(conn, args.symbol, args.start_date, args.end_date)
        
        if not results_df.empty:
            # Format and display results
            formatted_output = format_verification_results(results_df)
            print("\n--- Verification Details ---")
            print(formatted_output)
            print("\n" + "=" * 60)
            
            # Summarize results
            summarize_verification_results(results_df)
        else:
            logger.info("No verification results to display.")
            
    except Exception as e:
        logger.error(f"An error occurred during verification: {e}", exc_info=True)
    finally:
        if conn and close_conn_locally:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main() 