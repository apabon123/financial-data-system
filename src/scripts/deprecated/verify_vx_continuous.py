#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VX Continuous Contract Verification Script

This script verifies the rollover logic for VX continuous contracts by:
1. Checking the first month of data (March-April 2004)
2. Examining rollover days
3. Verifying consistency with underlying futures contracts
"""

import os
import sys
import pandas as pd
import duckdb
import logging
from datetime import datetime, timedelta, date
import yaml
import calendar
import argparse
from rich.console import Console
from rich.table import Table
from typing import Dict, Any, List, Tuple, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DB_PATH = "./data/financial_data.duckdb"
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"
PRICE_DIFF_THRESHOLD = 0.015 # Threshold for price difference warnings

MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
MONTH_MAP = {code: i+1 for i, code in enumerate(MONTH_CODES)}
INV_MONTH_MAP = {i+1: code for i, code in enumerate(MONTH_CODES)}

def connect_to_database():
    """Connect to the DuckDB database."""
    try:
        db_path = os.path.join('data', 'financial_data.duckdb')
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

def run_query(conn, query, description):
    """Run a SQL query and return the results as a DataFrame."""
    try:
        logger.info(f"Running {description}...")
        df = conn.execute(query).df()
        logger.info(f"Query completed. Found {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to run query: {e}")
        return pd.DataFrame()

def verify_first_month(conn):
    """Verify the first month of VXc1 data."""
    query = """
    WITH dates AS (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE timestamp BETWEEN '2004-04-21' AND '2004-04-30'
        AND interval_value = 1
        AND interval_unit = 'day'
    ),
    vx_contracts AS (
        SELECT DISTINCT
            symbol,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date
        FROM market_data
        WHERE symbol LIKE 'VX%'
        AND symbol NOT LIKE 'VXc%'
        AND interval_value = 1
        AND interval_unit = 'day'
        GROUP BY symbol
    ),
    vxc1_data AS (
        SELECT
            date,
            open,
            high,
            low,
            settle,
            volume
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        AND date BETWEEN '2004-04-21' AND '2004-04-30'
        ORDER BY date
    ),
    active_contracts AS (
        SELECT
            d.date,
            (
                SELECT symbol
                FROM vx_contracts
                WHERE first_date <= d.date
                AND last_date >= d.date
                ORDER BY last_date ASC
                LIMIT 1
            ) as active_contract
        FROM dates d
    )
    SELECT
        v.date,
        v.open as vxc1_open,
        v.high as vxc1_high,
        v.low as vxc1_low,
        v.settle as vxc1_settle,
        v.volume as vxc1_volume,
        a.active_contract,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = v.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as contract_settle,
        CASE
            WHEN LAG(a.active_contract) OVER (ORDER BY v.date) != a.active_contract 
                 AND LAG(a.active_contract) OVER (ORDER BY v.date) IS NOT NULL
            THEN 'ROLLOVER'
            ELSE 'NORMAL'
        END as day_type
    FROM vxc1_data v
    LEFT JOIN active_contracts a ON v.date = a.date
    ORDER BY v.date;
    """
    
    df = run_query(conn, query, "First month verification")
    if not df.empty:
        print("\nFirst Month Verification Results:")
        print("=================================")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        # Rename columns for clarity in output
        df.columns = ['Date', 'VXc1 Open', 'VXc1 High', 'VXc1 Low', 'VXc1 Settle', 'VXc1 Volume', 'Active Contract', 'Contract Settle', 'Day Type']
        print(df.to_string(index=False))
        print("\n")
    return df

def verify_rollover_days(conn):
    """Verify rollover days and calculate price differences."""
    query = """
    WITH dates AS (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE timestamp BETWEEN '2004-04-21' AND '2004-12-31'
        AND interval_value = 1
        AND interval_unit = 'day'
    ),
    vx_contracts AS (
        SELECT DISTINCT
            symbol,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date
        FROM market_data
        WHERE symbol LIKE 'VX%'
        AND symbol NOT LIKE 'VXc%'
        AND interval_value = 1
        AND interval_unit = 'day'
        GROUP BY symbol
    ),
    active_contracts AS (
        SELECT
            d.date,
            (
                SELECT symbol
                FROM vx_contracts
                WHERE first_date <= d.date
                AND last_date >= d.date
                ORDER BY last_date ASC
                LIMIT 1
            ) as active_contract
        FROM dates d
    ),
    contract_changes AS (
        SELECT
            date,
            active_contract,
            LAG(active_contract) OVER (ORDER BY date) as prev_contract
        FROM active_contracts
    ),
    rollover_days AS (
        SELECT *
        FROM contract_changes
        WHERE prev_contract IS NOT NULL
        AND active_contract != prev_contract
    )
    SELECT
        r.date as rollover_date,
        r.prev_contract,
        r.active_contract,
        (
            SELECT settle
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date
            LIMIT 1
        ) as vxc1_settle,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = r.prev_contract
            AND timestamp = r.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as prev_contract_settle,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = r.active_contract
            AND timestamp = r.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as new_contract_settle,
        (
            SELECT settle
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date - interval '1 day'
            LIMIT 1
        ) as vxc1_prev_settle,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = r.prev_contract
            AND timestamp = r.date - interval '1 day'
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as prev_contract_prev_settle,
        (
            SELECT settle
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date + interval '1 day'
            LIMIT 1
        ) as vxc1_next_settle,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = r.active_contract
            AND timestamp = r.date + interval '1 day'
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as new_contract_next_settle
    FROM rollover_days r
    ORDER BY r.date;
    """
    
    df = run_query(conn, query, "Rollover days verification")
    if not df.empty:
        print("\nRollover Day Verification Results:")
        print("==================================")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        # Rename columns for clarity in output
        df.columns = [
            'Rollover Date', 'Previous Contract', 'Active Contract', 
            'VXc1 Settle', 'Prev Contract Settle', 'New Contract Settle',
            'VXc1 Prev Settle', 'Prev Contract Prev Settle', 
            'VXc1 Next Settle', 'New Contract Next Settle'
        ]
        print(df.to_string(index=False))
        print("\n")
    return df

def check_consistency(conn):
    """Check for inconsistencies in VXc1 data against underlying futures contracts."""
    query = """
    WITH dates AS (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE interval_value = 1
        AND interval_unit = 'day'
    ),
    vx_contracts AS (
        SELECT DISTINCT
            symbol,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date
        FROM market_data
        WHERE symbol LIKE 'VX%'
        AND symbol NOT LIKE 'VXc%'
        AND interval_value = 1
        AND interval_unit = 'day'
        GROUP BY symbol
    ),
    active_contracts AS (
        SELECT
            d.date,
            (
                SELECT symbol
                FROM vx_contracts
                WHERE first_date <= d.date
                AND last_date >= d.date
                ORDER BY last_date ASC
                LIMIT 1
            ) as active_contract
        FROM dates d
    ),
    daily_continuous_contracts AS (
        SELECT
            date,
            symbol,
            MAX(settle) as settle
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        GROUP BY date, symbol
    )
    SELECT
        c.date,
        c.symbol as continuous_contract,
        a.active_contract,
        c.settle as continuous_settle,
        (
            SELECT settle
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = c.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as contract_settle,
        ABS(c.settle - (
            SELECT settle
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = c.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        )) as price_diff,
        CASE
            WHEN ABS(c.settle - (
                SELECT settle
                FROM market_data
                WHERE symbol = a.active_contract
                AND timestamp = c.date
                AND interval_value = 1
                AND interval_unit = 'day'
                LIMIT 1
            )) > c.settle * 0.01 THEN 'WARNING'
            ELSE 'OK'
        END as consistency_check
    FROM daily_continuous_contracts c
    LEFT JOIN active_contracts a ON c.date = a.date
    ORDER BY c.date;
    """
    
    df = run_query(conn, query, "Consistency check")
    if not df.empty:
        print("\nConsistency Check Results:")
        print("=========================")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
        
        # Count warnings
        warnings = df[df['consistency_check'] == 'WARNING']
        if not warnings.empty:
            print(f"\nFound {len(warnings)} days with significant price differences (>1%)")
            print("\nWarning Details:")
            print(warnings[['date', 'continuous_contract', 'active_contract', 'continuous_settle', 'contract_settle', 'price_diff']].to_string(index=False))
        print("\n")
    return df

def load_config(config_path: str) -> Dict[str, Any]:
    """Load the configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def get_futures_config(config: Dict[str, Any], root_symbol: str) -> Optional[Dict[str, Any]]:
    """Get the configuration for a specific futures root symbol."""
    if not config or 'futures' not in config:
        logger.error("Futures configuration not found in loaded config.")
        return None
    futures = config.get('futures', [])
    for future in futures:
        if future.get('base_symbol') == root_symbol:
            return future
    logger.error(f"Configuration for root symbol '{root_symbol}' not found in futures list.")
    return None
    
def get_nth_weekday_of_month(year: int, month: int, weekday_int: int, n: int) -> date:
    """
    Gets the date of the nth occurrence of a specific weekday in a given month and year.
    weekday_int: 0=Mon, 1=Tue, ..., 6=Sun
    n: 1 for 1st, 2 for 2nd, etc.
    """
    first_day_of_month = date(year, month, 1)
    day_of_week_first = first_day_of_month.weekday()
    days_to_add = (weekday_int - day_of_week_first + 7) % 7
    first_occurrence_date = first_day_of_month + timedelta(days=days_to_add)
    nth_occurrence_date = first_occurrence_date + timedelta(weeks=n - 1)
    if nth_occurrence_date.month != month:
        raise ValueError(f"The {n}th weekday {weekday_int} does not exist in {year}-{month:02d}")
    return nth_occurrence_date

def get_previous_weekday(target_date: date, weekday_int: int) -> date:
    """Gets the date of the most recent specified weekday before or on the target_date."""
    days_to_subtract = (target_date.weekday() - weekday_int + 7) % 7
    return target_date - timedelta(days=days_to_subtract)

def get_expiry_date(contract: str, future_config: dict) -> Optional[date]:
    """Get the expiry date for a contract based on config rules (returns date object).
       Handles VIX rule (Wed before 3rd Fri) and basic holiday adjustment.
    """
    try:
        if 'c' in contract[-3:]:
             return None # Skip continuous
             
        # Parse symbol
        match = re.match(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$", contract)
        if not match:
             logger.error(f"[Verifier] Could not parse contract symbol format: {contract}")
             return None
        _, month_code, year_str = match.groups()
        year = 2000 + int(year_str)
        month = MONTH_MAP.get(month_code)
        if not month: return None

        expiry_rule = future_config.get('expiry_rule')
        if not expiry_rule:
             logger.error(f"[Verifier] No expiry rule found for {contract}")
             return None
             
        # --- Implement Rule Logic (match generator's logic) --- 
        day_type = expiry_rule.get('day_type', '').lower()
        day_number = expiry_rule.get('day_number', 0)
        calculated_expiry = None
        
        # VIX Rule: Wednesday 30 days before 3rd Friday of *next* month 
        # (or more simply: Wed before 3rd Fri of contract month)
        if day_type == 'wednesday' and day_number == 3: # Assuming this config signifies the VIX rule
            try:
                # Find 3rd Friday of the contract month
                third_friday = get_nth_weekday_of_month(year, month, 4, 3) # Friday=4
                # Find Wednesday immediately preceding that Friday
                calculated_expiry = get_previous_weekday(third_friday - timedelta(days=1), 2) # Wednesday=2
                logger.debug(f"[Verifier: {contract}] Rule: Wed before 3rd Fri. 3rd Fri={third_friday}, Calculated Exp={calculated_expiry}")
            except ValueError as e:
                logger.error(f"[Verifier: {contract}] Error applying 'Wed before 3rd Fri' rule: {e}")
                return None
        else:
            # Add other rules here if needed for other contracts
            logger.warning(f"[Verifier] Using fallback expiry calculation (e.g., simple 3rd Wed) for {contract} rule: {expiry_rule}. May be inaccurate.")
            # Fallback/simplified logic (e.g., 3rd Wednesday of contract month)
            try:
                 c = calendar.monthcalendar(year, month)
                 wednesdays = [day for week in c for day_index, day in enumerate(week) if day_index == calendar.WEDNESDAY and day != 0]
                 if len(wednesdays) >= 3:
                      calculated_expiry = date(year, month, wednesdays[2])
                 else: return None
            except Exception: return None # Failed fallback

        if not calculated_expiry:
            return None

        # --- Holiday Adjustment (Simplified version) --- 
        # For verification, a precise holiday adjustment might not be critical,
        # but we should at least avoid weekends if the rule didn't.
        # A full holiday check would require loading holiday sets here too.
        if expiry_rule.get('adjust_for_holiday', False):
             # Simple check: If calculated expiry is Sat/Sun, move back to Friday
             if calculated_expiry.weekday() == 5: # Saturday
                  final_expiry = calculated_expiry - timedelta(days=1)
             elif calculated_expiry.weekday() == 6: # Sunday
                  final_expiry = calculated_expiry - timedelta(days=2)
             else:
                  final_expiry = calculated_expiry 
             if final_expiry != calculated_expiry:
                  logger.debug(f"[Verifier: {contract}] Adjusted expiry from {calculated_expiry} to {final_expiry} due to weekend.")
             return final_expiry
        else:
             return calculated_expiry # Return date object

    except Exception as e:
        logger.error(f"[Verifier] Error getting expiry date for {contract}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _get_theoretical_next_symbol(current_symbol: str) -> Optional[str]:
    """Calculates the next contract symbol in the theoretical sequence."""
    if not current_symbol or len(current_symbol) < 3:
        return None
    root = current_symbol[:-3]
    month_code = current_symbol[-3]
    year_str = current_symbol[-2:]
    try:
        year = int(year_str)
        current_month_index = MONTH_CODES.index(month_code)
    except (ValueError, IndexError):
        return None
    next_month_index = (current_month_index + 1) % 12
    next_month_code = MONTH_CODES[next_month_index]
    next_year = year
    if next_month_index < current_month_index:
        next_year += 1
    next_year_str = str(next_year % 100).zfill(2)
    return f"{root}{next_month_code}{next_year_str}"

def verify_continuous_contract(conn: duckdb.DuckDBPyConnection, continuous_symbol: str):
    console = Console()
    root_symbol = continuous_symbol[:-2]
    try:
        contract_number = int(continuous_symbol[-1])
    except ValueError:
        logger.error(f"Invalid continuous symbol format: {continuous_symbol}. Expected format like 'VXc1'.")
        return

    logger.info(f"Starting verification for {continuous_symbol} (Root: {root_symbol}, Number: {contract_number}) ")
    
    # Load config for expiry rules
    config = load_config(DEFAULT_CONFIG_PATH)
    future_config = get_futures_config(config, root_symbol)
    if not future_config:
         logger.error(f"Could not load futures config for {root_symbol} from {DEFAULT_CONFIG_PATH}")
         return
         
    # --- Get contract data (remove dependency on sorted_contracts_with_expiry_dt) ---
    try:
        # Fetch all continuous data first
        continuous_df = conn.execute(
            f"SELECT timestamp, settle as continuous_settle, underlying_symbol FROM continuous_contracts WHERE symbol = ? ORDER BY timestamp",
            [continuous_symbol]
        ).fetchdf()
        if continuous_df.empty:
            logger.error(f"No data found for continuous contract {continuous_symbol} in the database.")
            return
        continuous_df['date_dt'] = pd.to_datetime(continuous_df['timestamp'])
        continuous_df['date'] = continuous_df['date_dt'].dt.date
        min_date = continuous_df['date'].min()
        max_date = continuous_df['date'].max()
        logger.info(f"Verifying data for {continuous_symbol} from {min_date} to {max_date}.")

        # Filter out weekends
        original_count = len(continuous_df)
        continuous_df = continuous_df[continuous_df['date_dt'].dt.weekday < 5]
        filtered_count = len(continuous_df)
        logger.info(f"Filtered out {original_count - filtered_count} weekend rows.")
        if continuous_df.empty:
            logger.error(f"No non-weekend data found for {continuous_symbol} after filtering.")
            return
            
        # --- Initialize theoretical symbol tracking --- 
        # Determine the first *expected* symbol based on the first date in the data
        # Need a way to find the nth contract theoretically active on min_date
        # This still requires iterating through potential contracts
        # Let's start simpler: assume the first *generator* underlying is the first *theoretical* one
        first_generator_underlying = continuous_df['underlying_symbol'].dropna().iloc[0]
        if not first_generator_underlying:
             logger.error("Cannot determine starting theoretical symbol: No non-null underlying_symbol found in data.")
             return
        theoretical_expected_symbol = first_generator_underlying
        expiry_date_for_theoretical = get_expiry_date(theoretical_expected_symbol, future_config)
        if not expiry_date_for_theoretical:
             logger.error(f"Could not determine expiry date for initial symbol {theoretical_expected_symbol}. Aborting.")
             return
        logger.info(f"Initial theoretical symbol: {theoretical_expected_symbol} (Expires: {expiry_date_for_theoretical})")

    except Exception as e:
        logger.error(f"Error fetching or processing continuous contract data for {continuous_symbol}: {e}")
        return

    # --- Main Verification Loop (Revised Logic) --- 
    results = []
    calculated_rollover_dates = {}
    last_generator_underlying = None
    last_date = None
    previous_expiry_date_for_theoretical = expiry_date_for_theoretical # Store initial expiry

    for i, row in continuous_df.iterrows():
        current_date = row['date']
        continuous_settle = row['continuous_settle']
        is_settle_nan = pd.isna(continuous_settle)
        generator_underlying = row.get('underlying_symbol') 
    
        # Determine theoretical rollover BEFORE potentially updating the theoretical symbol
        is_theoretical_rollover = (current_date == previous_expiry_date_for_theoretical)
    
        # --- Calculate actual rollover BEFORE using it in checks --- 
        is_actual_rollover = (last_generator_underlying and generator_underlying is not None and generator_underlying != last_generator_underlying)

        # --- Update Theoretical Symbol based on Expiry --- 
        # Check if the *current* date is AT or past the expiry of the *current* theoretical symbol
        if current_date >= expiry_date_for_theoretical: # Roll ON expiry day (theoretical)
            next_theoretical = _get_theoretical_next_symbol(theoretical_expected_symbol)
            if next_theoretical:
                new_expiry = get_expiry_date(next_theoretical, future_config)
                if new_expiry:
                    logger.info(f"Theoretical rollover on {current_date}: {theoretical_expected_symbol} -> {next_theoretical} (Expires: {new_expiry})")
                    theoretical_expected_symbol = next_theoretical
                    previous_expiry_date_for_theoretical = expiry_date_for_theoretical # Store the expiry that *was* active
                    expiry_date_for_theoretical = new_expiry 
                else:
                    logger.warning(f"Could not get expiry for {next_theoretical} during theoretical advancement on {current_date}. Theoretical expiry might be incorrect going forward.")

        # Initialize variables
        price_diff = None
        consistency_check = "OK"
        underlying_settle_generator = None # Settle for the symbol the generator *used*
        underlying_settle_theoretical = None # Settle for the *theoretically* expected symbol
        comparison_settle = None # Value to display in the 'Compared Settle' column

        # Fetch Settle Price for Generator's Underlying (if it exists)
        if generator_underlying:
            query_generator = f"SELECT settle FROM market_data WHERE symbol = '{generator_underlying}' AND timestamp::DATE = '{current_date}' LIMIT 1"
            try:
                result_generator = conn.execute(query_generator).fetchone()
                if result_generator:
                    underlying_settle_generator = result_generator[0]
            except Exception as e:
                logger.error(f"DB Error fetching generator underlying ({generator_underlying}) on {current_date}: {e}")

        # Fetch Settle Price for Theoretical Underlying (if different from generator's)
        query_theoretical = f"SELECT settle FROM market_data WHERE symbol = '{theoretical_expected_symbol}' AND timestamp::DATE = '{current_date}' LIMIT 1"
        try:
            result_theoretical = conn.execute(query_theoretical).fetchone()
            if result_theoretical:
                underlying_settle_theoretical = result_theoretical[0]
        except Exception as e:
            logger.error(f"DB Error fetching theoretical underlying ({theoretical_expected_symbol}) on {current_date}: {e}")

        # --- Determine Consistency Check Status (Revised Logic V2) ---
        # Priority 1: Check if theoretical data is missing
        if underlying_settle_theoretical is None:
            consistency_check = "THEORETICAL_DATA_MISSING"
            comparison_settle = None # Cannot compare
            price_diff = None
        # Priority 2: Handle NaN Continuous Settle (implies generator failed somehow)
        elif is_settle_nan:
            if generator_underlying is None:
                consistency_check = "NaN+NO_GEN_SYMBOL"
            elif generator_underlying != theoretical_expected_symbol:
                consistency_check = "NaN+GEN_SYMBOL_MISMATCH" # Generator skipped, resulted in NaN (Expected if theory data missing, but we checked that)
            else: # Symbols match but result is NaN? Data issue.
                consistency_check = "NaN+SYMBOLS_MATCH"
            comparison_settle = None # Cannot compare if settle is NaN
            price_diff = None
        # Priority 3: Rollover Day (both theoretically and actually)
        elif is_theoretical_rollover and is_actual_rollover:
            consistency_check = "ROLLOVER_DAY"
            comparison_settle = underlying_settle_generator # Compare against generator on rollover
            if comparison_settle is not None:
                price_diff = continuous_settle - comparison_settle
                if abs(price_diff) > PRICE_DIFF_THRESHOLD:
                    consistency_check += "_PRICE_DIFF_WARN"
            else:
                consistency_check = "ROLLOVER_MISSING_GEN_DATA" # Generator data missing on rollover
                price_diff = None
        # Priority 4: Generator used different symbol than theoretical (and not rollover)
        elif generator_underlying != theoretical_expected_symbol:
            consistency_check = "GENERATOR_SYMBOL_MISMATCH"
            comparison_settle = None # Cannot compare against theoretical
            price_diff = None
        # Priority 5: Mismatched Rollover? (e.g., theoretical roll but no actual roll) - Less critical for now
        # elif is_theoretical_rollover != is_actual_rollover:
            # consistency_check = "ROLLOVER_MISMATCH" 
            # comparison_settle = None
            # price_diff = None
        # Case 5: Settle HAS value. Symbols match. Underlying data found.
        elif theoretical_expected_symbol == generator_underlying: # Default case: Symbols match, not rollover
            comparison_settle = underlying_settle_theoretical # Compare against theoretical
            price_diff = continuous_settle - comparison_settle
            if abs(price_diff) > PRICE_DIFF_THRESHOLD:
                consistency_check = "PRICE_DIFF_WARNING"
        else: # Should not happen if logic above is exhaustive
            consistency_check = "UNHANDLED_STATE"
            comparison_settle = None
            price_diff = None

        # --- Store Rollover Info --- 
        # Moved outside the status overwrite block, still triggered by is_actual_rollover
        if is_actual_rollover and last_date:
            if last_date not in calculated_rollover_dates:
                calculated_rollover_dates[last_date] = (last_generator_underlying, generator_underlying)
            else: 
                logger.warning(f"Multiple generator rolls detected ending on the same date {last_date}. Keeping first detected ({calculated_rollover_dates[last_date][0]} -> {calculated_rollover_dates[last_date][1]}). Ignoring {last_generator_underlying} -> {generator_underlying}")

        # Append results
        results.append({
            'date': current_date,
            'continuous_symbol': continuous_symbol,
            'expected_contract': theoretical_expected_symbol,
            'generator_contract': generator_underlying if generator_underlying else 'N/A',
            'continuous_settle': continuous_settle,
            'contract_settle': comparison_settle, # Use the determined comparison value
            'price_diff': price_diff,
            'consistency_check': consistency_check
        })

        # Update trackers
        last_date = current_date
        last_generator_underlying = generator_underlying
        
    # --- Output Results --- 
    results_df = pd.DataFrame(results)
    
    # Display results table (Adjust styling based on new Check statuses)
    table = Table(title=f"Verification Results for {continuous_symbol}")
    table.add_column("Date", style="dim")
    table.add_column("Theoretical Expected")
    table.add_column("Actual Underlying")
    table.add_column("Continuous Settle", justify="right")
    table.add_column("Compared Settle", justify="right")
    table.add_column("Price Diff", justify="right")
    table.add_column("Check", justify="center")
    
    for _, row in results_df.iterrows():
        style = ""
        check = row['consistency_check']
        check_str = check # Default check string
        
        # Original values from the dataframe
        theoretical_expected = row['expected_contract']
        generator_actual = row['generator_contract']
        continuous_settle_val = row['continuous_settle']
        underlying_settle_val = row['contract_settle'] # This now holds the relevant comparison settle
        price_diff_val = row['price_diff']

        # Determine formatting based on Check status and NaN values
        if check == "THEORETICAL_DATA_MISSING":
            style = "bold red"
            check_str = "THEORY_MISSING" # Shorten display name
            # comparison_settle and price_diff are already N/A
        elif "NaN" in check: # Handle all NaN cases generically
            style = "cyan"
            check_str = check # Show full NaN status (e.g., NaN+NO_GEN_SYMBOL)
            # comparison_settle and price_diff are already N/A
        elif check == "GENERATOR_SYMBOL_MISMATCH":
            style = "magenta"
            check_str = "GEN_SYM_MISMATCH" # Shorten display name
            # comparison_settle and price_diff are already N/A
        elif check == "SKIPPED_CONTRACT_WRONG_DATA": # Keep old status mapping if needed? Or remove? Let's remove for now.
            # This status should ideally be replaced by GENERATOR_SYMBOL_MISMATCH if theoretical data exists
            # Or THEORETICAL_DATA_MISSING if it doesn't. Add a fallback just in case.
            style = "grey50"
            check_str = "OLD_SKIPPED_STATUS" 
            compared_settle_str = "N/A" 
            price_diff_str = "N/A" 
        elif "ROLLOVER" in check:
            style = "yellow" if "WARN" not in check and "MISSING" not in check else "bold orange"
            check_str = check # Keep full status
            if "MISSING" in check:
                 compared_settle_str = "[bold red]MISSING[/]"
                 price_diff_str = "N/A"
        elif check == "ROLLOVER_MISSING_GEN_DATA":
            # Specific handling for missing data on rollover day
            style = "bold magenta"
            check_str = check
            compared_settle_str = "[bold red]MISSING[/]"
            price_diff_str = "N/A"
        elif check == "PRICE_DIFF_WARNING":
            style = "bold red"
            check_str = check # Keep full status
        elif check == "OK": # Explicitly check for OK case
            style = ""
            check_str = "OK"
            continuous_settle_str = f"{continuous_settle_val:.2f}" if pd.notna(continuous_settle_val) else "[bold red]ERR_NaN[/]" # Should not be NaN if OK
            compared_settle_str = f"{underlying_settle_val:.2f}" if pd.notna(underlying_settle_val) else "[bold red]ERR_N/A[/]" # Should exist if OK
            price_diff_str = f"{price_diff_val:.2f}" if pd.notna(price_diff_val) else "[bold red]ERR_N/A[/]" # Should exist if OK
        else:
            # Fallback for any unhandled check status to prevent error
            style = "grey50" # Dim unhandled status
            check_str = f"UNHANDLED: {check}"
            continuous_settle_str = f"{continuous_settle_val}" if pd.notna(continuous_settle_val) else "NaN"
            compared_settle_str = f"{underlying_settle_val}" if pd.notna(underlying_settle_val) else "N/A"
            price_diff_str = f"{price_diff_val}" if pd.notna(price_diff_val) else "N/A"

        table.add_row(
            str(row['date']),
            theoretical_expected,
            generator_actual if generator_actual else 'N/A', # Ensure N/A if None
            continuous_settle_str,
            compared_settle_str,
            price_diff_str,
            check_str,
            style=style
        )

    console.print(table)

    # --- Verify Rollover Dates Table (using actual generator rollovers) --- 
    # ... (Keep existing Rollover Table generation logic using calculated_rollover_dates) ...
    logger.info("--- Rollover Date Verification (Based on Actual Generator Changes) ---")
    rollover_table = Table(title="Actual Rollover Dates vs Config (3rd Wednesday)")
    rollover_table.add_column("Generator Rollover Date (Day Before Switch)")
    rollover_table.add_column("Old Contract (Actual)")
    rollover_table.add_column("New Contract (Actual)")
    rollover_table.add_column("Is Old Contract Expiry 3rd Wednesday?", justify="center")

    sorted_actual_rollover_dates = sorted(calculated_rollover_dates.keys())

    for rollover_trigger_date in sorted_actual_rollover_dates:
        old_contract, new_contract = calculated_rollover_dates[rollover_trigger_date]
        
        # Check 3rd Wednesday rule for the OLD contract
        calculated_expiry = get_expiry_date(old_contract, future_config)
        is_third_wednesday = False
        status_text = "[bold red]NO[/bold red]"
        if calculated_expiry:
            if calculated_expiry == rollover_trigger_date:
                is_third_wednesday = True
                status_text = "[green]YES[/green]"
            else:
                 status_text += f" [dim](Expected {calculated_expiry})[/dim]" # Show mismatch
        else:
             status_text += " [dim](Calc Fail)[/dim]" 

        rollover_table.add_row(
            str(rollover_trigger_date),
            old_contract,
            new_contract,
            status_text
        )
        
    console.print(rollover_table)
    logger.info("Note: 3rd Wednesday check uses expiry of the OLD contract. Mismatches may occur due to holidays, data gaps, or rule variations.")

def clean_weekend_data(conn: duckdb.DuckDBPyConnection):
    """Removes weekend data from the continuous_contracts table."""
    try:
        logger.info("Attempting to delete weekend data (Saturday/Sunday) from continuous_contracts...")
        # DuckDB DAYOFWEEK: Sunday=0, Saturday=6
        delete_query = "DELETE FROM continuous_contracts WHERE DAYOFWEEK(timestamp) = 0 OR DAYOFWEEK(timestamp) = 6;"
        with conn.cursor() as cur:
            cur.execute(delete_query)
            deleted_count = cur.rowcount
            logger.info(f"Deleted {deleted_count} weekend rows from continuous_contracts.")
            conn.commit() # Explicitly commit the change
    except Exception as e:
        logger.error(f"Error deleting weekend data: {e}")
        conn.rollback() # Rollback on error

def main():
    parser = argparse.ArgumentParser(description='Verify VX continuous futures contracts against underlying data.')
    parser.add_argument('--symbol', type=str, default='VXc1', help='Continuous contract symbol to verify (e.g., VXc1, VXc2)')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
    parser.add_argument('--clean-weekends', action='store_true', help='Run the weekend data cleaning step.')
    args = parser.parse_args()

    conn = None # Initialize conn
    try:
        # Connect with read/write permissions if cleaning
        read_only_mode = not args.clean_weekends 
        conn = duckdb.connect(args.db_path, read_only=read_only_mode) 
        
        if args.clean_weekends:
            clean_weekend_data(conn)
            # Optionally exit after cleaning or continue to verify
            logger.info("Weekend cleaning complete. Run without --clean-weekends to verify.")
            # return # Uncomment this line if you only want to clean

        # Proceed with verification if not exiting after cleaning
        verify_continuous_contract(conn, args.symbol)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 