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
from datetime import datetime, timedelta
import yaml
import calendar
import argparse
from rich.console import Console
from rich.table import Table
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DB_PATH = "./data/financial_data.duckdb"
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"

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
        WHERE timestamp BETWEEN '2004-03-26' AND '2004-04-30'
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
            close,
            volume
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        AND date BETWEEN '2004-03-26' AND '2004-04-30'
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
        v.close as vxc1_close,
        v.volume as vxc1_volume,
        a.active_contract,
        (
            SELECT close
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = v.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as contract_close,
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
        print(df.to_string(index=False))
        print("\n")
    return df

def verify_rollover_days(conn):
    """Verify rollover days and calculate price differences."""
    query = """
    WITH dates AS (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE timestamp BETWEEN '2004-03-26' AND '2004-12-31'
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
            SELECT close
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date
            LIMIT 1
        ) as vxc1_close,
        (
            SELECT close
            FROM market_data
            WHERE symbol = r.prev_contract
            AND timestamp = r.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as prev_contract_close,
        (
            SELECT close
            FROM market_data
            WHERE symbol = r.active_contract
            AND timestamp = r.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as new_contract_close,
        (
            SELECT close
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date - interval '1 day'
            LIMIT 1
        ) as vxc1_prev_close,
        (
            SELECT close
            FROM market_data
            WHERE symbol = r.prev_contract
            AND timestamp = r.date - interval '1 day'
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as prev_contract_prev_close,
        (
            SELECT close
            FROM continuous_contracts
            WHERE symbol = 'VXc1'
            AND date = r.date + interval '1 day'
            LIMIT 1
        ) as vxc1_next_close,
        (
            SELECT close
            FROM market_data
            WHERE symbol = r.active_contract
            AND timestamp = r.date + interval '1 day'
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as new_contract_next_close
    FROM rollover_days r
    ORDER BY r.date;
    """
    
    df = run_query(conn, query, "Rollover days verification")
    if not df.empty:
        print("\nRollover Days Verification Results:")
        print("===================================")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
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
            MAX(close) as close
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        GROUP BY date, symbol
    )
    SELECT
        c.date,
        c.symbol as continuous_contract,
        a.active_contract,
        c.close as continuous_close,
        (
            SELECT close
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = c.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        ) as contract_close,
        ABS(c.close - (
            SELECT close
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = c.date
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        )) as price_diff,
        CASE
            WHEN ABS(c.close - (
                SELECT close
                FROM market_data
                WHERE symbol = a.active_contract
                AND timestamp = c.date
                AND interval_value = 1
                AND interval_unit = 'day'
                LIMIT 1
            )) > c.close * 0.01 THEN 'WARNING'
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
            print(warnings[['date', 'continuous_contract', 'active_contract', 'continuous_close', 'contract_close', 'price_diff']].to_string(index=False))
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
    """Get the configuration for a specific futures contract."""
    futures = config.get('futures', [])
    for future in futures:
        if future.get('base_symbol') == root_symbol:
            return future
    logger.error(f"No configuration found for root symbol {root_symbol}")
    return None

def get_expiry_date(contract: str, future_config: dict) -> Optional[datetime]:
    """Get the expiry date for a contract based on config (e.g., 3rd Wednesday)."""
    try:
        if 'c' in contract[-3:]: # Skip already continuous contracts if passed mistakenly
            return None
            
        month_code = contract[-3]
        year_str = contract[-2:]
        year = 2000 + int(year_str)
        
        month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                     'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
        month = month_map.get(month_code)
        if not month:
            logger.error(f"Invalid month code {month_code} in contract {contract}")
            return None
        
        expiry_rule = future_config.get('expiry_rule', {})
        if not expiry_rule:
            logger.error(f"No expiry rule found for {contract} in config")
            return None
        
        day_type = expiry_rule.get('day_type', '').lower()
        day_number = expiry_rule.get('day_number', 0)
        
        if day_type == 'wednesday' and day_number == 3:
            # Calculate the third Wednesday
            c = calendar.monthcalendar(year, month)
            wednesdays = [day for week in c for day_index, day in enumerate(week) if day_index == calendar.WEDNESDAY and day != 0]
            if len(wednesdays) < 3:
                logger.error(f"Could not find third Wednesday for {year}-{month}")
                return None
            third_wednesday_day = wednesdays[2]
            expiry_date = datetime(year, month, third_wednesday_day)
            # Note: Holiday adjustments are not implemented in this basic check yet.
            return expiry_date
        else:
            logger.error(f"Unsupported expiry rule in config: {expiry_rule} for {contract}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting expiry date for {contract}: {e}")
        return None

def get_available_contracts(conn: duckdb.DuckDBPyConnection, root_symbol: str) -> List[Tuple[str, datetime]]:
    """Get all available contracts for a root symbol and their expiry dates, sorted by expiry."""
    try:
        query = """
            SELECT DISTINCT symbol
            FROM market_data
            WHERE symbol LIKE ?
            AND interval_value = 1 AND interval_unit = 'day'
            AND symbol NOT LIKE '%c_' -- Exclude continuous symbols
            ORDER BY symbol -- Initial sort helps, but final sort is by date
        """
        contracts_df = conn.execute(query, [f"{root_symbol}%"]).fetchdf()
        available_symbols = contracts_df['symbol'].tolist()

        config = load_config(DEFAULT_CONFIG_PATH)
        future_config = get_futures_config(config, root_symbol)
        if not future_config:
            return []

        contracts_with_expiry = []
        for contract in available_symbols:
            expiry = get_expiry_date(contract, future_config)
            if expiry:
                contracts_with_expiry.append((contract, expiry))
            else:
                logger.warning(f"Could not determine expiry for {contract}. Skipping.")

        # Sort contracts chronologically by expiry date
        contracts_with_expiry.sort(key=lambda x: x[1])
        
        if not contracts_with_expiry:
            logger.error(f"No contracts with valid expiry dates found for {root_symbol}.")
            return []
            
        logger.info(f"Found {len(contracts_with_expiry)} contracts with expiry dates for {root_symbol}.")
        return contracts_with_expiry

    except Exception as e:
        logger.error(f"Error getting available contracts: {e}")
        return []

def verify_continuous_contract(conn: duckdb.DuckDBPyConnection, continuous_symbol: str):
    """Verifies the continuous contract against underlying contracts."""
    console = Console()
    root_symbol = continuous_symbol[:-2]
    try:
        contract_number = int(continuous_symbol[-1])
    except ValueError:
        logger.error(f"Invalid continuous symbol format: {continuous_symbol}. Expected format like 'VXc1'.")
        return

    logger.info(f"Starting verification for {continuous_symbol} (Root: {root_symbol}, Number: {contract_number})")

    # 1. Get sorted list of underlying contracts with their expiry dates
    sorted_contracts_with_expiry = get_available_contracts(conn, root_symbol)
    if not sorted_contracts_with_expiry:
        logger.error(f"Could not retrieve or sort contracts for {root_symbol}. Aborting verification.")
        return

    # 2. Fetch the continuous contract data, determine actual date range
    try:
        continuous_df = conn.execute(
            f"SELECT date, close as continuous_close FROM continuous_contracts WHERE symbol = ? ORDER BY date", 
            [continuous_symbol]
        ).fetchdf()
        if continuous_df.empty:
            logger.error(f"No data found for continuous contract {continuous_symbol} in the database.")
            return
        # Convert date column to datetime.datetime objects for weekday checks
        continuous_df['date_dt'] = pd.to_datetime(continuous_df['date'])
        continuous_df['date'] = continuous_df['date_dt'].dt.date # Keep original date for comparison
        min_date = continuous_df['date'].min()
        max_date = continuous_df['date'].max()
        logger.info(f"Verifying data for {continuous_symbol} from {min_date} to {max_date}.")
    except Exception as e:
        logger.error(f"Error fetching continuous contract data for {continuous_symbol}: {e}")
        return

    # 3. Iterate through dates and verify
    results = []
    calculated_rollover_dates = {} # Store expiry date -> (old_contract, new_contract)
    gap_details = []
    last_verified_date = None

    # Determine starting index for the contracts based on contract_number
    current_base_index = contract_number - 1
    if current_base_index >= len(sorted_contracts_with_expiry):
        logger.error(f"Not enough contracts ({len(sorted_contracts_with_expiry)}) to generate {continuous_symbol} (needs base index {current_base_index}).")
        return
        
    active_contract_index = current_base_index
    last_active_symbol = None

    for index, row in continuous_df.iterrows():
        current_date_dt = row['date_dt'] # Use datetime object for checks
        current_date = row['date'] # Use date object for comparisons/fetching
        continuous_close = row['continuous_close']
        
        # Gap Check
        if last_verified_date:
            # Calculate business days difference (approximate)
            days_diff = (current_date - last_verified_date).days
            business_days_approx = sum(1 for i in range(1, days_diff) if (last_verified_date + timedelta(days=i)).weekday() < 5)
            if business_days_approx > 3: # Flag gaps larger than 3 business days
                gap_details.append(f"Gap detected: {business_days_approx} business days between {last_verified_date} and {current_date}")
                logger.warning(f"Large gap detected: {business_days_approx} business days between {last_verified_date} and {current_date}")
        last_verified_date = current_date
        
        # Sunday Check
        is_sunday = current_date_dt.weekday() == 6 # Monday is 0 and Sunday is 6

        # Ensure we have enough contracts
        if active_contract_index >= len(sorted_contracts_with_expiry):
             logger.warning(f"Ran out of contracts to determine active contract for {continuous_symbol} on date {current_date}. Stopping verification loop.")
             break 

        current_active_info = sorted_contracts_with_expiry[active_contract_index]
        current_expiry_date = current_active_info[1].date() # Compare date part only
        active_symbol_for_this_date = current_active_info[0]

        # Rollover Logic (same as generator): Rollover happens ON expiry day
        # Check if current_date is on or after the expiry date of the contract at active_contract_index
        needs_rollover = current_date >= current_expiry_date

        if needs_rollover:
            # Check if there's a next contract to roll into
            if active_contract_index + 1 < len(sorted_contracts_with_expiry):
                 # Perform the rollover for the *next* iteration's check
                 previous_active_info = current_active_info
                 active_contract_index += 1
                 new_active_info = sorted_contracts_with_expiry[active_contract_index]
                 active_symbol_for_this_date = new_active_info[0] # Use the NEW contract on the rollover day
                 logger.debug(f"{continuous_symbol}: Rollover check triggered on {current_date}. Previous contract {previous_active_info[0]} expired {previous_active_info[1].date()}. New active contract for this date: {active_symbol_for_this_date}")
                 # Record the rollover
                 calculated_rollover_dates[previous_active_info[1].date()] = (previous_active_info[0], new_active_info[0])
            else:
                # No more contracts to roll into
                logger.info(f"Reached end of available contracts for {continuous_symbol} after expiry of {current_active_info[0]} on {current_expiry_date}. Stopping verification loop.")
                break
        
        # Fetch underlying contract data for the determined active symbol on this date
        try:
            underlying_data = conn.execute(
                "SELECT close FROM market_data WHERE symbol = ? AND timestamp::DATE = ? AND interval_value = 1 AND interval_unit = 'day' LIMIT 1",
                [active_symbol_for_this_date, current_date]
            ).fetchone()
        except Exception as e:
             logger.error(f"Error fetching underlying data for {active_symbol_for_this_date} on {current_date}: {e}")
             underlying_data = None

        contract_close = underlying_data[0] if underlying_data else None
        price_diff = None
        consistency_check = "OK"

        if is_sunday:
            consistency_check = "SUNDAY_DATA"
        elif contract_close is not None and continuous_close is not None:
            price_diff = round(continuous_close - contract_close, 2)
            if abs(price_diff) > 0.01 and not needs_rollover: 
                consistency_check = "WARNING"
            elif needs_rollover:
                 consistency_check = "ROLLOVER_DAY"
        elif contract_close is None and continuous_close is not None:
             consistency_check = "MISSING_UNDERLYING"
        elif contract_close is not None and continuous_close is None:
             consistency_check = "MISSING_CONTINUOUS"
        else:
             consistency_check = "MISSING_BOTH"

        results.append({
            'date': current_date,
            'continuous_symbol': continuous_symbol,
            'active_contract': active_symbol_for_this_date,
            'continuous_close': continuous_close,
            'contract_close': contract_close,
            'price_diff': price_diff,
            'consistency_check': consistency_check
        })

        last_active_symbol = active_symbol_for_this_date

    # 4. Output Results
    results_df = pd.DataFrame(results)
    
    # Display results table (modify styling for SUNDAY_DATA)
    table = Table(title=f"Verification Results for {continuous_symbol}")
    table.add_column("Date", style="dim")
    # table.add_column("Continuous Symbol") # Redundant as it's in the title
    table.add_column("Active Contract")
    table.add_column("Continuous Close", justify="right")
    table.add_column("Contract Close", justify="right")
    table.add_column("Price Diff", justify="right")
    table.add_column("Check", justify="center")

    # Add rows, highlighting warnings and rollovers
    for _, row in results_df.iterrows():
        style = ""
        if row['consistency_check'] == "WARNING":
            style = "bold red"
        elif row['consistency_check'] == "ROLLOVER_DAY":
            style = "yellow"
        elif row['consistency_check'] == "SUNDAY_DATA":
             style = "bold blue"
        elif "MISSING" in row['consistency_check']:
            style = "magenta"

        table.add_row(
            str(row['date']),
            row['active_contract'],
            f"{row['continuous_close']:.2f}" if row['continuous_close'] is not None else "N/A",
            f"{row['contract_close']:.2f}" if row['contract_close'] is not None else "N/A",
            f"{row['price_diff']:.2f}" if row['price_diff'] is not None else "N/A",
            row['consistency_check'],
            style=style
        )

    console.print(table)
    
    # Report Gaps
    if gap_details:
        logger.warning("--- Large Gaps Detected ---")
        for gap_message in gap_details:
            logger.warning(gap_message)
        logger.warning("---------------------------")

    # 5. Verify Rollover Dates against 3rd Wednesday Rule
    logger.info("--- Rollover Date Verification ---")
    rollover_table = Table(title="Rollover Date Check vs Config (3rd Wednesday)")
    rollover_table.add_column("Expected Rollover (Expiry Date)")
    rollover_table.add_column("Old Contract")
    rollover_table.add_column("New Contract")
    rollover_table.add_column("Is 3rd Wednesday?", justify="center")

    config = load_config(DEFAULT_CONFIG_PATH)
    future_config = get_futures_config(config, root_symbol)

    sorted_rollover_dates = sorted(calculated_rollover_dates.keys())

    for expiry_date in sorted_rollover_dates:
        old_contract, new_contract = calculated_rollover_dates[expiry_date]
        
        # Check if expiry_date is the 3rd Wednesday
        is_third_wednesday = False
        if future_config:
            calc_expiry = get_expiry_date(old_contract, future_config) # Recalculate to be sure
            if calc_expiry and calc_expiry.date() == expiry_date:
                 is_third_wednesday = True
        
        status_text = "[green]YES[/green]" if is_third_wednesday else "[bold red]NO[/bold red]"
        rollover_table.add_row(
            str(expiry_date),
            old_contract,
            new_contract,
            status_text
        )
        
    console.print(rollover_table)
    logger.info("Note: 3rd Wednesday check does not currently account for holidays.")

def clean_weekend_data(conn: duckdb.DuckDBPyConnection):
    """Removes weekend data from the continuous_contracts table."""
    try:
        logger.info("Attempting to delete weekend data (Saturday/Sunday) from continuous_contracts...")
        # DuckDB DAYOFWEEK: Sunday=0, Saturday=6
        delete_query = "DELETE FROM continuous_contracts WHERE DAYOFWEEK(date) = 0 OR DAYOFWEEK(date) = 6;"
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