#!/usr/bin/env python
"""
View Futures Contracts in Database

This script displays a summary of all futures contracts in the database,
with details like symbol, date range, and row counts.
"""

import os
import sys
import duckdb
import pandas as pd
from datetime import datetime, date
import re
import argparse
from rich.console import Console
from rich.table import Table
import logging
from pathlib import Path
from typing import Optional

# --- Imports needed for temporary update ---
# REMOVED: Unused imports causing ModuleNotFoundError
# from src.scripts.market_data.vix.update_vx_futures import (
#     download_cboe_data, 
#     prepare_data_for_db, 
#     update_market_data, 
#     ROOT_SYMBOL # Import ROOT_SYMBOL ('VX') if needed for queries
# )
# --- End temporary imports ---

# Database path
# REMOVED: Incorrect global DB_PATH variable. Path is handled in __main__.
# DB_PATH = "C:/temp/data/financial_data.duckdb" # Point back to original DB in temp location

# Configure logger
logger = logging.getLogger(__name__)

def _get_metadata_table(conn, base_symbol: str, interval_unit: str, interval_value: int) -> Optional[str]:
    """Helper to query symbol_metadata for the correct data table based on symbol and interval."""
    try:
        query = "SELECT data_table FROM symbol_metadata WHERE base_symbol = ? AND interval_unit = ? AND interval_value = ? LIMIT 1"
        # Ensure base_symbol is uppercase for the query to match how it's stored from market_symbols.yaml
        params = [base_symbol.upper(), interval_unit, interval_value]
        result = conn.execute(query, params).fetchone()
        if result:
            return result[0]
        else:
            # Fallback logic if metadata not found
            logging.warning(f"Metadata not found for {base_symbol} ({interval_value} {interval_unit}). Defaulting to 'market_data'.")
            return 'market_data'
    except Exception as e:
        logging.error(f"Error querying symbol_metadata for {base_symbol} ({interval_value} {interval_unit}): {e}")
        return 'market_data' # Fallback on error

def get_futures_contracts(conn, pattern=None, interval_value=1, interval_unit='daily'):
    """Get a list of futures contracts matching the specified pattern and interval."""
    
    # --- Determine target table based on pattern AND interval using METADATA ---    
    target_table = 'market_data' # Default for 'all' or if pattern is None
    if pattern:
        # Use the original interval unit passed to this function for the metadata lookup
        target_table = _get_metadata_table(conn, pattern, interval_unit, interval_value) # Use original unit
        # Print the targeted table if it's CBOE (similar to previous logic)
        if target_table == 'market_data_cboe':
             print(f"Targeting CBOE table: {target_table}")
    
    # Base WHERE clause for interval matching - USE original unit here too
    interval_where_clause = f"AND interval_value = {interval_value} AND interval_unit = '{interval_unit}'"
    
    if pattern:
        if re.match(r'^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$', pattern):
            # Exact futures symbol
            query = f"""
                SELECT symbol
                FROM {target_table}
                WHERE symbol = '{pattern}'
                    {interval_where_clause}
                LIMIT 1
            """
            # print(f"DEBUG: Using exact futures symbol match: {query}")
        elif pattern.startswith("^") and pattern.endswith("$"):
            # Regex pattern
            query = f"""
                SELECT DISTINCT symbol
                FROM {target_table}
                WHERE regexp_matches(symbol, '{pattern}')
                    {interval_where_clause}
                ORDER BY symbol
            """
            # print(f"DEBUG: Using regex pattern: {query}")
        else:
            # Partial symbol match (base symbol) - Use LIKE for broader matching
            like_pattern = f'{pattern}%' # e.g., VX%
            query = f"""
                SELECT DISTINCT symbol
                FROM {target_table}
                WHERE symbol LIKE '{like_pattern}' 
                    {interval_where_clause}
                ORDER BY symbol
            """
            # print(f"DEBUG: Using partial symbol match with LIKE pattern: {like_pattern}")
    else:
        # Default pattern for any standard futures contract (from market_data)
        target_table = 'market_data' # Override for default pattern - only check main table
        # print(f"DEBUG: Overriding target table to {target_table} for default pattern.")
        query = f"""
            SELECT DISTINCT symbol
            FROM {target_table}
            WHERE regexp_matches(symbol, '^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$')
                {interval_where_clause}
            ORDER BY symbol
        """
        # print(f"DEBUG: Using default futures pattern.")
    
    # print(f"DEBUG: Executing query: {query}")
    
    try:
        result_df = conn.execute(query).fetchdf()
        return result_df, target_table # Return both the DataFrame and the determined target table
    except Exception as e:
        logging.error(f"Error executing get_futures_contracts query: {e}")
        return pd.DataFrame(columns=['symbol']), target_table # Return empty df but still return table

def get_contract_details(conn, symbol, interval_value=1, interval_unit='daily', target_table='market_data'):
    """Get details for a specific futures contract at a specific interval from the specified table."""
    db_interval_unit = interval_unit # Use the input unit directly
    
    # --- Use the provided target_table --- # 
    # Remove the logic that tries to determine table based on symbol or metadata lookup here
    # base_match = re.match(r'^([A-Z]{2,3})', symbol)
    # base_symbol_for_lookup = base_match.group(1) if base_match else symbol
    # target_table = _get_metadata_table(conn, base_symbol_for_lookup)
    # --- End removal ---
    
    # Use parameterized query with the passed target_table
    query = f"""
        SELECT
            COUNT(*) as row_count,
            MIN(timestamp)::DATE as first_date,  -- Cast to DATE for consistency
            MAX(timestamp)::DATE as last_date,   -- Cast to DATE for consistency
            ROUND(AVG(high-low), 4) as avg_range -- Increased precision for range
        FROM {target_table}
        WHERE symbol = ?
            AND interval_value = ?
            AND interval_unit = ?
    """
    
    params = [symbol, int(interval_value), db_interval_unit]

    try:
        result = conn.execute(query, params).fetchone()
        # Adjust result tuple index since volume was removed
        # Original: (row_count, first_date, last_date, avg_range, avg_volume) -> len 5
        # New:      (row_count, first_date, last_date, avg_range)            -> len 4
        
        # Pad with None for missing avg_volume to maintain structure downstream
        if result is not None:
            result_padded = result + (None,) # Add None for avg_volume
        else:
            result_padded = (0, None, None, None, None) # Default error tuple

        # print(f"DEBUG: Details for {symbol} ({interval_value}{interval_unit}): {result_padded}")
        return result_padded
    except Exception as e:
        print(f"Error getting contract details for {symbol}: {e}")
        # Return a tuple matching the expected structure but with error indicators
        return (0, None, None, None, None)

def parse_futures_symbol(symbol):
    """Parse a futures symbol into its components."""
    # Match the typical futures symbol pattern
    match = re.match(r'^([A-Z]{2,3})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    
    if not match:
        # print(f"DEBUG: Could not parse symbol: {symbol}")
        return None
    
    base_symbol, month_code, year_code = match.groups()
    
    # Convert month code to month name
    month_map = {
        'F': 'January',
        'G': 'February', 
        'H': 'March',
        'J': 'April',
        'K': 'May',
        'M': 'June',
        'N': 'July',
        'Q': 'August',
        'U': 'September',
        'V': 'October',
        'X': 'November',
        'Z': 'December'
    }
    
    month_name = month_map.get(month_code, 'Unknown')
    
    # Convert 2-digit year to 4-digit year
    year_num = int(year_code)
    if year_num < 50:  # Assume 20xx for years less than 50
        year = 2000 + year_num
    else:  # Assume 19xx for years 50 and greater
        year = 1900 + year_num
    
    # Debug: Print when parsing NQH25
    # if symbol == 'NQH25':
        # print(f"DEBUG: Parsed NQH25 as {base_symbol} {month_name} {year}")
    
    return {
        'base_symbol': base_symbol,
        'month_code': month_code,
        'month_name': month_name,
        'year_code': year_code,
        'year': year,
        'full_name': f"{base_symbol} {month_name} {year}"
    }

# Month code mapping for sorting
MONTH_CODE_ORDER = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

def add_placeholder_contracts(conn, symbols):
    """Add placeholder entries for futures contracts that aren't in the database yet."""
    print(f"\nAdding placeholder entries for missing contracts: {', '.join(symbols)}")
    
    today = date.today()
    
    for symbol in symbols:
        # Parse the symbol
        parsed = parse_futures_symbol(symbol)
        if not parsed:
            print(f"Could not parse symbol: {symbol}")
            continue
        
        # Check if the contract already exists
        check_query = f"""
            SELECT COUNT(*) FROM market_data 
            WHERE symbol = '{symbol}'
        """
        count = conn.execute(check_query).fetchone()[0]
        
        if count > 0:
            print(f"Contract {symbol} already exists with {count} rows")
            continue
        
        # Create a placeholder entry for today's date
        timestamp = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Insert a single placeholder row
        insert_query = f"""
            INSERT INTO market_data (
                timestamp, symbol, open, high, low, close, settle,
                volume, interval_value, interval_unit, source, quality
            ) VALUES (
                '{timestamp}', '{symbol}', 0, 0, 0, 0, 0, 
                0, 1, 'daily', 'Placeholder', 0
            )
        """
        try:
            conn.execute(insert_query)
            print(f"Added placeholder entry for {symbol} ({parsed['full_name']})")
        except Exception as e:
            print(f"Error adding placeholder for {symbol}: {e}")
    
    print("Done adding placeholders\n")

def format_value(value) -> str:
    """Helper function to format table cell values nicely."""
    if pd.isna(value):
        return "[dim]N/A[/dim]"
    if isinstance(value, (int, float)):
        if abs(value) > 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if abs(value) > 1_000_000:
             return f"{value / 1_000_000:.2f}M"
        if abs(value) > 1_000:
             return f"{value / 1_000:.1f}K"
        if isinstance(value, float):
            return f"{value:.4f}" 
    return str(value)

def list_futures_contracts(base_symbol=None, add_missing=False, interval_value=1, interval_unit='daily', conn=None):
    """List futures contracts with details."""
    # This function now directly receives interval parameters
    # print(f"DEBUG: list_futures_contracts called with base: {base_symbol}, iv: {interval_value}, iu: {interval_unit}")
    
    if conn is None:
        print("Error: Database connection not provided.")
        return

    # Get contracts based on base_symbol (pattern) and interval
    # Now returns a tuple: (DataFrame, target_table_name)
    contracts_df, target_table_for_details = get_futures_contracts(conn, base_symbol, interval_value, interval_unit)

    if contracts_df.empty:
        print(f"No contracts found matching pattern: {base_symbol or '[All Futures]'} for {interval_value} {interval_unit}")
        return

    # Collect details and parsed info for sorting
    contracts_info = []
    for symbol in contracts_df['symbol']:
        # Pass the determined target_table to get_contract_details
        details = get_contract_details(conn, symbol, interval_value, interval_unit, target_table=target_table_for_details) 
        parsed = parse_futures_symbol(symbol)
        contracts_info.append({
            'symbol': symbol,
            'details': details, # Tuple: (rows, start_date, end_date, avg_range, avg_volume)
            'parsed': parsed    # Dict or None
        })

    # Sort the contracts_info list
    def sort_key(contract):
        parsed = contract['parsed']
        if parsed:
            year = parsed['year']
            month_order = MONTH_CODE_ORDER.get(parsed['month_code'], 99) # Place unknown months last
            return (year, month_order)
        else:
            return (9999, 99) # Place unparsable symbols at the very end

    contracts_info.sort(key=sort_key)

    # Ensure get_contract_details is called with the correct interval
    # contract_details = [get_contract_details(conn, symbol, interval_value, interval_unit) 
    #                     for symbol in contracts_df['symbol']]

    # Create console for rich display
    console = Console()
    
    # Create table for display
    table = Table(
        title=f"Futures Contracts for {base_symbol or 'All'} ({interval_value} {interval_unit})",
        show_header=True, 
        header_style="bold magenta",
        border_style="blue"
    )
    
    # Add columns
    table.add_column("Symbol", style="cyan")
    table.add_column("Contract", style="green")
    table.add_column("Start Date", style="green")
    table.add_column("End Date", style="green")
    table.add_column("Rows", justify="right", style="yellow")
    table.add_column("Avg Range", justify="right", style="yellow")
    
    # Count of displayed contracts
    displayed_count = 0
    
    # Add rows for each contract from the sorted list
    # for idx, symbol in enumerate(contracts_df['symbol']):
    for contract in contracts_info:
        symbol = contract['symbol']
        # Get details from our tuple
        # rows, start_date, end_date, avg_range, avg_volume = contract_details[idx]
        rows, start_date, end_date, avg_range, avg_volume = contract['details']
        
        # Parse the symbol for better display
        # parsed = parse_futures_symbol(symbol)
        parsed = contract['parsed']
        contract_name = parsed['full_name'] if parsed else symbol
        
        # Format values for display
        start_str = start_date.strftime('%Y-%m-%d') if start_date else 'N/A'
        end_str = end_date.strftime('%Y-%m-%d') if end_date else 'N/A'
        rows_str = format_value(rows) if rows else '0'
        range_str = format_value(avg_range) if avg_range else 'N/A'
        
        # Add row to table
        table.add_row(
            symbol,
            contract_name,
            start_str,
            end_str,
            rows_str,
            range_str
        )
        displayed_count += 1
    
    # Display the table
    console.print("\n")
    console.print(table)
    
    # Print summary
    print(f"\nTotal Contracts: {displayed_count}")
    
    # Optionally add missing contracts if requested
    if add_missing:
        missing_symbols = []
        # Calculate missing symbols based on contract pattern
        # Implementation would go here
        
        # Add missing contracts if any found
        if missing_symbols:
            add_placeholder_contracts(conn, missing_symbols)

# Placeholder for temporary update function (if still needed, otherwise remove)
# def temporary_update_contract(conn, symbol, contract_details):
#     # ... (implementation) ...

if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="View futures contract details stored in the database.")
    parser.add_argument('base_symbol', nargs='?', default=None, help='Optional base symbol (e.g., ES, NQ, VX) or specific contract (e.g., ESH24) to filter by.')
    parser.add_argument("--add-missing", action="store_true", help="Attempt to add missing placeholder contracts (Use with caution). Currently disabled.")
    parser.add_argument("--db-path", default="data/financial_data.duckdb", help="Path to the DuckDB database file.")
    # Add arguments for interval - support both hyphenated and underscore versions for compatibility
    parser.add_argument("--interval_value", "--interval-value", type=int, default=1, help="Interval value (e.g., 1, 15).")
    parser.add_argument("--interval_unit", "--interval-unit", type=str, default='daily', choices=['min', 'minute', 'daily', 'weekly', 'monthly'], help="Interval unit (e.g., daily, min).")
    
    args = parser.parse_args()

    # --- Database Connection --- #
    db_file = Path(args.db_path).resolve() # Use Path object for better handling
    if not db_file.exists():
        print(f"Error: Database file not found at {db_file}")
        sys.exit(1)
    
    conn = None # Initialize conn
    try:
        conn = duckdb.connect(database=str(db_file), read_only=True) # Connect read-only by default
        print(f"Connected to database: {db_file}")
        
        # --- Call Listing Function with Parsed Args --- #
        list_futures_contracts(
            base_symbol=args.base_symbol, 
            add_missing=args.add_missing, 
            interval_value=args.interval_value, 
            interval_unit=args.interval_unit, # Use the parsed and normalized value
            conn=conn
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

