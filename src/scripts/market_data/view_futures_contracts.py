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

# Database path
DB_PATH = "./data/financial_data.duckdb"

def get_futures_contracts(conn, pattern=None, interval_value=1, interval_unit='daily'):
    """Get a list of futures contracts matching the specified pattern and interval."""
    print(f"DEBUG: get_futures_contracts called with pattern: {pattern}, interval: {interval_value} {interval_unit}")
    
    # Base WHERE clause for interval matching
    # No longer need mapping - database is standardized to 'daily'
    interval_where_clause = f"AND interval_value = {interval_value} AND interval_unit = '{interval_unit}'"
    
    if pattern:
        if re.match(r'^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$', pattern):
            # Exact futures symbol
            query = f"""
                SELECT symbol
                FROM market_data
                WHERE symbol = '{pattern}'
                    {interval_where_clause}
                LIMIT 1
            """
            print(f"DEBUG: Using exact futures symbol match: {query}")
        elif pattern.startswith("^") and pattern.endswith("$"):
            # Regex pattern
            query = f"""
                SELECT DISTINCT symbol
                FROM market_data
                WHERE regexp_matches(symbol, '{pattern}')
                    {interval_where_clause}
                ORDER BY symbol
            """
            print(f"DEBUG: Using regex pattern: {query}")
        else:
            # Partial symbol match (base symbol) - Use LIKE for broader matching
            like_pattern = f'{pattern}%' # e.g., VX%
            query = f"""
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE '{like_pattern}' 
                    {interval_where_clause}
                ORDER BY symbol
            """
            print(f"DEBUG: Using partial symbol match with LIKE pattern: {like_pattern}")
    else:
        # Default pattern for any standard futures contract
        query = f"""
            SELECT DISTINCT symbol
            FROM market_data
            WHERE regexp_matches(symbol, '^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$')
                {interval_where_clause}
            ORDER BY symbol
        """
        print(f"DEBUG: Using default futures pattern.")
    
    print(f"DEBUG: Executing query: {query}")
    
    try:
        result = conn.execute(query).fetchdf()
        print(f"DEBUG: Query returned {len(result)} rows")
        if not result.empty:
            print(f"DEBUG: First few symbols in result: {', '.join(result['symbol'].head().tolist())}")
        # Simplified check
        test_symbol = 'NQH25' if pattern != 'NQH25' else 'ESH25' # Avoid self-check if NQH25 is pattern
        if test_symbol in result['symbol'].values:
            print(f"DEBUG: {test_symbol} is in the result set")
        else:
            print(f"DEBUG: {test_symbol} is NOT in the result set")
        return result
    except Exception as e:
        print(f"DEBUG: Error executing query: {e}")
        return pd.DataFrame(columns=['symbol'])

def get_contract_details(conn, symbol, interval_value=1, interval_unit='daily'):
    """Get details for a specific futures contract at a specific interval."""
    # No longer need mapping - database is standardized to 'daily'
    # db_interval_unit = 'day' if interval_unit == 'daily' and symbol.startswith('VX') else interval_unit # Map 'daily' to 'day' only for VX for now
    db_interval_unit = interval_unit # Use the input unit directly
    
    query = f"""
        SELECT 
            COUNT(*) as row_count,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            ROUND(AVG(high-low), 2) as avg_range,
            ROUND(AVG(volume), 0) as avg_volume
        FROM market_data
        WHERE symbol = '{symbol}'
            AND interval_value = {interval_value} 
            AND interval_unit = '{db_interval_unit}'
    """
    
    if symbol == 'NQH25':
        print(f"DEBUG: Getting details for NQH25 ({interval_value} {interval_unit}) with query: {query}")
        result = conn.execute(query).fetchone()
        print(f"DEBUG: NQH25 details result: {result}")
        return result
    
    return conn.execute(query).fetchone()

def parse_futures_symbol(symbol):
    """Parse a futures symbol into its components."""
    # Match the typical futures symbol pattern
    match = re.match(r'^([A-Z]{2,3})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    
    if not match:
        print(f"DEBUG: Could not parse symbol: {symbol}")
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
    if symbol == 'NQH25':
        print(f"DEBUG: Parsed NQH25 as {base_symbol} {month_name} {year}")
    
    return {
        'base_symbol': base_symbol,
        'month_code': month_code,
        'month_name': month_name,
        'year_code': year_code,
        'year': year,
        'full_name': f"{base_symbol} {month_name} {year}"
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

def list_futures_contracts(base_symbol=None, add_missing=False, interval_value=1, interval_unit='daily'):
    """List futures contracts, optionally filtered by base symbol and interval."""
    interval_str = f"{interval_value}{interval_unit[0]}"
    console = Console()
    try:
        conn = duckdb.connect(DB_PATH, read_only=False if add_missing else True)
        
        if add_missing:
            add_placeholder_contracts(conn, ["ESH25", "NQH25"])
        
        pattern = None
        title_symbol = "All"
        if base_symbol:
            if re.match(r'^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$', base_symbol):
                pattern = base_symbol  
                title_symbol = base_symbol
                print(f"\n=== Futures Contract {title_symbol} ({interval_str}) ===\n")
            else:
                pattern = base_symbol # Use base symbol directly for pattern matching in get_futures_contracts
                title_symbol = base_symbol
                print(f"\n=== Futures Contracts for {title_symbol} ({interval_str}) ===\n")
        else:
            print(f"\n=== All Futures Contracts in Database ({interval_str}) ===\n")
        
        contracts_df = get_futures_contracts(conn, pattern, interval_value, interval_unit)
        
        if contracts_df.empty:
            print(f"No {interval_str} futures contracts found matching '{pattern or 'any'}'.")
            return
        
        grouped_contracts = {}
        all_contracts = []
        
        print(f"DEBUG: Processing {len(contracts_df)} contracts for {interval_str} interval")
        
        for _, row in contracts_df.iterrows():
            symbol = row['symbol']
            
            parsed = parse_futures_symbol(symbol)
            if not parsed:
                print(f"DEBUG: Skipping {symbol} due to parsing failure")
                continue
            
            details = get_contract_details(conn, symbol, interval_value, interval_unit)
            
            if not details or details[0] == 0:
                 print(f"DEBUG: No details found for {symbol} at {interval_str}, skipping.")
                 continue
            
            contract_info = {
                'symbol': symbol,
                'base_symbol': parsed['base_symbol'],
                'full_name': parsed['full_name'],
                'row_count': details[0],
                'first_date': details[1].strftime('%Y-%m-%d') if details[1] else 'N/A',
                'last_date': details[2].strftime('%Y-%m-%d') if details[2] else 'N/A',
                'avg_range': details[3] if details[3] is not None else 'N/A',
                'avg_volume': details[4] if details[4] is not None else 'N/A'
            }
            
            # Debug: Print contract info for NQH25
            if symbol == 'NQH25':
                print(f"DEBUG: NQH25 contract_info: {contract_info}")
            
            # Add to group
            base = parsed['base_symbol']
            if base not in grouped_contracts:
                grouped_contracts[base] = []
            grouped_contracts[base].append(contract_info)
            all_contracts.append(contract_info)
        
        # Display logic
        if not grouped_contracts:
             print(f"No contract details found for the {interval_str} interval.")
             return

        overall_total_contracts = 0
        overall_total_rows = 0

        for base in sorted(grouped_contracts.keys()):
            contracts = grouped_contracts[base]
            contracts.sort(key=lambda x: parse_futures_symbol(x['symbol'])['year'] * 100 + 
                         list("FGHJKMNQUVXZ").index(parse_futures_symbol(x['symbol'])['month_code']))
            
            # Create rich table
            table_title = f"[bold cyan]== {base} Futures Contracts ({len(contracts)} contracts) - Interval: {interval_str} ==[/bold cyan]"
            table = Table(title=table_title, show_header=True, header_style="bold blue", border_style="dim blue")
            
            # Define columns
            columns_to_display = ['symbol', 'full_name', 'row_count', 'first_date', 'last_date', 'avg_range', 'avg_volume']
            justify_map = {'row_count': 'right', 'avg_range': 'right', 'avg_volume': 'right'}
            style_map = {'symbol': 'cyan', 'full_name': 'white', 'row_count': 'green', 'first_date': 'yellow', 'last_date': 'yellow', 'avg_range': 'magenta', 'avg_volume': 'magenta'}
            
            for col in columns_to_display:
                table.add_column(col, style=style_map.get(col, 'white'), justify=justify_map.get(col, 'left'))

            # Add rows
            group_total_rows = 0
            for contract in contracts:
                row_values = [format_value(contract.get(col, 'N/A')) for col in columns_to_display]
                table.add_row(*row_values)
                group_total_rows += contract.get('row_count', 0)
            
            console.print(table) # Print table for the current base symbol
            console.print(f"[dim]Total rows for {base}: {group_total_rows:,}[/dim]\n")
            
            overall_total_contracts += len(contracts)
            overall_total_rows += group_total_rows
        
        # Overall Summary Table
        summary_table = Table(title=f"[bold magenta]=== Summary for Interval {interval_str} ===[/bold magenta]", show_header=False, border_style="green", show_edge=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")
        summary_table.add_row("Total contracts displayed:", f"{overall_total_contracts:,}")
        summary_table.add_row("Total rows displayed:", f"{overall_total_rows:,}")
        summary_table.add_row("Unique base symbols displayed:", f"{len(grouped_contracts):,}")
        console.print(summary_table)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description='View futures contracts in the database.')
    parser.add_argument('base_symbol', nargs='?', default=None, 
                        help='Optional base symbol (e.g., ES) or specific contract (e.g., ESH24) to filter by.')
    parser.add_argument('--interval-value', type=int, default=1, 
                        help='Interval value (default: 1).')
    parser.add_argument('--interval-unit', choices=['daily', 'minute'], default='daily', 
                        help='Interval unit (default: daily).')
    parser.add_argument('--add-missing', action='store_true',
                        help='Add placeholder entries for March 2025 contracts (if needed).')
    
    args = parser.parse_args()
    
    # Debug print args
    print(f"DEBUG: Running with args: base_symbol={args.base_symbol}, interval_value={args.interval_value}, interval_unit={args.interval_unit}, add_missing={args.add_missing}")
    
    list_futures_contracts(
        base_symbol=args.base_symbol, 
        add_missing=args.add_missing,
        interval_value=args.interval_value,
        interval_unit=args.interval_unit
    ) 