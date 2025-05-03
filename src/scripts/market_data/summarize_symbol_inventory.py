import duckdb
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional
import argparse
from rich.console import Console
from rich.table import Table

# Determine project root and database path
# This assumes the script is located in src/scripts/market_data/
project_root = Path(__file__).resolve().parent.parent.parent.parent # Go up four levels
db_path = project_root / "data" / "financial_data.duckdb"
METADATA_TABLE_NAME = "symbol_metadata" # Added constant

def _get_metadata_table(conn, base_symbol: str, interval_unit: str, interval_value: int) -> Optional[str]:
    """Helper to query symbol_metadata for the correct data table based on symbol and interval."""
    try:
        query = f"SELECT data_table FROM {METADATA_TABLE_NAME} WHERE base_symbol = ? AND interval_unit = ? AND interval_value = ? LIMIT 1"
        params = [base_symbol, interval_unit, interval_value]
        result = conn.execute(query, params).fetchone()
        if result:
            return result[0]
        else:
            # Fallback logic if metadata not found
            print(f"[Warning] Metadata not found for {base_symbol} ({interval_value} {interval_unit}). Cannot determine target table.", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[Error] querying symbol_metadata for {base_symbol} ({interval_value} {interval_unit}): {e}", file=sys.stderr)
        return None

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

def summarize_inventory(db_connection: duckdb.DuckDBPyConnection, base_symbol_filter: Optional[str] = None, continuous_only: bool = False):
    """Fetches symbol metadata and prints a concise summary using Rich."""
    console = Console()
    
    if continuous_only:
        print("--- Fetching Continuous Contract Inventory Metadata ---")
        target_table = "continuous_contracts"
        symbol_column = "symbol"
        query = f"""
        SELECT 
            symbol, 
            interval_unit,
            interval_value,
            adjusted,
            built_by,
            strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
            strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
            COUNT(*) as record_count
        FROM {target_table} 
        GROUP BY symbol, interval_unit, interval_value, adjusted, built_by
        ORDER BY symbol, interval_unit, interval_value, adjusted, built_by;
        """
    else:
        print("--- Fetching Symbol Inventory Metadata ---")
        if base_symbol_filter:
            print(f"--- Filtering for Base Symbol: {base_symbol_filter} ---")
        
        # Determine target table based on filter (existing logic)
        # target_table = "market_data"
        # symbol_column = "symbol"
        # if base_symbol_filter and base_symbol_filter.startswith('@'):
        #     target_table = "continuous_contracts"
        #     print(f"--- Querying continuous_contracts table for {base_symbol_filter} ---")
        #     # Query for specific continuous base symbol
        #     query = f\"\"\" ... \"\"\" # Old query kept for reference
        #     query_params = (f"{base_symbol_filter}%",)
        # elif base_symbol_filter:
        #      target_table = "market_data" # Assume specific non-continuous is in market_data for now
        #      print(f"--- Querying {target_table} table for Base Symbol: {base_symbol_filter} ---")
        #      # This part might need refinement if filtered base symbols exist in market_data_cboe
        #      query = f\"\"\" ... \"\"\" # Old query kept for reference
        #      query_params = (f"{base_symbol_filter}%",)
        # else: # <-- This part remains the same for Option O1 (no filter)
        #      print(f"--- Querying market_data and market_data_cboe tables for all symbols ---")
        #      # Query to get symbol, interval, counts, and dates from BOTH tables
        #      query = f\"\"\" ... \"\"\" # UNION ALL Query
        #      query_params = ()

        # --- NEW LOGIC using Metadata Table --- # 
        if base_symbol_filter:
            # When filtering, we need *an* interval to look up the table.
            # Default to daily for this summary view, or make it configurable?
            # For now, assume daily (day, 1) is the primary interval for filtered summary.
            lookup_interval_unit = 'day'
            lookup_interval_value = 1
            # Special case for continuous - they use a placeholder interval
            if base_symbol_filter.startswith('@'):
                lookup_interval_unit = 'continuous'
                lookup_interval_value = 0
                
            target_table = _get_metadata_table(db_connection, base_symbol_filter, lookup_interval_unit, lookup_interval_value)
            if not target_table:
                print(f"Could not find metadata for base symbol {base_symbol_filter} and interval ({lookup_interval_value} {lookup_interval_unit}). Aborting.", file=sys.stderr)
                return # Exit if no metadata found
                
            print(f"--- Querying table '{target_table}' for Base Symbol: {base_symbol_filter} (metadata for {lookup_interval_value} {lookup_interval_unit}) ---")
            symbol_column = "symbol" # Column name is usually symbol
            # Query the specific target table based on metadata
            query = f"""
            SELECT
                {symbol_column} as symbol, 
                interval_unit,
                interval_value,
                strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
                strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
                COUNT(*) as record_count
            FROM {target_table}
            WHERE {symbol_column} LIKE ? -- Parameter binding for safety
            GROUP BY {symbol_column}, interval_unit, interval_value
            ORDER BY {symbol_column}, interval_unit, interval_value;
            """
            # Adjust LIKE pattern based on symbol type if necessary (e.g., continuous)
            like_pattern = f"{base_symbol_filter}%" 
            query_params = (like_pattern,)
        else:
            # --- UNFILTERED O1 Case - Keep UNION ALL --- #
            print(f"--- Querying market_data and market_data_cboe tables for all symbols (UNION ALL) ---")
            query = f"""
            WITH combined_data AS (
                 SELECT
                     symbol,
                     interval_unit,
                     interval_value,
                     timestamp
                 FROM market_data
                 UNION ALL
                 SELECT
                     symbol,
                     'day' as interval_unit, -- Assume CBOE data is daily
                     1 as interval_value,   -- Assume CBOE data is daily
                     timestamp
                 FROM market_data_cboe
             )
             SELECT
                 symbol,
                 interval_unit,
                 interval_value,
                 strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
                 strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
                 COUNT(*) as record_count
             FROM combined_data
             GROUP BY symbol, interval_unit, interval_value
             ORDER BY symbol, interval_unit, interval_value;
            """
            query_params = () # No parameters for the combined query
            # --- END UNFILTERED O1 Case ---

    try:
        # Use execute with parameters if they exist
        if query_params:
            metadata_df = db_connection.execute(query, query_params).fetchdf()
        else:
            metadata_df = db_connection.execute(query).fetchdf()

        if metadata_df.empty:
            if continuous_only:
                print("No data found in the continuous_contracts table.")
            elif base_symbol_filter:
                 print(f"No data found for filter: {base_symbol_filter}")
            else:
                 print("No market data found in the database.")
            return

        # --- START CONTINUOUS ONLY DISPLAY --- 
        if continuous_only:
            print(f"\nFound metadata for {metadata_df['symbol'].nunique()} unique continuous symbols.")
            cont_table = Table(title="[bold blue]Continuous Contract Summary[/bold blue]", show_header=True, header_style="bold magenta", border_style="dim cyan")
            cont_table.add_column("Symbol", style="cyan", no_wrap=True)
            cont_table.add_column("Interval", style="white")
            cont_table.add_column("Adjusted", style="yellow")
            cont_table.add_column("Built By", style="yellow")
            cont_table.add_column("First Date", style="yellow")
            cont_table.add_column("Last Date", style="yellow")
            cont_table.add_column("Records", style="green", justify="right")

            # Helper to format interval
            def format_interval(row):
                unit = row.get('interval_unit', '').lower()
                val = row.get('interval_value', '?')
                if unit == 'day': return f"{val}d"
                if unit == 'minute': return f"{val}m"
                # Add other units if necessary
                return f"{val}{unit[:1] if unit else ''}"
            
            for _, row in metadata_df.iterrows():
                cont_table.add_row(
                    row['symbol'],
                    format_interval(row),
                    str(row.get('adjusted', 'N/A')),
                    str(row.get('built_by', 'N/A')),
                    format_value(row['first_date']),
                    format_value(row['last_date']),
                    format_value(row['record_count'])
                )
            console.print("\n")
            console.print(cont_table)
            return # End here for continuous only
        # --- END CONTINUOUS ONLY DISPLAY --- 

        # --- START ORIGINAL DISPLAY LOGIC (for market_data or filtered continuous) --- 
        print(f"\nFound metadata for {metadata_df['symbol'].nunique()} unique symbols/contracts across various intervals.")
        
        # Extract base symbol
        def get_base(symbol):
            if symbol.startswith('@') and '=' in symbol:
                 return symbol.split('=')[0] # Handle continuous like @ES=101XN -> @ES
            elif symbol.startswith('$VIX'): # Explicitly handle VIX index
                return "$VIX.X"
            elif symbol.startswith('VX'): # Handle VX futures (e.g., VXK25)
                 # Check if the rest looks like a contract code (e.g., M25)
                 if len(symbol) >= 4 and symbol[2].isalpha() and symbol[3:].isdigit():
                     return "VX"
                 # Fallback for other VX symbols if necessary, or return original
                 return symbol # Or maybe "VX" if all VX... should group
            elif symbol == 'SPY' or symbol == 'QQQ' or symbol == 'AAPL' or symbol == 'GS': # Handle specific equities
                 return symbol
            # General futures pattern (e.g., ESM25 -> ES)
            elif len(symbol) >= 4 and symbol[-3].isalpha() and symbol[-2:].isdigit():
                 # Crude check for futures code like M24, Z23 etc.
                 if symbol[-3] in 'FGHJKMNQUVXZ':
                     return symbol[:-3]
            # Default: return original symbol if no pattern matches
            return symbol

        metadata_df['base_symbol'] = metadata_df['symbol'].apply(get_base)
        
        # Create the combined interval string (e.g., '1d', '15m') early on
        # Handle potential missing interval_unit/value (e.g., from CBOE part if query changes)
        def safe_interval_format(row):
            unit = getattr(row, 'interval_unit', None)
            value = getattr(row, 'interval_value', None)
            if unit and value is not None:
                unit_char = unit[0] if isinstance(unit, str) and len(unit) > 0 else '?'
                return f"{value}{unit_char}"
            return 'N/A' # Fallback if columns are missing

        metadata_df['interval'] = metadata_df.apply(safe_interval_format, axis=1)
        
        # Filter by base symbol if provided *after* combining data and extracting base
        if base_symbol_filter:
            metadata_df = metadata_df[metadata_df['base_symbol'] == base_symbol_filter].copy()
            if metadata_df.empty:
                print(f"No data found for base symbol: {base_symbol_filter}")
                return
            print(f"Filtered down to {metadata_df['symbol'].nunique()} contracts for base symbol {base_symbol_filter}.")
            
            # --- Add Table for Specific Contracts Found --- 
            specific_contracts_table = Table(title=f"[bold blue]Specific Continuous Contracts Found for {base_symbol_filter}[/bold blue]", show_header=True, header_style="bold magenta", border_style="dim cyan")
            specific_contracts_table.add_column("Specific Symbol", style="cyan", no_wrap=True)
            specific_contracts_table.add_column("Interval", style="white")
            specific_contracts_table.add_column("First Date", style="yellow")
            specific_contracts_table.add_column("Last Date", style="yellow")
            specific_contracts_table.add_column("Records", style="green", justify="right")
            
            # Sort by symbol then interval before displaying
            metadata_df_sorted = metadata_df.sort_values(by=['symbol', 'interval'])
            
            for _, row in metadata_df_sorted.iterrows():
                specific_contracts_table.add_row(
                    row['symbol'],
                    row['interval'], # Use the calculated interval (e.g., 1d, 15m)
                    format_value(row['first_date']),
                    format_value(row['last_date']),
                    format_value(row['record_count'])
                )
            console.print("\n")
            console.print(specific_contracts_table)
            # --- End Specific Contracts Table ---
        
        # --- Overall Summary --- 
        if not base_symbol_filter:
            overall_summary_df = metadata_df.groupby('base_symbol').agg(
                overall_first_date=('first_date', 'min'),
                overall_last_date=('last_date', 'max'),
                total_records=('record_count', 'sum'),
                unique_contracts=('symbol', 'nunique')
            ).reset_index()
            
            overall_table = Table(title="[bold magenta]Overall Symbol Summary[/bold magenta]", show_header=True, header_style="bold blue", border_style="green")
            overall_table.add_column("Base Symbol", style="cyan")
            overall_table.add_column("Unique Contracts", style="green", justify="right")
            overall_table.add_column("First Date", style="yellow")
            overall_table.add_column("Last Date", style="yellow")
            overall_table.add_column("Total Records", style="green", justify="right")
            
            for _, row in overall_summary_df.iterrows():
                overall_table.add_row(
                    row['base_symbol'],
                    format_value(row['unique_contracts']),
                    format_value(row['overall_first_date']),
                    format_value(row['overall_last_date']),
                    format_value(row['total_records'])
                )
            console.print("\n")
            console.print(overall_table)
        
        # --- Interval Detail Summary --- 
        title_suffix = f" for {base_symbol_filter}" if base_symbol_filter else ""
        
        interval_detail = metadata_df[[
            'base_symbol', 'symbol', 'interval', 'record_count', 'first_date', 'last_date'
        ]].rename(columns={
            'record_count': 'interval_records',
            'first_date': 'interval_first_date',
            'last_date': 'interval_last_date'
        })
        aggregated_interval_summary_df = interval_detail.groupby(['base_symbol', 'interval']).agg(
            interval_first_date=('interval_first_date', 'min'),
            interval_last_date=('interval_last_date', 'max'),
            interval_records=('interval_records', 'sum'),
            contracts_in_interval=('symbol', 'nunique')
        ).reset_index()
        
        interval_table = Table(title=f"[bold magenta]Interval Details by Base Symbol{title_suffix}[/bold magenta]", show_header=True, header_style="bold blue", border_style="dim blue")
        interval_table.add_column("Base Symbol", style="cyan")
        interval_table.add_column("Interval", style="white")
        interval_table.add_column("First Date", style="yellow")
        interval_table.add_column("Last Date", style="yellow")
        interval_table.add_column("Records", style="green", justify="right")
        interval_table.add_column("Contracts", style="green", justify="right")
        
        for _, row in aggregated_interval_summary_df.iterrows():
            interval_table.add_row(
                row['base_symbol'],
                row['interval'],
                format_value(row['interval_first_date']),
                format_value(row['interval_last_date']),
                format_value(row['interval_records']),
                format_value(row['contracts_in_interval'])
            )
        console.print("\n")
        console.print(interval_table)
        
    except duckdb.Error as e:
        print(f"DuckDB Error fetching inventory metadata: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error summarizing inventory: {e}", file=sys.stderr)

def main():
    # Add argparse for base_symbol filter
    parser = argparse.ArgumentParser(description='Summarize symbol inventory in the database.')
    parser.add_argument('--base-symbol', type=str, default=None,
                        help='Optional base symbol (e.g., ES, @VX) to filter results for.')
    parser.add_argument('--continuous-only', action='store_true',
                        help='Summarize only the continuous_contracts table.')
    args = parser.parse_args()
    
    if not db_path.is_file():
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = None
    try:
        conn = duckdb.connect(database=str(db_path), read_only=True)
        print(f"Connected to database: {db_path} (Read-Only)")
        # Pass the filter argument to summarize_inventory
        summarize_inventory(conn, base_symbol_filter=args.base_symbol, continuous_only=args.continuous_only)
    except duckdb.Error as e:
        print(f"Failed to connect to database {db_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main() 