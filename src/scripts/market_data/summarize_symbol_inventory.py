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
        target_table = "market_data"
        symbol_column = "symbol"
        if base_symbol_filter and base_symbol_filter.startswith('@'):
            target_table = "continuous_contracts"
            print(f"--- Querying continuous_contracts table for {base_symbol_filter} ---")
        elif base_symbol_filter: 
             print(f"--- Querying market_data table for {base_symbol_filter} ---")
        else:
             print(f"--- Querying market_data table for all symbols ---")

        # Query to get symbol, interval, counts, and dates (existing logic)
        query = f"""
        SELECT 
            {symbol_column} as symbol, 
            interval_unit,
            interval_value,
            strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
            strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
            COUNT(*) as record_count
        FROM {target_table} 
        GROUP BY {symbol_column}, interval_unit, interval_value
        ORDER BY {symbol_column}, interval_unit, interval_value;
        """
        
    try:
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
            elif symbol.startswith('$VIX'):
                return "$VIX.X"
            elif symbol.startswith('VX') and len(symbol) >= 4 and not symbol.startswith('VXc'): # Avoid matching VXc1 etc.
                return "VX"
            elif len(symbol) >= 4 and symbol[-3:-2].isalpha() and symbol[-2:].isdigit():
                 return symbol[:-3]
            else:
                return symbol # Return original if no pattern matches
        
        metadata_df['base_symbol'] = metadata_df['symbol'].apply(get_base)
        
        # Create the combined interval string (e.g., '1d', '15m') early on
        metadata_df['interval'] = metadata_df.apply(lambda row: f"{row['interval_value']}{row['interval_unit'][0]}", axis=1)
        
        # Filter by base symbol if provided *before* generating summaries
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