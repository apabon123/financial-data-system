import duckdb
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
import yaml # Added import

# Add project root to sys.path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming utils might have helper functions in the future, though not strictly needed now
# from src.utils.logging_config import setup_logging

# --- Configuration ---
CONFIG_PATH = Path(project_root) / "config" / "market_symbols.yaml"
DEFAULT_DB_PATH = Path(project_root) / "data" / "financial_data.duckdb"

def get_continuous_futures_symbols(db_path: Path) -> list[str]:
    """Reads config for base symbols, queries DB for actual continuous symbols, 
       and selects the primary ones based on heuristics."""
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return []
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return []

    base_symbols = []
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        futures = config.get('futures', [])
        for future in futures:
            if future.get('is_continuous', False):
                base = future.get('base_symbol')
                if base:
                    base_symbols.append(base)
    except Exception as e:
        print(f"Error reading or parsing config file {CONFIG_PATH}: {e}")
        return []

    if not base_symbols:
        print("No base symbols marked as continuous found in config.")
        return []

    print(f"Configured continuous base symbols: {base_symbols}")
    selected_continuous_symbols = []
    conn = None
    try:
        conn = duckdb.connect(database=str(db_path), read_only=True)
        print(f"Connected to {db_path} to find actual continuous symbols.")

        for base in base_symbols:
            print(f"Looking for continuous symbols starting with '@{base}='...")
            query = f"SELECT DISTINCT symbol FROM continuous_contracts WHERE symbol LIKE '@{base}=%' ORDER BY symbol;"
            results_df = conn.execute(query).fetchdf()
            
            if results_df.empty:
                print(f" -> No symbols found matching '@{base}=%'")
                continue

            candidates = results_df['symbol'].tolist()
            print(f" -> Found candidates: {candidates}")
            
            # Heuristic to select the 'primary' symbol
            selected_symbol = None
            if base in ['ES', 'NQ']:
                # Prefer adjusted continuous ending in XC
                preferred = [s for s in candidates if s.endswith('XC')]
                if preferred:
                    selected_symbol = preferred[0] # Pick the first one if multiple
                elif candidates: 
                    selected_symbol = candidates[0] # Fallback to first available
            elif base == 'VX':
                # Return ALL VX continuous contracts instead of just the front month
                if candidates:
                    print(f" -> Including all VX continuous symbols")
                    selected_continuous_symbols.extend(candidates)
                    continue  # Skip the selected_symbol code below since we directly added all candidates
            else:
                # Default fallback for other base symbols
                if candidates:
                    selected_symbol = candidates[0]
            
            if selected_symbol:
                print(f" -> Selected primary symbol: {selected_symbol}")
                selected_continuous_symbols.append(selected_symbol)
            else:
                print(f" -> Could not select a primary symbol for base '{base}' from candidates.")

    except duckdb.Error as e:
        print(f"Database error while finding continuous symbols: {e}")
        return [] # Return empty on error
    except Exception as e:
        print(f"Unexpected error finding continuous symbols: {e}")
        return []
    finally:
        if conn:
            conn.close()
            print(f"Closed DB connection used for finding symbols.")

    print(f"Selected continuous symbols for export: {selected_continuous_symbols}")
    return selected_continuous_symbols

def export_daily_closes(db_path: str, output_csv_path: str):
    """Fetches daily close data for SPY, VIX, and primary continuous futures, then exports to CSV."""
    print(f"Starting export of daily closes to {output_csv_path}...")
    
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        print(f"Error: Database file not found at {db_path}")
        return
        
    # Get the specific continuous symbols to query from the database
    continuous_futures = get_continuous_futures_symbols(db_path_obj)
    # No need to check if empty here, get_continuous_futures_symbols handles errors
    
    symbols_to_fetch = ['SPY', '$VIX.X'] + continuous_futures
    print(f"Final symbols to fetch: {symbols_to_fetch}")

    conn = None
    all_data_frames = []

    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        print(f"Connected to database: {db_path}")

        # --- Check continuous_contracts schema if it exists ---
        try:
            print("\n--- Describing continuous_contracts table schema ---")
            schema_df = conn.execute("DESCRIBE continuous_contracts;").fetchdf()
            print(schema_df.to_string(index=False))
            print("----------------------------------------------------\n")
        except duckdb.CatalogException:
            print("Table 'continuous_contracts' not found.")
        except Exception as e:
            print(f"Error describing continuous_contracts: {e}")
        # --- End schema check ---

        for symbol in symbols_to_fetch:
            print(f"Processing symbol: {symbol}...")
            
            # Determine table and close column based on symbol type
            table = 'market_data'
            symbol_col = 'symbol' # Default symbol column name
            close_col = 'close' # Default to 'close' for non-futures like SPY

            if symbol in continuous_futures:
                table = 'continuous_contracts'
                symbol_col = 'symbol' # Corrected: Assume 'symbol' column in continuous_contracts too
                # Check if 'close' or 'settle' exists in continuous_contracts
                try:
                    # Use the determined symbol_col in the WHERE clause
                    conn.execute(f"SELECT close FROM {table} WHERE {symbol_col} = ? LIMIT 1", [symbol])
                    close_col = 'close'
                except duckdb.CatalogException:
                    try:
                        # Use the determined symbol_col in the WHERE clause
                        conn.execute(f"SELECT settle FROM {table} WHERE {symbol_col} = ? LIMIT 1", [symbol])
                        close_col = 'settle'
                    except duckdb.CatalogException:
                        print(f"Warning: Could not find 'close' or 'settle' column for {symbol} in {table}. Skipping.")
                        continue
                except Exception as e:
                     print(f"Warning: Error checking columns for {symbol} in {table}: {e}. Skipping.")
                     continue
            elif symbol == '$VIX.X':
                # VIX specifically uses 'settle' in market_data
                close_col = 'settle'
                table = 'market_data' # Ensure table is market_data for VIX
                symbol_col = 'symbol'
            # Default case (e.g. SPY) uses table='market_data', symbol_col='symbol', close_col='close'

            print(f" -> Querying table='{table}', symbol_col='{symbol_col}', close_col='{close_col}'")

            # Construct query carefully based on table
            select_list = f"timestamp, \"{close_col.lower()}\""
            if table == 'continuous_contracts':
                 select_list += ", underlying_symbol"
                 
            query = f"""
            SELECT {select_list}
            FROM {table} 
            WHERE {symbol_col} = ? 
              AND interval_unit = 'daily' 
              AND interval_value = 1
            ORDER BY timestamp;
            """
            
            try:
                # Use lowercase column names for consistency after fetch
                conn.execute("SET preserve_insertion_order=false;") # Ensure consistent column order
                df = conn.execute(query, [symbol]).fetchdf()
                df.columns = [col.lower() for col in df.columns] # Lowercase columns

                if not df.empty:
                    print(f" -> Found {len(df)} daily records for {symbol}.")
                    
                    # Check for NaN values specifically in the close/settle column being used
                    close_col_lower = close_col.lower()
                    # REMOVED: NaN diagnostic check block

                    # --- Print raw timestamp samples ---
                    # Keep this for now, it can be useful
                    print(f"   Raw timestamp samples for {symbol}:")
                    print("   First 3 rows:")
                    print(df.head(3).to_string(index=False))
                    print("   Last 3 rows:")
                    print(df.tail(3).to_string(index=False))
                    # --- End print raw timestamp samples ---
                    
                    # Convert timestamp to Date and set as index
                    df['Date'] = pd.to_datetime(df['timestamp']).dt.date
                    
                    # Sanitize symbol name for column header
                    sanitized_symbol = symbol.replace('$VIX.X', 'VIX') # Handle VIX first
                    sanitized_symbol = sanitized_symbol.replace('@', '_').replace('=', '_')
                    print(f" -> Renaming column '{close_col}' to '{sanitized_symbol}'")
                    
                    df = df.rename(columns={close_col: sanitized_symbol})
                    # Select only Date (new) and sanitized_symbol column, set Date as index
                    df = df[['Date', sanitized_symbol]].set_index('Date') 
                    all_data_frames.append(df)
                else:
                    print(f" -> No daily data found for {symbol} in table '{table}' with interval_unit='daily' and interval_value=1.")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        if not all_data_frames:
            print("No data fetched for any symbol. Cannot create CSV.")
            return

        # Merge all dataframes using an outer join on the Date index
        print("Merging data...")
        merged_df = pd.concat(all_data_frames, axis=1, join='outer')
        merged_df = merged_df.sort_index() # Sort by date

        # Save to CSV
        print(f"Saving merged data to {output_csv_path}...")
        merged_df.to_csv(output_csv_path, index=True, date_format='%Y-%m-%d')
        print("Export complete.")

    except duckdb.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


def export_symbol_data(db_path: str, symbol: str, interval_unit: str, interval_value: int, output_csv_path: str):
    """Exports OHLCV data for a specific symbol and interval to CSV."""
    print(f"Starting export of {interval_value} {interval_unit} data for {symbol} to {output_csv_path}...")

    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        print(f"Error: Database file not found at {db_path}")
        return

    conn = None
    try:
        conn = duckdb.connect(database=str(db_path), read_only=True)
        print(f"Connected to database: {db_path}")

        # Determine table and symbol column
        if symbol.startswith('@') and '=' in symbol:
            table = 'continuous_contracts'
            symbol_col = 'symbol' # Assuming 'symbol' in continuous_contracts
        else:
            table = 'market_data'
            symbol_col = 'symbol' # Assuming 'symbol' in market_data
        print(f" -> Determined table: {table}")
        
        # Determine primary close/settle column
        # Try settle first for market_data, close first for continuous
        # Fallback to the other if the preferred one doesn't exist
        preferred_col = 'settle' if table == 'market_data' else 'close'
        fallback_col = 'close' if preferred_col == 'settle' else 'settle'
        final_close_col = None
        try:
            conn.execute(f'SELECT "{preferred_col}" FROM {table} WHERE {symbol_col} = ? LIMIT 1', [symbol])
            final_close_col = preferred_col
            print(f" -> Using primary price column: {final_close_col}")
        except duckdb.Error:
            print(f" -> Primary column '{preferred_col}' not found or error querying. Trying fallback '{fallback_col}'.")
            try:
                conn.execute(f'SELECT "{fallback_col}" FROM {table} WHERE {symbol_col} = ? LIMIT 1', [symbol])
                final_close_col = fallback_col
                print(f" -> Using fallback price column: {final_close_col}")
            except duckdb.Error:
                print(f" -> Fallback column '{fallback_col}' also not found or error querying. Cannot determine price column.")
                return
        
        # Build the query
        # Add more columns as needed (open, high, low, volume, open_interest?)
        query = f"""
            SELECT 
                timestamp,
                open, 
                high, 
                low, 
                "{final_close_col}" AS price, -- Use determined close/settle column
                volume
            FROM {table}
            WHERE {symbol_col} = ? 
              AND interval_unit = ?
              AND interval_value = ?
            ORDER BY timestamp;
        """
        
        print(f" -> Executing query...")
        df = conn.execute(query, [symbol, interval_unit, interval_value]).fetchdf()

        if df.empty:
            print(f"No data found for {symbol} with interval {interval_value} {interval_unit} in table {table}.")
            return

        print(f"Found {len(df)} records. Saving to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False, date_format='%Y-%m-%d %H:%M:%S') # Keep time if present
        print("Export complete.")

    except duckdb.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


def inspect_data(db_path, table_name='market_data', symbol_pattern=None, root_symbol=None, start_date=None, end_date=None, interval_unit=None, interval_value=None, source=None, asset_type=None):
    """
    Connects to the database and retrieves data summary based on filters.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    try:
        con = duckdb.connect(database=db_path, read_only=True)

        # Check if table exists first
        tables_df = con.execute("SHOW TABLES;").fetchdf()
        if table_name not in tables_df['name'].tolist():
             print(f"Error: Table '{table_name}' not found in database {db_path}.")
             con.close()
             return

        # Determine columns based on table
        # Basic check - assumes continuous_contracts has similar basic columns
        # A more robust approach would DESCRIBE the specific table
        select_cols = ["symbol", "interval_value", "interval_unit", "source", "MIN(CAST(timestamp AS DATE)) AS StartDate", "MAX(CAST(timestamp AS DATE)) AS EndDate"]
        if table_name == 'market_data':
            # Calculate MinSettle excluding zeros
            select_cols.extend(["MIN(CASE WHEN settle <> 0 THEN settle ELSE NULL END) AS MinSettleNonZero", "MAX(settle) AS MaxSettle"])
        elif table_name == 'continuous_contracts':
            # Continuous contracts table might use 'close' or 'settle', check schema if needed
            # Assuming 'close' for now based on typical continuous contract generation
            # If 'settle' exists and is preferred, change 'close' to 'settle' below
            try:
                 # Check if settle exists in continuous_contracts
                 con.execute(f"SELECT settle FROM {table_name} LIMIT 1;")
                 # Calculate MinSettle excluding zeros
                 select_cols.extend(["MIN(CASE WHEN settle <> 0 THEN settle ELSE NULL END) AS MinSettleNonZero", "MAX(settle) AS MaxSettle"])
            except Exception:
                 try:
                    # Fallback to close if settle doesn't exist
                    con.execute(f"SELECT close FROM {table_name} LIMIT 1;")
                    select_cols.extend(["MIN(close) AS MinClose", "MAX(close) AS MaxClose"])
                 except Exception:
                     print(f"Warning: Neither 'settle' nor 'close' found in {table_name} for min/max check.")

        # Add conditional extraction for contract year/month for futures-like symbols
        # Assumes format like '...YY' or '...MYY' (e.g., VXM24 -> M=month_code, 24=year)
        # This requires the symbol to have at least 2 characters for the year.
        select_cols.append((
            "CASE "
            "    WHEN (%(asset_type)s = 'future' OR %(asset_type)s = 'all') AND regexp_matches(symbol, '.*[A-Z][0-9]{2}$') THEN CAST('20' || SUBSTRING(symbol, LENGTH(symbol)-1, 2) AS INTEGER) "
            "    ELSE NULL "
            "END AS contract_year"
        ) % {'asset_type': repr(asset_type)})
        select_cols.append((
            "CASE "
            "    WHEN (%(asset_type)s = 'future' OR %(asset_type)s = 'all') AND regexp_matches(symbol, '.*[A-Z][0-9]{2}$') THEN SUBSTRING(symbol, LENGTH(symbol)-2, 1) "
            "    ELSE NULL "
            "END AS contract_month_code"
        ) % {'asset_type': repr(asset_type)})

        select_cols.append("COUNT(*) AS RowCount")
        select_str = ", ".join(select_cols)

        query = f"""
        SELECT
            {select_str}
        FROM {table_name}
        WHERE 1=1
        """

        params_list = [] # Use a list to maintain order

        if symbol_pattern:
            query += " AND symbol LIKE ?"
            params_list.append(symbol_pattern)

        # NOTE: Removed root_symbol logic here for clarity as it uses direct string formatting
        # If combining LIKE and starts_with, parameter handling needs care
        # For now, assume only one of symbol_pattern or root_symbol is used per run
        if root_symbol and not symbol_pattern: # Only apply if symbol_pattern wasn't used
            query += f" AND starts_with(symbol, '{root_symbol}')"
            # No parameter needed for starts_with as it's formatted directly

        if start_date:
            query += " AND CAST(timestamp AS DATE) >= ?"
            params_list.append(start_date)
        if end_date:
            query += " AND CAST(timestamp AS DATE) <= ?"
            params_list.append(end_date)
        if interval_unit:
            query += " AND interval_unit = ?"
            params_list.append(interval_unit)
        if interval_value:
            query += " AND interval_value = ?"
            params_list.append(interval_value)
        if source:
            query += " AND source = ?"
            params_list.append(source)

        # --- Asset Type Filtering (Heuristic based on symbol) ---
        # This is basic and may need significant improvement based on actual symbols used
        if asset_type == 'future':
            # Assuming futures have numbers in them (e.g., VXF24, ESH24)
            # and are typically shorter than, say, CUSIPs or long option symbols.
            # This is a weak heuristic.
            query += " AND regexp_matches(symbol, '.*[0-9].*') AND LENGTH(symbol) < 8" # Example heuristic
        elif asset_type == 'equity':
             # Assuming equities generally DON'T have numbers and are shorter
            query += " AND NOT regexp_matches(symbol, '.*[0-9].*') AND LENGTH(symbol) < 6" # Example heuristic
        # Add more heuristics for options, etc. if needed

        # --- Asset Type Filtering --- (May not apply well to continuous_contracts)
        if asset_type == 'future':
            # Modify heuristic if needed for continuous symbols like VXc1
            query += " AND (regexp_matches(symbol, '.*[0-9].*') OR symbol LIKE '%c_')" # Adjusted heuristic
        elif asset_type == 'equity':
            query += " AND NOT (regexp_matches(symbol, '.*[0-9].*') OR symbol LIKE '%c_')" # Adjusted heuristic

        # Adjust GROUP BY - Remove source if not present in continuous_contracts
        group_by_cols = ["symbol", "interval_value", "interval_unit"]
        if 'source' in [col.split(' ')[0] for col in select_cols]: # Check if source is selected
             group_by_cols.append("source")
        group_by_str = ", ".join(group_by_cols)

        # Define base order by columns
        order_by_base = ["symbol", "interval_value", "interval_unit"]
        if 'source' in [col.split(' ')[0] for col in select_cols]:
            order_by_base.append("source")

        # Add conditional sorting by year and month code for futures
        order_by_list = []
        if asset_type == 'future' or asset_type == 'all':
             # Using CASE statement for month code sorting within SQL
             month_sort_case = ("CASE contract_month_code "
                                "WHEN 'F' THEN 1 WHEN 'G' THEN 2 WHEN 'H' THEN 3 WHEN 'J' THEN 4 "
                                "WHEN 'K' THEN 5 WHEN 'M' THEN 6 WHEN 'N' THEN 7 WHEN 'Q' THEN 8 "
                                "WHEN 'U' THEN 9 WHEN 'V' THEN 10 WHEN 'X' THEN 11 WHEN 'Z' THEN 12 "
                                "ELSE 99 END NULLS LAST")
             order_by_list.extend([
                 "contract_year NULLS LAST", # Sort by year first
                 month_sort_case # Then by month code order
             ])
        order_by_list.extend(order_by_base) # Finally sort by the base columns
        order_by_str = ", ".join(order_by_list)

        query += f" GROUP BY {group_by_str} ORDER BY {order_by_str};"

        print("--- Query ---")
        print(query)
        # print("--- Params List ---") # Debugging
        # print(params_list)
        print("-------------")

        # Pass parameters as a list/tuple
        df = con.execute(query, params_list).fetchdf()
        con.close()

        if df.empty:
            print("No data found matching the criteria.")
        else:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df.to_string(index=False))

    except Exception as e:
        print(f"An error occurred: {e}")
        if 'con' in locals() and con:
            con.close()

def main():
    parser = argparse.ArgumentParser(description="Inspect market data, export daily closes, or export specific symbol data from the DuckDB database.")
    parser.add_argument("-db", "--database", default=str(DEFAULT_DB_PATH), help=f"Path to the database file (default: {DEFAULT_DB_PATH}).")
    
    # --- Operation Mode Arguments (Mutually Exclusive) ---
    mode_group = parser.add_mutually_exclusive_group(required=True) # Require one mode
    mode_group.add_argument("--inspect", action='store_true', help="Run in inspection mode (requires inspection options).")
    mode_group.add_argument("--export-daily-summary", action='store_true', help="Export daily closes summary (SPY, VIX, primary continuous).")
    mode_group.add_argument("--export-symbol", help="Export specific symbol data (requires --export-interval and --export-output).")

    # --- Inspection Options (Only relevant if --inspect is True) ---
    inspect_group = parser.add_argument_group('Inspection Options')
    inspect_group.add_argument("--table", default="market_data", help="Database table to inspect (e.g., market_data, continuous_contracts).")
    inspect_group.add_argument("-s", "--symbol-pattern", dest='symbol_pattern', help="Symbol pattern to match (e.g., 'VX%', 'SPY'). Uses SQL LIKE.")
    inspect_group.add_argument("-r", "--root-symbol", dest='root_symbol', help="Root symbol prefix (e.g., 'VX', 'ES') to find related contracts.")
    inspect_group.add_argument("--start", dest='start_date', help="Start date (YYYY-MM-DD).")
    inspect_group.add_argument("--end", dest='end_date', help="End date (YYYY-MM-DD).")
    inspect_group.add_argument("--unit", dest='interval_unit', help="Interval unit (e.g., 'day', 'minute').")
    inspect_group.add_argument("--value", dest='interval_value', type=int, help="Interval value (e.g., 1, 5).")
    inspect_group.add_argument("--source", help="Data source.")
    inspect_group.add_argument("-t", "--type", dest='asset_type', choices=['equity', 'future', 'all'], default='all', help="Filter by asset type (heuristic based on symbol).")

    # --- Daily Summary Export Options (Only relevant if --export-daily-summary is True) ---
    daily_export_group = parser.add_argument_group('Daily Summary Export Options')
    daily_export_group.add_argument("--daily-output", default="output/reports/daily_closes_export.csv", help="Output file path for the daily summary export.")

    # --- Specific Symbol Export Options (Required if --export-symbol is used) ---
    symbol_export_group = parser.add_argument_group('Specific Symbol Export Options')
    symbol_export_group.add_argument("--export-interval", default="daily,1", help="Interval to export (format: unit,value e.g., 'daily,1', 'minute,15').")
    symbol_export_group.add_argument("--export-output", help="Output file path for specific symbol export (required if --export-symbol is set).")
    
    args = parser.parse_args()

    # --- Execute based on mode ---
    if args.inspect:
        # --- Execute Inspection ---
        print(f"Inspecting table '{args.table}' in database: {args.database}")
        inspect_data(
            db_path=args.database,
            table_name=args.table,
            symbol_pattern=args.symbol_pattern,
            root_symbol=args.root_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval_unit=args.interval_unit,
            interval_value=args.interval_value,
            source=args.source,
            asset_type=args.asset_type
        )
    elif args.export_daily_summary:
        # --- Execute Daily Summary Export ---
        # Ensure output directory exists (the function doesn't create it)
        output_path = Path(args.daily_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_daily_closes(db_path=args.database, output_csv_path=args.daily_output)
        
    elif args.export_symbol:
        # --- Execute Specific Symbol Export ---
        if not args.export_output:
            parser.error("--export-output is required when using --export-symbol.")
            
        try:
            unit, value_str = args.export_interval.split(',')
            value = int(value_str)
        except ValueError:
            parser.error("--export-interval must be in the format 'unit,value' (e.g., 'daily,1').")
        
        # Ensure output directory exists
        output_path = Path(args.export_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_symbol_data(
            db_path=args.database,
            symbol=args.export_symbol,
            interval_unit=unit.lower(),
            interval_value=value,
            output_csv_path=args.export_output
        )

if __name__ == "__main__":
    main() 