import duckdb
import argparse
import pandas as pd
import os
import sys

# Add project root to sys.path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming utils might have helper functions in the future, though not strictly needed now
# from src.utils.logging_config import setup_logging

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
    parser = argparse.ArgumentParser(description="Inspect market data in the DuckDB database.")
    parser.add_argument("-db", "--database", default="data/financial_data.duckdb", help="Path to the database file.")
    parser.add_argument("--table", default="market_data", help="Database table to inspect (e.g., market_data, continuous_contracts).")
    parser.add_argument("-s", "--symbol", dest='symbol_pattern', help="Symbol pattern to match (e.g., 'VX%', 'SPY'). Uses SQL LIKE.")
    parser.add_argument("-r", "--root", dest='root_symbol', help="Root symbol prefix (e.g., 'VX', 'ES') to find related contracts.")
    parser.add_argument("--start", dest='start_date', help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest='end_date', help="End date (YYYY-MM-DD).")
    parser.add_argument("--unit", dest='interval_unit', help="Interval unit (e.g., 'day', 'minute').")
    parser.add_argument("--value", dest='interval_value', type=int, help="Interval value (e.g., 1, 5).")
    parser.add_argument("--source", help="Data source.")
    parser.add_argument("-t", "--type", dest='asset_type', choices=['equity', 'future', 'all'], default='all', help="Filter by asset type (heuristic based on symbol).")

    args = parser.parse_args()

    # setup_logging() # Add if logging is needed

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

if __name__ == "__main__":
    main() 