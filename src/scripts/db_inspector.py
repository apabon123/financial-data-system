import duckdb
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
import re
from rich.console import Console
from rich.table import Table

# Determine project root based on script location
# This assumes the script is located in src/scripts/
project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "data" / "financial_data.duckdb"
sql_dir = project_root / "sql"

def format_value(value) -> str:
    """Helper function to format table cell values nicely."""
    if pd.isna(value):
        return "[dim]N/A[/dim]" # Style nulls
    if isinstance(value, (int, float)):
        if abs(value) > 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if abs(value) > 1_000_000:
             return f"{value / 1_000_000:.2f}M"
        if abs(value) > 1_000:
             return f"{value / 1_000:.1f}K"
        if isinstance(value, float):
            return f"{value:.4f}" # Format floats
    return str(value)

def display_dataframe_rich(df: pd.DataFrame, title: str):
    """Displays a pandas DataFrame using rich.table.Table."""
    if df.empty:
        print(f"Query for '{title}' returned no results.")
        return

    console = Console()
    table = Table(title=f"[bold magenta]{title}[/bold magenta]", show_header=True, header_style="bold blue", border_style="dim blue")

    # Add columns to the table
    for column in df.columns:
        # Basic type check for justification (could be more sophisticated)
        justify = "right" if pd.api.types.is_numeric_dtype(df[column]) else "left"
        style = "cyan" if justify == "left" else "green"
        table.add_column(str(column), style=style, justify=justify)

    # Add rows to the table
    for index, row in df.iterrows():
        table.add_row(*[format_value(value) for value in row])

    console.print(table)

def execute_sql_from_file(sql_file_path: Path, db_connection: duckdb.DuckDBPyConnection):
    """Executes SQL statements from a file and prints results for each SELECT using Rich."""
    if not sql_file_path.is_file():
        print(f"Error: SQL file not found at {sql_file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Executing {sql_file_path.name} ---")
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            full_sql_script = f.read()

        # Basic split by semicolon. Assumes semicolons don't appear in strings/comments.
        # More robust parsing could be added if needed.
        sql_statements = [stmt.strip() for stmt in full_sql_script.split(';') if stmt.strip()] # Remove empty statements

        print(f"Found {len(sql_statements)} SQL statement(s) in the file.")

        statement_counter = 0
        for sql_query in sql_statements:
            statement_counter += 1
            print(f"\n--- Statement {statement_counter}/{len(sql_statements)} ---")
            # Display the query being executed (optional, can be long)
            # query_preview = (sql_query[:100] + '...') if len(sql_query) > 100 else sql_query
            # print(f"Executing: {query_preview.replace('\n', ' ')}")

            try:
                result_relation = db_connection.execute(sql_query)

                if result_relation:
                    # Try to fetch results if the execution returned a relation
                    result_df = result_relation.fetchdf()
                    if not result_df.empty:
                        display_dataframe_rich(result_df, f"Results for Statement {statement_counter} ({sql_file_path.name})")
                    else:
                        print(f"Statement {statement_counter} executed successfully (SELECT returned no results).")
                else:
                    # If execute returned None (likely a non-SELECT statement)
                    print(f"Statement {statement_counter} executed successfully (non-SELECT).")

            except duckdb.Error as stmt_err:
                print(f"DuckDB Error executing statement {statement_counter}: {stmt_err}", file=sys.stderr)
                print(f"Problematic SQL: \n{sql_query}")
                # Optionally stop execution on first error, or continue
                # break
            except Exception as stmt_gen_err:
                 print(f"Unexpected Error executing statement {statement_counter}: {stmt_gen_err}", file=sys.stderr)
                 print(f"Problematic SQL: \n{sql_query}")
                 # break

    except Exception as e:
        print(f"Error processing file {sql_file_path.name}: {e}", file=sys.stderr)
    finally:
        print(f"\n--- Finished execution of {sql_file_path.name} ---")

def main():
    parser = argparse.ArgumentParser(description="Execute a SQL query file against the financial data DuckDB.")
    parser.add_argument("sql_file_name", help="Name of the SQL file located in the 'sql/' directory (e.g., 'data_gaps.sql').")
    parser.add_argument("--db", default=str(db_path), help=f"Path to the DuckDB database file (default: {db_path}).")

    args = parser.parse_args()

    sql_file_to_run = sql_dir / args.sql_file_name
    db_file = Path(args.db)

    if not db_file.is_file():
        print(f"Error: Database file not found at {db_file}", file=sys.stderr)
        sys.exit(1)

    conn = None
    # Determine if connection should be read-only based on SQL file name or content?
    # For now, default to read-only for safety in an inspection tool.
    # Maintenance tasks might need write access.
    read_only_mode = True
    if args.sql_file_name == 'database_maintenance.sql': # Example condition
         print("Opening database in read-write mode for maintenance.")
         read_only_mode = False

    try:
        conn = duckdb.connect(database=str(db_file), read_only=read_only_mode)
        print(f"Connected to database: {db_file} (Read-Only: {read_only_mode})")
        execute_sql_from_file(sql_file_to_run, conn)
    except duckdb.Error as e:
        print(f"Failed to connect to database {db_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 