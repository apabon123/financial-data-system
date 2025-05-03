#!/usr/bin/env python
"""
View Symbol Metadata Table

This script displays the contents of the symbol_metadata table.
"""

import os
import sys
import duckdb
import pandas as pd
from pathlib import Path
import argparse
import logging
from rich.console import Console
from rich.table import Table

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")
METADATA_TABLE_NAME = "symbol_metadata"

def view_metadata(conn: duckdb.DuckDBPyConnection):
    """Fetch and display the symbol_metadata table."""
    console = Console()
    try:
        # Check for column existence
        cols_exist = conn.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{METADATA_TABLE_NAME}'").fetchall()
        col_names = [col[0] for col in cols_exist]
        
        # Build SELECT list dynamically, including desired original columns
        select_cols = ['base_symbol', 'asset_type']
        # Add optional/original columns if they exist
        if 'interval_unit' in col_names: select_cols.append('interval_unit')
        if 'interval_value' in col_names: select_cols.append('interval_value')
        if 'data_table' in col_names: select_cols.append('data_table')
        if 'data_source' in col_names: select_cols.append('data_source')
        # Add new script paths if they exist
        if 'historical_script_path' in col_names: select_cols.append('historical_script_path')
        if 'update_script_path' in col_names: select_cols.append('update_script_path')
        # Add other potentially useful columns
        if 'config_path' in col_names: select_cols.append('config_path')
        if 'last_updated' in col_names: select_cols.append('last_updated')
        # Include old update_script only if new ones aren't present
        if 'update_script' in col_names and 'historical_script_path' not in col_names:
             select_cols.append('update_script')

        # Build ORDER BY clause dynamically
        order_by_cols = ['base_symbol']
        if 'interval_unit' in select_cols: order_by_cols.append('interval_unit')
        if 'interval_value' in select_cols: order_by_cols.append('interval_value')
        order_by_clause = ', '.join(order_by_cols)

        query = f"""
            SELECT {', '.join(select_cols)} 
            FROM {METADATA_TABLE_NAME} 
            ORDER BY {order_by_clause};
        """
        metadata_df = conn.execute(query).fetchdf()

        if metadata_df.empty:
            print(f"Table '{METADATA_TABLE_NAME}' is empty or does not exist.")
            return

        table = Table(
            title=f"Contents of '{METADATA_TABLE_NAME}'",
            show_header=True,
            header_style="bold cyan",
            border_style="dim blue"
        )

        for col in metadata_df.columns:
            style = "white"
            justify = "left"
            if "symbol" in col:
                style = "bold magenta"
            elif "script_path" in col or ("script" in col and "path" not in col):
                 style = "green"
                 justify = "left"
            elif "updated" in col or "value" in col:
                 style = "yellow"
                 justify = "right"
            elif "type" in col:
                 style = "blue"
            elif "interval" in col:
                 style = "cyan"
                 justify = "right"
            elif "table" in col or "source" in col:
                 style = "blue"
            
            col_title = col.replace('_',' ').title()
            if col == 'historical_script_path':
                 col_title = "Historical Fetch Script"
            elif col == 'update_script_path':
                 col_title = "Raw Update Script"
            elif col == 'update_script':
                 col_title = "Update Script (Old)"

            table.add_column(col_title, style=style, justify=justify, overflow="fold")

        for _, row in metadata_df.iterrows():
             row_values = [str(item) if item is not None else "[dim]N/A[/dim]" for item in row]
             table.add_row(*row_values)

        console.print(table)

    except duckdb.CatalogException:
         print(f"Error: Table '{METADATA_TABLE_NAME}' not found in the database.")
    except duckdb.Error as e:
        logging.error(f"Database error querying {METADATA_TABLE_NAME}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='View the symbol_metadata table.')
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the DuckDB database file.")
    args = parser.parse_args()

    db_file = Path(args.db_path).resolve()
    if not db_file.exists():
        logging.error(f"Error: Database file not found at {db_file}")
        sys.exit(1)

    conn = None
    try:
        conn = duckdb.connect(database=str(db_file), read_only=True)
        logging.info(f"Connected to database: {db_file} (Read-Only)")
        view_metadata(conn)
    except duckdb.Error as e:
        logging.error(f"Failed to connect to database {db_file}: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main() 