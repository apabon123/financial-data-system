#!/usr/bin/env python
"""
Script to inspect VX volume-based roll dates from the futures_roll_dates table.
"""
import duckdb
import pandas as pd
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table

# Determine project root and database path
# This assumes the script is located in src/scripts/analysis/
project_root = Path(__file__).resolve().parent.parent.parent.parent
db_path = project_root / "data" / "financial_data.duckdb"

def main():
    console = Console()
    
    if not db_path.is_file():
        console.print(f"[bold red]Error: Database file not found at {db_path}[/bold red]")
        sys.exit(1)

    conn = None
    try:
        conn = duckdb.connect(database=str(db_path), read_only=True)
        console.print(f"[green]Connected to database: {db_path} (Read-Only)[/green]")

        query = """
            SELECT SymbolRoot, Contract, RollDate, RollType, CalculationTimestamp
            FROM futures_roll_dates
            WHERE SymbolRoot = 'VX'
            ORDER BY RollDate, RollType;
        """
        
        console.print(f"\n[cyan]Executing query:[/cyan]\n{query}")
        
        results_df = conn.execute(query).fetchdf()

        if results_df.empty:
            console.print("\n[yellow]No VX roll dates found in the 'futures_roll_dates' table (for any RollType).[/yellow]")
        else:
            console.print(f"\n[bold green]Found {len(results_df)} VX Roll Dates (All RollTypes):[/bold green]")
            
            table = Table(title="VX Roll Dates - All Types (futures_roll_dates)", show_header=True, header_style="bold magenta", border_style="dim cyan")
            table.add_column("SymbolRoot", style="cyan", no_wrap=True)
            table.add_column("Contract", style="white")
            table.add_column("RollDate", style="yellow")
            table.add_column("RollType", style="blue")
            table.add_column("CalculationTimestamp", style="green")

            for _, row in results_df.iterrows():
                table.add_row(
                    str(row.get('SymbolRoot', 'N/A')),
                    str(row.get('Contract', 'N/A')),
                    str(pd.to_datetime(row.get('RollDate')).date()) if pd.notna(row.get('RollDate')) else 'N/A',
                    str(row.get('RollType', 'N/A')),
                    str(row.get('CalculationTimestamp', 'N/A'))
                )
            console.print(table)

    except duckdb.Error as e:
        console.print(f"\n[bold red]DuckDB Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            console.print("\n[green]Database connection closed.[/green]")

if __name__ == "__main__":
    main() 