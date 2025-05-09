#!/usr/bin/env python
"""
Script to inspect contents of the futures_roll_dates table.
"""
import duckdb
import pandas as pd
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table

# Determine project root and database path
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
            SELECT SymbolRoot, RollType, COUNT(*) as Count, MIN(RollDate) as MinRollDate, MAX(RollDate) as MaxRollDate
            FROM futures_roll_dates
            GROUP BY SymbolRoot, RollType
            ORDER BY SymbolRoot, RollType;
        """
        
        console.print(f"\n[cyan]Executing summary query:[/cyan]\n{query}")
        
        results_df = conn.execute(query).fetchdf()

        if results_df.empty:
            console.print("\n[bold red]The 'futures_roll_dates' table is completely empty.[/bold red]")
        else:
            console.print(f"\n[bold green]Summary of 'futures_roll_dates' table by SymbolRoot and RollType:[/bold green]")
            
            table = Table(title="futures_roll_dates Summary", show_header=True, header_style="bold magenta", border_style="dim cyan")
            table.add_column("SymbolRoot", style="cyan", no_wrap=True)
            table.add_column("RollType", style="blue")
            table.add_column("Count", style="green", justify="right")
            table.add_column("MinRollDate", style="yellow")
            table.add_column("MaxRollDate", style="yellow")

            for _, row in results_df.iterrows():
                table.add_row(
                    str(row.get('SymbolRoot', 'N/A')),
                    str(row.get('RollType', 'N/A')),
                    str(row.get('Count', 'N/A')),
                    str(pd.to_datetime(row.get('MinRollDate')).date()) if pd.notna(row.get('MinRollDate')) else 'N/A',
                    str(pd.to_datetime(row.get('MaxRollDate')).date()) if pd.notna(row.get('MaxRollDate')) else 'N/A'
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