#!/usr/bin/env python
"""Find VX futures data in all tables."""

import duckdb
import pandas as pd
from rich.console import Console
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    console = Console()
    conn = duckdb.connect('./data/financial_data.duckdb')
    
    # Get all tables
    tables_df = conn.execute("SHOW TABLES").fetchdf()
    tables = tables_df['name'].tolist()
    
    console.print("\n[bold]Searching for VX futures data in all tables:[/bold]")
    
    for table in tables:
        try:
            # Check if table has VX futures data
            count_df = conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM {table} 
                WHERE symbol LIKE 'VX%'
            """).fetchdf()
            
            if count_df['count'].iloc[0] > 0:
                console.print(f"\n[green]Found VX futures data in {table}:[/green]")
                console.print(count_df)
                
                # Show sample of the data
                sample_df = conn.execute(f"""
                    SELECT DISTINCT symbol, COUNT(*) as rows
                    FROM {table}
                    WHERE symbol LIKE 'VX%'
                    GROUP BY symbol
                    ORDER BY symbol
                    LIMIT 5
                """).fetchdf()
                console.print("\nSample symbols:")
                console.print(sample_df)
                
                # Show table structure
                console.print("\nTable structure:")
                structure_df = conn.execute(f"DESCRIBE {table}").fetchdf()
                console.print(structure_df)
                
        except Exception as e:
            logger.error(f"Error checking table {table}: {e}")
            continue

if __name__ == '__main__':
    main() 