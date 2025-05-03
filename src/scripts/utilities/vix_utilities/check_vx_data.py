#!/usr/bin/env python
"""Check VX futures data in market_data table."""

import duckdb
import pandas as pd
from rich.console import Console

def main():
    console = Console()
    conn = duckdb.connect('./data/financial_data.duckdb')
    
    # Check VX futures data count
    console.print("\n[bold]VX Futures Data Count:[/bold]")
    count_df = conn.execute("""
        SELECT COUNT(*) as count 
        FROM market_data 
        WHERE symbol LIKE 'VX%'
        AND interval_value = 1 
        AND interval_unit = 'day'
    """).fetchdf()
    console.print(count_df)
    
    # Check distinct VX symbols
    console.print("\n[bold]Distinct VX Symbols:[/bold]")
    symbols_df = conn.execute("""
        SELECT DISTINCT symbol
        FROM market_data 
        WHERE symbol LIKE 'VX%'
        AND interval_value = 1 
        AND interval_unit = 'day'
        ORDER BY symbol
    """).fetchdf()
    console.print(symbols_df)
    
    # Check date range for VX futures
    console.print("\n[bold]VX Futures Date Range:[/bold]")
    range_df = conn.execute("""
        SELECT 
            MIN(timestamp) as earliest_date,
            MAX(timestamp) as latest_date,
            COUNT(DISTINCT symbol) as num_contracts
        FROM market_data 
        WHERE symbol LIKE 'VX%'
        AND interval_value = 1 
        AND interval_unit = 'day'
    """).fetchdf()
    console.print(range_df)

if __name__ == '__main__':
    main() 