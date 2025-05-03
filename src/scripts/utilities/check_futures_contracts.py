#!/usr/bin/env python
"""
Check Futures Contracts

This script checks the status of futures contracts in the database,
focusing on the date ranges and row counts.
"""

import duckdb
import pandas as pd
import sys

# Database path
DB_PATH = "./data/financial_data.duckdb"

def check_specific_contracts(symbols=None):
    """Check specific futures contracts."""
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH, read_only=True)
        
        if symbols:
            symbols_str = "', '".join(symbols)
            where_clause = f"WHERE symbol IN ('{symbols_str}')"
        else:
            where_clause = "WHERE regexp_matches(symbol, '^(ES|NQ)[HMUZ][0-9]{2}$')"
        
        # Query to get contract details
        query = f"""
            SELECT 
                symbol,
                COUNT(*) as row_count,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                ROUND(AVG(high-low), 2) as avg_range,
                ROUND(AVG(volume), 0) as avg_volume
            FROM market_data
            {where_clause}
                AND interval_value = 1 
                AND interval_unit = 'daily'
            GROUP BY symbol
            ORDER BY 
                CASE 
                    WHEN LEFT(symbol, 2) = 'ES' THEN 1
                    WHEN LEFT(symbol, 2) = 'NQ' THEN 2
                    ELSE 3
                END,
                last_date DESC
        """
        
        results_df = conn.execute(query).fetchdf()
        
        # Display results
        if results_df.empty:
            print("No matching contracts found in database")
            return
        
        # Parse contract symbols to add full names
        results_df['full_name'] = results_df['symbol'].apply(parse_futures_symbol)
        
        # Reorder columns for display
        display_df = results_df[['symbol', 'full_name', 'row_count', 'first_date', 'last_date', 'avg_range', 'avg_volume']]
        
        # Print results
        print("\nContract Details:\n")
        pd.set_option('display.width', 140)
        pd.set_option('display.max_rows', None)
        print(display_df.to_string(index=False))
        
        # Print summary
        print(f"\nTotal contracts: {len(results_df)}")
        print(f"Total rows: {results_df['row_count'].sum()}")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.close()

def parse_futures_symbol(symbol):
    """Parse a futures symbol to get the full name."""
    try:
        # Match pattern: 2-3 letters + month code + 2 digit year
        import re
        match = re.match(r'^([A-Z]{2,3})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
        
        if not match:
            return symbol
        
        base, month_code, year_code = match.groups()
        
        # Month code to name mapping
        month_map = {
            'F': 'January',
            'G': 'February', 
            'H': 'March',
            'J': 'April',
            'K': 'May',
            'M': 'June',
            'N': 'July',
            'Q': 'August',
            'U': 'September',
            'V': 'October',
            'X': 'November',
            'Z': 'December'
        }
        
        # Convert 2-digit year to 4-digit year
        year = int(year_code)
        year = 2000 + year if year < 50 else 1900 + year
        
        return f"{base} {month_map[month_code]} {year}"
    except:
        return symbol

if __name__ == "__main__":
    # Get contract symbols from command line args if provided
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    check_specific_contracts(symbols) 