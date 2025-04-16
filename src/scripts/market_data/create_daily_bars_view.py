#!/usr/bin/env python3
"""
Script to create a view for daily bars from the market_data table.
"""

import duckdb

def create_daily_bars_view(db_path: str = "./data/financial_data.duckdb"):
    """Create a view for daily bars from the market_data table."""
    try:
        conn = duckdb.connect(db_path)
        
        # Drop the table if it exists
        conn.execute("DROP TABLE IF EXISTS daily_bars")
        
        # Create the view
        conn.execute("""
        CREATE VIEW daily_bars AS
        SELECT 
            timestamp as date,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            source,
            interval_unit,
            interval_value
        FROM market_data
        WHERE interval_unit = 'day' 
        AND interval_value = 1
        """)
        
        print("Successfully created daily_bars view")
        
        # Verify the view was created
        result = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
        print(f"Number of rows in daily_bars view: {result[0]}")
        
    except Exception as e:
        print(f"Error creating daily_bars view: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_daily_bars_view() 