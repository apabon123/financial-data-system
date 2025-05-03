import argparse
import duckdb
import sys
from datetime import datetime

def connect_db(db_path, read_only=True):
    """Connects to the DuckDB database."""
    try:
        conn = duckdb.connect(database=db_path, read_only=read_only)
        return conn
    except duckdb.Error as e:
        print(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def lookup_contract(conn, continuous_symbol, date_str):
    """Looks up the underlying contract for a continuous symbol on a specific date."""
    try:
        # Convert date string to date object
        lookup_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Query the mapping table
        query = """
            SELECT date, continuous_symbol, underlying_symbol
            FROM continuous_contract_mapping
            WHERE continuous_symbol = ?
            AND date = ?
        """
        result = conn.execute(query, [continuous_symbol, lookup_date]).fetchdf()
        
        if result.empty:
            # If no exact match, find the nearest date before the lookup date
            query = """
                SELECT date, continuous_symbol, underlying_symbol
                FROM continuous_contract_mapping
                WHERE continuous_symbol = ?
                AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            result = conn.execute(query, [continuous_symbol, lookup_date]).fetchdf()
            
            if result.empty:
                print(f"No mapping found for {continuous_symbol} on or before {date_str}")
                return None
            else:
                print(f"Using nearest previous date mapping ({result['date'].iloc[0]})")
        
        return result
    except ValueError:
        print(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
        return None
    except Exception as e:
        print(f"Error looking up contract: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Look up underlying contract for continuous symbol.')
    parser.add_argument('--db-path', default='data/financial_data.duckdb', help='Path to DuckDB database')
    parser.add_argument('--symbol', required=True, help='Continuous contract symbol (e.g., @VX=101XN)')
    parser.add_argument('--date', required=True, help='Date to look up (YYYY-MM-DD)')
    parser.add_argument('--show-all', action='store_true', help='Show all continuous contracts for this date')
    
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_db(args.db_path)
    
    try:
        if args.show_all:
            # Show all continuous contracts for the specified date
            try:
                lookup_date = datetime.strptime(args.date, '%Y-%m-%d').date()
                query = """
                    SELECT date, continuous_symbol, underlying_symbol
                    FROM continuous_contract_mapping
                    WHERE date = ?
                    ORDER BY continuous_symbol
                """
                result = conn.execute(query, [lookup_date]).fetchdf()
                
                if result.empty:
                    print(f"No mappings found for date {args.date}")
                else:
                    print(f"\nActive contracts on {args.date}:")
                    print("=" * 50)
                    for _, row in result.iterrows():
                        print(f"{row['continuous_symbol']:12} -> {row['underlying_symbol']}")
                    print("=" * 50)
            except ValueError:
                print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
        else:
            # Look up specific continuous contract on the date
            result = lookup_contract(conn, args.symbol, args.date)
            
            if result is not None and not result.empty:
                row = result.iloc[0]
                print(f"\nContract Lookup Result:")
                print("=" * 50)
                print(f"Date:              {row['date']}")
                print(f"Continuous Symbol: {row['continuous_symbol']}")
                print(f"Underlying Symbol: {row['underlying_symbol']}")
                print("=" * 50)
    finally:
        # Close connection
        conn.close()

if __name__ == "__main__":
    main() 