import duckdb
import pandas as pd
import argparse
import os
import glob

def load_vix_index(source_file, db_path):
    """Loads VIX index data from a CSV file into the database."""
    if not os.path.isfile(source_file):
        print(f"Error: Source file not found: {source_file}")
        return

    print(f"Processing file: {os.path.basename(source_file)}...")
    canonical_symbol = '$VIX.X' # Canonical symbol from config
    source_name = 'CBOE'

    try:
        # Read the CSV, parsing dates correctly
        df = pd.read_csv(source_file, parse_dates=['DATE'], dayfirst=False) # Assumes M/D/YYYY
        print(f"  Initial rows read: {len(df)}")

        # Rename columns to match database schema (lowercase)
        df.rename(columns={
            'DATE': 'timestamp',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close'
        }, inplace=True)

        # --- Data Preparation ---
        # Use 'close' for 'settle' as index has no separate settlement
        df['settle'] = df['close']

        # Add missing columns with appropriate values
        df['symbol'] = canonical_symbol
        df['source'] = source_name
        df['interval_value'] = 1
        df['interval_unit'] = 'day'
        df['adjusted'] = False
        df['changed'] = False
        df['volume'] = 0  # Volume not applicable for index
        df['open_interest'] = 0 # Open Interest not applicable for index

        # Ensure correct data types for numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential data might be NaN after coercion (e.g., close/settle)
        initial_rows = len(df)
        df.dropna(subset=['settle'], inplace=True) # Use settle (derived from close) for check
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"  Warning: Dropped {dropped_rows} rows due to NaN in essential columns (close/settle).")

        if df.empty:
            print("No valid data remaining after cleaning. Exiting.")
            return

        # Define final columns order matching the likely table structure
        final_cols = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume',
            'open_interest', 'source', 'interval_value', 'interval_unit', 'adjusted', 'changed'
        ]
        df_final = df[final_cols]
        print(f"Processed {len(df_final)} valid rows.")

        # --- Database Interaction ---
        print(f"Connecting to database: {db_path}")
        con = duckdb.connect(database=db_path, read_only=False)

        # Delete existing data for this symbol
        print(f"Deleting existing data for symbol: {canonical_symbol}...")
        try:
            con.execute("DELETE FROM market_data WHERE symbol = ?", (canonical_symbol,))
            print(f"  Deleted existing records for {canonical_symbol}.")
        except Exception as delete_e:
            print(f"Error deleting existing data for {canonical_symbol}: {delete_e}")
            con.close()
            return # Stop if deletion fails

        # Insert new data
        print("Registering DataFrame and inserting data into market_data table...")
        db_columns_str = ', '.join(f'"{col}"' for col in final_cols)

        con.register('vix_df_view', df_final)
        con.execute(f"INSERT INTO market_data ({db_columns_str}) SELECT {db_columns_str} FROM vix_df_view")

        con.unregister('vix_df_view')
        con.close()
        print("Data insertion complete.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        if 'con' in locals() and con and not con.is_closed():
            con.close()

def main():
    parser = argparse.ArgumentParser(description="Load CBOE VIX index data from a CSV file into DuckDB.")
    parser.add_argument("source_file", help="Path to the source CSV file (e.g., VIX_History.csv).")
    parser.add_argument("-db", "--database", default="data/financial_data.duckdb", help="Path to the database file.")

    args = parser.parse_args()
    load_vix_index(args.source_file, args.database)

if __name__ == "__main__":
    main() 