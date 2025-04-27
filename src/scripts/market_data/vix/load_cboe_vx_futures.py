import duckdb
import pandas as pd
import argparse
import os
import glob

def load_vx_futures(source_dir, db_path):
    """Loads VX futures data from CSV files in source_dir into the database, deriving symbol from filename."""
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    all_files = glob.glob(os.path.join(source_dir, 'VX*.csv')) # Ensure we only grab VX files
    if not all_files:
        print(f"No CSV files starting with 'VX' found in {source_dir}")
        return

    print(f"Found {len(all_files)} CSV files to process.")

    all_data_to_load = []
    final_cols = [] # Define outside loop

    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"Processing file: {filename}...")
        try:
            # Derive symbol directly from filename (e.g., "VXF13.csv" -> "VXF13")
            symbol = filename[:-4] # Remove the last 4 characters (.csv)
            print(f"  Using symbol: {symbol}")

            # Read the full CSV - skip bad lines
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
            except Exception as read_e:
                # Handle cases where even skipping bad lines fails (e.g., file totally unreadable)
                print(f"  Critical Error reading file {filename}: {read_e}. Skipping entire file.")
                continue # Skip to the next file

            # Check for necessary columns before proceeding - Using CORRECTED names including Settle
            required_csv_cols = ['Trade Date', 'Open', 'High', 'Low', 'Close', 'Settle', 'Total Volume', 'Open Interest'] # Added Settle
            if not all(col in df.columns for col in required_csv_cols):
                print(f"  Warning: Missing one or more required columns ({required_csv_cols}) in {filename}. Available: {list(df.columns)}. Skipping.")
                continue

            # Select the required columns - Using CORRECTED names including Settle
            df_renamed = df[required_csv_cols].copy()
            # Rename to database schema columns
            df_renamed.columns = ['timestamp', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest'] # Added settle

            df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
            df_renamed['symbol'] = symbol
            df_renamed['source'] = 'CBOE'
            df_renamed['interval_value'] = 1
            df_renamed['interval_unit'] = 'daily'
            df_renamed['adjusted'] = False

            # Ensure correct data types for numeric columns - Including Settle
            numeric_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest'] # Added settle
            for col in numeric_cols:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')

            # Define final_cols based on the first processed DataFrame
            if not final_cols:
                 final_cols = [
                    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume',
                    'open_interest', 'source', 'interval_value', 'interval_unit', 'adjusted'
                 ] # Added settle

            # Handle potential NaN values introduced by coerce, fill with 0 or appropriate value if needed
            # Example: df_renamed[numeric_cols] = df_renamed[numeric_cols].fillna(0)

            df_final = df_renamed[final_cols]
            all_data_to_load.append(df_final)

        except Exception as e:
            print(f"  Error processing file {filename}: {e}")

    if not all_data_to_load:
        print("No data successfully processed. Exiting.")
        return

    combined_df = pd.concat(all_data_to_load, ignore_index=True)
    # Drop rows where essential numeric data might be NaN after coercion, e.g., close price
    combined_df.dropna(subset=['close'], inplace=True)
    print(f"Successfully processed {len(combined_df)} valid rows from {len(all_data_to_load)} files.")

    try:
        print(f"Connecting to database: {db_path}")
        con = duckdb.connect(database=db_path, read_only=False)
        print("Registering DataFrame and inserting data into market_data table...")

        # Define the columns we are inserting into the database table - including settle
        db_columns = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume',
            'open_interest', 'source', 'interval_value', 'interval_unit', 'adjusted'
        ] # Added settle
        db_columns_str = ', '.join(f'"{col}"' for col in db_columns) # Quote column names

        # Register DataFrame as a temporary view
        # Make sure combined_df has the columns defined in db_columns
        con.register('combined_df_view', combined_df[db_columns]) # Use the updated db_columns list

        # Execute INSERT INTO specifying columns
        con.execute(f"INSERT INTO market_data ({db_columns_str}) SELECT {db_columns_str} FROM combined_df_view")

        con.unregister('combined_df_view') # Clean up the view
        con.close()
        print("Data insertion complete.")

    except Exception as e:
        print(f"Database error: {e}")
        if 'con' in locals() and con:
            con.close()

def main():
    parser = argparse.ArgumentParser(description="Load CBOE VX futures data from CSV files into DuckDB (expects VX<MonthCode><YY>.csv filenames).")
    parser.add_argument("source_dir", help="Directory containing the source CSV files.")
    parser.add_argument("-db", "--database", default="data/financial_data.duckdb", help="Path to the database file.")

    args = parser.parse_args()
    load_vx_futures(args.source_dir, args.database)

if __name__ == "__main__":
    main() 