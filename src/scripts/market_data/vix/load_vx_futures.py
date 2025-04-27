import pandas as pd
import os
import glob
import duckdb
import argparse # Ensure argparse is imported

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

            # Check for necessary columns before proceeding - Using CORRECTED names including Settle
            required_csv_cols = ['Trade Date', 'Open', 'High', 'Low', 'Close', 'Settle', 'Total Volume', 'Open Interest'] # Added Settle

            # Read the full CSV - skip bad lines and only use required columns
            try:
                # Reinstate on_bad_lines='skip' and add usecols
                df = pd.read_csv(filepath, on_bad_lines='skip', usecols=required_csv_cols)
                print(f"  Initial rows read (using specific columns) from {filename}: {len(df)}") # Updated log message
            except ValueError as ve:
                 # Handle case where a required column itself is missing in the header
                 print(f"  Critical Error reading file {filename}: Required column missing or other ValueError: {ve}. Skipping entire file.")
                 continue
            except Exception as read_e:
                # Handle other potential read errors
                print(f"  Critical Error reading file {filename}: {read_e}. Skipping entire file.")
                continue # Skip to the next file

            # Now df should only contain the required columns, proceed with renaming and processing
            # Check if DataFrame is empty after skipping bad lines
            if df.empty:
                print(f"  Warning: No valid rows could be read from {filename} using required columns. Skipping.")
                continue

            # Select the required columns (already done by usecols, but copy for safety)
            df_renamed = df.copy()
            # Rename to database schema columns
            df_renamed.columns = ['timestamp', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest'] # Added settle

            df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
            df_renamed['symbol'] = symbol
            df_renamed['source'] = 'CBOE'
            df_renamed['interval_value'] = 1
            df_renamed['interval_unit'] = 'day'
            df_renamed['adjusted'] = False

            # Ensure correct data types for numeric columns - Including Settle
            numeric_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest'] # Added settle
            for col in numeric_cols:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')

            # Log rows with NaN in 'close' after coercion
            nan_close_count = df_renamed['close'].isna().sum()
            if nan_close_count > 0:
                print(f"  Warning: Found {nan_close_count} rows with NaN in 'close' column after numeric conversion in {filename}.")

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
    print(f"Total rows combined from all files before dropna: {len(combined_df)}") # Log total rows before dropna

    # Log rows with NaN in 'settle' in the combined DataFrame before dropna
    combined_nan_settle_count = combined_df['settle'].isna().sum()
    if combined_nan_settle_count > 0:
        print(f"  Warning: Found {combined_nan_settle_count} rows with NaN in 'settle' column in combined data before dropna.")

    # Drop rows where essential numeric data might be NaN after coercion, e.g., settle price
    original_row_count = len(combined_df)
    combined_df.dropna(subset=['settle'], inplace=True)
    dropped_rows_count = original_row_count - len(combined_df)
    if dropped_rows_count > 0:
        print(f"  Dropped {dropped_rows_count} rows due to NaN in 'settle' column.")

    print(f"Total rows remaining after dropna: {len(combined_df)}") # Renamed log message for clarity

    try:
        print(f"Connecting to database: {db_path}")
        con = duckdb.connect(database=db_path, read_only=False)

        # --- Add Deletion Logic --- 
        processed_symbols = combined_df['symbol'].unique().tolist()
        if processed_symbols:
            print(f"Deleting existing data for {len(processed_symbols)} symbols...")
            # Use parameter substitution to avoid SQL injection vulnerabilities
            symbols_tuple = tuple(processed_symbols)
            placeholders = ', '.join('?' * len(symbols_tuple))
            delete_query = f"DELETE FROM market_data WHERE symbol IN ({placeholders})"
            try:
                con.execute(delete_query, symbols_tuple)
                print(f"  Deleted records for symbols: {', '.join(processed_symbols[:10])}{'...' if len(processed_symbols) > 10 else ''}")
            except Exception as delete_e:
                print(f"Error deleting existing data: {delete_e}")
                # Optionally, decide if you want to stop the script here or continue with insertion
                # For now, we'll print the error and continue
        # --- End Deletion Logic ---

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
    print("--- Script starting execution ---") # Add print at the very start
    parser = argparse.ArgumentParser(description="Load CBOE VX futures data from CSV files into DuckDB (expects VX<MonthCode><YY>.csv filenames).")
    parser.add_argument("source_dir", help="Directory containing the source CSV files.")
    parser.add_argument("-db", "--database", default="data/financial_data.duckdb", help="Path to the database file.")

    args = parser.parse_args()

    print(f"--- Arguments received ---") # Log received arguments
    print(f"Source Directory: {args.source_dir}")
    print(f"Database Path: {args.database}")
    print("-------------------------")

    load_vx_futures(args.source_dir, args.database)

if __name__ == "__main__":
    main() 