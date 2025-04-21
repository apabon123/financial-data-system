import pandas as pd
import argparse
import os
import glob
import re
import sys

# Mapping from Month Abbreviation to Futures Month Code
MONTH_TO_CODE = {
    'Jan': 'F', 'Feb': 'G', 'Mar': 'H', 'Apr': 'J',
    'May': 'K', 'Jun': 'M', 'Jul': 'N', 'Aug': 'Q',
    'Sep': 'U', 'Oct': 'V', 'Nov': 'X', 'Dec': 'Z'
}

def parse_futures_column(futures_str):
    """Parses the 'Futures' column string like 'F (Jan 13)' or 'F (Jan 2013)' to get month code and 2-digit year."""
    # Updated regex to capture 2 or 4 digit years
    match = re.search(r'\((\w{3})\s+(\d{2}|\d{4})\)', futures_str)
    if match:
        month_abbr = match.group(1)
        year_match = match.group(2) # This could be '13' or '2013'

        # Ensure we get a 2-digit year
        if len(year_match) == 4:
            year_short = year_match[-2:] # Take last two digits
        elif len(year_match) == 2:
            year_short = year_match
        else:
            return None, None # Should not happen with the regex, but safe check

        if month_abbr in MONTH_TO_CODE:
            month_code = MONTH_TO_CODE[month_abbr]
            return month_code, year_short
    return None, None

def rename_files(source_dir):
    """Renames CSV files in source_dir based on the Futures column content."""
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    all_files = glob.glob(os.path.join(source_dir, '*.csv'))
    if not all_files:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Found {len(all_files)} CSV files to process in {source_dir}")
    renamed_count = 0
    skipped_count = 0

    for current_filepath in all_files:
        current_filename = os.path.basename(current_filepath)
        print(f"Processing file: {current_filename}...")

        try:
            # Read only a few rows to get the Futures column value
            df_sample = pd.read_csv(current_filepath, nrows=5)
            if 'Futures' not in df_sample.columns:
                print(f"  Warning: 'Futures' column not found. Skipping.")
                skipped_count += 1
                continue

            # Assume the Futures value is consistent, take the first one
            futures_val = df_sample['Futures'].iloc[0]
            month_code, year_short = parse_futures_column(futures_val)

            if not month_code or not year_short:
                print(f"  Warning: Could not parse month/year from 'Futures' column ('{futures_val}'). Skipping.")
                skipped_count += 1
                continue

            # Construct the new filename
            new_filename = f"VX{month_code}{year_short}.csv"
            new_filepath = os.path.join(source_dir, new_filename)

            # Check if renaming is needed and if target exists
            if new_filename == current_filename:
                print(f"  Filename {current_filename} is already correct. Skipping.")
                skipped_count += 1
                continue

            if os.path.exists(new_filepath):
                print(f"  Warning: Target file {new_filename} already exists. Skipping rename of {current_filename}.")
                skipped_count += 1
                continue

            # Rename the file
            print(f"  Renaming {current_filename} to {new_filename}")
            os.rename(current_filepath, new_filepath)
            renamed_count += 1

        except Exception as e:
            print(f"  Error processing file {current_filename}: {e}")
            skipped_count += 1

    print("\nRenaming process complete.")
    print(f"Renamed: {renamed_count} files")
    print(f"Skipped: {skipped_count} files")

def main():
    parser = argparse.ArgumentParser(description="Rename CBOE VX futures CSV files to VX<MonthCode><YY>.csv format based on internal 'Futures' column.")
    parser.add_argument("source_dir", help="Directory containing the source CSV files to rename.")
    args = parser.parse_args()
    rename_files(args.source_dir)

if __name__ == "__main__":
    main() 