import subprocess
import os
import sys

# Assuming the database file is in the 'data' subdirectory relative to the script
db_path = os.path.join('data', 'financial_data.duckdb')

# Use the full path to the duckdb executable provided by the user
duckdb_exe_path = r'C:\Users\alexp\Tools\duckdb.exe'

# Base command parts
command = [duckdb_exe_path, db_path]

# Add any additional arguments passed to the script
# This allows passing flags like -c "SQL QUERY" directly
if len(sys.argv) > 1:
    command.extend(sys.argv[1:])

try:
    # print(f"Executing command: {' '.join(command)}") # Optional: for debugging
    subprocess.run(command, check=True)
except FileNotFoundError:
    print(f"Error: DuckDB executable not found at {duckdb_exe_path}")
except Exception as e:
    print(f"An error occurred: {e}") 