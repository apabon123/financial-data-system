# temp_delete_vx.py
import duckdb
import os
import sys

# Define the database path relative to the script location
db_path = os.path.join("data", "financial_data.duckdb")
conn = None

# Define the SQL command
sql_delete = "DELETE FROM continuous_contracts WHERE symbol IN ('@VX=102XC', '@VX=102XN');"

try:
    print(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)
    print("Executing SQL:", sql_delete)
    
    # Execute the DELETE command
    result = conn.execute(sql_delete)
    
    # DuckDB DELETE usually doesn't return a useful row count directly in this mode
    # We can check the connection's `last_changes` if available, but a simple message is fine
    print(f"Attempted to delete rows for '@VX=102XC' and '@VX=102XN'.")
    
    # Commit the changes
    conn.commit()
    print("Changes committed.")

except duckdb.Error as e:
    print(f"DuckDB Error: {e}", file=sys.stderr)
    if conn:
        conn.rollback() # Rollback on error
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    if conn:
        conn.rollback()
    sys.exit(1)
finally:
    if conn:
        conn.close()
        print("Database connection closed.")

print("Script finished.")