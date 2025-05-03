import duckdb
import pandas as pd

# --- Configuration ---
# Adjust this path if your database is located elsewhere
DB_PATH = "data/financial_data.duckdb"

def get_schema_info(conn, table_name):
    """Retrieves and formats schema info for a table."""
    try:
        print(f"--- Schema for table: {table_name} ---")
        # Use PRAGMA table_info
        schema_df = conn.execute(f"PRAGMA table_info('{table_name}');").fetchdf()

        if schema_df.empty:
            print(f"Table '{table_name}' not found or has no columns.")
            return

        # Display relevant columns
        print(schema_df[['name', 'type', 'notnull', 'pk']].to_string(index=False))

        # Identify and print primary key columns
        pk_columns = schema_df[schema_df['pk'] == 1]['name'].tolist()
        if pk_columns:
            print(f"\nPrimary Key: ({', '.join(pk_columns)})")
        else:
            print("\nPrimary Key: Not explicitly defined (or no columns marked as PK)")

    except duckdb.CatalogException:
        print(f"Error: Table '{table_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred while retrieving schema for {table_name}: {e}")
    finally:
        print("-" * (len(table_name) + 18)) # Separator line


# --- Main Execution ---
conn = None
try:
    conn = duckdb.connect(database=DB_PATH, read_only=True)
    print(f"Connected to database: {DB_PATH}")

    get_schema_info(conn, 'market_data')
    print("\n") # Add space between tables
    get_schema_info(conn, 'market_data_cboe')
    print("\n") # Add space between tables
    get_schema_info(conn, 'continuous_contracts')

except duckdb.Error as e:
    print(f"Error connecting to database {DB_PATH}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close()
        print(f"\nDatabase connection closed.")
