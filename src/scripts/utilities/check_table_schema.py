import duckdb
import argparse

def check_table_schema(db_path, table_name):
    """Check the schema of a table in DuckDB."""
    conn = duckdb.connect(db_path)
    
    try:
        # Get column information
        query = f"PRAGMA table_info('{table_name}')"
        columns = conn.execute(query).fetchdf()
        
        print(f"Schema for table '{table_name}':")
        print("="*50)
        for _, row in columns.iterrows():
            nullable = "NULL" if row['notnull'] == 0 else "NOT NULL"
            pk = "PRIMARY KEY" if row['pk'] == 1 else ""
            default = f"DEFAULT {row['dflt_value']}" if row['dflt_value'] is not None else ""
            print(f"{row['name']} {row['type']} {nullable} {default} {pk}".strip())
        
        # Get sample data
        query = f"SELECT * FROM {table_name} LIMIT 1"
        sample = conn.execute(query).fetchdf()
        
        print("\nSample data (1 row):")
        print("="*50)
        if not sample.empty:
            for column in sample.columns:
                value = sample.iloc[0][column]
                print(f"{column}: {value}")
        else:
            print("No data in table.")
            
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the schema of a table in DuckDB")
    parser.add_argument("--db-path", default="data/financial_data.duckdb", help="Path to the DuckDB database")
    parser.add_argument("--table", default="market_data", help="Name of the table to check")
    
    args = parser.parse_args()
    check_table_schema(args.db_path, args.table) 