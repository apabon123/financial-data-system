import duckdb

def verify_continuous_contract_sources():
    """Verify the source field for continuous contracts with _d suffix."""
    try:
        # Connect to the database
        conn = duckdb.connect('data/financial_data.duckdb')
        
        # Check the current state
        query = """
        SELECT source, COUNT(*) as count
        FROM continuous_contracts
        WHERE symbol LIKE '%_d'
        GROUP BY source
        ORDER BY count DESC
        """
        result = conn.execute(query).fetchdf()
        
        print("\nCurrent state of continuous contracts with _d suffix:")
        for _, row in result.iterrows():
            print(f"Source: {row['source']}, Count: {row['count']}")
        
        conn.close()
        print("\nDatabase connection closed")
        
    except Exception as e:
        print(f"Error verifying continuous contract sources: {e}")
        raise

if __name__ == "__main__":
    verify_continuous_contract_sources() 