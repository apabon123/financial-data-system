import duckdb

def check_es_data():
    con = duckdb.connect('data/financial_data.duckdb')
    
    print('Checking market_data table for ES contracts in March 2024:')
    market_data_query = """
    SELECT DISTINCT 
        symbol,
        MIN(timestamp) as first_date,
        MAX(timestamp) as last_date
    FROM market_data 
    WHERE symbol LIKE 'ES%' 
    AND timestamp BETWEEN '2024-03-01' AND '2024-03-31'
    GROUP BY symbol 
    ORDER BY symbol;
    """
    print(con.execute(market_data_query).fetchdf())
    
    print('\nChecking futures_roll_dates table for ES:')
    roll_dates_query = """
    SELECT * 
    FROM futures_roll_dates 
    WHERE SymbolRoot = 'ES' 
    ORDER BY RollDate DESC 
    LIMIT 10;
    """
    print(con.execute(roll_dates_query).fetchdf())

if __name__ == '__main__':
    check_es_data() 