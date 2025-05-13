# Examples and Usage Guide

This document provides comprehensive examples for using the Financial Data Management System, including both Python code examples and SQL queries.

## Table of Contents

1. [Python Usage Examples](#python-usage-examples)
   - [Basic Operations](#basic-operations)
   - [Command-Line Usage](#command-line-usage)
   - [Advanced Usage](#advanced-usage)
2. [SQL Query Examples](#sql-query-examples)
   - [Basic Queries](#basic-queries)
   - [Data Source Specific Queries](#data-source-specific-queries)
   - [Timeframe Specific Queries](#timeframe-specific-queries)
   - [Aggregation Queries](#aggregation-queries)
   - [Combining Multiple Symbols](#combining-multiple-symbols)
   - [Data Quality and Export](#data-quality-and-export)

## Python Usage Examples

### Basic Operations

#### Example 1: Fetch Daily Data for Specific Symbols
```python
# Fetch daily OHLCV data for specific symbols
from tradestation_market_data_agent import TradeStationMarketDataAgent

agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
result = agent.process_query("fetch daily data for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31")

# Check the result
print(f"Fetched {result['results']['data_fetched']} records")
print(f"Saved {result['results']['data_saved']} new records to database")
```

#### Example 2: Fetch Minute Data
```python
# Fetch 5-minute OHLCV data
agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
result = agent.process_query("fetch 5-minute data for SPY from 2023-09-01 to 2023-09-30")

# Access the results
if result["results"]["success"]:
    print("Data fetched successfully")
else:
    print("Errors:", result["results"]["errors"])
```

#### Example 3: Update Database with Latest Data
```python
# Daily update process for all active symbols
from data_retrieval_agent import DataRetrievalAgent

# First, get list of active symbols
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")
symbols = retrieval_agent.get_active_symbols()

# Then update data for each symbol
market_agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
for symbol_batch in chunks(symbols, 10):  # Process in batches of 10
    symbols_str = ", ".join(symbol_batch)
    market_agent.process_query(f"fetch daily data for {symbols_str} from latest")

# Helper function to chunk list
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
```

#### Example 4: Query Stored Data
```python
# Get OHLC data for analysis
from data_retrieval_agent import DataRetrievalAgent

retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")
data = retrieval_agent.process_query("get daily data for AAPL from 2023-01-01 to 2023-12-31")

# Convert to pandas DataFrame for analysis
import pandas as pd
df = pd.DataFrame(data["results"]["data"])
print(f"Retrieved {len(df)} records for analysis")
```

### Command-Line Usage

#### Example 5: Using the Command-Line Interface
```bash
# Basic usage with default parameters
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL from 2023-01-01 to 2023-12-31"

# With verbose output and more compute loops
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31" -c 5 -v

# Fetch minute data
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch 1-minute data for SPY from 2023-09-01 to 2023-09-02" -v
```

#### Example 6: Scheduled Data Collection
```bash
# Script to be added to crontab for daily updates at 9 PM
#!/bin/bash
# daily_update.sh

DB_PATH="/path/to/financial_data.duckdb"
LOG_PATH="/path/to/logs/daily_update_$(date +%Y%m%d).log"

echo "Starting daily update at $(date)" > $LOG_PATH

# Update market data for major indices
uv run tradestation_market_data_agent.py -d $DB_PATH -q "fetch daily data for SPY, QQQ, DIA, IWM from latest" >> $LOG_PATH 2>&1

# Update economic data
uv run economic_data_api_agent.py -d $DB_PATH -q "update economic indicators" >> $LOG_PATH 2>&1

echo "Completed daily update at $(date)" >> $LOG_PATH

# Add to crontab with:
# 0 21 * * 1-5 /path/to/daily_update.sh
```

### Advanced Usage

#### Example 7: Error Handling and Retries
```python
# Implementing robust error handling with retries
import time
from tradestation_market_data_agent import TradeStationMarketDataAgent

def fetch_with_retry(symbols, start_date, end_date, max_retries=3):
    """Fetch data with automatic retries on failure."""
    agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
    
    symbols_str = ", ".join(symbols)
    query = f"fetch daily data for {symbols_str} from {start_date} to {end_date}"
    
    for attempt in range(max_retries):
        try:
            result = agent.process_query(query)
            
            if result["results"]["success"]:
                return result
            
            # Check for rate limiting errors specifically
            if any("rate limit" in error.lower() for error in result["results"]["errors"]):
                wait_time = (attempt + 1) * 30  # Exponential backoff: 30s, 60s, 90s
                print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            else:
                # Other errors might not be resolved by waiting
                print(f"Error: {result['results']['errors']}. Retry {attempt + 1}/{max_retries}")
                time.sleep(5)
        except Exception as e:
            print(f"Exception: {e}. Retry {attempt + 1}/{max_retries}")
            time.sleep(5)
    
    raise Exception(f"Failed to fetch data after {max_retries} attempts")

# Usage
try:
    result = fetch_with_retry(
        symbols=["AAPL", "MSFT", "GOOGL"], 
        start_date="2023-01-01", 
        end_date="2023-12-31"
    )
    print(f"Successfully fetched {result['results']['data_fetched']} records")
except Exception as e:
    print(f"Failed: {e}")
```

#### Example 8: Combining Data from Multiple Sources
```python
# Combining market data with economic data for analysis
from tradestation_market_data_agent import TradeStationMarketDataAgent
from economic_data_api_agent import EconomicDataAPIAgent
from data_retrieval_agent import DataRetrievalAgent
import pandas as pd

# Fetch market data
market_agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
market_agent.process_query("fetch daily data for SPY from 2022-01-01 to 2023-12-31")

# Fetch economic data
econ_agent = EconomicDataAPIAgent(database_path="./financial_data.duckdb")
econ_agent.process_query("fetch economic indicators GDP, CPI, UNEMPLOYMENT_RATE from 2022-01-01 to 2023-12-31")

# Query and combine the data
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")
market_data = retrieval_agent.execute_query("""
    SELECT timestamp, symbol, close 
    FROM daily_bars 
    WHERE symbol = 'SPY' 
    AND timestamp BETWEEN '2022-01-01' AND '2023-12-31'
""")

econ_data = retrieval_agent.execute_query("""
    SELECT timestamp, indicator, value
    FROM economic_data
    WHERE indicator IN ('GDP', 'CPI', 'UNEMPLOYMENT_RATE')
    AND timestamp BETWEEN '2022-01-01' AND '2023-12-31'
""")

# Pivot economic data for easier joining
econ_df = pd.DataFrame(econ_data)
econ_pivoted = econ_df.pivot(index='timestamp', columns='indicator', values='value').reset_index()

# Join with market data
market_df = pd.DataFrame(market_data)
combined_data = pd.merge(
    market_df, 
    econ_pivoted,
    on='timestamp',
    how='left'
)

print(f"Combined dataset has {len(combined_data)} rows and {combined_data.columns.size} columns")
```

## SQL Query Examples

**Important Note:** Market data is stored in two primary tables:
*   `market_data`: Contains data for most instruments (e.g., ES, NQ) typically sourced from TradeStation or other non-CBOE sources.
*   `market_data_cboe`: Contains *only* daily VIX futures (`VX%` symbols) and VIX Index (`$VIX.X`) data downloaded directly from CBOE. This table has fewer columns (no volume/open interest).

Choose the correct table based on the symbol you are querying.

### Basic Queries

#### Get the most recent data for a symbol (Non-VIX Future)
```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
ORDER BY timestamp DESC
LIMIT 1;
```

#### Get the most recent data for a symbol (VIX Future)
```sql
SELECT * FROM market_data_cboe
WHERE symbol = 'VXK25'
ORDER BY timestamp DESC
LIMIT 1;
```

#### Get all data for a specific date (from both tables)
```sql
-- Query market_data for non-VIX CBOE symbols
SELECT * FROM market_data
WHERE date = '2023-04-22'
  AND source != 'CBOE' -- Exclude any potential CBOE data here
UNION ALL
-- Query market_data_cboe for VIX symbols and VIX index
SELECT 
    timestamp, symbol, open, high, low, 
    NULL as close, -- market_data_cboe doesn't have 'close' column
    settle, 
    NULL as volume, -- No volume
    NULL as open_interest, -- No OI
    interval_value, interval_unit, source, 
    strftime(timestamp, '%Y-%m-%d') as date, -- Recreate date string
    NULL as up_volume, NULL as down_volume, -- No tick volume
    FALSE as changed, -- Assume FALSE
    FALSE as adjusted, -- Assume FALSE
    100 as quality -- Assume 100
FROM market_data_cboe
WHERE strftime(timestamp, '%Y-%m-%d') = '2023-04-22' -- Filter by date string
ORDER BY symbol, timestamp;
```

### Data Source Specific Queries

#### Get all TradeStation data
```sql
SELECT * FROM market_data
WHERE source = 'tradestation'
ORDER BY symbol, timestamp DESC
LIMIT 100;
```

#### Get all CBOE data (VIX Futures & Index)
```sql
SELECT * FROM market_data_cboe
-- WHERE source = 'CBOE' -- Condition redundant as this table only has CBOE data
ORDER BY symbol, timestamp DESC
LIMIT 100;
```

### Timeframe Specific Queries

#### Get daily data for a specific symbol and date range (Non-VIX Future)
```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'daily'
AND date BETWEEN '2023-01-01' AND '2023-04-22'
ORDER BY timestamp;
```

#### Get daily data for a specific symbol and date range (VIX Future)
```sql
SELECT * FROM market_data_cboe
WHERE symbol = 'VXK25'
AND interval_unit = 'daily' -- Implicitly daily
AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22' -- Filter on timestamp::DATE
ORDER BY timestamp;
```

#### Get minute data for a specific symbol and time range (Non-VIX Future)
```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'minute'
AND timestamp BETWEEN '2023-04-22 09:30:00' AND '2023-04-22 16:00:00'
ORDER BY timestamp;
```

### Aggregation Queries

#### Get the average daily volume for a symbol by month (Non-VIX Future)
```sql
SELECT 
    DATE_TRUNC('month', timestamp) AS month,
    AVG(volume) AS avg_daily_volume
FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'daily'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month;
```

#### Get the highest and lowest prices by week (VIX Future)
```sql
SELECT 
    DATE_TRUNC('week', timestamp) AS week,
    MAX(high) AS highest_price,
    MIN(low) AS lowest_price
FROM market_data_cboe
WHERE symbol = 'VXK25'
GROUP BY DATE_TRUNC('week', timestamp)
ORDER BY week;
```

### Combining Multiple Symbols

#### Compare ES and NQ futures closing prices
```sql
WITH es_data AS (
    SELECT timestamp::DATE as date, close AS es_close
    FROM market_data
    WHERE symbol LIKE 'ES%'
    AND interval_unit = 'daily'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
),
nq_data AS (
    SELECT timestamp::DATE as date, close AS nq_close
    FROM market_data
    WHERE symbol LIKE 'NQ%'
    AND interval_unit = 'daily'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
)
SELECT 
    es_data.date,
    es_data.es_close,
    nq_data.nq_close,
    nq_data.nq_close / es_data.es_close AS ratio
FROM es_data
JOIN nq_data ON es_data.date = nq_data.date
ORDER BY es_data.date;
```

#### Compare VIX index with VIX futures
```sql
WITH vix_data AS (
    SELECT timestamp::DATE as date, settle AS vix_settle -- Use settle (mapped from CLOSE)
    FROM market_data_cboe 
    WHERE symbol = '$VIX.X'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
),
vx_futures AS (
    SELECT timestamp::DATE as date, settle AS vx_settle
    FROM market_data_cboe
    WHERE symbol LIKE 'VX%'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
)
SELECT 
    vix_data.date,
    vix_data.vix_settle AS vix_index_settle,
    vx_futures.vx_settle AS vx_future_settle,
    vx_futures.vx_settle - vix_data.vix_settle AS basis
FROM vix_data
JOIN vx_futures ON vix_data.date = vx_futures.date
ORDER BY vix_data.date;
```

### Data Quality and Export

#### Find records that have been changed or filled (in market_data)
```sql
SELECT * FROM market_data
WHERE changed = true
ORDER BY timestamp DESC
LIMIT 100;
```

#### Find records with lower quality scores (in market_data)
```sql
SELECT * FROM market_data
WHERE quality < 100
ORDER BY quality ASC, timestamp DESC
LIMIT 100;
```

#### Export daily bars for a specific symbol to CSV (VIX Future)
```sql
COPY (
    SELECT 
        timestamp::DATE as date, 
        open, 
        high, 
        low, 
        settle -- Use settle for CBOE data
    FROM market_data_cboe
    WHERE symbol = 'VXK25'
    ORDER BY date
) TO 'vxk25_data.csv' (HEADER, DELIMITER ',');
```