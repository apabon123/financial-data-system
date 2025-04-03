# Usage Examples

This document provides examples for using the Financial Data Management System. These examples cover various aspects of the system, from basic data retrieval to complex workflows involving multiple agents.

## Basic Operations

### Example 1: Fetch Daily Data for Specific Symbols
```python
# Fetch daily OHLCV data for specific symbols
from tradestation_market_data_agent import TradeStationMarketDataAgent

agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
result = agent.process_query("fetch daily data for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31")

# Check the result
print(f"Fetched {result['results']['data_fetched']} records")
print(f"Saved {result['results']['data_saved']} new records to database")
```

### Example 2: Fetch Minute Data
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

### Example 3: Update Database with Latest Data
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

### Example 4: Query Stored Data
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

## Command-Line Usage

### Example 5: Using the Command-Line Interface
```bash
# Basic usage with default parameters
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL from 2023-01-01 to 2023-12-31"

# With verbose output and more compute loops
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31" -c 5 -v

# Fetch minute data
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch 1-minute data for SPY from 2023-09-01 to 2023-09-02" -v
```

### Example 6: Scheduled Data Collection
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

## Advanced Usage

### Example 7: Error Handling and Retries
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

### Example 8: Combining Data from Multiple Sources
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

### Example 9: Using the Derived Indicators Agent
```python
# Calculate and store technical indicators
from derived_indicators_agent import DerivedIndicatorsAgent

# Initialize agent
indicators_agent = DerivedIndicatorsAgent(database_path="./financial_data.duckdb")

# Calculate RSI for a symbol
result = indicators_agent.process_query("""
    calculate RSI for AAPL using daily data from 2023-01-01 to 2023-12-31 
    with parameters: period=14
""")

# Calculate multiple indicators in one pass
result = indicators_agent.process_query("""
    calculate indicators SMA, EMA, MACD, RSI, BBANDS 
    for MSFT using daily data from 2023-01-01 to 2023-12-31
""")

print(f"Calculated {result['results']['indicators_calculated']} indicators")

# Query the calculated indicators
from data_retrieval_agent import DataRetrievalAgent
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")
indicator_data = retrieval_agent.execute_query("""
    SELECT di.timestamp, di.symbol, di.indicator_name, di.value, m.close
    FROM derived_indicators di
    JOIN market_data m ON di.timestamp = m.timestamp AND di.symbol = m.symbol
    WHERE di.symbol = 'AAPL'
    AND di.indicator_name = 'RSI'
    AND di.timestamp BETWEEN '2023-01-01' AND '2023-12-31'
    ORDER BY di.timestamp
""")

# Convert to DataFrame for analysis or visualization
import pandas as pd
indicator_df = pd.DataFrame(indicator_data)
```

### Example 10: Building a Complete Trading Analysis Workflow
```python
# Complete workflow: data retrieval, indicator calculation, and analysis
import pandas as pd
import matplotlib.pyplot as plt
from tradestation_market_data_agent import TradeStationMarketDataAgent
from derived_indicators_agent import DerivedIndicatorsAgent
from data_retrieval_agent import DataRetrievalAgent

# Step 1: Define symbols and timeframe
symbols = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2023-01-01'
end_date = '2023-12-31'

# Step 2: Fetch market data if not already in database
market_agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
for symbol in symbols:
    market_agent.process_query(f"fetch daily data for {symbol} from {start_date} to {end_date}")

# Step 3: Calculate technical indicators
indicators_agent = DerivedIndicatorsAgent(database_path="./financial_data.duckdb")
for symbol in symbols:
    indicators_agent.process_query(f"""
        calculate indicators SMA, RSI, BBANDS
        for {symbol} using daily data from {start_date} to {end_date}
        with parameters: SMA_period=50, RSI_period=14, BBANDS_period=20
    """)

# Step 4: Retrieve and analyze data
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")

for symbol in symbols:
    # Get price data with indicators
    query = f"""
        SELECT 
            m.timestamp, m.symbol, m.open, m.high, m.low, m.close, m.volume,
            di_sma.value as sma_50,
            di_rsi.value as rsi_14,
            di_bb_upper.value as bb_upper,
            di_bb_middle.value as bb_middle,
            di_bb_lower.value as bb_lower
        FROM market_data m
        LEFT JOIN derived_indicators di_sma 
            ON m.timestamp = di_sma.timestamp 
            AND m.symbol = di_sma.symbol 
            AND di_sma.indicator_name = 'SMA'
        LEFT JOIN derived_indicators di_rsi 
            ON m.timestamp = di_rsi.timestamp 
            AND m.symbol = di_rsi.symbol 
            AND di_rsi.indicator_name = 'RSI'
        LEFT JOIN derived_indicators di_bb_upper 
            ON m.timestamp = di_bb_upper.timestamp 
            AND m.symbol = di_bb_upper.symbol 
            AND di_bb_upper.indicator_name = 'BBANDS_UPPER'
        LEFT JOIN derived_indicators di_bb_middle 
            ON m.timestamp = di_bb_middle.timestamp 
            AND m.symbol = di_bb_middle.symbol 
            AND di_bb_middle.indicator_name = 'BBANDS_MIDDLE'
        LEFT JOIN derived_indicators di_bb_lower 
            ON m.timestamp = di_bb_lower.timestamp 
            AND m.symbol = di_bb_lower.symbol 
            AND di_bb_lower.indicator_name = 'BBANDS_LOWER'
        WHERE m.symbol = '{symbol}'
        AND m.interval_unit = 'daily'
        AND m.timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY m.timestamp
    """
    
    data = retrieval_agent.execute_query(query)
    df = pd.DataFrame(data)
    
    # Step 5: Perform analysis (example: find potential buy signals)
    df['rsi_oversold'] = df['rsi_14'] < 30
    df['price_below_lower_bb'] = df['close'] < df['bb_lower']
    df['buy_signal'] = df['rsi_oversold'] & df['price_below_lower_bb']
    
    signals = df[df['buy_signal']].copy()
    print(f"\nFound {len(signals)} potential buy signals for {symbol}:")
    if len(signals) > 0:
        print(signals[['timestamp', 'close', 'rsi_14']])

# Step 6: Visualize results for the first symbol
symbol = symbols[0]
df_viz = df.copy()
df_viz['timestamp'] = pd.to_datetime(df_viz['timestamp'])
df_viz.set_index('timestamp', inplace=True)

plt.figure(figsize=(14, 10))

# Price and Bollinger Bands
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df_viz.index, df_viz['close'], label='Close Price')
ax1.plot(df_viz.index, df_viz['bb_upper'], 'r--', label='Upper BB')
ax1.plot(df_viz.index, df_viz['bb_middle'], 'g--', label='Middle BB')
ax1.plot(df_viz.index, df_viz['bb_lower'], 'r--', label='Lower BB')
ax1.scatter(df_viz[df_viz['buy_signal']].index, df_viz.loc[df_viz['buy_signal'], 'close'], 
           color='green', marker='^', s=100, label='Buy Signal')
ax1.set_title(f'{symbol} Price with Bollinger Bands')
ax1.set_ylabel('Price')
ax1.legend()

# RSI
ax2 = plt.subplot(2, 1, 2)
ax2.plot(df_viz.index, df_viz['rsi_14'], label='RSI(14)')
ax2.axhline(y=70, color='r', linestyle='--', label='Overbought')
ax2.axhline(y=30, color='g', linestyle='--', label='Oversold')
ax2.set_title(f'{symbol} RSI(14)')
ax2.set_ylabel('RSI Value')
ax2.set_ylim(0, 100)
ax2.legend()

plt.tight_layout()
plt.savefig(f'{symbol}_analysis.png')
print(f"\nSaved analysis chart for {symbol} to {symbol}_analysis.png")
```

### Example 11: Working with TradeStation Account Data
```python
# Fetching and analyzing account positions and balances
from tradestation_account_data_agent import TradeStationAccountDataAgent
from data_retrieval_agent import DataRetrievalAgent

# Fetch account data
account_agent = TradeStationAccountDataAgent(database_path="./financial_data.duckdb")
account_id = "YOUR_ACCOUNT_ID"  # Replace with actual account ID

# Fetch positions
result = account_agent.process_query(f"fetch positions for account {account_id}")
print(f"Retrieved {result['results']['positions_fetched']} positions")

# Fetch balances
result = account_agent.process_query(f"fetch balances for account {account_id}")
print(f"Retrieved balance data: ${result['results']['balances']['equity']:.2f} equity")

# Analyze performance
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")
performance_data = retrieval_agent.execute_query(f"""
    SELECT 
        p.symbol,
        p.quantity,
        p.average_price,
        p.market_value,
        p.open_pl,
        p.open_pl_percent,
        m.close as current_price
    FROM positions p
    JOIN latest_prices m ON p.symbol = m.symbol
    WHERE p.account_id = '{account_id}'
    AND p.quantity != 0
    ORDER BY p.open_pl_percent DESC
""")

import pandas as pd
performance_df = pd.DataFrame(performance_data)
print("\nPosition Performance Summary:")
print(performance_df)

# Calculate portfolio statistics
if len(performance_df) > 0:
    total_value = performance_df['market_value'].sum()
    total_pl = performance_df['open_pl'].sum()
    
    print(f"\nPortfolio Statistics:")
    print(f"Total Market Value: ${total_value:.2f}")
    print(f"Total P&L: ${total_pl:.2f} ({total_pl/total_value*100:.2f}%)")
    print(f"Best Performing: {performance_df.iloc[0]['symbol']} ({performance_df.iloc[0]['open_pl_percent']:.2f}%)")
    print(f"Worst Performing: {performance_df.iloc[-1]['symbol']} ({performance_df.iloc[-1]['open_pl_percent']:.2f}%)")
```

### Example 12: Working with Web Scraping Agent
```python
# Scraping additional financial data from websites
from web_scraping_agent import WebScrapingAgent
from data_normalization_agent import DataNormalizationAgent

# Initialize agents
scraping_agent = WebScrapingAgent(database_path="./financial_data.duckdb")
normalization_agent = DataNormalizationAgent(database_path="./financial_data.duckdb")

# Scrape earnings calendar data
result = scraping_agent.process_query("scrape earnings calendar for next week")
raw_data = result['results']['scraped_data']

# Normalize the data
normalized_data = normalization_agent.process_query(f"""
    normalize earnings calendar data using format:
    - symbol (string)
    - report_date (date)
    - time (string: "before_market", "after_market", "unspecified")
    - estimated_eps (float)
""")

# Store in database
from schema_management_agent import SchemaManagementAgent
schema_agent = SchemaManagementAgent(database_path="./financial_data.duckdb")

# Create earnings_calendar table if it doesn't exist
schema_agent.process_query("""
    create table if not exists earnings_calendar (
        symbol VARCHAR,
        report_date DATE,
        report_time VARCHAR,
        estimated_eps DOUBLE,
        actual_eps DOUBLE NULL,
        surprise_percent DOUBLE NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Insert the normalized data
from duckdb_write_agent import DuckDBWriteAgent
write_agent = DuckDBWriteAgent(database_path="./financial_data.duckdb")
write_agent.process_query("""
    insert normalized_data into earnings_calendar 
    on conflict (symbol, report_date) do update
""")

print(f"Added {result['results']['items_scraped']} earnings calendar entries to database")
```

## Integration with External Tools

### Example 13: Exporting Data for Analysis in Jupyter Notebook
```python
# Export data to CSV for use in external tools
from data_retrieval_agent import DataRetrievalAgent
import pandas as pd

# Initialize agent
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")

# Execute a complex query
query = """
SELECT 
    m.timestamp, 
    m.symbol, 
    m.open, m.high, m.low, m.close, m.volume,
    e1.value as gdp,
    e2.value as unemployment,
    e3.value as interest_rate
FROM daily_bars m
LEFT JOIN economic_data e1 
    ON DATE_TRUNC('quarter', m.timestamp) = e1.timestamp 
    AND e1.indicator = 'GDP'
LEFT JOIN economic_data e2 
    ON DATE_TRUNC('month', m.timestamp) = e2.timestamp 
    AND e2.indicator = 'UNEMPLOYMENT_RATE'
LEFT JOIN economic_data e3 
    ON DATE_TRUNC('day', m.timestamp) = e3.timestamp 
    AND e3.indicator = 'FEDERAL_FUNDS_RATE'
WHERE m.symbol IN ('SPY', 'QQQ', 'DIA')
AND m.timestamp BETWEEN '2020-01-01' AND '2023-12-31'
ORDER BY m.timestamp, m.symbol
"""

data = retrieval_agent.execute_query(query)
df = pd.DataFrame(data)

# Export to CSV
export_path = "./financial_data_export.csv"
df.to_csv(export_path, index=False)
print(f"Exported {len(df)} rows to {export_path}")

# Example Jupyter code to use this data
jupyter_code = """
# In Jupyter:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the exported data
df = pd.read_csv('financial_data_export.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Pivot the data for comparison
pivoted = df.pivot(index='timestamp', columns='symbol', values='close')

# Calculate returns
returns = pivoted.pct_change().dropna()

# Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation of Daily Returns')
plt.show()

# Plot price trends with economic indicators
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Price trends
for symbol in pivoted.columns:
    ax1.plot(pivoted.index, pivoted[symbol]/pivoted[symbol].iloc[0], label=symbol)
ax1.set_title('Normalized Price Performance')
ax1.set_ylabel('Normalized Price')
ax1.legend()

# Economic indicator (unemployment rate)
unemployment = df[df['symbol'] == 'SPY'][['timestamp', 'unemployment']].drop_duplicates()
unemployment = unemployment.set_index('timestamp').dropna()
ax2.plot(unemployment.index, unemployment['unemployment'], 'r-', label='Unemployment Rate')
ax2.set_title('Unemployment Rate')
ax2.set_ylabel('Rate (%)')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.show()
"""

print("\nJupyter Notebook Example:")
print(jupyter_code)
```

### Example 14: Backtesting a Simple Strategy
```python
# Using the data for a simple backtesting example
from data_retrieval_agent import DataRetrievalAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize agent
retrieval_agent = DataRetrievalAgent(database_path="./financial_data.duckdb")

# Get historical data for backtesting
symbol = "SPY"
backtest_query = f"""
SELECT 
    timestamp, symbol, open, high, low, close, volume
FROM daily_bars
WHERE symbol = '{symbol}'
AND timestamp BETWEEN '2020-01-01' AND '2023-12-31'
ORDER BY timestamp
"""

data = retrieval_agent.execute_query(backtest_query)
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Calculate indicators
df['sma_50'] = df['close'].rolling(window=50).mean()
df['sma_200'] = df['close'].rolling(window=200).mean()
df['signal'] = np.where(df['sma_50'] > df['sma_200'], 1, 0)  # 1 = buy, 0 = sell/hold cash

# Backtest logic
df['signal_shift'] = df['signal'].shift(1)
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['returns'] * df['signal_shift']

# Calculate cumulative returns
df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# Calculate statistics
trading_days_per_year = 252
total_return = df['strategy_cumulative_returns'].iloc[-1]
total_days = len(df)
annual_return = (1 + total_return) ** (trading_days_per_year / total_days) - 1
annual_volatility = df['strategy_returns'].std() * np.sqrt(trading_days_per_year)
sharpe_ratio = annual_return / annual_volatility
max_drawdown = (df['strategy_cumulative_returns'] - df['strategy_cumulative_returns'].cummax()).min()

# Print results
print(f"\nBacktesting Results for {symbol} SMA Crossover Strategy:")
print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['cumulative_returns'], label='Buy and Hold')
plt.plot(df.index, df['strategy_cumulative_returns'], label='SMA Crossover Strategy')
plt.title(f'{symbol} SMA Crossover Strategy Backtest')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.savefig(f'{symbol}_backtest.png')
print(f"\nSaved backtest chart to {symbol}_backtest.png")
```

## Error Handling Examples

### Example 15: Handling API Rate Limits and Outages
```python
# Implement a robust API calling function with circuit breaker pattern
import time
import random
from functools import wraps
from datetime import datetime

# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_failures:
            self.state = "OPEN"
            print(f"Circuit OPEN at {datetime.now()} after {self.failures} failures")
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def allow_request(self):
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                print(f"Circuit HALF_OPEN at {datetime.now()}")
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False

# Initialize circuit breaker for TradeStation API
ts_circuit = CircuitBreaker(max_failures=5, reset_timeout=600)

# Decorator function for API calls with circuit breaker
def with_circuit_breaker(circuit):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not circuit.allow_request():
                raise Exception(f"Circuit breaker open. Try again later.")
            
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                raise e
        return wrapper
    return decorator

# Example API function with circuit breaker
@with_circuit_breaker(ts_circuit)
def fetch_market_data(symbol, timeframe, start_date, end_date):
    from tradestation_market_data_agent import TradeStationMarketDataAgent
    
    agent = TradeStationMarketDataAgent(database_path="./financial_data.duckdb")
    query = f"fetch {timeframe} data for {symbol} from {start_date} to {end_date}"
    
    result = agent.process_query(query)
    
    if not result["results"]["success"]:
        raise Exception(f"API call failed: {result['results']['errors']}")
    
    return result

# Usage with error handling
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "INTC", "AMD"]
start_date = "2023-01-01"
end_date = "2023-12-31"

for symbol in symbols:
    try:
        print(f"Fetching data for {symbol}...")
        result = fetch_market_data(symbol, "daily", start_date, end_date)
        print(f"Success! Fetched {result['results']['data_fetched']} records for {symbol}")
        
        # Add random delay to avoid hammering the API
        delay = random.uniform(1.0, 3.0)
        time.sleep(delay)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
```

### Example 16: Data Validation and Quality Checks
```python
# Implement data validation and quality checking
from data_validation_agent import DataValidationAgent
import pandas as pd

# Initialize agent
validation_agent = DataValidationAgent(database_path="./financial_data.duckdb")

# Define validation rules for market data
validation_rules = {
    "timestamp_check": "timestamp IS NOT NULL",
    "symbol_check": "symbol IS NOT NULL AND symbol != ''",
    "price_range_check": "open > 0 AND high > 0 AND low > 0 AND close > 0",
    "high_low_check": "high >= low",
    "high_open_close_check": "high >= open AND high >= close",
    "low_open_close_check": "low <= open AND low <= close",
    "volume_check": "volume >= 0"
}

# Validate recently added data
result = validation_agent.process_query("""
    validate market_data 
    where timestamp >= DATEADD(day, -7, CURRENT_DATE)
    using standard price and volume rules
""")

# Print validation results
print("\nData Validation Results:")
print(f"Records checked: {result['results']['records_checked']}")
print(f"Records passing all checks: {result['results']['records_valid']}")
print(f"Records with issues: {result['results']['records_invalid']}")

# If there are invalid records, show details
if result['results']['records_invalid'] > 0:
    invalid_data = validation_agent.execute_query("""
        SELECT * FROM validation_failures
        WHERE validation_batch_id = '{result['results']['batch_id']}'
        ORDER BY symbol, timestamp
    """)
    
    df_invalid = pd.DataFrame(invalid_data)
    print("\nValidation Failures:")
    print(df_invalid.groupby('rule_name').size().reset_index(name='count'))
    
    # Fix the most common issues automatically
    if 'high_low_check' in df_invalid['rule_name'].values:
        print("\nAttempting to fix high/low inconsistencies...")
        fix_result = validation_agent.process_query("""
            auto_fix market_data 
            where high < low
            using rule swap_high_low
        """)
        print(f"Fixed {fix_result['results']['records_fixed']} records")
```

## Data Processing Workflows

### Example 17: Building a Data Processing Pipeline
```python
# Build a complete data processing pipeline using multiple agents
import time
import pandas as pd
from datetime import datetime, timedelta

# Import all agents
from tradestation_market_data_agent import TradeStationMarketDataAgent
from economic_data_api_agent import EconomicDataAPIAgent
from data_normalization_agent import DataNormalizationAgent
from data_validation_agent import DataValidationAgent
from derived_indicators_agent import DerivedIndicatorsAgent
from duckdb_write_agent import DuckDBWriteAgent

class DataPipeline:
    def __init__(self, database_path="./financial_data.duckdb"):
        # Initialize agents
        self.market_agent = TradeStationMarketDataAgent(database_path=database_path)
        self.econ_agent = EconomicDataAPIAgent(database_path=database_path)
        self.norm_agent = DataNormalizationAgent(database_path=database_path)
        self.validation_agent = DataValidationAgent(database_path=database_path)
        self.indicators_agent = DerivedIndicatorsAgent(database_path=database_path)
        self.write_agent = DuckDBWriteAgent(database_path=database_path)
        
        # Pipeline configuration
        self.config = {
            "market_symbols": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
            "economic_indicators": ["GDP", "CPI", "UNEMPLOYMENT_RATE", "FEDERAL_FUNDS_RATE"],
            "indicators_to_calculate": ["SMA:50", "SMA:200", "RSI:14", "BBANDS:20"],
            "lookback_days": 30,
            "batch_size": 2
        }
    
    def process_market_data(self):
        """Fetch and process market data for configured symbols"""
        print(f"\n[{datetime.now()}] Processing market data...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.config["lookback_days"])
        
        results = {
            "symbols_processed": 0,
            "records_fetched": 0,
            "records_saved": 0
        }
        
        # Process symbols in batches
        symbols = self.config["market_symbols"]
        for i in range(0, len(symbols), self.config["batch_size"]):
            batch = symbols[i:i+self.config["batch_size"]]
            symbols_str = ", ".join(batch)
            
            try:
                print(f"Processing batch: {symbols_str}")
                query = f"fetch daily data for {symbols_str} from {start_date} to {end_date}"
                result = self.market_agent.process_query(query)
                
                if result["results"]["success"]:
                    results["symbols_processed"] += len(batch)
                    results["records_fetched"] += result["results"]["data_fetched"]
                    results["records_saved"] += result["results"]["data_saved"]
                else:
                    print(f"Error processing batch: {result['results']['errors']}")
                
                # Add delay between batches
                time.sleep(2)
                
            except Exception as e:
                print(f"Exception processing batch: {e}")
        
        print(f"Market data processing complete.")
        print(f"Symbols processed: {results['symbols_processed']}")
        print(f"Records fetched: {results['records_fetched']}")
        print(f"New records saved: {results['records_saved']}")
        return results
    
    def process_economic_data(self):
        """Fetch and process economic data for configured indicators"""
        print(f"\n[{datetime.now()}] Processing economic data...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)  # Longer timeframe for economic data
        
        indicators_str = ", ".join(self.config["economic_indicators"])
        query = f"fetch economic indicators {indicators_str} from {start_date} to {end_date}"
        
        try:
            result = self.econ_agent.process_query(query)
            
            print(f"Economic data processing complete.")
            print(f"Indicators processed: {len(self.config['economic_indicators'])}")
            print(f"Records fetched: {result['results'].get('data_fetched', 0)}")
            print(f"New records saved: {result['results'].get('data_saved', 0)}")
            return result["results"]
            
        except Exception as e:
            print(f"Exception processing economic data: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_data(self):
        """Validate recently added data"""
        print(f"\n[{datetime.now()}] Validating data...")
        
        try:
            result = self.validation_agent.process_query("""
                validate market_data 
                where timestamp >= DATEADD(day, -7, CURRENT_DATE)
                using standard price and volume rules
            """)
            
            print(f"Validation complete.")
            print(f"Records checked: {result['results'].get('records_checked', 0)}")
            print(f"Records valid: {result['results'].get('records_valid', 0)}")
            print(f"Records invalid: {result['results'].get('records_invalid', 0)}")
            
            # If there are invalid records, try to fix them automatically
            if result['results'].get('records_invalid', 0) > 0:
                print("Attempting automatic fixes for invalid records...")
                fix_result = self.validation_agent.process_query("""
                    auto_fix market_data 
                    from validation_batch {result['results']['batch_id']}
                    using standard_fixes
                """)
                print(f"Fixed {fix_result['results'].get('records_fixed', 0)} records")
            
            return result["results"]
            
        except Exception as e:
            print(f"Exception during validation: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_indicators(self):
        """Calculate technical indicators for market data"""
        print(f"\n[{datetime.now()}] Calculating technical indicators...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.config["lookback_days"])
        
        results = {
            "symbols_processed": 0,
            "indicators_calculated": 0
        }
        
        # Process each symbol
        for symbol in self.config["market_symbols"]:
            try:
                indicators_str = ", ".join(self.config["indicators_to_calculate"])
                query = f"""
                    calculate indicators {indicators_str} 
                    for {symbol} using daily data 
                    from {start_date} to {end_date}
                """
                result = self.indicators_agent.process_query(query)
                
                if result["results"]["success"]:
                    results["symbols_processed"] += 1
                    results["indicators_calculated"] += result["results"].get("indicators_calculated", 0)
                else:
                    print(f"Error calculating indicators for {symbol}: {result['results'].get('errors', ['Unknown error'])}")
                
                # Add delay between symbols
                time.sleep(1)
                
            except Exception as e:
                print(f"Exception calculating indicators for {symbol}: {e}")
        
        print(f"Indicator calculation complete.")
        print(f"Symbols processed: {results['symbols_processed']}")
        print(f"Indicators calculated: {results['indicators_calculated']}")
        return results
    
    def run_pipeline(self):
        """Run the complete data pipeline"""
        print(f"======================================")
        print(f"Starting data pipeline: {datetime.now()}")
        print(f"======================================")
        
        pipeline_start = time.time()
        
        # Step 1: Process market data
        market_results = self.process_market_data()
        
        # Step 2: Process economic data
        econ_results = self.process_economic_data()
        
        # Step 3: Validate data
        validation_results = self.validate_data()
        
        # Step 4: Calculate technical indicators
        indicator_results = self.calculate_indicators()
        
        pipeline_duration = time.time() - pipeline_start
        
        print(f"\n======================================")
        print(f"Pipeline completed: {datetime.now()}")
        print(f"Duration: {pipeline_duration:.2f} seconds")
        print(f"======================================")
        
        return {
            "market_results": market_results,
            "econ_results": econ_results,
            "validation_results": validation_results,
            "indicator_results": indicator_results,
            "pipeline_duration": pipeline_duration
        }

# Usage
if __name__ == "__main__":
    pipeline = DataPipeline(database_path="./financial_data.duckdb")
    results = pipeline.run_pipeline()
    
    # Save pipeline results to log
    log_filename = f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"Pipeline Run: {datetime.now()}\n")
        log_file.write(f"Duration: {results['pipeline_duration']:.2f} seconds\n\n")
        
        log_file.write("Market Data Results:\n")
        log_file.write(f"  Symbols processed: {results['market_results']['symbols_processed']}\n")
        log_file.write(f"  Records fetched: {results['market_results']['records_fetched']}\n")
        log_file.write(f"  Records saved: {results['market_results']['records_saved']}\n\n")
        
        log_file.write("Economic Data Results:\n")
        if isinstance(results['econ_results'], dict):
            for key, value in results['econ_results'].items():
                log_file.write(f"  {key}: {value}\n")
        else:
            log_file.write(f"  {results['econ_results']}\n\n")
        
        log_file.write("Validation Results:\n")
        if isinstance(results['validation_results'], dict):
            for key, value in results['validation_results'].items():
                log_file.write(f"  {key}: {value}\n")
        else:
            log_file.write(f"  {results['validation_results']}\n\n")
        
        log_file.write("Indicator Results:\n")
        log_file.write(f"  Symbols processed: {results['indicator_results']['symbols_processed']}\n")
        log_file.write(f"  Indicators calculated: {results['indicator_results']['indicators_calculated']}\n")
    
    print(f"\nPipeline log saved to {log_filename}")
```

### Example 18: Scheduled Data Collection with Monitoring
```bash
#!/bin/bash
# daily_data_collection.sh
# Description: Script to collect and process financial data, with monitoring and notifications

# Configuration
DB_PATH="/path/to/financial_data.duckdb"
LOG_DIR="/path/to/logs"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/data_collection_${TODAY}.log"
ERROR_LOG="${LOG_DIR}/errors_${TODAY}.log"
EMAIL="youremail@example.com"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to send email notification
send_notification() {
    subject="$1"
    message="$2"
    echo "$message" | mail -s "$subject" "$EMAIL"
}

# Start logging
echo "===== Data Collection Process Started at $(date) =====" > "$LOG_FILE"

# Step 1: Market data collection
echo "Collecting market data..." >> "$LOG_FILE"
uv run tradestation_market_data_agent.py -d "$DB_PATH" -q "fetch daily data for AAPL, MSFT, GOOGL, SPY, QQQ from latest" >> "$LOG_FILE" 2> "$ERROR_LOG"

if [ $? -ne 0 ]; then
    echo "ERROR: Market data collection failed. See error log." >> "$LOG_FILE"
    send_notification "ERROR: Market data collection failed" "The market data collection process failed on ${TODAY}. Check the error log at ${ERROR_LOG}."
else
    echo "Market data collection completed successfully." >> "$LOG_FILE"
fi

# Step 2: Economic data collection
echo "Collecting economic data..." >> "$LOG_FILE"
uv run economic_data_api_agent.py -d "$DB_PATH" -q "update economic indicators" >> "$LOG_FILE" 2>> "$ERROR_LOG"

if [ $? -ne 0 ]; then
    echo "ERROR: Economic data collection failed. See error log." >> "$LOG_FILE"
    send_notification "ERROR: Economic data collection failed" "The economic data collection process failed on ${TODAY}. Check the error log at ${ERROR_LOG}."
else
    echo "Economic data collection completed successfully." >> "$LOG_FILE"
fi

# Step 3: Data validation
echo "Validating collected data..." >> "$LOG_FILE"
uv run data_validation_agent.py -d "$DB_PATH" -q "validate market_data from last 7 days" >> "$LOG_FILE" 2>> "$ERROR_LOG"

INVALID_COUNT=$(grep "Records with issues:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
if [ -n "$INVALID_COUNT" ] && [ "$INVALID_COUNT" -gt 0 ]; then
    echo "WARNING: Found $INVALID_COUNT invalid records. Attempting auto-fix..." >> "$LOG_FILE"
    uv run data_validation_agent.py -d "$DB_PATH" -q "auto_fix last validation batch" >> "$LOG_FILE" 2>> "$ERROR_LOG"
    
    # Check if fixes were applied
    FIXED_COUNT=$(grep "Fixed records:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    if [ "$FIXED_COUNT" -lt "$INVALID_COUNT" ]; then
        send_notification "WARNING: Data validation issues" "Found $INVALID_COUNT invalid records and fixed $FIXED_COUNT on ${TODAY}. Manual review may be needed."
    fi
fi

# Step 4: Calculate technical indicators
echo "Calculating technical indicators..." >> "$LOG_FILE"
uv run derived_indicators_agent.py -d "$DB_PATH" -q "update all indicators for latest data" >> "$LOG_FILE" 2>> "$ERROR_LOG"

if [ $? -ne 0 ]; then
    echo "ERROR: Indicator calculation failed. See error log." >> "$LOG_FILE"
    send_notification "ERROR: Indicator calculation failed" "The indicator calculation process failed on ${TODAY}. Check the error log at ${ERROR_LOG}."
else
    echo "Indicator calculation completed successfully." >> "$LOG_FILE"
fi

# Generate summary report
DATA_COUNTS=$(uv run data_retrieval_agent.py -d "$DB_PATH" -q "count records from past 7 days" 2>> "$ERROR_LOG")
echo "Data collection summary:" >> "$LOG_FILE"
echo "$DATA_COUNTS" >> "$LOG_FILE"

# Check for any errors
if [ -s "$ERROR_LOG" ]; then
    echo "Process completed with errors. See error log for details." >> "$LOG_FILE"
    send_notification "Data collection completed with errors" "The data collection process completed on ${TODAY} but encountered some errors. Check the logs at ${LOG_DIR}."
else
    echo "Process completed successfully with no errors." >> "$LOG_FILE"
    rm "$ERROR_LOG"  # Remove empty error log
fi

echo "===== Data Collection Process Completed at $(date) =====" >> "$LOG_FILE"

# Add to crontab with:
# 0 18 * * 1-5 /path/to/daily_data_collection.sh
```

## Using the System for Analysis

### Example 19: Perform Advanced Market Analysis
```python
# Example script for advanced market analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_retrieval_agent import DataRetrievalAgent

def perform_correlation_analysis(database_path, symbols, start_date, end_date):
    """Analyze correlations between symbols and create visualization"""
    # Initialize the data retrieval agent
    agent = DataRetrievalAgent(database_path=database_path)
    
    # Create a comma-separated string of symbols for the query
    symbols_str = "'" + "','".join(symbols) + "'"
    
    # Query to get daily close prices for the selected symbols
    query = f"""
    SELECT timestamp, symbol, close
    FROM daily_bars
    WHERE symbol IN ({symbols_str})
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY timestamp, symbol
    """
    
    # Execute the query
    data = agent.execute_query(query)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Pivot the data to get a time series for each symbol
    pivot_df = df.pivot(index='timestamp', columns='symbol', values='close')
    
    # Calculate daily returns
    returns = pivot_df.pct_change().dropna()
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.matshow(corr_matrix, fignum=1)
    plt.title('Correlation of Daily Returns', fontsize=14, pad=20)
    
    # Add correlation values to the heatmap
    for (i, j), value in np.ndenumerate(corr_matrix):
        plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value) > 0.6 else 'black')
    
    # Set ticks to symbol names
    plt.xticks(range(len(symbols)), symbols, rotation=45)
    plt.yticks(range(len(symbols)), symbols)
    
    plt.colorbar()
    plt.savefig('correlation_analysis.png')
    print(f"Correlation analysis saved to correlation_analysis.png")
    
    # Calculate rolling correlations to SPY
    if 'SPY' in symbols:
        plt.figure(figsize=(14, 8))
        
        # Calculate 30-day rolling correlation to SPY for each symbol
        spy_correlations = pd.DataFrame()
        for symbol in symbols:
            if symbol != 'SPY':
                roll_corr = returns['SPY'].rolling(window=30).corr(returns[symbol])
                spy_correlations[symbol] = roll_corr
        
        # Plot rolling correlations
        spy_correlations.plot(figsize=(14, 8))
        plt.title('30-Day Rolling Correlation to SPY', fontsize=14)
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Date')
        plt.grid(True)
        plt.savefig('spy_rolling_correlations.png')
        print(f"Rolling correlation analysis saved to spy_rolling_correlations.png")
    
    return corr_matrix, pivot_df, returns

def perform_volatility_analysis(database_path, symbols, start_date, end_date):
    """Analyze volatility patterns for the given symbols"""
    # Initialize the data retrieval agent
    agent = DataRetrievalAgent(database_path=database_path)
    
    # Create a comma-separated string of symbols for the query
    symbols_str = "'" + "','".join(symbols) + "'"
    
    # Query to get daily OHLC prices for the selected symbols
    query = f"""
    SELECT timestamp, symbol, open, high, low, close
    FROM daily_bars
    WHERE symbol IN ({symbols_str})
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY timestamp, symbol
    """
    
    # Execute the query
    data = agent.execute_query(query)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Group by symbol
    symbols_data = {symbol: group for symbol, group in df.groupby('symbol')}
    
    # Calculate metrics for each symbol
    volatility_metrics = {}
    
    for symbol, symbol_df in symbols_data.items():
        # Set timestamp as index
        symbol_df = symbol_df.set_index('timestamp')
        
        # Calculate daily returns
        symbol_df['returns'] = symbol_df['close'].pct_change()
        
        # Calculate true range
        symbol_df['tr1'] = abs(symbol_df['high'] - symbol_df['low'])
        symbol_df['tr2'] = abs(symbol_df['high'] - symbol_df['close'].shift(1))
        symbol_df['tr3'] = abs(symbol_df['low'] - symbol_df['close'].shift(1))
        symbol_df['true_range'] = symbol_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate volatility metrics
        metrics = {
            'annualized_volatility': symbol_df['returns'].std() * np.sqrt(252) * 100,  # in percentage
            'avg_daily_range': (symbol_df['high'] - symbol_df['low']).mean() / symbol_df['close'].mean() * 100,  # in percentage
            'avg_true_range': symbol_df['true_range'].mean() / symbol_df['close'].mean() * 100,  # in percentage
            'max_drawdown': (symbol_df['close'] / symbol_df['close'].cummax() - 1).min() * 100,  # in percentage
        }
        
        # Calculate 20-day rolling volatility
        symbol_df['rolling_vol_20d'] = symbol_df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        volatility_metrics[symbol] = {
            'metrics': metrics,
            'rolling_vol': symbol_df['rolling_vol_20d']
        }
    
    # Create a summary DataFrame
    metrics_df = pd.DataFrame({symbol: data['metrics'] for symbol, data in volatility_metrics.items()})
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Plot rolling volatility for each symbol
    rolling_vol_df = pd.DataFrame({symbol: data['rolling_vol'] for symbol, data in volatility_metrics.items()})
    rolling_vol_df.plot(figsize=(14, 8))
    plt.title('20-Day Rolling Annualized Volatility', fontsize=14)
    plt.ylabel('Volatility (%)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig('rolling_volatility.png')
    print(f"Rolling volatility analysis saved to rolling_volatility.png")
    
    # Bar chart of annualized volatility
    plt.figure(figsize=(14, 8))
    metrics_df.loc['annualized_volatility'].plot(kind='bar')
    plt.title('Annualized Volatility Comparison', fontsize=14)
    plt.ylabel('Volatility (%)')
    plt.grid(True, axis='y')
    plt.savefig('annualized_volatility.png')
    print(f"Annualized volatility comparison saved to annualized_volatility.png")
    
    return volatility_metrics, metrics_df

# Usage example
if __name__ == "__main__":
    database_path = "./financial_data.duckdb"
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    print("Performing correlation analysis...")
    corr_matrix, prices, returns = perform_correlation_analysis(
        database_path, symbols, start_date, end_date
    )
    
    print("\nPerforming volatility analysis...")
    volatility_metrics, metrics_summary = perform_volatility_analysis(
        database_path, symbols, start_date, end_date
    )
    
    print("\nSummary of Volatility Metrics:")
    print(metrics_summary.T)
    
    print("\nAnalysis complete. Visualizations have been saved as PNG files.")
```

## Integrating with External Systems

### Example 20: Export Data to CSV for External Tools
```python
# Script to export data for use in external tools (Excel, Tableau, etc.)
import os
import pandas as pd
from datetime import datetime
from data_retrieval_agent import DataRetrievalAgent

def export_to_csv(database_path, output_dir="./exports"):
    """Export key data to CSV files for external tools"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize agent
    agent = DataRetrievalAgent(database_path=database_path)
    
    # Get timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define export queries
    exports = {
        "daily_market_data": {
            "query": """
                SELECT * FROM daily_bars
                WHERE timestamp >= DATEADD(month, -6, CURRENT_DATE)
                ORDER BY timestamp, symbol
            """,
            "filename": f"market_data_daily_{timestamp}.csv"
        },
        "latest_positions": {
            "query": """
                SELECT * FROM active_positions
                ORDER BY account_id, symbol
            """,
            "filename": f"active_positions_{timestamp}.csv"
        },
        "economic_indicators": {
            "query": """
                SELECT * FROM economic_calendar
                WHERE timestamp >= DATEADD(year, -2, CURRENT_DATE)
                ORDER BY timestamp, indicator
            """,
            "filename": f"economic_indicators_{timestamp}.csv"
        },
        "technical_indicators": {
            "query": """
                SELECT di.timestamp, di.symbol, di.indicator_name, di.value,
                       di.interval_value, di.interval_unit
                FROM derived_indicators di
                WHERE di.timestamp >= DATEADD(month, -3, CURRENT_DATE)
                ORDER BY di.timestamp, di.symbol, di.indicator_name
            """,
            "filename": f"technical_indicators_{timestamp}.csv"
        }
    }
    
    results = {}
    
    # Execute each export
    for name, export_info in exports.items():
        try:
            print(f"Exporting {name}...")
            
            # Execute query
            data = agent.execute_query(export_info["query"])
            
            if not data:
                print(f"No data returned for {name}")
                results[name] = {"success": False, "error": "No data returned", "count": 0}
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            output_path = os.path.join(output_dir, export_info["filename"])
            df.to_csv(output_path, index=False)
            
            print(f"Exported {len(df)} rows to {output_path}")
            results[name] = {"success": True, "path": output_path, "count": len(df)}
            
        except Exception as e:
            print(f"Error exporting {name}: {e}")
            results[name] = {"success": False, "error": str(e), "count": 0}
    
    # Create a summary report
    summary_path = os.path.join(output_dir, f"export_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Data Export Summary - {datetime.now()}\n")
        f.write("="*50 + "\n\n")
        
        for name, result in results.items():
            f.write(f"{name}:\n")
            if result["success"]:
                f.write(f"  Status: Success\n")
                f.write(f"  Rows: {result['count']}\n")
                f.write(f"  File: {result['path']}\n")
            else:
                f.write(f"  Status: Failed\n")
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
    
    print(f"\nExport summary saved to {summary_path}")
    return results

# Usage
if __name__ == "__main__":
    database_path = "./financial_data.duckdb"
    export_results = export_to_csv(database_path)
    
    # Print overall results
    print("\nExport Results Summary:")
    for name, result in export_results.items():
        status = "✓ Success" if result["success"] else "✗ Failed"
        print(f"{status} - {name}: {result.get('count', 0)} rows")
```

This comprehensive set of examples covers a wide range of scenarios for using the Financial Data Management System, from basic operations to advanced workflows and integration with external systems. These examples should help users understand how to leverage the system's capabilities for various financial data management and analysis tasks.