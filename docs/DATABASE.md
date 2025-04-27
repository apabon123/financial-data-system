# Database Structure

This document describes the database structure of the Financial Data System.

## Overview

The system uses DuckDB as its database engine, with a single file (`./data/financial_data.duckdb`) containing all data. The database is organized into several tables and views to efficiently store and manage financial data.

## Tables

### market_data
Primary table for storing raw market data.

```sql
CREATE TABLE market_data (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    date VARCHAR,            -- Date in YYYY-MM-DD format for compatibility
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    settle DOUBLE,           -- Settlement price (especially for futures)
    volume DOUBLE,
    open_interest DOUBLE,    -- For futures contracts
    up_volume DOUBLE,        -- TradeStation specific
    down_volume DOUBLE,      -- TradeStation specific
    interval_value INTEGER,
    interval_unit VARCHAR,
    source VARCHAR,
    changed BOOLEAN DEFAULT FALSE,  -- Flag for indicating changed/filled data
    adjusted BOOLEAN DEFAULT FALSE, -- Flag for adjusted prices
    quality INTEGER DEFAULT 100,    -- Quality score for data
    PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
);
```

Key fields:
- **timestamp**: The exact date and time of the data point
- **symbol**: Trading symbol (e.g., ES, NQ, VX, $VIX.X)
- **date**: String representation of the date in YYYY-MM-DD format (for backward compatibility)
- **open, high, low, close**: Price data
- **settle**: Settlement price (primarily for futures)
- **volume**: Trading volume
- **open_interest**: Number of open contracts (for futures)
- **up_volume, down_volume**: Volume on up/down ticks (TradeStation specific)
- **interval_value, interval_unit**: Timeframe specification (e.g., 1 day, 5 minute)
- **source**: Data source (e.g., 'tradestation', 'CBOE')
- **changed**: Flag indicating whether the data has been modified/filled
- **quality**: Data quality score (100 = original source data, lower values for derived/filled data)

### continuous_contracts
Stores generated continuous futures contracts.

```sql
CREATE TABLE continuous_contracts (
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (symbol, timestamp)
);
```

- Contains continuous futures contracts (e.g., VXc1, VXc2, etc.)
- Generated from individual futures contracts in market_data
- Used for analysis and trading strategies
- Supports multiple continuous contracts per base symbol (e.g., VXc1, VXc2)
- Handles rollovers on expiry days
- Includes metadata about data quality and adjustments

### account_balances
Stores account balance information.

```sql
CREATE TABLE account_balances (
    account_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    currency VARCHAR NOT NULL,
    cash_balance DOUBLE PRECISION,
    buying_power DOUBLE PRECISION,
    PRIMARY KEY (account_id, timestamp, currency)
);
```

### account_summary
Stores account summary information.

```sql
CREATE TABLE account_summary (
    account_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    net_liquidation_value DOUBLE PRECISION,
    gross_position_value DOUBLE PRECISION,
    net_position_value DOUBLE PRECISION,
    PRIMARY KEY (account_id, timestamp)
);
```

### active_positions
Stores current active positions.

```sql
CREATE TABLE active_positions (
    account_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    quantity INTEGER,
    average_price DOUBLE PRECISION,
    market_price DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    PRIMARY KEY (account_id, symbol, timestamp)
);
```

### historical_positions
Stores historical position data.

```sql
CREATE TABLE historical_positions (
    account_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    quantity INTEGER,
    average_price DOUBLE PRECISION,
    market_price DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    PRIMARY KEY (account_id, symbol, timestamp)
);
```

### trades
Stores trade information.

```sql
CREATE TABLE trades (
    account_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    quantity INTEGER,
    price DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    trade_id VARCHAR,
    PRIMARY KEY (account_id, symbol, timestamp, trade_id)
);
```

## Views

### daily_bars
View for accessing daily market data.

```sql
CREATE VIEW daily_bars AS
SELECT 
    date,
    symbol,
    open,
    high,
    low,
    close,
    volume,
    source
FROM 
    market_data
WHERE 
    interval_unit = 'day' AND interval_value = 1;
```

### weekly_bars
View for accessing weekly market data.

```sql
CREATE VIEW weekly_bars AS
SELECT 
    symbol,
    timestamp as date,
    open,
    high,
    low,
    close,
    volume,
    up_volume,
    down_volume,
    source,
    interval_value,
    interval_unit,
    adjusted,
    quality
FROM market_data
WHERE interval_value = 1 
AND interval_unit = 'week';
```

### monthly_bars
View for accessing monthly market data.

```sql
CREATE VIEW monthly_bars AS
SELECT 
    symbol,
    timestamp as date,
    open,
    high,
    low,
    close,
    volume,
    up_volume,
    down_volume,
    source,
    interval_value,
    interval_unit,
    adjusted,
    quality
FROM market_data
WHERE interval_value = 1 
AND interval_unit = 'month';
```

## Data Flow

1. Raw market data is loaded into the `market_data` table
2. Continuous contracts are generated and stored in the `continuous_contracts` table
3. Account data (balances, positions, trades) is stored in their respective tables
4. Views provide convenient access to data at different timeframes

## Common Queries

### Get Latest Data for a Symbol
```sql
SELECT * FROM market_data 
WHERE symbol = 'AAPL' 
AND interval_value = 1 
AND interval_unit = 'day'
ORDER BY timestamp DESC 
LIMIT 1;
```

### Get Continuous Contract Data
```sql
SELECT * FROM continuous_contracts 
WHERE symbol = 'VXc1' 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Get Account Positions
```sql
SELECT * FROM active_positions 
WHERE account_id = 'DU123456' 
ORDER BY timestamp DESC;
```

## Notes

1. The `market_data` table is the primary source of truth for all market data.
2. Views like `daily_bars`, `weekly_bars`, and `monthly_bars` provide convenient access to filtered data.
3. The `continuous_contracts` table stores derived data that is generated from the raw data.
4. All timestamps are stored in UTC to avoid timezone issues.
5. The quality field (0-100) indicates the reliability of the data point.
6. Continuous contracts are generated with proper rollover handling on expiry days.
7. Account data is stored with millisecond precision timestamps.
8. Views provide date-based access to market data for convenience. 