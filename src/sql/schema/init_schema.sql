-- Market Data Table
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    open_interest BIGINT,
    up_volume BIGINT,
    down_volume BIGINT,
    source VARCHAR,
    interval_value INTEGER,
    interval_unit VARCHAR,
    adjusted BOOLEAN DEFAULT FALSE,
    quality INTEGER DEFAULT 100,
    PRIMARY KEY (timestamp, symbol, interval_value, interval_unit)
);

-- Economic Data Table
CREATE TABLE IF NOT EXISTS economic_data (
    timestamp TIMESTAMP NOT NULL,
    indicator VARCHAR NOT NULL,
    value DOUBLE,
    source VARCHAR,
    frequency VARCHAR,
    revision_number INTEGER DEFAULT 0,
    PRIMARY KEY (timestamp, indicator)
);

-- Symbols Table
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id VARCHAR PRIMARY KEY,
    symbol VARCHAR UNIQUE,
    name VARCHAR,
    sector VARCHAR,
    type VARCHAR,
    active BOOLEAN DEFAULT TRUE,
    exchange VARCHAR,
    currency VARCHAR,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Sources Table
CREATE TABLE IF NOT EXISTS data_sources (
    source_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    type VARCHAR,
    last_updated TIMESTAMP,
    status VARCHAR,
    priority INTEGER,
    api_key_reference VARCHAR,
    rate_limit INTEGER
);

-- Derived Indicators Table
CREATE TABLE IF NOT EXISTS derived_indicators (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    indicator_name VARCHAR,
    value DOUBLE,
    parameters JSON,
    interval_value INTEGER,
    interval_unit VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol, indicator_name, interval_value, interval_unit)
);

-- Metadata Table
CREATE TABLE IF NOT EXISTS metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR
);

-- Symbol Tags Relation
CREATE TABLE IF NOT EXISTS symbol_tags (
    symbol_id VARCHAR,
    tag VARCHAR,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol_id, tag),
    FOREIGN KEY (symbol_id) REFERENCES symbols(symbol_id)
);

-- Indicator Metadata Table
CREATE TABLE IF NOT EXISTS indicator_metadata (
    indicator_name VARCHAR PRIMARY KEY,
    display_name VARCHAR,
    description TEXT,
    unit VARCHAR,
    source VARCHAR
);

-- Account Balances Table
CREATE TABLE IF NOT EXISTS account_balances (
    timestamp TIMESTAMP,
    account_id VARCHAR,
    cash_balance DOUBLE,
    buying_power DOUBLE,
    day_trading_buying_power DOUBLE,
    equity DOUBLE,
    margin_balance DOUBLE,
    real_time_buying_power DOUBLE,
    real_time_equity DOUBLE,
    real_time_cost_of_positions DOUBLE,
    day_trades_count INTEGER,
    day_trading_qualified BOOLEAN,
    source VARCHAR,
    currency VARCHAR,
    PRIMARY KEY (timestamp, account_id)
);

-- Positions Table
CREATE TABLE IF NOT EXISTS positions (
    timestamp TIMESTAMP,
    account_id VARCHAR,
    symbol VARCHAR,
    quantity DOUBLE,
    average_price DOUBLE,
    market_value DOUBLE,
    cost_basis DOUBLE,
    open_pl DOUBLE,
    open_pl_percent DOUBLE,
    day_pl DOUBLE,
    initial_margin DOUBLE,
    maintenance_margin DOUBLE,
    position_id VARCHAR,
    source VARCHAR,
    PRIMARY KEY (timestamp, account_id, symbol)
);

-- Orders Table
CREATE TABLE IF NOT EXISTS orders (
    timestamp TIMESTAMP,
    account_id VARCHAR,
    order_id VARCHAR,
    symbol VARCHAR,
    quantity DOUBLE,
    order_type VARCHAR,
    side VARCHAR,
    status VARCHAR,
    limit_price DOUBLE,
    stop_price DOUBLE,
    filled_quantity DOUBLE,
    remaining_quantity DOUBLE,
    average_fill_price DOUBLE,
    duration VARCHAR,
    route VARCHAR,
    execution_time TIMESTAMP,
    cancellation_time TIMESTAMP,
    source VARCHAR,
    PRIMARY KEY (order_id)
);

-- Trades Table
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMP,
    account_id VARCHAR,
    order_id VARCHAR,
    trade_id VARCHAR,
    symbol VARCHAR,
    quantity DOUBLE,
    price DOUBLE,
    side VARCHAR,
    commission DOUBLE,
    fees DOUBLE,
    trade_time TIMESTAMP,
    position_effect VARCHAR,
    source VARCHAR,
    PRIMARY KEY (trade_id)
);

-- Continuous Contracts Table
CREATE TABLE IF NOT EXISTS continuous_contracts (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,           -- Continuous contract symbol (e.g., VXc1)
    underlying_symbol VARCHAR,     -- Specific contract used for this row (e.g., VXF10)
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    open_interest BIGINT,
    up_volume BIGINT,
    down_volume BIGINT,
    source VARCHAR,
    built_by VARCHAR,        -- How the continuous row was built ('local_generator', 'tradestation')
    interval_value INTEGER,
    interval_unit VARCHAR,
    adjusted BOOLEAN DEFAULT FALSE,
    quality INTEGER DEFAULT 100,
    settle DOUBLE,           -- Settlement price. Added missing column
    PRIMARY KEY (symbol, timestamp, interval_value, interval_unit) -- Changed order
);

-- Data Cleaning Runs Log Table
CREATE TABLE IF NOT EXISTS data_cleaning_runs (
    run_id VARCHAR PRIMARY KEY,
    pipeline_name VARCHAR NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    records_count INTEGER,
    modifications_count INTEGER,
    elapsed_time DOUBLE, -- Or REAL, depending on DuckDB precision preference for seconds
    status VARCHAR, -- e.g., 'success', 'error', 'skipped'
    error_message TEXT, -- For storing error details if status is 'error'
    cleaners_applied TEXT, -- Comma-separated list of cleaner names
    fields_modified TEXT, -- Comma-separated list of modified fields
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- When this log entry was created
);

-- Symbol Metadata Table
CREATE TABLE IF NOT EXISTS symbol_metadata (
    symbol VARCHAR NOT NULL,
    base_symbol VARCHAR,          -- e.g. ES, NQ
    description VARCHAR,
    exchange VARCHAR,
    asset_type VARCHAR,           -- 'future', 'continuous_future', 'equity', 'forex', 'crypto', 'option', 'index'
    data_source VARCHAR,          -- Preferred data source for this symbol, e.g., 'tradestation', 'ibkr', 'polygon'
    data_table VARCHAR,           -- Table where data is stored, e.g. 'market_data', 'continuous_contracts'
    interval_unit VARCHAR,        -- e.g., 'minute', 'daily', 'tick'
    interval_value INTEGER,       -- e.g., 1, 5, 15 for minutes, or 1 for daily
    config_path VARCHAR,          -- Path to the specific YAML config for this symbol if applicable
    start_date DATE,              -- Earliest date for which historical data should be considered/fetched
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    historical_script_path VARCHAR, -- Path to script for historical data
    update_script_path VARCHAR,   -- Path to script for incremental updates
    additional_metadata JSON,     -- For any other source or symbol specific metadata
    PRIMARY KEY (symbol, interval_unit, interval_value)
);

-- Create standard views

-- Daily Bars View
CREATE OR REPLACE VIEW daily_bars AS
SELECT 
    DATE_TRUNC('day', timestamp) AS date,
    symbol,
    FIRST(open) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS open,
    MAX(high) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS high,
    MIN(low) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS low,
    LAST(close) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS close,
    SUM(volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS volume,
    SUM(up_volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS up_volume,
    SUM(down_volume) FILTER(WHERE interval_unit = 'daily' OR interval_value = 1440) AS down_volume,
    'daily' AS interval_unit,
    1 AS interval_value,
    source
FROM market_data
GROUP BY DATE_TRUNC('day', timestamp), symbol, source;

-- Minute Bars View
CREATE OR REPLACE VIEW minute_bars AS
SELECT *
FROM market_data
WHERE interval_unit = 'minute';

-- Five Minute Bars View
CREATE OR REPLACE VIEW five_minute_bars AS
SELECT 
    TIME_BUCKET('5 minutes', timestamp) AS timestamp,
    symbol,
    FIRST(open) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close) AS close,
    SUM(volume) AS volume,
    SUM(up_volume) AS up_volume,
    SUM(down_volume) AS down_volume,
    5 AS interval_value,
    'minute' AS interval_unit,
    source
FROM market_data
WHERE interval_unit = 'minute' AND interval_value = 1
GROUP BY TIME_BUCKET('5 minutes', timestamp), symbol, source;

-- Latest Prices View
CREATE OR REPLACE VIEW latest_prices AS
WITH ranked_prices AS (
    SELECT 
        symbol,
        timestamp,
        close,
        ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
    FROM market_data
)
SELECT symbol, timestamp, close 
FROM ranked_prices 
WHERE rn = 1;

-- Weekly Bars View
CREATE OR REPLACE VIEW weekly_bars AS
SELECT 
    DATE_TRUNC('week', timestamp) AS week_start,
    symbol,
    FIRST(open) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close) AS close,
    SUM(volume) AS volume,
    SUM(up_volume) AS up_volume,
    SUM(down_volume) AS down_volume,
    'weekly' AS interval_unit,
    1 AS interval_value,
    source
FROM market_data
WHERE interval_unit = 'daily' OR interval_value = 1440
GROUP BY DATE_TRUNC('week', timestamp), symbol, source;

-- Monthly Bars View
CREATE OR REPLACE VIEW monthly_bars AS
SELECT 
    DATE_TRUNC('month', timestamp) AS month_start,
    symbol,
    FIRST(open) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close) AS close,
    SUM(volume) AS volume,
    SUM(up_volume) AS up_volume,
    SUM(down_volume) AS down_volume,
    'monthly' AS interval_unit,
    1 AS interval_value,
    source
FROM market_data
WHERE interval_unit = 'daily' OR interval_value = 1440
GROUP BY DATE_TRUNC('month', timestamp), symbol, source;

-- Economic Calendar View
CREATE OR REPLACE VIEW economic_calendar AS
SELECT 
    e.timestamp,
    e.indicator,
    e.value,
    e.source,
    im.display_name,
    im.description,
    im.unit
FROM economic_data e
LEFT JOIN indicator_metadata im ON e.indicator = im.indicator_name;

-- Account Summary View
CREATE OR REPLACE VIEW account_summary AS
SELECT 
    ab.timestamp,
    ab.account_id,
    ab.cash_balance,
    ab.buying_power,
    ab.equity,
    ab.real_time_equity,
    COUNT(DISTINCT p.symbol) AS position_count,
    SUM(p.market_value) AS total_position_value,
    SUM(p.open_pl) AS total_open_pl,
    SUM(CASE WHEN p.open_pl > 0 THEN 1 ELSE 0 END) AS winning_positions,
    SUM(CASE WHEN p.open_pl < 0 THEN 1 ELSE 0 END) AS losing_positions
FROM account_balances ab
LEFT JOIN positions p ON ab.account_id = p.account_id AND DATE_TRUNC('day', ab.timestamp) = DATE_TRUNC('day', p.timestamp)
GROUP BY ab.timestamp, ab.account_id, ab.cash_balance, ab.buying_power, ab.equity, ab.real_time_equity;

-- Active Positions View
CREATE OR REPLACE VIEW active_positions AS
WITH latest_positions AS (
    SELECT 
        account_id,
        symbol,
        MAX(timestamp) AS max_timestamp
    FROM positions
    GROUP BY account_id, symbol
)
SELECT 
    p.*
FROM positions p
JOIN latest_positions lp ON 
    p.account_id = lp.account_id AND 
    p.symbol = lp.symbol AND 
    p.timestamp = lp.max_timestamp
WHERE p.quantity != 0;

-- Open Orders View
CREATE OR REPLACE VIEW open_orders AS
SELECT *
FROM orders
WHERE status IN ('Open', 'PartiallyFilled');

-- Symbol List View (for backward compatibility)
CREATE OR REPLACE VIEW symbol_list AS
SELECT DISTINCT symbol FROM market_data;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_economic_data_indicator ON economic_data(indicator);
CREATE INDEX IF NOT EXISTS idx_economic_data_timestamp ON economic_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_derived_indicators_name ON derived_indicators(indicator_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_account_balances_timestamp ON account_balances(timestamp, account_id);
CREATE INDEX IF NOT EXISTS idx_positions_account_symbol ON positions(account_id, symbol);
CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id, status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_account_symbol ON trades(account_id, symbol);