-- Market Data Table
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    up_volume BIGINT,
    down_volume BIGINT,
    source VARCHAR,
    PRIMARY KEY (timestamp, symbol)
);

-- Economic Data Table
CREATE TABLE IF NOT EXISTS economic_data (
    timestamp TIMESTAMP NOT NULL,
    indicator VARCHAR NOT NULL,
    value DOUBLE,
    source VARCHAR,
    PRIMARY KEY (timestamp, indicator)
);

-- Create useful views
CREATE VIEW IF NOT EXISTS daily_bars AS
SELECT 
    DATE_TRUNC('day', timestamp) AS date,
    symbol,
    FIRST(open) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close) AS close,
    SUM(volume) AS volume
FROM market_data
GROUP BY date, symbol;

-- Create symbols view
CREATE VIEW IF NOT EXISTS symbol_list AS
SELECT DISTINCT symbol FROM market_data;