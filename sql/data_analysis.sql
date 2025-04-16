-- ========================================================
-- DATA ANALYSIS QUERIES
-- ========================================================
-- This file contains queries for analyzing financial data,
-- including price movements, volume patterns, and market
-- statistics.
-- ========================================================

-- 2.1 Price Movement Analysis
-- ------------------------------------------------------------
-- Analyze price movements and volatility
WITH price_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        open,
        high,
        low,
        close,
        volume,
        (high - low) / NULLIF(open, 0) * 100 as daily_range_pct,
        (close - open) / NULLIF(open, 0) * 100 as daily_return_pct
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    daily_range_pct,
    daily_return_pct,
    volume,
    CASE 
        WHEN daily_return_pct > 0 THEN 'UP'
        WHEN daily_return_pct < 0 THEN 'DOWN'
        ELSE 'FLAT'
    END as direction
FROM price_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC, symbol;

-- 2.2 Volume Analysis
-- ------------------------------------------------------------
-- Analyze trading volume patterns
WITH volume_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        volume,
        up_volume,
        down_volume,
        CASE 
            WHEN up_volume > down_volume THEN 'UP'
            WHEN down_volume > up_volume THEN 'DOWN'
            ELSE 'NEUTRAL'
        END as volume_direction
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    volume,
    up_volume,
    down_volume,
    volume_direction,
    (up_volume - down_volume)::float / NULLIF(volume, 0) * 100 as volume_imbalance_pct
FROM volume_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC, symbol;

-- 2.3 Market Statistics
-- ------------------------------------------------------------
-- Calculate key market statistics
WITH market_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        close,
        volume,
        LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
        LAG(volume) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_volume
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    close,
    volume,
    (close - prev_close) / NULLIF(prev_close, 0) * 100 as price_change_pct,
    (volume - prev_volume)::float / NULLIF(prev_volume, 0) * 100 as volume_change_pct,
    CASE 
        WHEN close > prev_close AND volume > prev_volume THEN 'STRONG_UP'
        WHEN close > prev_close AND volume < prev_volume THEN 'WEAK_UP'
        WHEN close < prev_close AND volume > prev_volume THEN 'STRONG_DOWN'
        WHEN close < prev_close AND volume < prev_volume THEN 'WEAK_DOWN'
        ELSE 'NEUTRAL'
    END as market_condition
FROM market_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC, symbol;

-- 2.4 Price Distribution Analysis
-- ------------------------------------------------------------
-- Analyze price distribution and identify key levels
WITH price_distribution AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        close,
        NTILE(10) OVER (PARTITION BY symbol ORDER BY close) as price_decile
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    price_decile,
    MIN(close) as min_price,
    MAX(close) as max_price,
    AVG(close) as avg_price,
    COUNT(*) as frequency
FROM price_distribution
WHERE date >= current_date - interval '90 days'
GROUP BY symbol, price_decile
ORDER BY symbol, price_decile;

-- 2.5 Trading Session Analysis
-- ------------------------------------------------------------
-- Analyze trading patterns by session
WITH session_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        EXTRACT(HOUR FROM timestamp) as hour,
        open,
        high,
        low,
        close,
        volume
    FROM market_data
    WHERE interval_unit = 'minute'
)
SELECT 
    symbol,
    date,
    hour,
    COUNT(*) as trades,
    SUM(volume) as total_volume,
    AVG(close) as avg_price,
    MAX(high) - MIN(low) as price_range
FROM session_stats
WHERE date >= current_date - interval '7 days'
GROUP BY symbol, date, hour
ORDER BY date DESC, hour;

-- 2.6 Correlation Analysis
-- ------------------------------------------------------------
-- Calculate price correlations between symbols
WITH price_changes AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
        NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp), 0) as daily_return
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    a.symbol as symbol1,
    b.symbol as symbol2,
    CORR(a.daily_return, b.daily_return) as correlation
FROM price_changes a
JOIN price_changes b ON a.date = b.date AND a.symbol < b.symbol
WHERE a.date >= current_date - interval '90 days'
GROUP BY a.symbol, b.symbol
HAVING COUNT(*) > 20  -- Minimum number of data points for correlation
ORDER BY ABS(correlation) DESC;

-- 2.7 Volatility Analysis
-- ------------------------------------------------------------
-- Calculate rolling volatility metrics
WITH returns AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        LN(close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp), 0)) as log_return
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    log_return,
    STDDEV(log_return) OVER (
        PARTITION BY symbol 
        ORDER BY date 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) * SQRT(252) as annualized_volatility
FROM returns
WHERE date >= current_date - interval '90 days'
ORDER BY date DESC, symbol;

-- 2.8 Market Breadth Analysis
-- ------------------------------------------------------------
-- Analyze market breadth indicators
WITH daily_stats AS (
    SELECT 
        date_trunc('day', timestamp) as date,
        COUNT(DISTINCT symbol) as total_symbols,
        COUNT(DISTINCT CASE WHEN close > open THEN symbol END) as advancing_symbols,
        COUNT(DISTINCT CASE WHEN close < open THEN symbol END) as declining_symbols,
        COUNT(DISTINCT CASE WHEN close = open THEN symbol END) as unchanged_symbols
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    date,
    total_symbols,
    advancing_symbols,
    declining_symbols,
    unchanged_symbols,
    (advancing_symbols::float / NULLIF(total_symbols, 0) * 100) as advance_decline_ratio
FROM daily_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC;

-- 2.9 Price Momentum Analysis
-- ------------------------------------------------------------
-- Calculate momentum indicators
WITH momentum_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        close,
        LAG(close, 5) OVER (PARTITION BY symbol ORDER BY timestamp) as price_5d_ago,
        LAG(close, 10) OVER (PARTITION BY symbol ORDER BY timestamp) as price_10d_ago,
        LAG(close, 20) OVER (PARTITION BY symbol ORDER BY timestamp) as price_20d_ago
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    close,
    (close - price_5d_ago) / NULLIF(price_5d_ago, 0) * 100 as momentum_5d,
    (close - price_10d_ago) / NULLIF(price_10d_ago, 0) * 100 as momentum_10d,
    (close - price_20d_ago) / NULLIF(price_20d_ago, 0) * 100 as momentum_20d
FROM momentum_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC, symbol;

-- 2.10 Market Depth Analysis
-- ------------------------------------------------------------
-- Analyze market depth and liquidity
WITH depth_stats AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        volume,
        (high - low) / NULLIF(close, 0) * 100 as spread_pct,
        CASE 
            WHEN volume > LAG(volume) OVER (PARTITION BY symbol ORDER BY timestamp) THEN 'INCREASING'
            WHEN volume < LAG(volume) OVER (PARTITION BY symbol ORDER BY timestamp) THEN 'DECREASING'
            ELSE 'STABLE'
        END as volume_trend
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    date,
    volume,
    spread_pct,
    volume_trend,
    CASE 
        WHEN spread_pct < 1 THEN 'TIGHT'
        WHEN spread_pct < 2 THEN 'NORMAL'
        ELSE 'WIDE'
    END as spread_condition
FROM depth_stats
WHERE date >= current_date - interval '30 days'
ORDER BY date DESC, symbol; 