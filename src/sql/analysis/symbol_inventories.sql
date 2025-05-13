-- ========================================================
-- SYMBOL INVENTORY QUERIES
-- ========================================================
-- This file contains queries for analyzing the symbols in your
-- financial database, their metadata, and coverage information.
-- ========================================================

-- 1.1 Get a complete list of all unique symbols in the database
-- ------------------------------------------------------------
-- Use this to get a quick overview of what symbols exist
SELECT DISTINCT symbol
FROM market_data
ORDER BY symbol;

-- 1.2 Get basic metadata for all symbols
-- ------------------------------------------------------------
-- Shows first/last dates, record counts and date range for each symbol
SELECT 
    symbol,
    strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
    strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
    COUNT(*) as record_count,
    DATEDIFF('day', MIN(timestamp)::TIMESTAMP, MAX(timestamp)::TIMESTAMP) as date_range_days
FROM market_data
GROUP BY symbol
ORDER BY MAX(timestamp) DESC, symbol;

-- 1.3 Find symbols that haven't been updated recently
-- ------------------------------------------------------------
-- Identifies symbols that may need data refreshing
SELECT 
    symbol,
    strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
    strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
    COUNT(*) as record_count,
    DATEDIFF('day', MIN(timestamp)::TIMESTAMP, MAX(timestamp)::TIMESTAMP) as date_range_days
FROM market_data
GROUP BY symbol
HAVING DATEDIFF('day', MAX(timestamp)::TIMESTAMP, CURRENT_DATE) > 7  -- Symbols not updated in 7+ days
ORDER BY MAX(timestamp) DESC, symbol;

-- 1.4 Find symbols with most recent data
-- ------------------------------------------------------------
-- Shows which symbols have the most up-to-date data
SELECT 
    symbol,
    strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
    COUNT(*) as record_count
FROM market_data
GROUP BY symbol
ORDER BY MAX(timestamp) DESC
LIMIT 20;  -- Adjust limit as needed

-- 1.5 Get symbol counts by data interval
-- ------------------------------------------------------------
-- Shows what types of data intervals you have for symbols
SELECT 
    interval_unit,
    interval_value,
    COUNT(DISTINCT symbol) as symbol_count,
    COUNT(*) as record_count
FROM market_data
GROUP BY interval_unit, interval_value
ORDER BY interval_unit, interval_value;

-- 1.6 Get symbol metadata by interval type
-- ------------------------------------------------------------
-- Details about each symbol's data at specific intervals
SELECT 
    symbol, 
    interval_unit,
    interval_value,
    strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
    strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
    COUNT(*) as record_count
FROM market_data
WHERE interval_unit = 'day'  -- Change to desired interval
GROUP BY symbol, interval_unit, interval_value
ORDER BY symbol, interval_unit, interval_value;

-- 1.7 Find symbols matching a specific pattern
-- ------------------------------------------------------------
-- Useful for finding related symbols (futures, options, etc.)
-- Adjust the LIKE pattern as needed
SELECT DISTINCT symbol
FROM market_data
WHERE symbol LIKE 'VX%'  -- Change pattern as needed
ORDER BY symbol;

-- 1.8 Compare data coverage between two time periods
-- ------------------------------------------------------------
-- Useful for finding symbols with data gaps in specific periods
WITH symbols_list AS (
    SELECT DISTINCT symbol FROM market_data
),
period1 AS (
    SELECT 
        symbol,
        COUNT(*) as p1_count
    FROM market_data
    WHERE timestamp BETWEEN '2023-01-01' AND '2023-06-30'  -- Adjust dates
    AND interval_unit = 'day'
    GROUP BY symbol
),
period2 AS (
    SELECT 
        symbol,
        COUNT(*) as p2_count
    FROM market_data
    WHERE timestamp BETWEEN '2023-07-01' AND '2023-12-31'  -- Adjust dates
    AND interval_unit = 'day'
    GROUP BY symbol
)
SELECT 
    s.symbol,
    COALESCE(p1.p1_count, 0) as period1_count,
    COALESCE(p2.p2_count, 0) as period2_count,
    COALESCE(p2.p2_count, 0) - COALESCE(p1.p1_count, 0) as count_difference
FROM symbols_list s
LEFT JOIN period1 p1 ON s.symbol = p1.symbol
LEFT JOIN period2 p2 ON s.symbol = p2.symbol
ORDER BY count_difference;

-- 1.9 Find periods with highest data density
-- ------------------------------------------------------------
-- Shows which months have the most complete data coverage
SELECT 
    strftime(DATE_TRUNC('month', timestamp)::TIMESTAMP, '%Y-%m-%d') as month,
    COUNT(DISTINCT symbol) as symbol_count,
    COUNT(*) as record_count
FROM market_data
WHERE interval_unit = 'day'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY COUNT(DISTINCT symbol) DESC;

-- 1.10 Summarize continuous contracts
-- ------------------------------------------------------------
-- Get metadata for continuous contract symbols, broken down by interval
SELECT 
    symbol,
    interval_unit,
    interval_value,
    strftime(MIN(timestamp)::TIMESTAMP, '%Y-%m-%d') as first_date,
    strftime(MAX(timestamp)::TIMESTAMP, '%Y-%m-%d') as last_date,
    COUNT(*) as record_count,
    DATEDIFF('day', MIN(timestamp)::TIMESTAMP, MAX(timestamp)::TIMESTAMP) as date_range_days
FROM continuous_contracts
GROUP BY symbol, interval_unit, interval_value
ORDER BY symbol, interval_unit, interval_value;