-- ========================================================
-- CONTINUOUS CONTRACT ANALYSIS QUERIES
-- ========================================================
-- This file contains queries for analyzing continuous futures
-- contracts, detecting rollover issues, and checking data quality
-- in derived continuous contracts.
-- ========================================================

-- 4.1 Continuous contract overview
-- ------------------------------------------------------------
-- Basic metadata about all continuous contracts
SELECT 
    symbol,
    -- Format date as YYYY-MM-DD
    strftime(MIN(date)::TIMESTAMP, '%Y-%m-%d') AS first_date,
    strftime(MAX(date)::TIMESTAMP, '%Y-%m-%d') AS last_date,
    COUNT(*) as record_count,
    DATEDIFF('day', MIN(date)::TIMESTAMP, MAX(date)::TIMESTAMP) as date_range_days
FROM continuous_contracts
GROUP BY symbol
ORDER BY symbol;

-- 4.2 Detect gaps in continuous contracts
-- ------------------------------------------------------------
-- Identify missing days in continuous contract data
-- Replace 'VXc1' with your continuous contract symbol
WITH date_series AS (
    SELECT 
        symbol,
        date::TIMESTAMP as date_ts,
        LEAD(date::TIMESTAMP) OVER (PARTITION BY symbol ORDER BY date) as next_date_ts
    FROM continuous_contracts
    WHERE symbol = 'VXc1'
)
SELECT 
    symbol,
    strftime(date_ts, '%Y-%m-%d') as gap_start,
    strftime(next_date_ts, '%Y-%m-%d') as gap_end,
    DATEDIFF('day', date_ts, next_date_ts) as days_between,
    DATEDIFF('day', date_ts, next_date_ts) - 1 as gap_days
FROM date_series
WHERE next_date_ts IS NOT NULL
  AND DATEDIFF('day', date_ts, next_date_ts) > 1
ORDER BY date_ts;
-- 4.3 Detect potential rollover issues
-- ------------------------------------------------------------
-- Find days with unusual price changes that might indicate rollover problems
-- Replace 'VXc1' with your continuous contract symbol
WITH day_changes AS (
    SELECT 
        symbol,
        strftime(date::TIMESTAMP, '%Y-%m-%d') as formatted_date,
        date,
        close,
        LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
        (close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) - 1) * 100 as pct_change
    FROM continuous_contracts
    WHERE symbol = 'VXc1'  -- Replace with your symbol
)
SELECT 
    formatted_date,
    prev_close,
    close,
    pct_change
FROM day_changes
WHERE ABS(pct_change) > 5  -- Adjust threshold as needed
ORDER BY ABS(pct_change) DESC;

-- 4.4 Identify actual rollover dates
-- ------------------------------------------------------------
-- Find days when the underlying futures contract actually changed
-- This assumes the continuous contract symbol follows a pattern like 'XXXc1'
-- and the underlying contracts follow a pattern like 'XXXMYY' (e.g., ESH24)
-- Adjust the subqueries based on your actual symbol naming conventions
WITH contract_changes AS (
    SELECT 
        c.date,
        c.symbol as continuous_symbol,
        c.close as continuous_close,
        (
            -- Get the actual futures contract that would be active on this date
            -- This is a simplified approach - adjust based on your actual data
            SELECT symbol
            FROM market_data
            WHERE symbol LIKE REPLACE(c.symbol, 'c1', '') || '%'  -- e.g., 'VXc1'