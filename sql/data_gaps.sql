-- ========================================================
-- DATA GAPS ANALYSIS QUERIES
-- ========================================================
-- This file contains queries for identifying and analyzing gaps
-- in time series data for financial symbols.
-- ========================================================

-- 2.1 Basic gap detection for a specific symbol
-- ------------------------------------------------------------
-- Simple detection of missing days in daily data
-- Replace 'VXF24' with your symbol of interest
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
    AND symbol = 'VXF24'  -- Replace with your symbol
)
SELECT 
    symbol,
    date as gap_start,
    next_date as gap_end,
    (next_date - date - 1) as gap_days
FROM date_series
WHERE (next_date - date) > 1  -- More than 1 day between records
ORDER BY date;

-- 2.2 Gap detection with weekend filtering
-- ------------------------------------------------------------
-- More intelligent gap detection that excludes weekends
-- Replace 'VXF24' with your symbol of interest
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
    AND symbol = 'VXF24'  -- Replace with your symbol
),
filtered_gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as calendar_days,
        (
            -- Count business days in the gap (exclude weekends)
            (next_date - date - 1) - 
            -- Subtract weekends (count Saturdays in range)
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) = 6) -
            -- Subtract weekends (count Sundays in range)
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) = 0)
        ) as business_days
    FROM date_series
    WHERE (next_date - date) > 1  -- More than 1 day between records
)
SELECT *
FROM filtered_gaps
WHERE business_days > 0  -- Only show gaps in business days
ORDER BY gap_start;

-- 2.3 Advanced gap detection with holiday calendar
-- ------------------------------------------------------------
-- First create temporary table with holidays (use dates relevant to your data)
CREATE TEMP TABLE IF NOT EXISTS us_market_holidays (
    holiday_date DATE PRIMARY KEY
);

-- Insert major US market holidays (adjust years as needed)
-- These are common US market holidays
DELETE FROM us_market_holidays;
INSERT INTO us_market_holidays VALUES
    ('2023-01-02'), -- New Year's Day (observed)
    ('2023-01-16'), -- MLK Day
    ('2023-02-20'), -- Presidents Day
    ('2023-04-07'), -- Good Friday
    ('2023-05-29'), -- Memorial Day
    ('2023-06-19'), -- Juneteenth
    ('2023-07-04'), -- Independence Day
    ('2023-09-04'), -- Labor Day
    ('2023-11-23'), -- Thanksgiving
    ('2023-12-25'), -- Christmas
    ('2024-01-01'), -- New Year's Day
    ('2024-01-15'), -- MLK Day
    ('2024-02-19'), -- Presidents Day
    ('2024-03-29'), -- Good Friday
    ('2024-05-27'), -- Memorial Day
    ('2024-06-19'), -- Juneteenth
    ('2024-07-04'), -- Independence Day
    ('2024-09-02'), -- Labor Day
    ('2024-11-28'), -- Thanksgiving
    ('2024-12-25'); -- Christmas

-- Then detect gaps accounting for holidays and weekends
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
    AND symbol = 'VXF24'  -- Replace with your symbol
),
filtered_gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as raw_gap_days,
        (
            -- Count business days in the gap (exclude weekends)
            (next_date - date - 1) - 
            -- Subtract weekends (count Saturdays and Sundays in range)
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) IN (0, 6)) -
            -- Subtract holidays
            (SELECT COUNT(*) FROM us_market_holidays 
             WHERE holiday_date BETWEEN date + INTERVAL '1 day' AND next_date - INTERVAL '1 day')
        ) as business_day_gap
    FROM date_series
    WHERE (next_date - date) > 1  -- More than 1 day between records
)
SELECT 
    symbol,
    gap_start,
    gap_end,
    raw_gap_days as calendar_days,
    business_day_gap as business_days
FROM filtered_gaps
WHERE business_day_gap > 0  -- Only show gaps in business days
ORDER BY gap_start;

-- 2.4 Find symbols with the most gaps
-- ------------------------------------------------------------
-- Identify which symbols have the most missing data
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
),
gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as gap_days,
        (
            -- Count business days in the gap (exclude weekends)
            (next_date - date - 1) - 
            -- Subtract weekends
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) IN (0, 6))
        ) as business_day_gap
    FROM date_series
    WHERE (next_date - date) > 1  -- More than 1 day between records
)
SELECT 
    symbol,
    COUNT(*) as gap_count,
    SUM(business_day_gap) as total_missing_days,
    AVG(business_day_gap) as avg_gap_length,
    MAX(business_day_gap) as max_gap_length
FROM gaps
WHERE business_day_gap > 0
GROUP BY symbol
ORDER BY gap_count DESC;

-- 2.5 Find gaps by time period
-- ------------------------------------------------------------
-- Check for gaps in a specific date range
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
    AND timestamp BETWEEN '2023-01-01' AND '2023-12-31'  -- Adjust date range
),
gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as gap_days
    FROM date_series
    WHERE (next_date - date) > 1  -- More than 1 day between records
)
SELECT 
    symbol,
    gap_start,
    gap_end,
    gap_days
FROM gaps
ORDER BY gap_start;

-- 2.6 Gap analysis for continuous contracts
-- ------------------------------------------------------------
-- Identify gaps in continuous contract data
WITH date_series AS (
    SELECT 
        symbol,
        date,
        LEAD(date) OVER (PARTITION BY symbol ORDER BY date) as next_date
    FROM continuous_contracts
    WHERE symbol = 'VXc1'  -- Replace with your continuous contract symbol
),
gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as gap_days,
        (
            -- Count business days in the gap (exclude weekends)
            (next_date - date - 1) - 
            -- Subtract weekends
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) IN (0, 6))
        ) as business_day_gap
    FROM date_series
    WHERE (next_date - date) > 1  -- More than 1 day between records
)
SELECT 
    symbol,
    gap_start,
    gap_end,
    gap_days,
    business_day_gap
FROM gaps
WHERE business_day_gap > 0
ORDER BY gap_start;

-- 2.7 Compare data completeness between symbols
-- ------------------------------------------------------------
-- Shows which symbols have the most complete data
WITH date_range AS (
    -- Get the full date range from all symbols
    SELECT 
        MIN(timestamp::DATE) as min_date,
        MAX(timestamp::DATE) as max_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
),
business_days AS (
    -- Generate all business days in the range
    SELECT d::DATE as business_date
    FROM generate_series(
        (SELECT min_date FROM date_range),
        (SELECT max_date FROM date_range),
        INTERVAL '1 day'
    ) d
    WHERE EXTRACT(DOW FROM d) NOT IN (0, 6)  -- Exclude weekends
    AND d NOT IN (SELECT holiday_date FROM us_market_holidays)  -- Exclude holidays
),
symbol_coverage AS (
    -- Calculate coverage for each symbol
    SELECT 
        m.symbol,
        COUNT(DISTINCT m.timestamp::DATE) as actual_days,
        (SELECT COUNT(*) FROM business_days) as possible_days,
        COUNT(DISTINCT m.timestamp::DATE) * 100.0 / NULLIF((SELECT COUNT(*) FROM business_days), 0) as coverage_percent
    FROM market_data m
    WHERE interval_unit = 'day' AND interval_value = 1
    GROUP BY m.symbol
)
SELECT *
FROM symbol_coverage
ORDER BY coverage_percent DESC;

-- 2.8 Find symbols with large gaps in recent data
-- ------------------------------------------------------------
-- Identify symbols with significant gaps in the most recent period
WITH date_series AS (
    SELECT 
        symbol,
        timestamp::DATE as date,
        LEAD(timestamp::DATE) OVER (PARTITION BY symbol ORDER BY timestamp) as next_date
    FROM market_data
    WHERE interval_unit = 'day' AND interval_value = 1
    AND timestamp > CURRENT_DATE - INTERVAL '90 days'  -- Last 90 days
),
recent_gaps AS (
    SELECT 
        symbol,
        date as gap_start,
        next_date as gap_end,
        (next_date - date - 1) as gap_days,
        (
            -- Calculate business days
            (next_date - date - 1) - 
            (SELECT COUNT(*) FROM generate_series(date + INTERVAL '1 day', next_date - INTERVAL '1 day', INTERVAL '1 day') d 
             WHERE EXTRACT(DOW FROM d) IN (0, 6))
        ) as business_day_gap
    FROM date_series
    WHERE (next_date - date) > 1
)
SELECT 
    symbol,
    gap_start,
    gap_end,
    gap_days,
    business_day_gap
FROM recent_gaps
WHERE business_day_gap > 0
ORDER BY business_day_gap DESC;









