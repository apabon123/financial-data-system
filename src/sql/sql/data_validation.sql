-- ========================================================
-- DATA VALIDATION QUERIES
-- ========================================================
-- This file contains queries for validating data integrity,
-- checking for anomalies, and ensuring data quality.
-- ========================================================

-- 3.1 Data Completeness Check
-- ------------------------------------------------------------
-- Check for missing data points in time series
WITH time_series AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        COUNT(*) as records_per_day
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol, date_trunc('day', timestamp)
)
SELECT 
    symbol,
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(*) as total_days,
    COUNT(*) FILTER (WHERE records_per_day = 0) as missing_days,
    COUNT(*) FILTER (WHERE records_per_day > 1) as duplicate_days
FROM time_series
GROUP BY symbol
HAVING COUNT(*) FILTER (WHERE records_per_day = 0) > 0
   OR COUNT(*) FILTER (WHERE records_per_day > 1) > 0
ORDER BY symbol;

-- 3.2 Price Validity Check
-- ------------------------------------------------------------
-- Check for invalid price relationships
SELECT 
    symbol,
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    CASE 
        WHEN high < low THEN 'HIGH_LOW_ERROR'
        WHEN high < open THEN 'HIGH_OPEN_ERROR'
        WHEN high < close THEN 'HIGH_CLOSE_ERROR'
        WHEN low > open THEN 'LOW_OPEN_ERROR'
        WHEN low > close THEN 'LOW_CLOSE_ERROR'
        ELSE 'VALID'
    END as error_type
FROM market_data
WHERE 
    high < low OR
    high < open OR
    high < close OR
    low > open OR
    low > close
ORDER BY timestamp DESC;

-- 3.3 Volume Consistency Check
-- ------------------------------------------------------------
-- Check for volume inconsistencies
WITH volume_check AS (
    SELECT 
        symbol,
        timestamp,
        volume,
        up_volume,
        down_volume,
        CASE 
            WHEN up_volume + down_volume != volume THEN 'VOLUME_MISMATCH'
            WHEN volume < 0 THEN 'NEGATIVE_VOLUME'
            WHEN up_volume < 0 OR down_volume < 0 THEN 'NEGATIVE_COMPONENT'
            ELSE 'VALID'
        END as error_type
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    timestamp,
    volume,
    up_volume,
    down_volume,
    error_type
FROM volume_check
WHERE error_type != 'VALID'
ORDER BY timestamp DESC;

-- 3.4 Time Continuity Check
-- ------------------------------------------------------------
-- Check for gaps in time series
WITH time_gaps AS (
    SELECT 
        symbol,
        timestamp,
        LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as next_timestamp
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    timestamp,
    next_timestamp,
    EXTRACT(EPOCH FROM (next_timestamp - timestamp)) / 86400 as days_between
FROM time_gaps
WHERE next_timestamp - timestamp > interval '1 day'
ORDER BY timestamp DESC;

-- 3.5 Data Quality Score
-- ------------------------------------------------------------
-- Calculate data quality scores for each symbol
WITH quality_metrics AS (
    SELECT 
        symbol,
        COUNT(*) as total_records,
        COUNT(*) FILTER (WHERE quality < 90) as low_quality_records,
        COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL) as missing_prices,
        COUNT(*) FILTER (WHERE volume IS NULL) as missing_volume,
        COUNT(*) FILTER (WHERE high < low OR high < open OR high < close OR low > open OR low > close) as price_errors
    FROM market_data
    WHERE interval_unit = 'day'
)
SELECT 
    symbol,
    total_records,
    low_quality_records,
    missing_prices,
    missing_volume,
    price_errors,
    (100 - 
     (low_quality_records::float / NULLIF(total_records, 0) * 30) -
     (missing_prices::float / NULLIF(total_records, 0) * 30) -
     (missing_volume::float / NULLIF(total_records, 0) * 20) -
     (price_errors::float / NULLIF(total_records, 0) * 20)
    ) as quality_score
FROM quality_metrics
ORDER BY quality_score;

-- 3.6 Source Consistency Check
-- ------------------------------------------------------------
-- Check for inconsistencies between data sources
WITH source_comparison AS (
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        source,
        COUNT(*) as records,
        AVG(close) as avg_close,
        STDDEV(close) as stddev_close
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol, date_trunc('day', timestamp), source
)
SELECT 
    symbol,
    date,
    source,
    records,
    avg_close,
    stddev_close
FROM source_comparison
WHERE stddev_close > 0.01  -- Adjust threshold as needed
ORDER BY stddev_close DESC;

-- 3.7 Timestamp Validity Check
-- ------------------------------------------------------------
-- Check for invalid timestamps
SELECT 
    symbol,
    timestamp,
    EXTRACT(DOW FROM timestamp) as day_of_week,
    EXTRACT(HOUR FROM timestamp) as hour,
    CASE 
        WHEN EXTRACT(DOW FROM timestamp) IN (0, 6) THEN 'WEEKEND'
        WHEN EXTRACT(HOUR FROM timestamp) < 9 OR EXTRACT(HOUR FROM timestamp) > 16 THEN 'NON_MARKET_HOURS'
        ELSE 'VALID'
    END as timestamp_status
FROM market_data
WHERE 
    EXTRACT(DOW FROM timestamp) IN (0, 6) OR
    (EXTRACT(HOUR FROM timestamp) < 9 OR EXTRACT(HOUR FROM timestamp) > 16)
ORDER BY timestamp DESC;

-- 3.8 Data Freshness Check
-- ------------------------------------------------------------
-- Check how recent the data is for each symbol
WITH latest_data AS (
    SELECT 
        symbol,
        MAX(timestamp) as last_update,
        COUNT(*) FILTER (WHERE timestamp >= current_timestamp - interval '1 day') as records_last_24h,
        COUNT(*) FILTER (WHERE timestamp >= current_timestamp - interval '7 days') as records_last_7d
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol
)
SELECT 
    symbol,
    last_update,
    records_last_24h,
    records_last_7d,
    CASE 
        WHEN last_update >= current_timestamp - interval '1 day' THEN 'CURRENT'
        WHEN last_update >= current_timestamp - interval '7 days' THEN 'RECENT'
        ELSE 'STALE'
    END as data_status
FROM latest_data
ORDER BY last_update DESC;

-- 3.9 Symbol Consistency Check
-- ------------------------------------------------------------
-- Check for inconsistencies in symbol naming
SELECT 
    symbol,
    COUNT(DISTINCT interval_unit) as interval_types,
    COUNT(DISTINCT source) as source_count,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen,
    COUNT(*) as total_records
FROM market_data
GROUP BY symbol
HAVING COUNT(DISTINCT interval_unit) > 1
   OR COUNT(DISTINCT source) > 1
ORDER BY symbol;

-- 3.10 Data Completeness Report
-- ------------------------------------------------------------
-- Generate a comprehensive data completeness report
WITH completeness_metrics AS (
    SELECT 
        symbol,
        COUNT(*) as total_records,
        COUNT(DISTINCT date_trunc('day', timestamp)) as trading_days,
        COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL) as missing_prices,
        COUNT(*) FILTER (WHERE volume IS NULL) as missing_volume,
        COUNT(*) FILTER (WHERE quality < 90) as low_quality_records,
        MIN(timestamp) as first_record,
        MAX(timestamp) as last_record
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol
)
SELECT 
    symbol,
    total_records,
    trading_days,
    missing_prices,
    missing_volume,
    low_quality_records,
    first_record,
    last_record,
    (missing_prices::float / NULLIF(total_records, 0) * 100) as missing_prices_pct,
    (missing_volume::float / NULLIF(total_records, 0) * 100) as missing_volume_pct,
    (low_quality_records::float / NULLIF(total_records, 0) * 100) as low_quality_pct
FROM completeness_metrics
ORDER BY symbol; 