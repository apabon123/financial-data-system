-- ========================================================
-- DATA QUALITY ANALYSIS QUERIES
-- ========================================================
-- This file contains queries for analyzing the quality of
-- financial data, finding anomalies, and detecting potential errors.
-- ========================================================

-- 3.1 Basic quality metrics by symbol
-- ------------------------------------------------------------
-- Overview of quality measures for each symbol
SELECT 
    symbol,
    AVG(quality) as avg_quality,
    MIN(quality) as min_quality,
    MAX(quality) as max_quality,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE quality < 90) as low_quality_records,
    (COUNT(*) FILTER (WHERE quality < 90) * 100.0 / NULLIF(COUNT(*), 0)) as low_quality_percent
FROM market_data
GROUP BY symbol
ORDER BY low_quality_percent DESC;

-- 3.2 Find days with quality issues
-- ------------------------------------------------------------
-- Identify specific records with quality problems
SELECT 
    timestamp::DATE as date,
    symbol,
    quality,
    open, high, low, close, volume
FROM market_data
WHERE quality < 90  -- Adjust threshold as needed
AND interval_unit = 'day'
ORDER BY timestamp DESC, symbol;

-- 3.3 Check for missing OHLCV fields
-- ------------------------------------------------------------
-- Find records with missing critical data fields
SELECT 
    symbol,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE open IS NULL) as missing_open,
    COUNT(*) FILTER (WHERE high IS NULL) as missing_high,
    COUNT(*) FILTER (WHERE low IS NULL) as missing_low,
    COUNT(*) FILTER (WHERE close IS NULL) as missing_close,
    COUNT(*) FILTER (WHERE volume IS NULL) as missing_volume
FROM market_data
GROUP BY symbol
HAVING 
    COUNT(*) FILTER (WHERE open IS NULL) > 0 OR
    COUNT(*) FILTER (WHERE high IS NULL) > 0 OR
    COUNT(*) FILTER (WHERE low IS NULL) > 0 OR
    COUNT(*) FILTER (WHERE close IS NULL) > 0 OR
    COUNT(*) FILTER (WHERE volume IS NULL) > 0
ORDER BY (
    COUNT(*) FILTER (WHERE open IS NULL) + 
    COUNT(*) FILTER (WHERE high IS NULL) + 
    COUNT(*) FILTER (WHERE low IS NULL) + 
    COUNT(*) FILTER (WHERE close IS NULL) + 
    COUNT(*) FILTER (WHERE volume IS NULL)
) DESC;

-- 3.4 Detect unusual price movements (potential data errors)
-- ------------------------------------------------------------
-- Find suspiciously large price changes that may indicate data problems
-- Replace 'VXF24' with your symbol of interest
WITH price_changes AS (
    SELECT 
        symbol,
        timestamp,
        close,
        LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
        (close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp), 0) - 1) * 100 as pct_change
    FROM market_data
    WHERE interval_unit = 'day'
    AND symbol = 'VXF24'  -- Replace with your symbol
)
SELECT 
    timestamp,
    prev_close,
    close,
    pct_change
FROM price_changes
WHERE ABS(pct_change) > 10  -- Adjust threshold as needed
ORDER BY ABS(pct_change) DESC;

-- 3.5 Check OHLC consistency
-- ------------------------------------------------------------
-- Find records where OHLC values don't have proper relationships
-- (high should be >= open, close, and low; low should be <= open, close, and high)
SELECT 
    symbol,
    timestamp,
    open, high, low, close
FROM market_data
WHERE 
    high < open OR
    high < close OR
    low > open OR
    low > close OR
    high < low  -- This should never happen
ORDER BY timestamp DESC;

-- 3.6 Identify symbols with the most quality issues
-- ------------------------------------------------------------
-- Rank symbols by various quality metrics
SELECT 
    symbol,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE quality < 90) as low_quality_records,
    (COUNT(*) FILTER (WHERE quality < 90) * 100.0 / NULLIF(COUNT(*), 0)) as low_quality_percent,
    COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL) as missing_fields,
    COUNT(*) FILTER (WHERE high < open OR high < close OR low > open OR low > close OR high < low) as ohlc_inconsistencies
FROM market_data
GROUP BY symbol
HAVING 
    COUNT(*) FILTER (WHERE quality < 90) > 0 OR
    COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL) > 0 OR
    COUNT(*) FILTER (WHERE high < open OR high < close OR low > open OR low > close OR high < low) > 0
ORDER BY (
    (COUNT(*) FILTER (WHERE quality < 90) * 100.0 / NULLIF(COUNT(*), 0)) +
    (COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL) * 100.0 / NULLIF(COUNT(*), 0)) +
    (COUNT(*) FILTER (WHERE high < open OR high < close OR low > open OR low > close OR high < low) * 100.0 / NULLIF(COUNT(*), 0))
) DESC;

-- 3.7 Find zero volume trading days
-- ------------------------------------------------------------
-- Identify days when volume was reported as zero (potentially suspicious)
SELECT 
    symbol,
    timestamp,
    open, high, low, close, volume
FROM market_data
WHERE volume = 0
AND interval_unit = 'day'
ORDER BY timestamp DESC;

-- 3.8 Detect flat price days
-- ------------------------------------------------------------
-- Find days when price didn't change at all (potentially suspicious)
SELECT 
    symbol,
    timestamp,
    open, high, low, close, volume
FROM market_data
WHERE open = high AND high = low AND low = close  -- Completely flat prices
AND interval_unit = 'day'
AND volume > 0  -- There was actual trading
ORDER BY timestamp DESC;

-- 3.9 Find data sources with quality issues
-- ------------------------------------------------------------
-- Analyze data quality grouped by data source
SELECT 
    source,
    COUNT(*) as total_records,
    AVG(quality) as avg_quality,
    MIN(quality) as min_quality,
    COUNT(*) FILTER (WHERE quality < 90) as low_quality_records,
    (COUNT(*) FILTER (WHERE quality < 90) * 100.0 / NULLIF(COUNT(*), 0)) as low_quality_percent
FROM market_data
GROUP BY source
ORDER BY avg_quality ASC;

-- 3.10 Detect extreme price outliers
-- ------------------------------------------------------------
-- Find values that are statistically very far from normal for a symbol
-- This uses a simple Z-score approach for detection
WITH symbol_stats AS (
    SELECT 
        symbol,
        AVG(close) as avg_close,
        STDDEV(close) as stddev_close
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol
),
z_scores AS (
    SELECT 
        m.symbol,
        m.timestamp,
        m.close,
        (m.close - s.avg_close) / NULLIF(s.stddev_close, 0) as z_score
    FROM market_data m
    JOIN symbol_stats s ON m.symbol = s.symbol
    WHERE m.interval_unit = 'day'
)
SELECT 
    symbol,
    timestamp,
    close,
    z_score
FROM z_scores
WHERE ABS(z_score) > 3  -- Adjust threshold as needed (3 is ~99.7% confidence)
ORDER BY ABS(z_score) DESC;

-- 3.11 Volume anomaly detection
-- ------------------------------------------------------------
-- Find trading days with unusually high or low volume
WITH volume_stats AS (
    SELECT 
        symbol,
        AVG(volume) as avg_volume,
        STDDEV(volume) as stddev_volume
    FROM market_data
    WHERE interval_unit = 'day'
    GROUP BY symbol
),
z_scores AS (
    SELECT 
        m.symbol,
        m.timestamp,
        m.volume,
        (m.volume - v.avg_volume) / NULLIF(v.stddev_volume, 0) as volume_z_score
    FROM market_data m
    JOIN volume_stats v ON m.symbol = v.symbol
    WHERE m.interval_unit = 'day'
)
SELECT 
    symbol,
    timestamp,
    volume,
    volume_z_score
FROM z_scores
WHERE ABS(volume_z_score) > 3  -- Adjust threshold as needed
ORDER BY ABS(volume_z_score) DESC;

-- 3.12 Identify duplicated records
-- ------------------------------------------------------------
-- Find potentially duplicated data points
WITH duplicates AS (
    SELECT 
        symbol,
        timestamp::DATE,
        interval_unit,
        interval_value,
        COUNT(*) as record_count
    FROM market_data
    GROUP BY 
        symbol,
        timestamp::DATE,
        interval_unit,
        interval_value
    HAVING COUNT(*) > 1
)
SELECT 
    d.*,
    m.open, m.high, m.low, m.close, m.volume, m.source
FROM duplicates d
JOIN market_data m ON 
    d.symbol = m.symbol AND 
    d.timestamp::DATE = m.timestamp::DATE AND
    d.interval_unit = m.interval_unit AND
    d.interval_value = m.interval_value
ORDER BY 
    d.symbol,
    d.timestamp::DATE;
