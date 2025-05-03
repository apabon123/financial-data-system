# Query Examples for Market Data

This document provides examples of how to query the market data tables to retrieve various types of data.

**Important Note:** Market data is stored in two primary tables:
*   `market_data`: Contains data for most instruments (e.g., ES, NQ) typically sourced from TradeStation or other non-CBOE sources.
*   `market_data_cboe`: Contains *only* daily VIX futures (`VX%` symbols) and VIX Index (`$VIX.X`) data downloaded directly from CBOE. This table has fewer columns (no volume/open interest).

Choose the correct table based on the symbol you are querying.

## Basic Queries

### Get the most recent data for a symbol (Non-VIX Future)

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
ORDER BY timestamp DESC
LIMIT 1;
```

### Get the most recent data for a symbol (VIX Future)

```sql
SELECT * FROM market_data_cboe
WHERE symbol = 'VXK25'
ORDER BY timestamp DESC
LIMIT 1;
```

### Get all data for a specific date (from both tables)

```sql
-- Query market_data for non-VIX CBOE symbols (none in this case, but pattern shown)
SELECT * FROM market_data
WHERE date = '2023-04-22'
  AND source != 'CBOE' -- Exclude any potential CBOE data here
UNION ALL
-- Query market_data_cboe for VIX symbols and VIX index
SELECT 
    timestamp, symbol, open, high, low, 
    NULL as close, -- market_data_cboe doesn't have 'close' column
    settle, 
    NULL as volume, -- No volume
    NULL as open_interest, -- No OI
    interval_value, interval_unit, source, 
    strftime(timestamp, '%Y-%m-%d') as date, -- Recreate date string
    NULL as up_volume, NULL as down_volume, -- No tick volume
    FALSE as changed, -- Assume FALSE
    FALSE as adjusted, -- Assume FALSE
    100 as quality -- Assume 100
FROM market_data_cboe
WHERE strftime(timestamp, '%Y-%m-%d') = '2023-04-22' -- Filter by date string
ORDER BY symbol, timestamp;
```

### Get data for multiple symbols (mixed sources)

```sql
-- Example: ES future (market_data), VIX future (market_data_cboe), VIX Index (market_data_cboe)
SELECT timestamp, symbol, open, high, low, close, settle, volume, source
FROM market_data
WHERE symbol = 'ESH26' -- Only non-CBOE symbols here
  AND date = '2023-04-22'
UNION ALL 
SELECT timestamp, symbol, open, high, low, NULL as close, settle, NULL as volume, source
FROM market_data_cboe
WHERE symbol IN ('VXK25', '$VIX.X') -- Specify CBOE symbols here
  AND strftime(timestamp, '%Y-%m-%d') = '2023-04-22'
ORDER BY symbol, timestamp;
```

## Data Source Specific Queries

### Get all TradeStation data

```sql
SELECT * FROM market_data
WHERE source = 'tradestation'
ORDER BY symbol, timestamp DESC
LIMIT 100;
```

### Get all CBOE data (VIX Futures & Index)

```sql
SELECT * FROM market_data_cboe
-- WHERE source = 'CBOE' -- Condition redundant as this table only has CBOE data
ORDER BY symbol, timestamp DESC
LIMIT 100;
```

## Timeframe Specific Queries

### Get daily data for a specific symbol and date range (Non-VIX Future)

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'daily'
AND date BETWEEN '2023-01-01' AND '2023-04-22'
ORDER BY timestamp;
```

### Get daily data for a specific symbol and date range (VIX Future)

```sql
SELECT * FROM market_data_cboe
WHERE symbol = 'VXK25'
AND interval_unit = 'daily' -- Implicitly daily
AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22' -- Filter on timestamp::DATE
ORDER BY timestamp;
```

### Get minute data for a specific symbol and time range (Non-VIX Future)

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'minute'
AND timestamp BETWEEN '2023-04-22 09:30:00' AND '2023-04-22 16:00:00'
ORDER BY timestamp;
```

## Aggregation Queries

### Get the average daily volume for a symbol by month (Non-VIX Future)

```sql
SELECT 
    DATE_TRUNC('month', timestamp) AS month,
    AVG(volume) AS avg_daily_volume
FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'daily'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month;
```

### Get the highest and lowest prices by week (VIX Future)

```sql
SELECT 
    DATE_TRUNC('week', timestamp) AS week,
    MAX(high) AS highest_price,
    MIN(low) AS lowest_price
FROM market_data_cboe
WHERE symbol = 'VXK25'
GROUP BY DATE_TRUNC('week', timestamp)
ORDER BY week;
```

## Combining Multiple Symbols

### Compare ES and NQ futures closing prices

```sql
-- (This query remains the same as ES/NQ are in market_data)
WITH es_data AS (
    SELECT timestamp::DATE as date, close AS es_close
    FROM market_data
    WHERE symbol LIKE 'ES%'
    AND interval_unit = 'daily'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
),
nq_data AS (
    SELECT timestamp::DATE as date, close AS nq_close
    FROM market_data
    WHERE symbol LIKE 'NQ%'
    AND interval_unit = 'daily'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
)
SELECT 
    es_data.date,
    es_data.es_close,
    nq_data.nq_close,
    nq_data.nq_close / es_data.es_close AS ratio
FROM es_data
JOIN nq_data ON es_data.date = nq_data.date
ORDER BY es_data.date;
```

### Compare VIX index with VIX futures

```sql
-- Both VIX Index and VIX Futures are now in market_data_cboe
WITH vix_data AS (
    SELECT timestamp::DATE as date, settle AS vix_settle -- Use settle (mapped from CLOSE)
    FROM market_data_cboe 
    WHERE symbol = '$VIX.X'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
),
vx_futures AS (
    SELECT timestamp::DATE as date, settle AS vx_settle
    FROM market_data_cboe
    WHERE symbol LIKE 'VX%'
    AND timestamp::DATE BETWEEN '2023-01-01' AND '2023-04-22'
)
SELECT 
    vix_data.date,
    vix_data.vix_settle AS vix_index_settle,
    vx_futures.vx_settle AS vx_future_settle,
    vx_futures.vx_settle - vix_data.vix_settle AS basis
FROM vix_data
JOIN vx_futures ON vix_data.date = vx_futures.date
ORDER BY vix_data.date;
```

## Finding Data Quality Issues

### Find records that have been changed or filled (in market_data)

```sql
SELECT * FROM market_data
WHERE changed = true
ORDER BY timestamp DESC
LIMIT 100;
```

### Find records with lower quality scores (in market_data)

```sql
SELECT * FROM market_data
WHERE quality < 100
ORDER BY quality ASC, timestamp DESC
LIMIT 100;
```

## Exporting Data

### Export daily bars for a specific symbol to CSV (VIX Future)

```sql
COPY (
    SELECT 
        timestamp::DATE as date, 
        open, 
        high, 
        low, 
        settle -- Use settle for CBOE data
    FROM market_data_cboe
    WHERE symbol = 'VXK25'
    ORDER BY date
) TO 'vxk25_data.csv' (HEADER, DELIMITER ',');
``` 