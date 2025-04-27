# Query Examples for Market Data

This document provides examples of how to query the unified market data table to retrieve various types of data.

## Basic Queries

### Get the most recent data for a symbol

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
ORDER BY timestamp DESC
LIMIT 1;
```

### Get all data for a specific date

```sql
SELECT * FROM market_data
WHERE date = '2023-04-22'
ORDER BY symbol, timestamp;
```

### Get data for multiple symbols

```sql
SELECT * FROM market_data
WHERE symbol IN ('ESH26', 'NQH26', '$VIX.X')
AND date = '2023-04-22'
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

### Get all CBOE data

```sql
SELECT * FROM market_data
WHERE source = 'CBOE'
ORDER BY symbol, timestamp DESC
LIMIT 100;
```

## Timeframe Specific Queries

### Get daily data for a specific symbol and date range

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'day'
AND date BETWEEN '2023-01-01' AND '2023-04-22'
ORDER BY timestamp;
```

### Get minute data for a specific symbol and time range

```sql
SELECT * FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'minute'
AND timestamp BETWEEN '2023-04-22 09:30:00' AND '2023-04-22 16:00:00'
ORDER BY timestamp;
```

## Aggregation Queries

### Get the average daily volume for a symbol by month

```sql
SELECT 
    DATE_TRUNC('month', timestamp) AS month,
    AVG(volume) AS avg_daily_volume
FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'day'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month;
```

### Get the highest and lowest prices by week

```sql
SELECT 
    DATE_TRUNC('week', timestamp) AS week,
    MAX(high) AS highest_price,
    MIN(low) AS lowest_price
FROM market_data
WHERE symbol = 'ESH26'
AND interval_unit = 'day'
GROUP BY DATE_TRUNC('week', timestamp)
ORDER BY week;
```

## Combining Multiple Symbols

### Compare ES and NQ futures closing prices

```sql
WITH es_data AS (
    SELECT date, close AS es_close
    FROM market_data
    WHERE symbol LIKE 'ES%'
    AND interval_unit = 'day'
    AND date BETWEEN '2023-01-01' AND '2023-04-22'
),
nq_data AS (
    SELECT date, close AS nq_close
    FROM market_data
    WHERE symbol LIKE 'NQ%'
    AND interval_unit = 'day'
    AND date BETWEEN '2023-01-01' AND '2023-04-22'
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

### Compare VIX index with futures

```sql
WITH vix_data AS (
    SELECT date, close AS vix_close
    FROM market_data
    WHERE symbol = '$VIX.X'
    AND interval_unit = 'day'
    AND date BETWEEN '2023-01-01' AND '2023-04-22'
),
vx_futures AS (
    SELECT date, close AS vx_close
    FROM market_data
    WHERE symbol LIKE 'VX%'
    AND interval_unit = 'day'
    AND date BETWEEN '2023-01-01' AND '2023-04-22'
)
SELECT 
    vix_data.date,
    vix_data.vix_close,
    vx_futures.vx_close,
    vx_futures.vx_close - vix_data.vix_close AS basis
FROM vix_data
JOIN vx_futures ON vix_data.date = vx_futures.date
ORDER BY vix_data.date;
```

## Finding Data Quality Issues

### Find records that have been changed or filled

```sql
SELECT * FROM market_data
WHERE changed = true
ORDER BY timestamp DESC
LIMIT 100;
```

### Find records with lower quality scores

```sql
SELECT * FROM market_data
WHERE quality < 100
ORDER BY quality ASC, timestamp DESC
LIMIT 100;
```

## Exporting Data

### Export daily bars for a specific symbol to CSV

```sql
COPY (
    SELECT 
        date, 
        open, 
        high, 
        low, 
        close, 
        volume
    FROM market_data
    WHERE symbol = 'ESH26'
    AND interval_unit = 'day'
    ORDER BY date
) TO 'es_data.csv' (HEADER, DELIMITER ',');
``` 