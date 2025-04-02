# DuckDB Schema

## Main Tables

### market_data
- timestamp (TIMESTAMP) - Primary time index
- symbol (VARCHAR) - Trading symbol
- open (DOUBLE) - Opening price
- high (DOUBLE) - High price
- low (DOUBLE) - Low price
- close (DOUBLE) - Closing price
- volume (BIGINT) - Trading volume
- up_volume (BIGINT) - Optional up volume
- down_volume (BIGINT) - Optional down volume
- source (VARCHAR) - Data source identifier

### economic_data
- timestamp (TIMESTAMP) - Primary time index
- indicator (VARCHAR) - Economic indicator name
- value (DOUBLE) - Indicator value
- source (VARCHAR) - Data source

## Views
- daily_bars - Aggregated daily view