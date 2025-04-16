# Financial Data System Scripts

This document provides detailed information about all the scripts available in the Financial Data System.

## Directory Structure

The scripts are organized into the following directories:

- `src/scripts/market_data/`: Scripts for fetching and managing market data
- `src/scripts/database/`: Scripts for database operations
- `src/scripts/analysis/`: Scripts for data analysis and visualization
- `src/scripts/api/`: Scripts for API interactions and environment setup

## Market Data Scripts

### fetch_market_data.py

Fetches market data for specified symbols and intervals.

```bash
python src/scripts/market_data/fetch_market_data.py --symbol SYMBOL --interval-value VALUE --interval-unit UNIT [options]
```

Options:
- `--symbol`: Symbol to fetch data for (e.g., ESH24, SPY)
- `--interval-value`: Interval value (e.g., 1 for daily, 15 for 15-minute)
- `--interval-unit`: Interval unit (daily, minute, hour)
- `--start-date`: Start date for data fetch (YYYY-MM-DD)
- `--end-date`: End date for data fetch (YYYY-MM-DD)
- `--updatehistory`: Update historical data
- `--force`: Force update even if data exists

Example:
```bash
python src/scripts/market_data/fetch_market_data.py --symbol ESH24 --interval-value 15 --interval-unit minute --updatehistory --force
```

### check_futures_contracts.py

Analyzes futures contracts data and can fetch missing contracts.

```bash
python src/scripts/market_data/check_futures_contracts.py BASE_SYMBOL [options]
```

Options:
- `--start-year`: Start year for analysis
- `--end-year`: End year for analysis
- `--interval-value`: Interval value (e.g., 1 for daily, 15 for 15-minute)
- `--interval-unit`: Interval unit (daily, minute, hour)
- `--missing-only`: Show only missing contracts
- `--fetch-missing`: Fetch data for missing contracts

Example:
```bash
python src/scripts/market_data/check_futures_contracts.py NQ --start-year 2004 --end-year 2025 --interval-value 15 --interval-unit minute --fetch-missing
```

### update_es_contracts.py

Updates ES futures contracts data.

```bash
python src/scripts/market_data/update_es_contracts.py
```

This script updates all ES contracts from 2024 to 2026.

### force_update_contracts.py

Forces an update of specified contracts.

```bash
python src/scripts/market_data/force_update_contracts.py --symbols SYMBOL1 SYMBOL2 [options]
```

Options:
- `--symbols`: List of symbols to update
- `--interval-value`: Interval value (e.g., 1 for daily, 15 for 15-minute)
- `--interval-unit`: Interval unit (daily, minute, hour)
- `--start-date`: Start date for data fetch (YYYY-MM-DD)
- `--end-date`: End date for data fetch (YYYY-MM-DD)

Example:
```bash
python src/scripts/market_data/force_update_contracts.py --symbols ESH24 ESM24 --interval-value 15 --interval-unit minute
```

### generate_continuous_futures.py

Generates continuous futures contracts for a given root symbol.

```bash
python src/scripts/market_data/generate_continuous_futures.py --root-symbol SYMBOL --config CONFIG_FILE
```

Options:
- `--root-symbol`: Root symbol (e.g., VX for VIX futures)
- `--config`: Path to the market symbols configuration file

The script:
- Generates continuous contracts (c1, c2) for the specified root symbol
- Handles rollovers based on contract expiry dates
- Maintains gaps in data when underlying contracts are missing
- Uses only daily data (interval_value=1, interval_unit='day')
- Stores results in the continuous_contracts table

Example:
```bash
python src/scripts/market_data/generate_continuous_futures.py --root-symbol VX --config config/market_symbols.yaml
```

This will generate VXc1 and VXc2 continuous contracts, rolling over to the next contract on expiry dates.

### generate_futures_symbols.py

Generates futures symbols based on configuration.

```bash
python src/scripts/market_data/generate_futures_symbols.py [options]
```

Options:
- `--base-symbol`: Base symbol to generate (e.g., ES, NQ)
- `--start-year`: Start year for generation
- `--end-year`: End year for generation

Example:
```bash
python src/scripts/market_data/generate_futures_symbols.py --base-symbol ES --start-year 2020 --end-year 2025
```

### redownload_es_contracts.py

Redownloads specific ES contracts.

```bash
python src/scripts/market_data/redownload_es_contracts.py
```

## Database Scripts

### backup_database.py

Backs up the database.

```bash
python src/scripts/database/backup_database.py [options]
```

Options:
- `-d, --database PATH`: Path to the database file
- `-o, --output DIR`: Output directory for backups
- `-r, --retention DAYS`: Number of days to keep backups
- `-v, --verbose`: Enable verbose output

Example:
```bash
python src/scripts/database/backup_database.py -d ./data/financial_data.duckdb -o ./backups -r 30 -v
```

### scheduled_backup.py

Runs scheduled database backups.

```bash
python src/scripts/database/scheduled_backup.py
```

This script reads configuration from environment variables:
- `BACKUP_DIR`: Directory to store backups
- `RETENTION_DAYS`: Number of days to keep backups
- `DATABASE_PATH`: Path to the database file

### check_db.py

Checks database status and connectivity.

```bash
python src/scripts/database/check_db.py
```

## Data Analysis Scripts

### check_market_data.py

Analyzes market data for quality and completeness.

```bash
python src/scripts/analysis/check_market_data.py --symbol SYMBOL [options]
```

Options:
- `--symbol`: Symbol to check
- `--interval-value`: Interval value
- `--interval-unit`: Interval unit
- `--start-date`: Start date for analysis
- `--end-date`: End date for analysis

Example:
```bash
python src/scripts/analysis/check_market_data.py --symbol SPY --interval-value 1 --interval-unit daily
```

### visualize_data.py

Visualizes market data.

```bash
python src/scripts/analysis/visualize_data.py --symbol SYMBOL [options]
```

Options:
- `--symbol`: Symbol to visualize
- `--interval-value`: Interval value
- `--interval-unit`: Interval unit
- `--start-date`: Start date for visualization
- `--end-date`: End date for visualization
- `--chart-type`: Type of chart (line, candlestick, etc.)

Example:
```bash
python src/scripts/analysis/visualize_data.py --symbol SPY --interval-value 1 --interval-unit daily --chart-type line
```

### cleanup_market_data.py

Cleans up market data.

```bash
python src/scripts/analysis/cleanup_market_data.py --symbol SYMBOL [options]
```

Options:
- `--symbol`: Symbol to clean up
- `--interval-value`: Interval value
- `--interval-unit`: Interval unit
- `--start-date`: Start date for cleanup
- `--end-date`: End date for cleanup

Example:
```bash
python src/scripts/analysis/cleanup_market_data.py --symbol SPY --interval-value 1 --interval-unit daily
```

## API and Environment Scripts

### test_api_connections.py

Tests API connections.

```bash
python src/scripts/api/test_api_connections.py
```

### check_env.py

Checks environment setup.

```bash
python src/scripts/api/check_env.py
```

### check_env_vars.py

Checks environment variables.

```bash
python src/scripts/api/check_env_vars.py
```