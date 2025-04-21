# Scripts Documentation

This document provides detailed information about the scripts available in the Financial Data System project.

## Market Data Scripts

Located in `src/scripts/market_data/`:

### VIX Data Update and Processing

- **update_vx_futures.py**: Updates VIX futures data from CBOE
  ```
  python -m src.scripts.market_data.update_vx_futures [--full-regen] [--db-path PATH] [--config-path PATH]
  ```
  - `--full-regen`: Triggers full regeneration of continuous contracts and runs gap/zero filling
  - `--db-path`: Custom database path (default: data/financial_data.duckdb)
  - `--config-path`: Custom config path (default: config/market_symbols.yaml)

- **update_vix_index.py**: Updates the VIX index data
  ```
  python -m src.scripts.market_data.update_vix_index [--db-path PATH]
  ```

- **generate_vix_roll_calendar.py**: Generates roll calendar for VIX futures
  ```
  python -m src.scripts.market_data.generate_vix_roll_calendar [--force] [--db-path PATH]
  ```
  - `--force`: Forces regeneration of the entire roll calendar

- **generate_continuous_futures.py**: Generates continuous futures contracts
  ```
  python -m src.scripts.market_data.generate_continuous_futures --root-symbol SYMBOL [--db-path PATH] [--start-date DATE] [--end-date DATE] [--force]
  ```
  - `--root-symbol`: Root symbol for the futures (e.g., VX)
  - `--start-date`: Start date for generation (default: from config)
  - `--end-date`: End date for generation (default: current date)
  - `--force`: Forces regeneration by deleting existing data

### Data Quality and Cleanup

- **fill_vx_continuous_gaps.py**: Fills historical gaps in VXc1/VXc2 data from 2004-2005
  ```
  python -m src.scripts.market_data.fill_vx_continuous_gaps [--db-path PATH] [--start-date DATE] [--end-date DATE]
  ```

- **fill_vx_zero_prices.py**: Fills zero prices in VXc1-VXc5 using interpolation and reference data
  ```
  python -m src.scripts.market_data.fill_vx_zero_prices [--db-path PATH] [--start-date DATE] [--end-date DATE]
  ```

- **verify_continuous_futures.py**: Verifies continuous contracts against underlying data
  ```
  python -m src.scripts.market_data.verify_continuous_futures --root-symbol SYMBOL [--contract-code CODE] [--db-path PATH]
  ```

## Analysis Scripts

Located in `src/scripts/analysis/`:

- **show_vix_continuous_data.py**: Displays VIX and continuous contracts data for analysis
  ```
  python -m src.scripts.analysis.show_vix_continuous_data [--start-date DATE] [--end-date DATE] [--limit N]
  ```
  - `--start-date`: Start date for data display
  - `--end-date`: End date for data display
  - `--limit`: Limit the number of rows displayed

- **detect_vx_outliers.py**: Identifies potential outliers in VIX futures data
  ```
  python -m src.scripts.analysis.detect_vx_outliers [--start-date DATE] [--end-date DATE]
  ```

- **verify_vx_continuous.py**: Performs validation checks on VX continuous contracts
  ```
  python -m src.scripts.analysis.verify_vx_continuous [--symbol-prefix PREFIX] [--gap-threshold VALUE]
  ```
  - `--symbol-prefix`: Symbol prefix to check (e.g., VXc)
  - `--gap-threshold`: Price gap threshold for outlier detection (default: 0.1 or 10%)

- **view_vix_data.py**: Simple viewer for VIX data
  ```
  python -m src.scripts.analysis.view_vix_data [--date-range RANGE]
  ```

- **view_vix_data_formatted.py**: Enhanced viewer with formatting for VIX data
  ```
  python -m src.scripts.analysis.view_vix_data_formatted [--date-range RANGE] [--format {table,csv}]
  ```

## Utility Scripts

Located in `src/scripts/utility/`:

- **check_data_counts.py**: Checks record counts in the database
  ```
  python -m src.scripts.utility.check_data_counts [--table TABLE]
  ```

- **check_roll_calendar.py**: Validates the futures roll calendar
  ```
  python -m src.scripts.utility.check_roll_calendar [--root-symbol SYMBOL]
  ```

- **create_valid_trading_days.py**: Generates a list of valid trading days
  ```
  python -m src.scripts.utility.create_valid_trading_days [--start-year YEAR] [--end-year YEAR] [--output PATH]
  ```

- **check_table_schema.py**: Displays the schema of database tables
  ```
  python -m src.scripts.utility.check_table_schema [--table TABLE]
  ```

- **duckdb_launcher.py**: Launches an interactive DuckDB session
  ```
  python -m src.scripts.utility.duckdb_launcher [--db-path PATH]
  ```

## Database Management

Located in `src/scripts/database/`:

- **backup_database.py**: Backs up the database
  ```
  python -m src.scripts.database.backup_database [--db-path PATH] [--output DIR] [--retention DAYS]
  ```
  - `--db-path`: Database path to backup
  - `--output`: Output directory for backups
  - `--retention`: Number of days to keep backups

- **scheduled_backup.py**: Runs a scheduled backup (intended for cron/task scheduler)
  ```
  python -m src.scripts.database.scheduled_backup
  ```

## Maintenance Scripts

Located in project root:

- **reorganize_project.py**: Cleans up project structure by moving files to appropriate directories
  ```
  python reorganize_project.py
  ```

## Documentation Updates

To ensure documentation stays current as scripts are added or modified:

1. Update this document when adding or changing scripts
2. Keep script help text and docstrings complete and accurate
3. Run scripts with `--help` flag to see available options