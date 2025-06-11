# Database Documentation

This document provides comprehensive documentation for the Financial Data System's database, including schema details, maintenance procedures, and common operations.

## Database Overview

The system uses DuckDB as its primary database, stored at `data/financial_data.duckdb`. The database is designed to efficiently store and manage financial market data, with a focus on futures contracts and continuous series.

## Schema

### Core Data Tables

#### `market_data`
Stores raw OHLCV (Open, High, Low, Close, Volume) market data for various instruments and intervals. **Note:** This table primarily holds data from sources like TradeStation (e.g., ES, NQ futures) or other non-CBOE sources. VIX futures and VIX Index data from CBOE are stored in `market_data_cboe`.

| Column Name     | Data Type | Description                                                                 | Example             | Notes                                     |
|-----------------|-----------|-----------------------------------------------------------------------------|---------------------|-------------------------------------------|
| `timestamp`     | TIMESTAMP | Date/time of the beginning of the bar (UTC).                                | `2023-10-27 14:30:00` | Part of Primary Key                       |
| `symbol`        | VARCHAR   | Trading symbol (e.g., `SPY`, `ESH25`).                                      | `ESH25`             | Part of Primary Key                       |
| `open`          | DOUBLE    | Opening price for the interval.                                             | `4500.50`           |                                           |
| `high`          | DOUBLE    | Highest price during the interval.                                          | `4510.00`           |                                           |
| `low`           | DOUBLE    | Lowest price during the interval.                                           | `4495.25`           |                                           |
| `close`         | DOUBLE    | Closing price for the interval (may be same as `settle` for daily futures). | `4505.75`           | Use `settle` for futures daily price.     |
| `settle`        | DOUBLE    | Official settlement price (primarily for daily futures).                     | `4506.00`           | Preferred 'close' for daily futures.      |
| `volume`        | DOUBLE    | Volume traded during the interval.                                          | `1502340.0`         | Stored as DOUBLE for flexibility.         |
| `open_interest` | DOUBLE    | Open interest (primarily for futures).                                      | `2501000.0`         | Stored as DOUBLE.                         |
| `interval_value`| INTEGER   | Numeric value of the interval (e.g., 1, 5, 15).                             | `1`                 | Part of Primary Key                       |
| `interval_unit` | VARCHAR   | Unit of the interval (e.g., 'daily', 'minute').                             | `daily`             | Part of Primary Key                       |
| `source`        | VARCHAR   | Source of the data (e.g., 'tradestation').                                  | `tradestation`      |                                           |
| `date`          | VARCHAR   | Date string YYYY-MM-DD (redundant, kept for compatibility).                | `2023-10-27`        | Derived from `timestamp`.                 |
| `up_volume`     | DOUBLE    | Volume on up ticks (TradeStation specific).                                | `800000.0`          | Optional, may be NULL.                    |
| `down_volume`   | DOUBLE    | Volume on down ticks (TradeStation specific).                               | `700000.0`          | Optional, may be NULL.                    |
| `changed`       | BOOLEAN   | Flag indicating if data was modified post-ingestion (e.g., filled).         | `FALSE`             | Default `FALSE`.                          |
| `adjusted`      | BOOLEAN   | Flag indicating price adjustments (e.g., splits/dividends).                 | `FALSE`             | Not currently used for futures.           |
| `quality`       | INTEGER   | Data quality score (100 = original source).                                 | `100`               | Default `100`.                            |

**Primary Key:** (`timestamp`, `symbol`, `interval_value`, `interval_unit`)

#### `market_data_cboe`
Stores daily OHLC market data specifically for VIX futures (`VX` symbols) and the VIX Index (`$VIX.X`) downloaded directly from the CBOE website. This table has a slightly different schema than `market_data`, notably lacking `volume` and `open_interest`.

| Column Name     | Data Type | Description                                                                 | Example             | Notes                                     |
|-----------------|-----------|-----------------------------------------------------------------------------|---------------------|-------------------------------------------|
| `timestamp`     | TIMESTAMP | Date/time of the beginning of the bar (UTC - typically 00:00:00 for daily). | `2025-04-25 00:00:00` | Part of Primary Key                       |
| `symbol`        | VARCHAR   | Trading symbol (e.g., `VXK25`, `VXM25`, `$VIX.X`).                          | `VXK25`             | Part of Primary Key                       |
| `open`          | DOUBLE    | Opening price for the day.                                                  | `18.55`             |                                           |
| `high`          | DOUBLE    | Highest price during the day.                                               | `18.60`             |                                           |
| `low`           | DOUBLE    | Lowest price during the day.                                                | `18.40`             |                                           |
| `settle`        | DOUBLE    | Official settlement price for the day.                                       | `18.48`             | CBOE files provide 'Settle' column.       |
| `interval_value`| INTEGER   | Numeric value of the interval (always 1 for this table).                    | `1`                 | Part of Primary Key                       |
| `interval_unit` | VARCHAR   | Unit of the interval (always 'daily' for this table).                       | `daily`             | Part of Primary Key                       |
| `source`        | VARCHAR   | Source of the data (always 'CBOE' for this table).                          | `CBOE`              |                                           |

**Primary Key:** (`timestamp`, `symbol`, `interval_value`, `interval_unit`)

#### `continuous_contracts`
Stores generated or fetched continuous futures contract data for various symbols and intervals.

| Column Name       | Data Type | Description                                                                 | Example             | Notes                                       |
|-------------------|-----------|-----------------------------------------------------------------------------|---------------------|---------------------------------------------|
| `timestamp`       | TIMESTAMP | Date/time of the beginning of the bar (UTC).                                | `2023-10-27 00:00:00` | Part of Primary Key                         |
| `symbol`          | VARCHAR   | Continuous contract symbol (e.g., `@VX=101XN`, `@ES=102XC`).                | `@VX=101XN`         | Part of Primary Key                         |
| `underlying_symbol`| VARCHAR  | Specific individual contract used for this row's data (e.g., `VXF10`).      | `VXF10`             | Especially for locally generated series.    |
| `open`            | DOUBLE    | Opening price for the interval.                                             | `18.50`             |                                             |
| `high`            | DOUBLE    | Highest price during the interval.                                          | `19.00`             |                                             |
| `low`             | DOUBLE    | Lowest price during the interval.                                           | `18.25`             |                                             |
| `close`           | DOUBLE    | Closing price for the interval.                                             | `18.75`             |                                             |
| `volume`          | BIGINT    | Summed or estimated volume during the interval.                             | `350000`            |                                             |
| `open_interest`   | BIGINT    | Summed or estimated open interest.                                          | `450000`            |                                             |
| `source`          | VARCHAR   | Origin of the data (e.g., 'in_house', 'tradestation', 'polygon').      | `in_house`          | Indicates how the continuous series was built.|
| `built_by`        | VARCHAR   | Method used to build the continuous row (e.g., 'local_generator', 'tradestation'). | `local_generator` |                                     |
| `interval_value`  | INTEGER   | Numeric value of the interval.                                              | `1`                 | Part of Primary Key                         |
| `interval_unit`   | VARCHAR   | Unit of the interval (e.g., 'daily', 'minute').                           | `daily`             | Part of Primary Key                         |
| `adjusted`        | BOOLEAN   | Whether prices were adjusted during generation/fetching.                    | `TRUE`              | e.g., for Panama method.                    |
| `quality`         | INTEGER   | Data quality score.                                                         | `100`               | Default `100`.                              |
| `settle`          | DOUBLE    | Settlement price (often same as close).                                     | `18.75`             | Included for consistency.                   |

**Primary Key:** (`symbol`, `timestamp`, `interval_value`, `interval_unit`)

#### `futures_roll_calendar`
Defines the roll criteria for futures contracts, primarily used for VIX continuous generation.

| Column Name             | Data Type | Description                                                     | Example        |
|-------------------------|-----------|-----------------------------------------------------------------|----------------|
| `contract_code`         | VARCHAR   | The specific futures contract symbol (e.g., `VXK25`).          | `VXK25`        |
| `root_symbol`           | VARCHAR   | The base symbol (e.g., `VX`).                                   | `VX`           |
| `year`                  | INTEGER   | Contract expiration year.                                       | `2025`         |
| `month`                 | INTEGER   | Contract expiration month (1-12).                               | `5`            |
| `expiry_date`           | DATE      | Calculated expiry date based on config rules.                   | `2025-05-21`   |
| `roll_date`             | DATE      | Calculated date when continuous should roll *from* this contract. | `2025-05-19`   |
| `final_settlement_date` | DATE      | Official final settlement date (often same as expiry).          | `2025-05-21`   |
| `last_trading_day`      | DATE      | Last day the contract actively trades (often before expiry).    | `2025-05-20`   |

**Primary Key:** (`contract_code`)

#### `continuous_contract_mapping`
Maps each trading date to the specific underlying contract providing data for each continuous series (1st to Nth month). Used only for locally generated contracts (VIX).

| Column Name        | Data Type | Description                                                                 | Example        |
|--------------------|-----------|-----------------------------------------------------------------------------|----------------|
| `date`             | DATE      | The trading date.                                                          | `2025-04-15`   |
| `root_symbol`      | VARCHAR   | The base symbol (e.g., `VX`).                                              | `VX`           |
| `continuous_c1`    | VARCHAR   | Underlying contract symbol for the 1st continuous series on this date.       | `VXK25`        |
| `continuous_c2`    | VARCHAR   | Underlying contract symbol for the 2nd continuous series on this date.       | `VXM25`        |
| `...`              | VARCHAR   | Columns up to `continuous_c9` (or as configured).                         | `...`          |
| `continuous_cN`    | VARCHAR   | Underlying contract symbol for the Nth continuous series on this date.       | `VXZ25`        |

**Primary Key:** (`date`, `root_symbol`)

#### `symbol_metadata`
Acts as a central registry for all symbols managed by the system, detailing their properties, data sources, storage locations, and update mechanisms for various intervals. This table is primarily populated and updated by the `src/scripts/database/populate_symbol_metadata.py` script based on `config/market_symbols.yaml`.

| Column Name                | Data Type | Description                                                                    | Example                     | Notes                                         |
|----------------------------|-----------|--------------------------------------------------------------------------------|-----------------------------|-----------------------------------------------|
| `symbol`                   | VARCHAR   | The specific symbol identifier (e.g., `@ES=102XC`, `SPY`, `VXK25`).             | `@ES=102XC`                 | Part of Primary Key                           |
| `base_symbol`              | VARCHAR   | The root or base symbol (e.g., `ES`, `SPY`, `VX`).                             | `ES`                        | For grouping or reference.                    |
| `description`              | VARCHAR   | A human-readable description of the symbol.                                    | `E-mini S&P 500 Continuous` |                                               |
| `exchange`                 | VARCHAR   | The exchange where the symbol is traded.                                       | `CME`                       |                                               |
| `asset_type`               | VARCHAR   | Type of asset (e.g., 'future', 'continuous_future', 'equity', 'index').    | `continuous_future`         |                                               |
| `data_source`              | VARCHAR   | Preferred data source (e.g., 'tradestation', 'ibkr', 'cboe', 'polygon', 'in_house'). | `tradestation`              |                                               |
| `data_table`               | VARCHAR   | The database table where data for this symbol/interval is stored.                | `continuous_contracts`      | e.g., `market_data`, `continuous_contracts`   |
| `interval_unit`            | VARCHAR   | The unit of the data interval (e.g., 'minute', 'daily', 'tick').            | `daily`                     | Part of Primary Key                           |
| `interval_value`           | INTEGER   | The value of the data interval (e.g., 1, 5, 15 for minute; 1 for daily).       | `1`                         | Part of Primary Key                           |
| `config_path`              | VARCHAR   | Path to the configuration file that defines this symbol (e.g. `market_symbols.yaml`). | `config/market_symbols.yaml`|                                               |
| `start_date`               | DATE      | The earliest date for which historical data should be considered/fetched.      | `2000-01-01`                | From `market_symbols.yaml` or default.        |
| `last_updated`             | TIMESTAMP | Timestamp of the last update to this metadata entry.                           | `2023-10-27 15:00:00`       | Updated by `populate_symbol_metadata.py`.     |
| `historical_script_path`   | VARCHAR   | Path to the script used for fetching historical data.                          | `src/s./m./fetch_market_data.py` | Relative to project root.                   |
| `update_script_path`       | VARCHAR   | Path to the script used for updating data (if different from historical).      | `src/s./m./c./cont_loader.py`| Optional, relative to project root.         |
| `additional_metadata`      | JSON      | A JSON blob containing the original configuration item from `market_symbols.yaml`. | `{"type": "future", ...}`   | Stores the full config snippet.             |

**Primary Key:** (`symbol`, `interval_unit`, `interval_value`)

### Other Tables
- `economic_data`: (Future Use) For storing economic indicators
- `symbols`: (Future Use/Optional) Metadata about traded symbols
- `data_sources`: (Future Use/Optional) Metadata about data sources
- `derived_indicators`: (Future Use) Storage for calculated technical indicators
- `metadata`: (Internal Use/Optional) Key-value store for system settings
- Account Tables (`account_balances`, `positions`, `orders`, `trades`, etc.): (Future Use) Intended for storing brokerage account information

### Views
- `daily_bars`: Convenience view filtering `market_data` for `interval_unit = 'daily'`
- *(Other views like `weekly_bars`, `monthly_bars` may exist but might need updates based on current table structures and usage patterns)*

### Indexes
Appropriate indexes are created on primary keys and frequently queried columns (e.g., `timestamp`, `symbol`) for performance. The `symbol_metadata` table's primary key ensures efficient lookups for symbol configurations.

## Data Management and Maintenance

The primary interface for managing data and performing maintenance tasks is the `DB_inspect_enhanced.bat` script, which provides a menu-driven approach to various operations, including data updates and inspection.

### Key Scripts:

#### 1. `src/scripts/database/populate_symbol_metadata.py`
This script is crucial for initializing and maintaining the `symbol_metadata` table. It reads configurations from `config/market_symbols.yaml` and populates `symbol_metadata` with detailed entries for each symbol and supported data interval. It determines data sources, storage tables, and appropriate scripts for data fetching and updates.

**Usage (typically run via `DB_inspect_enhanced.bat` or as part of a larger update process):**
```bash
python src/scripts/database/populate_symbol_metadata.py
```

#### 2. `src/scripts/market_data/update_all_market_data_v2.py`
This script orchestrates the updating of market data for symbols defined in `symbol_metadata`. It iterates through relevant entries, checks for the last update times, and calls the appropriate data fetching/processing scripts (e.g., `fetch_market_data.py`, `continuous_contract_loader.py`, specific VIX scripts) based on the metadata.

**Usage (typically run via `DB_inspect_enhanced.bat`):**
```bash
python src/scripts/market_data/update_all_market_data_v2.py
```

#### 3. `src/scripts/market_data/continuous_contract_loader.py`
Responsible for loading and updating data in the `continuous_contracts` table. It handles data from various sources, including TradeStation continuous contracts and locally generated series (e.g., adjusted contracts).

### Deprecated/Old Maintenance Tools
Previous versions of the system used scripts like `view_futures_contracts.py`, `update_active_es_nq_futures.py`, and `maintain_futures_db.bat`. These are generally superseded by the `DB_inspect_enhanced.bat` interface and the `update_all_market_data_v2.py` and `populate_symbol_metadata.py` scripts. While they might still exist in the codebase, they are not part of the current primary maintenance workflow.

## Common Maintenance Tasks

### Regular Maintenance (e.g., Daily/Weekly)

1.  **Run `DB_inspect_enhanced.bat`**: Use the menu options to:
    *   Update/Repopulate `symbol_metadata` (if `market_symbols.yaml` has changed).
    *   Run the main market data update process (which invokes `update_all_market_data_v2.py`).
    *   Inspect data, view logs, or perform other database operations.

## Database Structure Notes

The market data tables (`market_data`, `market_data_cboe`, `continuous_contracts`) use composite primary keys to ensure data uniqueness for a given symbol at a specific timestamp and interval. The `symbol_metadata` table is key to understanding how data for each symbol and interval is sourced, stored, and maintained. 