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
Stores generated or fetched continuous futures contract data.

| Column Name       | Data Type | Description                                                    | Example             | Notes                                       |
|-------------------|-----------|----------------------------------------------------------------|---------------------|---------------------------------------------|
| `timestamp`       | TIMESTAMP | Date/time of the beginning of the bar (UTC).                                | `2023-10-27 00:00:00` | Part of Primary Key                         |
| `symbol`          | VARCHAR   | Continuous contract symbol (e.g., `@VX=101XN`, `@ES=102XC`).       | `@VX=101XN`         | Part of Primary Key                         |
| `open`            | DOUBLE    | Opening price for the interval.                                | `18.50`             |                                             |
| `high`            | DOUBLE    | Highest price during the interval.                               | `19.00`             |                                             |
| `low`             | DOUBLE    | Lowest price during the interval.                                | `18.25`             |                                             |
| `close`           | DOUBLE    | Closing price for the interval (use this, often same as settle). | `18.75`             |                                             |
| `settle`          | DOUBLE    | Settlement price (often same as close).                         | `18.75`             | Included for consistency with `market_data`. |
| `volume`          | DOUBLE    | Summed or estimated volume during the interval.                  | `350000.0`          | May be approximate for generated contracts. |
| `open_interest`   | DOUBLE    | Summed or estimated open interest.                             | `450000.0`          | May be approximate for generated contracts. |
| `interval_value`  | INTEGER   | Numeric value of the interval (typically 1).                   | `1`                 | Part of Primary Key (implicit daily).     |
| `interval_unit`   | VARCHAR   | Unit of the interval (typically 'daily').                      | `daily`             | Part of Primary Key (implicit daily).     |
| `source`          | VARCHAR   | Origin: 'generated' (VIX) or 'tradestation' (ES/NQ).         | `generated`         |                                             |
| `underlying_symbol`| VARCHAR  | Underlying contract used for this bar (for 'generated').       | `VXZ23`             | NULL for source='tradestation'.           |
| `adjusted`        | BOOLEAN   | Whether prices were adjusted during generation/fetching.       | `TRUE`              | Typically TRUE.                             |

**Primary Key:** (`timestamp`, `symbol`) *Note: interval is implicitly daily.*

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

### Other Tables
- `economic_data`: (Future Use) For storing economic indicators
- `symbols`: (Future Use/Optional) Metadata about traded symbols
- `data_sources`: (Future Use/Optional) Metadata about data sources
- `derived_indicators`: (Future Use) Storage for calculated technical indicators
- `metadata`: (Internal Use/Optional) Key-value store for system settings
- Account Tables (`account_balances`, `positions`, `orders`, `trades`, etc.): (Future Use) Intended for storing brokerage account information

### Views
- `daily_bars`: Convenience view filtering `market_data` for `interval_unit = 'daily'`
- *(Other views like `weekly_bars`, `monthly_bars` may exist but might need updates)*

### Indexes
Appropriate indexes are created on primary keys and frequently queried columns (e.g., `timestamp`, `symbol`) for performance.

## Maintenance Tools

### 1. View Futures Contracts (`view_futures_contracts.py`)

This script displays a summary of all futures contracts in the database, with details like symbol, date range, and row counts.

**Usage:**
```bash
# View all contracts
python view_futures_contracts.py

# View only ES contracts
python view_futures_contracts.py ES

# View only NQ contracts
python view_futures_contracts.py NQ
```

### 2. Update Active ES and NQ Futures (`update_active_es_nq_futures.py`)

This script identifies and updates only the active ES and NQ futures contracts based on the CME quarterly cycle (March, June, September, December). It automatically determines which contracts are currently active and fetches data for the next two upcoming contracts per symbol.

Key features:
- Follows the proper CME futures cycle (H, M, U, Z)
- Handles contract rollovers based on the 3rd Friday expiration rule
- Pulls 90 days of historical data to ensure continuity if the script hasn't run recently
- Avoids duplicates through database primary key constraints

**Usage:**
```bash
# Update active contracts
python update_active_es_nq_futures.py
```

## Common Maintenance Tasks

### Regular Maintenance (Weekly)

1. Update active contracts:
   ```bash
   python update_active_es_nq_futures.py
   ```

2. View updated contracts:
   ```bash
   python view_futures_contracts.py ES NQ
   ```

### Special Maintenance Operations

For special one-off maintenance scenarios, there are additional scripts available:

- `clean_futures_database.py`: For removing duplicate entries (only needed in special circumstances)
- `purge_old_futures_contracts.py`: For removing old contracts to save space (only as a one-off operation)

These operations should rarely be needed as the database enforces uniqueness through primary keys and historical data is typically valuable to retain.

### Windows Batch File Interface

For Windows users, a convenient batch file is provided that offers a menu-driven interface:

```bash
# Run the maintenance menu
maintain_futures_db.bat
```

This provides easy access to viewing and updating contracts without needing to remember command-line syntax.

## Database Structure Notes

The market data is stored in a DuckDB database with the following primary key structure:

- Primary Key: (`timestamp`, `symbol`, `interval_value`, `interval_unit`)

This composite primary key ensures that there can never be duplicate entries in the database, as the database engine enforces uniqueness on this combination of fields. 