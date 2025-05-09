# Futures Configuration Guide (`config/market_symbols.yaml`)

This document provides a comprehensive guide to configuring futures contracts in the `config/market_symbols.yaml` file, which dictates how the financial data system fetches, processes, and stores data.

## `futures` Section Structure

Each entry under the `futures:` list in the YAML file represents a futures instrument and requires several fields:

### Basic Information
- `base_symbol`: The root symbol (e.g., 'ES', 'NQ', 'VX'). *Required*.
- `description`: Human-readable description (e.g., 'E-mini S&P 500 Futures'). *Required*.
- `exchange`: The primary exchange (e.g., 'CME', 'CBOE', 'NYMEX'). *Required*.
- `calendar`: The specific trading calendar identifier used for date calculations and holiday adjustments (e.g., 'CME_Equity', 'CFE', 'CMEGlobex_CL'). Get valid names from `exchange_calendars` or `pandas_market_calendars`. *Required*.
- `start_date`: The earliest date (YYYY-MM-DD) for which data should ideally be available or fetched. *Required*.
- `source`: Specifies where the data for this future comes from. This dictates which fetching logic is used. Valid values:
    - `cboe`: Data fetched from CBOE website CSVs (Used for VIX).
    - `tradestation`: Data fetched via TradeStation API (Used for ES, NQ).
    *Required*.

### Data Collection Parameters
- `frequencies`: List of data frequencies to collect for this symbol (affects individual contracts fetched from TradeStation and potentially local generation if implemented).
    - Available options configured in `settings.data_frequencies`: typically '1min', '15min', 'daily'. *Required*.
- `num_active_contracts`: Number of nearest-expiry contracts the system should track/update (e.g., 2 for ES/NQ means front month and second month). *Required*.
- `is_continuous`: Boolean flag (true/false) indicating if continuous contract data should be maintained for this symbol. *Required*.

### Historical Contract Identification (Primarily for Local Generation/Mapping)
- `historical_contracts`: Configuration for identifying historical contract symbols.
    - `patterns`: List of standard futures month codes (e.g., ['H', 'M', 'U', 'Z'] for quarterly contracts). *Required if local generation is used*.
    - `start_year`: The year when historical data collection/generation should begin. *Required if local generation is used*.
    - `start_month`: (Optional) The starting month within the `start_year` (1-12). Defaults to 1 if omitted.

### Expiry Rules (Primarily for VIX Local Continuous Generation/Mapping)
The `expiry_rule` section defines when individual contracts expire, which is crucial for determining roll dates when generating continuous contracts *locally* (like for VIX). It may be less critical for symbols where continuous data is fetched directly (ES/NQ).

- `day_type`: Defines the type of rule. Options:
    - `friday`: Nth Friday of the month (e.g., `day_number: 3` for 3rd Friday).
    - `wednesday`: Nth Wednesday of the month (e.g., `day_number: 3` for 3rd Wednesday).
    - `business_day`: N business days before/after a reference point.
        - `days_before`: Number of business days *before* the reference.
        - `reference_day`: (Optional) Specific day of the month (e.g., 25) to count back from.
        - `reference_point`: (Optional) Can be 'last_business_day' to count back from the last business day of the month.
    - `special_rule`: Custom rule identifier (e.g., `VX_expiry` handled by specific logic).
*Required if local generation is used*.

- `day_number`: (Used with `friday`/`wednesday`) The occurrence number (e.g., 3 for the third).
- `adjust_for_holiday`: (true/false) Whether to adjust the calculated expiry date if it falls on a holiday according to the specified `calendar`. Defaults to `true`.
- `holiday_calendar`: (Deprecated - use `calendar`) Calendar for holidays.

## Month Codes

Standard futures month codes used in `historical_contracts.patterns`:
- F: Jan, G: Feb, H: Mar, J: Apr, K: May, M: Jun
- N: Jul, Q: Aug, U: Sep, V: Oct, X: Nov, Z: Dec

## Example Configurations

```yaml
futures:
  - base_symbol: ES
    calendar: CME_Equity # Specific calendar
    description: E-mini S&P 500 Futures
    exchange: CME
    frequencies: # Daily and Intraday for individual contracts
      - 1min
      - 15min
      - daily
    historical_contracts: # Define patterns for identifying historical individual contracts
      patterns: [H, M, U, Z]
      start_year: 2003
    num_active_contracts: 2
    expiry_rule: # Rule for determining active individual contracts
      day_type: friday
      day_number: 3
      adjust_for_holiday: true
    start_date: '2003-01-01'
    # is_continuous: true # This flag is deprecated at this level. Continuous contracts are defined by their own entries or inferred.
    source: tradestation # Individual contracts fetched via TradeStation

  - base_symbol: VX
    calendar: CFE # Specific calendar
    description: CBOE Volatility Index Futures
    exchange: CBOE
    frequencies: # Daily and 15min for individual contracts if fetched from TS, or for CBOE raw data
      - 15min
      - daily
    historical_contracts: # Required for identifying historical individual contracts
      patterns: [F, G, H, J, K, M, N, Q, U, V, X, Z] # All months
      start_year: 2004
    num_active_contracts: 9 # Track 9 near individual contracts
    expiry_rule: # Crucial for identifying active individual contracts
      day_type: wednesday
      special_rule: VX_expiry # Specific logic applies
      adjust_for_holiday: true
    start_date: '2004-01-01'
    # is_continuous: true # Deprecated here
    source: cboe # Raw individual VIX futures data is fetched from CBOE.
                 # For VIX continuous contracts built from TradeStation underlying data,
                 # ensure a separate entry or mechanism in symbol_metadata designates them
                 # with data_source = 'tradestation' and asset_type = 'continuous_future'.

# Continuous Contract Specific Entries (Example)
# These would typically be generated by populate_symbol_metadata.py based on rules,
# or could be explicitly defined if needed. The key is that `asset_type` becomes 'continuous_future'
# and `data_source` indicates how they are processed.

  - symbol: '@ES=102XC' # This is the actual symbol for the continuous contract data_table
    base_symbol: ES
    description: 'E-mini S&P 500 Continuous, Front Month, Calendar Adjusted'
    asset_type: continuous_future # Crucial for routing to continuous_contract_loader
    data_source: tradestation      # Indicates continuous_contract_loader uses TS underlying
    # Frequencies for continuous are typically daily, but could be configured
    frequencies:
      - unit: daily
        interval: 1
    # Other fields like exchange, calendar might be inherited or specified
    exchange: CME
    calendar: CME_Equity
    # Expiry rules, num_active_contracts are not directly applicable here as this IS the continuous contract

  - symbol: '@VX=101XN' # Actual symbol for the VIX continuous contract data_table
    base_symbol: VX
    description: 'VIX Continuous, Front Month, Nearest'
    asset_type: continuous_future
    data_source: tradestation      # VIX continuous now also built by continuous_contract_loader from TS VIX underlying
    frequencies:
      - unit: daily
        interval: 1
    exchange: CFE
    calendar: CFE
```

## Best Practices

1.  **Source Accuracy:** Ensure the `source` field correctly reflects where the *individual contract* data originates (`cboe` or `tradestation`). For continuous contracts, `data_source: tradestation` in their `symbol_metadata` entry means they will be built by `continuous_contract_loader.py` using TradeStation underlying data.
2.  **Calendar Specificity:** Use precise calendar names (e.g., `CME_Equity`, `CFE`) for accurate date calculations for individual contracts.
3.  **Expiry Rules:** Define `expiry_rule` carefully for individual futures to allow `MarketDataFetcher` to determine active contracts correctly.
4.  **Frequencies:** List all desired frequencies for *individual contracts*. Continuous contracts are typically daily but can be configured.
5.  **Consistency:** Keep configurations aligned with the actual data fetching and processing logic implemented in the scripts.
6.  **Symbol Metadata as the Driver:** Remember that `config/market_symbols.yaml` is processed by `populate_symbol_metadata.py` into the `symbol_metadata` table. The `asset_type` and `data_source` columns in `symbol_metadata` are key for how the `update_all_market_data.py` script routes processing, especially for continuous futures.

## Common Futures Configurations

### Equity Index Futures (ES, NQ)
- Expire on the third Friday of the month
- Two active contracts
- Available in 1-minute, 15-minute, and daily frequencies

### Commodity Futures (CL, GC)
- Expire three business days before month-end
- Multiple active contracts (6-12 months)
- Available in 1-minute, 15-minute, and daily frequencies

### Volatility Futures (VX)
- Expire on the third Wednesday of the month
- Nine active contracts
- Available in daily frequency only

## Month Codes
Standard futures month codes:
- F: January
- G: February
- H: March
- J: April
- K: May
- M: June
- N: July
- Q: August
- U: September
- V: October
- X: November
- Z: December

## Example: Populating `symbol_metadata` for a Continuous Future

While `config/market_symbols.yaml` describes the base instruments, the `symbol_metadata` table needs entries for the continuous contracts themselves. These are typically created by `populate_symbol_metadata.py` based on conventions or specific logic within that script. An entry in `symbol_metadata` for a continuous future that should be processed by `continuous_contract_loader.py` would look something like this (conceptual SQL representation):

```sql
INSERT INTO symbol_metadata (symbol, base_symbol, asset_type, data_source, data_table, ...) 
VALUES 
('@ES', 'ES', 'future', 'tradestation', '@ES', ...), -- Entry for the generic continuous symbol from YAML
('@ES=102XC', 'ES', 'continuous_future', 'tradestation', '@ES=102XC', ...); -- Specific continuous variant
```

The `update_all_market_data.py` script will specifically look for `asset_type = 'continuous_future'` and `data_source = 'tradestation'` to identify symbols to pass to `continuous_contract_loader.py`.

## Best Practices (Consolidated)

1.  **Calendar Selection:** Always specify the appropriate trading calendar for individual contracts.
2.  **Historical Data:** Set reasonable start dates for individual contracts based on data availability. Include all relevant month codes in patterns.
3.  **Expiry Rules:** Define for individual contracts. Enable holiday adjustments for US contracts. Use specific day rules for index futures, business day rules for commodity futures.
4.  **Data Frequencies:** Include all necessary frequencies for individual contracts. Consider storage. Continuous contracts usually daily.
5.  **`symbol_metadata` is Key:** Understand that this YAML populates `symbol_metadata`, which is the direct driver for the update orchestrator. Ensure `asset_type` and `data_source` in `symbol_metadata` correctly reflect how each symbol (individual or continuous) should be handled.

```yaml
# Example for individual ES contract definition in market_symbols.yaml
- base_symbol: ES
  calendar: CME_Equity # Use specific exchange calendar for individual contracts
  description: E-mini S&P 500 Futures
  exchange: CME
  frequencies: # For individual contracts
    - unit: minute # Frequencies now as dicts
      interval: 1
    - unit: minute
      interval: 15
    - unit: daily
      interval: 1
  historical_contracts:
    patterns:
      - H
      - M
      - U
      - Z
    start_month: 1
    start_year: 2003
  num_active_contracts: 2
  expiry_rule:
    day_type: friday
    day_number: 3
    adjust_for_holiday: true
  start_date: '2003-01-01'
  source: tradestation # For individual contracts

# Note: Continuous contract definitions like '@ES=102XC' are now primarily managed
# by their entries in the `symbol_metadata` table, where `asset_type` is 'continuous_future'
# and `data_source` is 'tradestation'. The `populate_symbol_metadata.py` script
# is responsible for creating these entries, potentially based on conventions or flags
# in the base symbol definition (like an implicit understanding if a `base_symbol` has continuous versions).
``` 