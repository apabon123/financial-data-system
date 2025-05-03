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
    frequencies: # Daily and Intraday
      - 1min
      - 15min
      - daily
    historical_contracts: # Define patterns even if not locally generated
      patterns: [H, M, U, Z]
      start_year: 2003
    num_active_contracts: 2
    expiry_rule: # Rule defined, but less critical for direct fetch
      day_type: friday
      day_number: 3
      adjust_for_holiday: true
    start_date: '2003-01-01'
    is_continuous: true
    source: tradestation # Fetched via TradeStation

  - base_symbol: VX
    calendar: CFE # Specific calendar
    description: CBOE Volatility Index Futures
    exchange: CBOE
    frequencies: # Daily and 15min as per current config
      - 15min
      - daily
    historical_contracts: # Required for local generation
      patterns: [F, G, H, J, K, M, N, Q, U, V, X, Z] # All months
      start_year: 2004
    num_active_contracts: 9 # Track 9 near contracts
    expiry_rule: # Crucial for local generation mapping
      day_type: wednesday
      special_rule: VX_expiry # Specific logic applies
      adjust_for_holiday: true
    start_date: '2004-01-01'
    is_continuous: true
    source: cboe # Fetched from CBOE website
```

## Best Practices

1.  **Source Accuracy:** Ensure the `source` field correctly reflects where the data originates (`cboe` or `tradestation`).
2.  **Calendar Specificity:** Use precise calendar names (e.g., `CME_Equity`, `CFE`) for accurate date calculations.
3.  **Expiry Rules:** Define `expiry_rule` carefully, especially for symbols like VX where continuous contracts are generated locally based on these rules.
4.  **Frequencies:** List all desired frequencies. Note that availability might depend on the `source`.
5.  **Consistency:** Keep configurations aligned with the actual data fetching and processing logic implemented in the scripts.

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

## Best Practices

1. **Calendar Selection**
   - Always specify the appropriate trading calendar
   - Use 'US' for contracts trading on US exchanges

2. **Historical Data**
   - Set reasonable start dates based on data availability
   - Include all relevant month codes in patterns

3. **Expiry Rules**
   - Always enable holiday adjustments for US contracts
   - Use specific day rules for index futures
   - Use business day rules for commodity futures

4. **Data Frequencies**
   - Include all necessary frequencies for your analysis
   - Consider storage requirements when selecting frequencies

## Example Configuration

```yaml
- base_symbol: ES
  calendar: US
  description: E-mini S&P 500 Futures
  exchange: CME
  frequencies:
    - 1min
    - 15min
    - daily
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
    holiday_calendar: US
  start_date: '2003-01-01'
``` 