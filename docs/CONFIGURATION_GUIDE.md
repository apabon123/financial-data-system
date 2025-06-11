# Configuration Guide

This document provides a comprehensive guide to configuring the Financial Data System, including the structure of configuration files, how to configure different types of instruments, and best practices for system configuration.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Directory Structure](#configuration-directory-structure)
3. [Core Configuration Files](#core-configuration-files)
4. [Instrument-Specific Configuration](#instrument-specific-configuration)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)

## Overview

The Financial Data System uses a modular, YAML-based configuration system that provides:

- **Separation of Concerns**: Different aspects of the system are configured in separate files
- **Inheritance and Templates**: Reduce duplication with template-based configurations
- **Cross-File References**: Reference settings from other configuration files
- **Environment-Specific Overrides**: Customize configuration for different environments
- **Schema Validation**: Ensure configuration validity with JSON Schema validation

## Configuration Directory Structure

```
config/
├── market_symbols.yaml   # Main symbol configuration (processed by populate_symbol_metadata.py)
├── schemas/             # JSON Schema validation files (primarily for new structure)
│   ├── exchanges.json
│   ├── futures.json
│   └── ...
└── yaml/                # New structured YAML files (aspirational/future use)
    ├── exchanges.yaml
    ├── futures.yaml
    ├── indices.yaml
    ├── etfs.yaml
    ├── equities.yaml
    ├── data_sources.yaml
    ├── cleaning_rules.yaml
    └── environments/    # Environment-specific overrides
        ├── development/
        │   └── ...
        └── production/
            └── ...
```

## Core Configuration Files

While the system is designed to move towards a new structured YAML configuration (within the `config/yaml/` directory), the primary configuration file currently processed by key scripts like `src/scripts/database/populate_symbol_metadata.py` is `config/market_symbols.yaml`. This file defines all instruments the system manages.

### `config/market_symbols.yaml` (Active Primary Configuration)

This YAML file is central to defining symbols, their properties, data sources, and how their metadata is stored in the `symbol_metadata` table. The `populate_symbol_metadata.py` script reads this file directly.

**Overall Structure:**

```yaml
settings:
  default_start_date: '2000-01-01' # Optional: Fallback start date if not in symbol config

indices:
  - symbol: $VIX.X
    base_symbol: $VIX.X  # Often same as symbol for indices/equities
    description: CBOE Volatility Index
    exchange: CBOE
    type: index            # Used to determine asset_type in symbol_metadata
    source: cboe           # Preferred data source
    frequencies: [daily]   # Or [ { unit: daily, interval: 1, source: cboe, raw_table: market_data_cboe } ]
    start_date: '1990-01-02'
    # ... other specific fields for VIX ...

futures:
  - symbol: ES
    base_symbol: ES
    description: E-mini S&P 500 Futures (active month)
    exchange: CME
    type: future
    default_source: tradestation # Can be overridden by specific frequency
    frequencies: ['1min', '15min', 'daily']
    start_date: '2003-01-01'
    historical_contracts:
      patterns: [H, M, U, Z]
      start_year: 2003
    # ... other future-specific fields ...

  - continuous_group: # Defines a group of related continuous future contracts
      identifier_base: "@VX"         # e.g., "@VX", "@ES"
      description_template: "{nth_month} VIX Continuous Future"
      exchange: CBOE
      type: continuous_future
      source: in_house         # Or 'tradestation' if applicable
      month_codes: ["1", "2", "3", "4", "5", "6", "7", "8"] # Corresponds to Nth month
      settings_code: "01XN"         # Suffix for symbol, e.g. "01XN" for @VX=101XN, "02XC" for @ES=102XC
      frequencies: [daily]          # Frequencies apply to all generated symbols
      start_date: '2006-01-01'      # Start date for the whole group
      # Base properties for all generated symbols in this group
      # These will be part of the `additional_metadata` for each generated symbol

equities:
  - symbol: SPY
    base_symbol: SPY
    description: SPDR S&P 500 ETF Trust
    exchange: NYSE
    type: equity
    source: tradestation
    frequencies: [daily, '1min']
    start_date: '1993-01-29'
    # ... other equity-specific fields ...

# ... other categories like forex, crypto ...
```

**Key Fields Interpreted by `populate_symbol_metadata.py`:**

*   **Top-level Categories**: `indices`, `futures`, `equities`, etc. The script iterates through these.
*   **Common Item Fields (for each entry in a category list):**
    *   `symbol`: (Required) The primary unique identifier for the instrument (e.g., `$VIX.X`, `ES`, `SPY`, or for continuous groups, this is constructed, e.g. `@VX=101XN`). This populates `symbol_metadata.symbol`.
    *   `base_symbol`: The root symbol (e.g., `ES`, `VX`). If not provided, defaults to `symbol`. Populates `symbol_metadata.base_symbol`.
    *   `description`: Human-readable description. Populates `symbol_metadata.description`.
    *   `exchange`: Trading exchange. Populates `symbol_metadata.exchange`.
    *   `type`: Asset type (e.g., `future`, `continuous_future`, `index`, `equity`). Populates `symbol_metadata.asset_type`.
    *   `source` / `default_source`: Preferred data source (e.g., `tradestation`, `cboe`, `in_house`, `polygon`). Used by `determine_metadata_for_interval` to set `symbol_metadata.data_source`.
    *   `frequencies`: A list defining the data intervals. Can be:
        *   A list of strings (e.g., `['1min', '15min', 'daily']`). `parse_frequency` helper is used.
        *   A list of dictionaries, each specifying `unit`, `interval`, and optionally `source`, `raw_table`, etc. to override defaults for that specific frequency.
        This list determines how many rows are created in `symbol_metadata` for this base item (one per frequency).
    *   `start_date`: (Important) The earliest date for historical data for this symbol. Populates `symbol_metadata.start_date`. If missing, the script may use `settings.default_start_date`.
    *   Other fields: Any other fields in the item's configuration (e.g., `historical_contracts`, `target_table`, `historical_script_path`, `update_script_path`) are used by `determine_metadata_for_interval` or stored in the `symbol_metadata.additional_metadata` JSON column.

*   **`continuous_group` specific fields:**
    *   `identifier_base`: The prefix for generated symbols (e.g., `@VX`).
    *   `month_codes`: A list of strings (e.g., `["1", "2"]`) representing the Nth month for continuous contracts. Each code generates a unique symbol (e.g., `@VX=1...`, `@VX=2...`).
    *   `settings_code`: A suffix appended to the generated symbol (e.g., `01XN` leads to `@VX=101XN`).
    *   `description_template`: A template string like `"{nth_month} VIX Continuous Future"` used to generate descriptions. `{nth_month}` is replaced (e.g., "1st", "2nd").
    *   The `frequencies`, `start_date`, `source`, `type`, `exchange` defined within the `continuous_group` apply to all symbols generated from that group.
    *   The entire `continuous_group` block (minus the generation-specific keys like `identifier_base`, `month_codes`) forms the basis of the `item_config` that gets stored in `additional_metadata` for each generated continuous symbol.

**Relationship to `symbol_metadata` Table:**

The `config/market_symbols.yaml` file is the direct input for the `src/scripts/database/populate_symbol_metadata.py` script. This script processes each symbol and `continuous_group` defined in the YAML, and for each frequency specified, it creates or updates an entry in the `symbol_metadata` table. The `symbol_metadata` table then acts as the central registry that other data loading and processing scripts (like `update_all_market_data_v2.py`) use to understand how to handle each instrument at each interval.

The `additional_metadata` column in `symbol_metadata` stores the full YAML configuration snippet for the specific symbol/interval as a JSON string, providing a complete record of its original definition.

### `config/yaml/*.yaml` (New Structured Configuration - Aspirational/Future Use)

The files within `config/yaml/` (e.g., `exchanges.yaml`, `futures.yaml`, `indices.yaml`) represent a newer, more structured approach to configuration. While defined in this guide, key operational scripts like `populate_symbol_metadata.py` and `update_all_market_data_v2.py` **currently rely on `config/market_symbols.yaml`**.

Future development may transition these scripts to use this new structured format. For now, ensure `config/market_symbols.yaml` is maintained accurately for current system operation.

### exchanges.yaml

Defines trading exchanges with their calendars, timezones, and session information.

```yaml
version: "1.0"

exchanges:
  CME:
    name: "Chicago Mercantile Exchange"
    timezone: "America/Chicago"
    country: "US"
    holidays:
      - name: "New Year's Day"
        rule: "USNewYearsDay"
    trading_sessions:
      regular:
        start: "08:30"
        end: "15:15"
    calendars:
      CME_Equity:
        description: "CME Equity Index Futures Trading Calendar"
        holidays: "inherit"  # Inherit from parent exchange

# Calendar calculation rules
calendar_rules:
  USNewYearsDay:
    date: "January 1"
    observed_if_weekend: true
```

### data_sources.yaml

Defines data sources and their connection parameters.

```yaml
version: "1.0"

data_sources:
  tradestation:
    type: "api"
    base_url: "https://api.tradestation.com/v3"
    auth_type: "oauth2"
    rate_limit: 60
    retry_strategy:
      max_retries: 3
      backoff_factor: 2
```

### cleaning_rules.yaml

Defines data cleaning rules and thresholds.

```yaml
version: "1.0"

cleaning_rules:
  price_spikes:
    max_change_percent: 20.0
    min_price: 0.01
    max_price: 1000000.0
```

## Instrument-Specific Configuration

### Futures Configuration

The `futures.yaml` file defines futures contracts with their specifications, roll rules, and exchange mappings.

#### Basic Structure

```yaml
version: "1.0"

# Templates for common futures properties
templates:
  equity_index_futures:
    default_source: "tradestation"
    default_raw_table: "market_data"
    frequencies: ["1min", "15min", "daily"]
    
# Futures definitions
futures:
  ES:
    inherit: "equity_index_futures"
    name: "E-mini S&P 500 Futures"
    description: "E-mini S&P 500 Futures"
    exchange: "CME"
    exchange_ref: "${exchanges.CME}"
    calendar: "CME_Equity"
    calendar_ref: "${exchanges.CME.calendars.CME_Equity}"
    contract_info:
      patterns: ["H", "M", "U", "Z"]
      start_year: 2003
      num_active_contracts: 2
    continuous_contracts:
      - identifier: "@ES=1P75V"
        description: "ES Continuous Contract (Panama Method)"
        method: "panama"
        ratio_limit: 0.75
```

#### Required Fields

1. **Basic Information**
   - `base_symbol`: Root symbol (e.g., 'ES', 'NQ', 'VX')
   - `description`: Human-readable description
   - `exchange`: Primary exchange
   - `calendar`: Trading calendar identifier
   - `start_date`: Earliest date for data
   - `source`: Data source ('cboe' or 'tradestation')

2. **Data Collection Parameters**
   - `frequencies`: List of data frequencies
   - `num_active_contracts`: Number of contracts to track
   - `is_continuous`: Whether to maintain continuous contracts

3. **Historical Contract Identification**
   - `historical_contracts.patterns`: Month codes
   - `historical_contracts.start_year`: Start year
   - `historical_contracts.start_month`: (Optional) Start month

4. **Expiry Rules**
   - `day_type`: Type of rule ('friday', 'wednesday', 'business_day')
   - `day_number`: Occurrence number
   - `adjust_for_holiday`: Whether to adjust for holidays

#### Month Codes

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

### Indices Configuration

The `indices.yaml` file defines market indices with their calculation methodologies.

```yaml
version: "1.0"

# Templates for common index properties
templates:
  equity_index:
    type: "index"
    default_source: "tradestation"
    
# Index definitions
indices:
  VIX:
    inherit: "volatility_index"
    symbol: "$VIX.X"
    name: "CBOE Volatility Index"
    exchange: "CBOE"
    exchange_ref: "${exchanges.CBOE}"
    calculation_methodology:
      type: "implied_volatility"
      source_options: "SPX"
      term: "30 days"
    related_futures: "VX"
```

### ETFs Configuration

The `etfs.yaml` file defines ETFs with their NAV calculation details.

```yaml
version: "1.0"

# Templates for common ETF properties
templates:
  equity_index_etf:
    type: "ETF"
    asset_class: "equity"
    default_source: "tradestation"
    
# ETF definitions
etfs:
  SPY:
    inherit: "equity_index_etf"
    symbol: "SPY"
    name: "SPDR S&P 500 ETF Trust"
    exchange: "NYSE"
    exchange_ref: "${exchanges.NYSE}"
    expense_ratio: 0.0945
    tracking_index: "SPX"
    tracking_index_ref: "${indices.SPX}"
```

### Equities Configuration

The `equities.yaml` file defines equities with their corporate action handling.

```yaml
version: "1.0"

# Templates for common equity properties
templates:
  common_stock:
    type: "Stock"
    asset_class: "equity"
    default_source: "tradestation"
    
# Equity definitions
equities:
  AAPL:
    inherit: "common_stock"
```

## Advanced Features

### Templates and Inheritance

Use templates to define common properties and inherit from them:

```yaml
templates:
  equity_index_futures:
    default_source: "tradestation"
    frequencies: ["1min", "15min", "daily"]

futures:
  ES:
    inherit: "equity_index_futures"
    # Add or override specific properties
```

### Cross-File References

Reference settings from other configuration files:

```yaml
futures:
  ES:
    exchange_ref: "${exchanges.CME}"
    calendar_ref: "${exchanges.CME.calendars.CME_Equity}"
```

### Environment Variables

Use environment variables in configuration:

```yaml
data_sources:
  tradestation:
    api_key: "${TRADESTATION_API_KEY}"
    api_secret: "${TRADESTATION_API_SECRET}"
```

### Environment-Specific Configuration

Override settings for different environments:

```yaml
# config/yaml/environments/development/exchanges.yaml
exchanges:
  CME:
    trading_sessions:
      regular:
        start: "09:30"  # Override for development
```

## Best Practices

1. **Source Accuracy**
   - Ensure `source` field correctly reflects data origin
   - Use `data_source: tradestation` for continuous contracts built from TradeStation data

2. **Calendar Specificity**
   - Use precise calendar names (e.g., `CME_Equity`, `CFE`)
   - Ensure calendars match exchange requirements

3. **Expiry Rules**
   - Define carefully for individual futures
   - Enable holiday adjustments for US contracts
   - Use specific day rules for index futures
   - Use business day rules for commodity futures

4. **Data Frequencies**
   - Include all necessary frequencies for individual contracts
   - Consider storage implications
   - Use daily frequency for continuous contracts

5. **Symbol Metadata**
   - Understand that YAML populates `symbol_metadata` table
   - Ensure `asset_type` and `data_source` correctly reflect processing requirements
   - Use appropriate values for continuous contracts

6. **Configuration Organization**
   - Use templates to reduce duplication
   - Maintain consistent structure across files
   - Document any non-standard configurations

7. **Validation**
   - Use JSON Schema validation
   - Test configurations in development environment
   - Validate cross-file references

8. **Security**
   - Store sensitive data in environment variables
   - Use secure credential management
   - Implement proper access controls

## Common Configurations

### Equity Index Futures (ES, NQ)
```yaml
futures:
  ES:
    inherit: "equity_index_futures"
    name: "E-mini S&P 500 Futures"
    exchange: "CME"
    calendar: "CME_Equity"
    contract_info:
      patterns: ["H", "M", "U", "Z"]
      start_year: 2003
      num_active_contracts: 2
    expiry_rule:
      day_type: "friday"
      day_number: 3
      adjust_for_holiday: true
```

### Volatility Futures (VX)
```yaml
futures:
  VX:
    inherit: "volatility_futures"
    name: "CBOE Volatility Index Futures"
    exchange: "CBOE"
    calendar: "CFE"
    contract_info:
      patterns: ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
      start_year: 2004
      num_active_contracts: 9
    expiry_rule:
      day_type: "wednesday"
      day_number: 3
      adjust_for_holiday: true
```

### Continuous Contracts
```yaml
futures:
  ES:
    continuous_contracts:
      - identifier: "@ES=1P75V"
        description: "ES Continuous Contract (Panama Method)"
        method: "panama"
        ratio_limit: 0.75
      - identifier: "@ES=102XC"
        description: "ES Continuous Contract (Calendar Method)"
        method: "calendar"
        roll_days: 2
``` 