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
├── market_symbols.yaml   # Legacy configuration (for backward compatibility)
├── schemas/             # JSON Schema validation files
│   ├── exchanges.json
│   ├── futures.json
│   └── ...
└── yaml/                # New structured YAML files
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