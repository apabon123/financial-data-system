# Futures Configuration Guide

This document provides a comprehensive guide to configuring futures contracts in the financial data system.

## Contract Configuration Fields

### Basic Information
- `base_symbol`: The root symbol for the futures contract (e.g., 'ES' for E-mini S&P 500)
- `calendar`: The trading calendar to use (e.g., 'US' for United States)
- `description`: Human-readable description of the contract
- `exchange`: The exchange where the contract trades (e.g., 'CME', 'NYMEX')
- `start_date`: The earliest date for which data should be collected

### Data Collection
- `frequencies`: List of data frequencies to collect
  - Available options: '1min', '15min', 'daily'
- `num_active_contracts`: Number of front-month contracts to maintain

### Historical Contracts
- `historical_contracts`: Configuration for historical data collection
  - `patterns`: List of month codes (e.g., 'H' for March, 'M' for June)
  - `start_month`: Starting month (1-12)
  - `start_year`: Year to begin collecting historical data

### Expiry Rules
The `expiry_rule` section defines when contracts expire:

#### Day Type Options
1. **Specific Day of Month**
   ```yaml
   day_type: friday
   day_number: 3  # Third Friday
   ```

2. **Business Days Before Date**
   ```yaml
   day_type: business_day
   days_before: 3  # Three business days before the target date
   ```

#### Common Settings
- `adjust_for_holiday`: Whether to adjust for holidays (true/false)
- `holiday_calendar`: Calendar to use for holiday adjustments

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