# Guide to Fixing Missing VX Continuous Contracts Data

## Issue Analysis

After investigating the issue with missing VX continuous contracts data, we've identified several key problems:

1. Many days where VIX data exists have missing continuous futures contracts data
2. The underlying raw VX futures contract data is missing for these dates in the `market_data` table
3. There may be issues with the continuous contract mapping table for some dates

## Data Requirements

For continuous futures contracts to be generated correctly, you need:

1. **Raw Futures Data**: The individual VX futures contracts data (VXF24, VXG24, etc.) must be present in the `market_data` table
2. **Contract Mapping**: The `continuous_contract_mapping` table must correctly map continuous symbols (like `@VX=101XN`) to underlying contracts
3. **Date Coverage**: All trading dates should be included in the calendar used for generation

## Solution Steps

### 1. Import Missing VX Futures Data

First, ensure you have all the necessary raw VX futures contract data:

```bash
# Check what VX contracts you currently have data for
python -c "import duckdb; conn = duckdb.connect('data/financial_data.duckdb', read_only=True); print(conn.execute('SELECT DISTINCT symbol FROM market_data WHERE symbol LIKE \'VX%\'').fetchdf())"

# Import missing VX futures data from your data source
# Use your data import script/tool to fetch the missing contracts
python src/scripts/market_data/fetch_futures_data.py --symbol VX --start-date 2023-01-01
```

### 2. Regenerate Contract Mapping Table

Update or regenerate the continuous contract mapping:

```bash
# This will create mappings for VX contracts
python src/scripts/market_data/create_continuous_contract_mapping.py --root-symbol VX --start-date 2023-01-01
```

### 3. Regenerate Continuous Contracts

After the above steps, regenerate the continuous contracts data:

```bash
# Regenerate continuous contracts with the force flag to overwrite existing data
python src/scripts/market_data/generate_continuous_futures.py --symbol VX --start-date 2023-01-01 --force
```

### 4. Verify the Fix

Run the missing data check script to verify the fix:

```bash
# Check for missing dates
python src/scripts/vix_utilities/find_missing_vx_continuous.py
```

## Common Issues and Troubleshooting

1. **Missing Contract Data**: The most common reason for missing continuous data is that the underlying futures contract data is not available. Make sure to import all necessary contracts.

2. **Calendar Issues**: If trading days are missing, check the exchange calendar being used for generation. For VIX/VX, the CBOE calendar should be used.

3. **Mapping Issues**: Sometimes the continuous contract mapping table might not contain entries for certain dates, especially around holidays or weekends. Ensure the mapping table is correctly populated.

4. **Database Consistency**: Ensure all tables (`market_data`, `continuous_contract_mapping`, and `continuous_contracts`) have consistent formats and data types.

## Future Data Pipeline Improvements

To prevent this issue in the future:

1. Set up a daily data pipeline that:
   - Imports new VX futures data
   - Updates the continuous contract mapping
   - Generates continuous contracts

2. Implement data validation checks that verify the completeness of continuous contracts data

3. Create alerts for missing data that notify you when there are gaps in the data

By following these steps and implementing these improvements, you can ensure that your VX continuous contracts data remains complete and accurate. 