# Market Data Update Guide

This guide explains how to use the `update_all_market_data.py` script to keep your financial database up-to-date.

## Overview

The script orchestrates the full update process for financial market data:
1. Updates the VIX Index (`$VIX.X`)
2. Updates active VX futures contracts (approx. 8 contracts)
3. Updates continuous VX contracts (VXc1, VXc2)
4. Fills historical gaps in VXc1 and VXc2 for 2004-2005 (if full update requested)

## Update Scenarios

### Daily Update

For daily updates to keep your database current:

```bash
python -m src.scripts.market_data.update_all_market_data
```

This will:
- Update the VIX Index data
- Update active VX futures contracts
- Update continuous contracts for the last 90 days

### Full Database Update

To perform a complete update of all historical data:

```bash
python -m src.scripts.market_data.update_all_market_data --full-update --verify
```

This will:
- Update the VIX Index data
- Update active VX futures contracts with full regeneration
- Regenerate continuous contracts from 2004-01-01 to present
- Fill historical gaps in 2004-2005 data for VXc1 and VXc2
- Verify the data after updates

### Custom Date Range

To update a specific date range:

```bash
python -m src.scripts.market_data.update_all_market_data --start-date 2023-01-01 --end-date 2023-12-31
```

This will update continuous contracts for the specified date range while still updating the latest VIX and futures data.

### Selective Updates

You can skip specific parts of the update process using the following flags:

- `--skip-vix`: Skip VIX Index update
- `--skip-futures`: Skip VX futures update
- `--skip-continuous`: Skip continuous contracts update
- `--skip-historical`: Skip historical gap filling (only applies with `--full-update`)

For example, to only update VX futures:

```bash
python -m src.scripts.market_data.update_all_market_data --skip-vix --skip-continuous
```

## Verification

To run verification after an update:

```bash
python -m src.scripts.market_data.update_all_market_data --verify
```

This will:
- Run the improved verification script for VXc1 and VXc2
- Display data counts for key symbols

## Troubleshooting

### Missing Configuration

If you get an error about missing configuration files, make sure the `config/market_symbols.yaml` file exists. You can specify a different location with:

```bash
python -m src.scripts.market_data.update_all_market_data --config-path /path/to/config.yaml
```

### Database Errors

If you encounter database errors, you can try:

1. Using a different database path:
```bash
python -m src.scripts.market_data.update_all_market_data --db-path /path/to/database.duckdb
```

2. Rebuilding the database from scratch with a full update:
```bash
python -m src.scripts.market_data.update_all_market_data --full-update
```

### Missing Data or Verification Errors

If verification shows issues with the data:

1. Check that the CBOE website is accessible and data is available
2. Run a full update to ensure all components are synchronized:
```bash
python -m src.scripts.market_data.update_all_market_data --full-update --verify
```

## Additional Documentation

For detailed information about individual components:

- VIX Index updates: `python -m src.scripts.market_data.update_vix_index --help`
- VX futures updates: `python -m src.scripts.market_data.update_vx_futures --help`
- Continuous contract generation: `python -m src.scripts.market_data.generate_continuous_futures --help`
- Gap filling: `python -m src.scripts.market_data.fill_vx_continuous_gaps --help`
- Verification: `python -m src.scripts.market_data.improved_verify_continuous --help` 