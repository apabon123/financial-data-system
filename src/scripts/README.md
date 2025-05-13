# scripts/

This directory contains automation and utility scripts for database management, data updates, analysis, and other operational tasks.

## Subdirectories
- `analysis/`: Scripts for data analysis and reporting.
- `database/`: Scripts for database setup, migration, and maintenance.
- `market_data/`: Scripts for updating and managing market data.
- `utilities/`: Miscellaneous utility scripts.

Scripts are organized by their primary function to facilitate maintenance and discovery.

## Directory Organization

```
scripts/
├── market_data/           # Scripts for market data retrieval and processing
│   ├── vix/               # VIX/VX-specific market data scripts
│   ├── es/                # ES-specific market data scripts (future)
│   └── ...                # Other symbol-specific directories (future)
├── analysis/              # Scripts for data analysis and visualization
│   ├── vix/               # VIX/VX-specific analysis scripts
│   ├── es/                # ES-specific analysis scripts (future)
│   └── ...                # Other symbol-specific directories (future)
├── database/              # Database management scripts
└── utility/               # Utility scripts for various purposes
```

## Main Scripts

- `market_data/update_all_market_data.py`: The main script for daily updates of all market data. This script coordinates the update process for all symbols, calling the appropriate symbol-specific scripts as needed.

## Symbol-Specific Scripts

Scripts that are specific to a particular symbol or instrument (e.g., VIX, ES) are placed in their own subdirectories to keep the codebase organized and make it easy to add support for new symbols.

### VIX/VX Scripts

- `market_data/vix/update_vix_index.py`: Updates the VIX Index
- `market_data/vix/update_vx_futures.py`: Updates VX futures contracts
- `market_data/vix/fill_vx_continuous_gaps.py`: Fills historical gaps in VXc1/VXc2

### Generic Scripts

Scripts that apply to multiple symbols or provide generic functionality remain in the parent directories.

- `market_data/generate_continuous_futures.py`: Generates continuous futures contracts for any symbol
- `market_data/fetch_market_data.py`: Generic market data fetching utilities 