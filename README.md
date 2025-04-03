# Financial Data Management System

A comprehensive data management system that retrieves, processes, stores, and analyzes financial market data from multiple sources using a modular agent-based architecture.

## Overview

This system provides tools for:
- Fetching market data from TradeStation API
- Generating futures contract symbols
- Checking data quality and identifying gaps
- Storing and analyzing financial data

## Project Structure

```
financial-data-system/
├── config/                  # Configuration files
│   └── market_symbols.yaml  # Market symbols configuration
├── data/                    # Data storage (DuckDB)
├── docs/                    # Documentation
├── logs/                    # Log files
├── src/                     # Source code
│   ├── agents/              # Agent modules
│   ├── config/              # Source-specific configuration
│   └── scripts/             # Command-line scripts
├── templates/               # Template files
├── tests/                   # Test files
└── venv/                    # Virtual environment
```

## Getting Started

See the [documentation](docs/README.md) for detailed setup and usage instructions.

## Command-Line Tools

### Fetch Market Data

```bash
# Fetch data for a specific symbol
python src/scripts/fetch_market_data.py --symbol ES

# Fetch data for all symbols defined in the config
python src/scripts/fetch_market_data.py
```

### Check Market Data

```bash
# List all available symbols
python src/scripts/check_market_data.py --list-symbols

# Analyze a specific symbol
python src/scripts/check_market_data.py --symbol ES
```

### Generate Futures Symbols

```bash
# Generate symbols for ES starting from 2020
python src/scripts/generate_futures_symbols.py --base-symbol ES --start-year 2020
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 