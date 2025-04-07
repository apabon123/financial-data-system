# Financial Data Management System

A comprehensive data management system that retrieves, processes, stores, and analyzes financial market data from multiple sources using a modular agent-based architecture.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

![System Architecture Diagram](docs/images/architecture.png)

## Features

- **Data Collection**: Retrieve market data from TradeStation API, economic data from FRED, and additional data through web scraping
- **Data Processing**: Normalize, validate, and transform financial data
- **Data Storage**: Efficiently store time-series financial data using DuckDB
- **Data Analysis**: Calculate technical indicators and perform financial analysis
- **Modular Design**: Built with single-file agents for maximum flexibility and maintainability
- **Command-Line Interface**: Interact with the system through intuitive natural language queries
- **Data Validation**: Check for data gaps and inconsistencies with built-in validation tools
- **Holiday Calendar Support**: Account for market holidays when analyzing data gaps
- **Continuous Contract Generation**: Create continuous futures contracts with multiple rollover methods
- **Price Discrepancy Detection**: Identify and log suspicious price differences at rollover points

## Getting Started

### Prerequisites

- Python 3.9 or higher
- TradeStation API credentials
- FRED API key (for economic data)
- Additional API keys based on your data sources
- DuckDB 1.2.1 or higher (for database operations)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-data-system.git
cd financial-data-system
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
# Install UV package manager
curl -sSf https://install.pydantic.dev | python3

# Install dependencies
uv pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following:
```
TRADESTATION_API_KEY=your-api-key-here
TRADESTATION_API_SECRET=your-api-key-here
FRED_API_KEY=your-api-key-here
```

5. Initialize the database:
```bash
python init_database.py -d ./financial_data.duckdb
```

For detailed setup instructions, please see [SETUP.md](SETUP.md).

## Usage

### Basic Usage

The system uses a collection of agents that can be run via command line:

```bash
# Fetch daily market data for specific symbols
uv run tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31"

# Fetch economic data
uv run economic_data_api_agent.py -d ./financial_data.duckdb -q "fetch economic indicators GDP, CPI, UNEMPLOYMENT_RATE from 2022-01-01 to 2023-12-31"

# Calculate technical indicators
uv run derived_indicators_agent.py -d ./financial_data.duckdb -q "calculate RSI for AAPL using daily data from 2023-01-01 to 2023-12-31 with parameters: period=14"

# Query stored data
uv run data_retrieval_agent.py -d ./financial_data.duckdb -q "get daily close prices for AAPL, MSFT, GOOGL from 2023-01-01 to 2023-12-31"
```

### Advanced Usage

For more complex operations and workflows, check out the [EXAMPLES.md](EXAMPLES.md) file for detailed examples.

## System Architecture

The system is built around the single-file agent pattern, where each agent focuses on a specific responsibility:

1. **Data Collection Agents**:
   - `tradestation_market_data_agent.py`: Retrieves market data from TradeStation API
   - `tradestation_account_data_agent.py`: Retrieves account data from TradeStation API
   - `economic_data_api_agent.py`: Retrieves economic data from sources like FRED
   - `web_scraping_agent.py`: Collects additional financial data from websites

2. **Data Processing Agents**:
   - `data_normalization_agent.py`: Transforms and standardizes data from various sources
   - `data_validation_agent.py`: Ensures data quality and integrity
   - `continuous_contract_generator.py`: Creates continuous futures contracts with multiple rollover methods

3. **Storage Agents**:
   - `duckdb_write_agent.py`: Writes data to DuckDB efficiently
   - `schema_management_agent.py`: Manages database schema evolution

4. **Query Agents**:
   - `data_retrieval_agent.py`: Retrieves and formats data from DuckDB
   - `analysis_agent.py`: Performs analysis on financial data

For a detailed overview of all agents, see [AGENTS.md](AGENTS.md).

## Database Schema

The system uses DuckDB with a well-defined schema for storing:
- Market data (OHLCV)
- Economic indicators
- Account information
- Positions and trades
- Derived technical indicators
- Continuous contracts

See [SCHEMA.md](SCHEMA.md) for the complete database schema documentation.

## Development

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=agents tests/
```

For detailed testing information, see the Testing section in [SETUP.md](SETUP.md).

### Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TradeStation for their market data API
- FRED for economic data access
- DuckDB team for their excellent embedded database

## Support

For issues, questions, or feedback, please create an issue in the GitHub repository.

## Command-Line Tools

The system includes several command-line tools for data management:

### Fetch Market Data

```bash
# Fetch data for a specific symbol
python src/scripts/fetch_market_data.py --symbol ES

# Fetch data for all symbols defined in the config
python src/scripts/fetch_market_data.py

# Use a custom configuration file
python src/scripts/fetch_market_data.py --config path/to/custom_config.yaml
```

### Check Market Data

```bash
# List all available symbols
python src/scripts/check_market_data.py --list-symbols

# Analyze a specific symbol
python src/scripts/check_market_data.py --symbol ES

# Check for gaps with a custom threshold
python src/scripts/check_market_data.py --symbol ES --max-gap-days 5

# Use a specific holiday calendar
python src/scripts/check_market_data.py --symbol ES --calendar EU

# Analyze all symbols
python src/scripts/check_market_data.py
```

### Generate Continuous Contracts

```bash
# Generate ES continuous contract using volume-based rollover
python src/scripts/generate_continuous_contract.py --symbol ES --output ES_backadj --rollover-method volume

# Generate NQ continuous contract using fixed rollover
python src/scripts/generate_continuous_contract.py --symbol NQ --output NQ_backadj --rollover-method fixed

# Force rebuild of existing ES continuous contract
python src/scripts/generate_continuous_contract.py --symbol ES --output ES_backadj --force

# Generate contract with custom date range
python src/scripts/generate_continuous_contract.py --symbol ES --output ES_backadj --start-date 2020-01-01 --end-date 2023-12-31
```

Features:
- Multiple rollover methods:
  * Volume-based: Rolls over when next contract's volume exceeds current within 5 days of expiry
  * Fixed: Rolls over one day before expiration
- Price discrepancy detection with configurable thresholds
- Comprehensive logging of rollover events
- Force mode for rebuilding existing contracts
- Custom date range support
- Interval-specific processing (15minute, 1daily, etc.)

### Generate Futures Symbols

```bash
# Generate symbols for ES starting from 2020
python src/scripts/generate_futures_symbols.py --base-symbol ES --start-year 2020

# Generate symbols for NQ starting from March 2021
python src/scripts/generate_futures_symbols.py --base-symbol NQ --start-year 2021 --start-month 3

# Use an existing config file and save to a custom path
python src/scripts/generate_futures_symbols.py --base-symbol CL --start-year 2020 --config existing_config.yaml --output custom_output.yaml
```

For more detailed information about each tool, use the `--help` flag:

```bash
python src/scripts/fetch_market_data.py --help
python src/scripts/check_market_data.py --help
python src/scripts/generate_continuous_contract.py --help
python src/scripts/generate_futures_symbols.py --help
```

## Holiday Calendars

The system includes holiday calendars for different markets to account for market closures when analyzing data gaps:

### US Market Holidays

The US calendar includes holidays for NYSE, CME, and other US markets:
- New Year's Day (January 1)
- Martin Luther King Jr. Day (Third Monday in January)
- Presidents Day (Third Monday in February)
- Good Friday (Friday before Easter Sunday)
- Memorial Day (Last Monday in May)
- Juneteenth (June 19)
- Independence Day (July 4)
- Labor Day (First Monday in September)
- Thanksgiving Day (Fourth Thursday in November)
- Christmas Day (December 25)

### European Market Holidays

The EU calendar includes common European market holidays:
- New Year's Day (January 1)
- Good Friday (Friday before Easter Sunday)
- Easter Monday (Monday after Easter Sunday)
- Labor Day (May 1)
- Christmas Day (December 25)
- Boxing Day (December 26)

### Asian Market Holidays

The ASIA calendar includes common Asian market holidays:
- New Year's Day (January 1)
- Chinese New Year (Varies by year)
- Labor Day (May 1)
- Mid-Autumn Festival (Varies by year)
- National Day (October 1)
- Christmas Day (December 25)

You can specify which calendar to use with the `--calendar` option in the `check_market_data.py` script:

```bash
python src/scripts/check_market_data.py --symbol ES --calendar EU
```