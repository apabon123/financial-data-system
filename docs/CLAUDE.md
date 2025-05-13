# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This project is a Financial Data System that manages market data for futures contracts (VIX, ES, NQ) and other financial instruments using DuckDB. The system:

1. **Fetches market data** from TradeStation API, CBOE, and other sources
2. **Processes and stores** data in DuckDB tables
3. **Builds continuous contracts** from individual futures contracts 
4. **Provides tools to analyze and visualize** the stored data

## Common Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Database Operations

```bash
# Initialize database (creates tables and schema)
python src/database/init_database.py -d ./data/financial_data.duckdb

# Backup database
python src/scripts/database/backup_database.py -d ./data/financial_data.duckdb -o ./backups

# Run database queries
python src/scripts/db_inspector.py data_quality.sql
```

### Market Data Update

```bash
# Full market data update (runs all update processes)
update_market_data.bat
# Or on Unix/MacOS
python src/scripts/market_data/update_all_market_data.py --verify

# Check specific symbol contracts
python src/scripts/market_data/view_futures_contracts.py --interval-unit "daily" --interval-value "1" "ES"

# Update VIX futures
python src/scripts/market_data/vix/update_vx_futures.py --config-path ./config/market_symbols.yaml --db-path ./data/financial_data.duckdb

# Build continuous contracts
python src/scripts/market_data/continuous_contract_loader.py @ES=102XC --interval-unit daily --interval-value 1
```

### Data Inspection

```bash
# Interactive database inspection (Windows)
DB_inspect.bat

# View symbol inventory
python src/scripts/market_data/summarize_symbol_inventory.py

# View continuous contracts
python src/scripts/market_data/summarize_symbol_inventory.py --continuous-only

# Check for data quality issues
python src/scripts/analysis/check_market_data.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_market_data_agent.py -v

# Run specific test function
pytest tests/test_schema.py::test_initialize_database -v
```

## Database Structure

The DuckDB database has three main tables:

1. **market_data**: Stores data for most futures contracts (ES, NQ) and other instruments
2. **market_data_cboe**: Specifically for VIX futures and VIX Index data from CBOE
3. **continuous_contracts**: Holds derived continuous contract data series

Key views include:
- daily_bars
- minute_bars 
- five_minute_bars
- weekly_bars
- monthly_bars

## Code Architecture

### Core Components

1. **Update Process** (`update_all_market_data.py`): Orchestrates the entire data update workflow
   - Updates symbol metadata
   - Fetches individual futures and other instruments data
   - Updates CBOE VX futures data
   - Updates continuous contracts

2. **Market Data Fetcher** (`fetch_market_data.py`): Handles fetching data from TradeStation API

3. **Continuous Contract Generator** (`continuous_contract_loader.py`): Builds continuous contracts

4. **Database Interface** (`database.py`): Provides utilities for database access

5. **Inspection Tools** (`view_futures_contracts.py`, `summarize_symbol_inventory.py`): Display and analyze data

### Configuration

The system is configured via:
- **market_symbols.yaml**: Defines symbols, exchanges, and data sources
- **.env**: Contains API credentials (TradeStation, FRED)

## Development Guidelines

1. **Data Flow:**
   - Raw data → market_data or market_data_cboe tables → continuous_contracts table → analysis

2. **Adding New Data Sources:**
   - Update market_symbols.yaml with new symbols
   - Implement or adapt a fetcher for the data source
   - Update the main orchestrator if needed

3. **Improving Continuous Contracts:**
   - Contract roll logic is in continuous_contract_loader.py
   - Roll dates are calculated based on expiration dates

4. **Performance Considerations:**
   - Large data operations should be done with DuckDB's vectorized operations
   - Batch operations when possible to reduce API calls
   - Use database indexes for frequent query patterns