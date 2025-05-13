# Financial Data System Overview

## Introduction

A comprehensive system for downloading, updating, managing, and inspecting financial market data, primarily focusing on futures contracts (VIX, ES, NQ) using DuckDB. The system automates key tasks related to maintaining a financial market database and provides tools for data analysis and inspection.

## System Requirements

### Hardware Requirements
- CPU: Multi-core processor recommended for parallel processing
- Memory: Minimum 8GB RAM, 16GB+ recommended for large datasets
- Storage: SSD recommended for improved database performance
- Network: Stable internet connection for API access

### Software Requirements
- Python 3.9 or higher
- UV package manager
- DuckDB
- Required Python packages (see Installation section)
- Required API keys:
  - TradeStation API (client ID and secret)
  - FRED API key for economic data
  - Optional: Other economic data APIs as needed

## Project Structure

```
financial-data-system/
├── backups/              # Database backup files
├── config/               # Configuration files
│   └── market_symbols.yaml
├── data/                 # Data directory
│   ├── financial_data.duckdb  # Main database
│   └── roll_calendars/   # Roll calendar data
├── deprecated/           # Deprecated files (kept for reference)
├── docs/                 # Documentation
├── logs/                 # Log files
├── output/               # Output files (reports, exports, etc.)
│   ├── data/
│   ├── exports/
│   ├── plots/
│   └── reports/
├── src/                  # Source code
│   ├── agents/           # Agent implementations
│   ├── ai/               # AI interface
│   ├── core/             # Core system components
│   │   ├── app.py        # Application class
│   │   ├── config.py     # Configuration management
│   │   ├── database.py   # Database connectivity
│   │   └── logging.py    # Logging system
│   ├── data_sources/     # Data source providers
│   ├── processors/       # Data processing components
│   │   ├── cleaners/     # Data cleaning pipeline
│   │   └── continuous/   # Continuous futures implementation
│   ├── scripts/          # Command-line scripts
│   │   ├── analysis/     # Data analysis scripts
│   │   ├── api/          # API-related scripts
│   │   ├── database/     # Database management scripts
│   │   ├── market_data/  # Market data processing scripts
│   │   ├── resources/    # Resource files (XML, templates, etc.)
│   │   ├── scripts/      # Batch scripts and utilities
│   │   └── utilities/    # Utility scripts
│   ├── sql/              # SQL scripts
│   │   └── sql/
│   ├── templates/        # Template files
│   └── validation/       # Data validation components
├── tests/                # Test code
│   ├── integration/      # Integration tests
│   ├── legacy/           # Legacy tests (kept for reference)
│   ├── unit/             # Unit tests
│   └── validation/       # Validation tests
├── update_market_data_v2.bat  # Main entry point script
└── requirements.txt      # Python dependencies
```

## Core Components

### Data Storage
- `data/financial_data.duckdb`: The central DuckDB database containing:
  - `market_data`: Most futures contracts (ES, NQ) and non-CBOE instruments
  - `market_data_cboe`: Daily VIX futures and VIX Index data
  - Other tables for configuration, continuous contracts, etc.

### Configuration
- `config/market_symbols.yaml`: Defines symbols, exchanges, update sources, and parameters
- `config/config.yaml`: System-wide configuration settings
- Environment variables for API keys and sensitive data

### Update System
- `update_market_data.bat`: Main update script
- `src/scripts/market_data/update_all_market_data.py`: Core orchestrator
- Various data fetchers and processors in `src/scripts/market_data/`

### Inspection Tools
- `DB_inspect.bat`: Interactive command-line interface
- Analysis scripts in `src/scripts/analysis/`
- Market data inspection in `src/scripts/market_data/`

## Installation

### 1. Install UV Package Manager
```bash
curl -sSf https://install.pydantic.dev | python3
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/financial-data-system.git
cd financial-data-system
```

### 3. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. API Configuration
1. Register for required API accounts:
   - [TradeStation Developer Portal](https://developer.tradestation.com/)
   - [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)

2. Create `.env` file with API keys:
```
TRADESTATION_API_KEY=your-client-id
TRADESTATION_API_SECRET=your-client-secret
FRED_API_KEY=your-fred-api-key
```

### 5. Database Initialization
```bash
python init_database.py -d ./financial_data.duckdb
```

## Usage

### 1. Update Market Data
```bash
update_market_data.bat
```

This orchestrates:
- Symbol metadata updates
- Individual instrument updates
- VIX futures updates
- Continuous contract generation
- Intraday data updates

### 2. Inspect Database
```bash
DB_inspect.bat
```

Provides tools for:
- Viewing inventory summaries
- Listing contract details
- Performing data quality checks
- Exporting data

## Technical Specifications

### Performance Expectations
- Data Collection: 500+ symbols daily within 30 minutes
- Minute Data: 10+ symbols per trading day within 15 minutes
- Query Performance: < 1 second for simple queries
- Complex Operations: < 5 minutes for 100 symbols
- Database Size: < 10GB for 1000 symbols/year

### Error Handling
- Comprehensive error categorization
- Exponential backoff for API limits
- Structured logging with rotation
- Input validation and sanitization

### Testing Requirements
- 85% minimum code coverage
- 100% coverage for critical components
- Unit, integration, and mock tests
- Performance and regression testing
- Continuous integration checks

### Data Backup Strategy
- Daily database snapshots
- Write-Ahead Logging (WAL)
- Periodic CSV exports
- Configurable retention policies

### Security
- API credentials in environment variables
- Input sanitization
- Rate limiting
- Read-only mode for production

## Troubleshooting

### Common Issues
1. API Authentication Errors
   - Verify API keys in `.env`
   - Check TradeStation permissions
   - Verify app approval status

2. Database Connection Errors
   - Check directory permissions
   - Verify DuckDB installation

3. Missing Dependencies
   - Reinstall requirements
   - Install specific packages as needed

4. Rate Limiting
   - Implement backoff strategies
   - Use smaller data batches

### Getting Help
1. Check logs in `logs/` directory
2. Review API documentation
3. Open GitHub issues with details

## Next Steps
1. Review examples in `EXAMPLES.md`
2. Explore agent documentation
3. Study database schema
4. Try demo scripts 