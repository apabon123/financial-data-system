# Financial Data System - VIX Futures Data

A comprehensive system for downloading, processing, and maintaining VIX futures market data.

## Overview

This system provides automated tools for:
- Downloading historical VIX futures data from CBOE
- Maintaining a local database of all VIX futures contracts
- Generating continuous futures contracts (VXc1, VXc2, etc.)
- Filling data gaps and zero prices
- Analyzing and verifying data quality

## Key Components

### Data Download and Update
- `src/scripts/market_data/vix/update_vx_futures.py`: Downloads VIX futures data from CBOE and updates the database
- `src/scripts/market_data/vix/update_vix_index.py`: Updates the VIX Index data

### Continuous Contract Generation
- `src/scripts/market_data/vix/generate_vix_roll_calendar.py`: Creates a roll calendar for VIX futures
- `src/scripts/market_data/generate_continuous_futures.py`: Generates continuous contracts based on the roll calendar

### Data Quality and Cleanup
- `src/scripts/market_data/vix/fill_vx_continuous_gaps.py`: Fills historical gaps in early continuous data using VIX index
- `src/scripts/market_data/vix/fill_vx_zero_prices.py`: Fixes zero prices in continuous contracts using interpolation and reference data

### Analysis and Verification
- `src/scripts/market_data/improved_verify_continuous.py`: Verifies continuous contracts against underlying data
- `src/scripts/analysis/vix/show_vix_continuous_data.py`: Displays VIX and continuous contracts data for analysis
- `src/scripts/analysis/vix/verify_vx_continuous.py`: Verifies VX continuous contracts specifically

## Database Structure

The system uses DuckDB for data storage with the following key tables:
- `market_data`: Raw price data for all instruments
- `futures_roll_calendar`: Roll dates for futures contracts
- `continuous_futures`: Generated continuous contracts

## Usage

### Basic Update
```
python -m src.scripts.market_data.update_vx_futures
```

### Daily Update (Recommended)
The main script for daily updates is `update_all_market_data.py`. This script handles:
- Updating VIX Index data
- Updating VX futures contracts
- Rebuilding continuous VX continuous contracts for the last ~90 days
- Optional verification of the data

Basic usage:
```bash
# Regular daily update (updates last ~90 days of continuous contracts)
python src/scripts/market_data/update_all_market_data.py

# With verification
python src/scripts/market_data/update_all_market_data.py --verify

# Full historical update (rebuilds all continuous contracts from 2004)
python src/scripts/market_data/update_all_market_data.py --full-update
```

For automated scheduled updates, run the setup_tasks.bat script:
```bash
setup_tasks.bat
```

This will create two scheduled tasks:
- VIXUpdate1: Runs daily at 3:50 PM CST
- VIXUpdate2: Runs daily at 7:00 PM CST

Available options for the update script:
- `--db-path`: Custom database path (default: data/financial_data.duckdb)
- `--config-path`: Custom config path (default: config/market_symbols.yaml)
- `--start-date`: Custom start date for continuous contracts (YYYY-MM-DD)
- `--end-date`: Custom end date for continuous contracts (YYYY-MM-DD)
- `--skip-vix`: Skip VIX Index update
- `--skip-futures`: Skip VX futures update
- `--skip-continuous`: Skip continuous contracts update
- `--skip-historical`: Skip historical gap filling

### Full Regeneration
```
python -m src.scripts.market_data.update_vx_futures --full-regen
```

### Display VIX/VXc1/VXc2 Data
```
python -m src.scripts.analysis.show_vix_continuous_data --start-date 2022-01-01 --end-date 2022-01-15
```

## Data Processing Pipeline

1. Download VIX futures data from CBOE
2. Update or insert data in the market_data table 
3. Generate/update roll calendar if needed
4. Generate continuous contracts (VXc1, VXc2, etc.)
5. Fill historical gaps for early data
6. Fix zero prices in continuous contracts
7. Verify continuous contracts against the underlying data

## Maintenance

The system is designed to be run daily to keep the data current. The update script can be scheduled via cron/task scheduler to automate this process.

### Project Organization

To maintain a clean project structure:

1. All Python scripts are organized in appropriate directories:
   - `src/scripts/market_data/`: Scripts for downloading and processing market data
   - `src/scripts/analysis/`: Scripts for data analysis and visualization
   - `src/scripts/utility/`: Utility scripts for database and file operations
   - `src/scripts/database/`: Database management scripts

2. Data files are stored in designated locations:
   - `data/`: Database and core data files
   - `output/data/`: Generated data files
   - `output/reports/`: Analysis reports and CSV outputs
   - `logs/`: Log files from script execution

3. To clean up the project structure, you can run:
   ```
   python reorganize_project.py
   ```

## Features

- Market data retrieval and processing
- Economic data retrieval from various APIs
- Account data management
- Data validation and cleaning
- Technical analysis and derived indicators
- Database management with DuckDB ([Database Documentation](docs/DATABASE.md))
- AI-driven natural language interface for interacting with the system
- Automated database backup with retention policy
- Advanced continuous contract generation with multiple rollover methods
- Price discrepancy detection and validation

## Project Structure

```
financial-data-system/
├── config/                 # Configuration files
├── data/                   # Data storage
│   └── archive/            # Archived data files
├── docs/                   # Documentation
│   ├── README.md           # Main documentation
│   ├── SCRIPTS.md          # Script documentation
│   └── DATABASE.md         # Database documentation
├── logs/                   # Log files
├── output/                 # Output files
│   ├── data/               # Generated data files
│   └── reports/            # Analysis reports
├── src/                    # Source code
│   ├── ai/                 # AI interface
│   │   ├── ai_interface.py         # Basic AI interface
│   │   └── ai_interface_llm.py     # LLM-powered AI interface
│   ├── agents/             # Agent modules
│   ├── api/                # API modules
│   ├── database/           # Database modules
│   ├── scripts/            # Python scripts
│   │   ├── analysis/       # Data analysis scripts
│   │   │   └── vix/        # VIX-specific analysis scripts
│   │   ├── database/       # Database management scripts
│   │   ├── market_data/    # Market data scripts
│   │   │   └── vix/        # VIX-specific market data scripts
│   │   └── utility/        # Utility scripts
│   ├── tradestation/       # TradeStation API modules
│   └── utils/              # Utility modules
├── sql/                    # SQL queries
├── tasks/                  # Scheduled task configuration files
├── templates/              # Template files
├── tests/                  # Test files
├── wrappers/               # Wrapper scripts for easier command-line use
├── .env.template           # Environment variables template
├── .gitignore              # Git ignore file
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── setup.py                # Setup script
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/financial-data-system.git
   cd financial-data-system
   ```

2. Create a virtual environment and activate it:
   ```