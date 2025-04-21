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
- `src/scripts/market_data/update_vx_futures.py`: Downloads VIX futures data from CBOE and updates the database

### Continuous Contract Generation
- `src/scripts/market_data/generate_vix_roll_calendar.py`: Creates a roll calendar for VIX futures
- `src/scripts/market_data/generate_continuous_futures.py`: Generates continuous contracts based on the roll calendar

### Data Quality and Cleanup
- `src/scripts/market_data/fill_vx_continuous_gaps.py`: Fills historical gaps in early continuous data using VIX index
- `src/scripts/market_data/fill_vx_zero_prices.py`: Fixes zero prices in continuous contracts using interpolation and reference data

### Analysis and Verification
- `src/scripts/market_data/verify_continuous_futures.py`: Verifies continuous contracts against underlying data
- `src/scripts/analysis/show_vix_continuous_data.py`: Displays VIX and continuous contracts data for analysis
- `src/scripts/analysis/detect_vx_outliers.py`: Identifies potential outliers in VIX futures data

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
│   │   ├── database/       # Database management scripts
│   │   ├── market_data/    # Market data scripts
│   │   └── utility/        # Utility scripts
│   ├── tradestation/       # TradeStation API modules
│   └── utils/              # Utility modules
├── sql/                    # SQL queries
├── templates/              # Template files
├── tests/                  # Test files
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
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API credentials:
   ```
   cp .env.template .env
   # Edit .env with your API keys
   ```

## Usage

### Scripts

The system includes numerous scripts for various tasks. For detailed information about all available scripts and their usage, see [SCRIPTS.md](docs/SCRIPTS.md).

### AI Interface

The system provides a natural language interface for interacting with the various tools and agents. You can use it with simple English commands.

To list available tools:
```
./scripts/ai --list
```

### Continuous Contract Generation

The system supports continuous futures contract generation based on configurable expiry rules defined in `config/market_symbols.yaml`. Rollovers occur *on* the expiry date of the front contract.

```bash
# Generate VXc1 and VXc2 continuous contracts based on config
python -m src.scripts.market_data.generate_continuous_futures --root-symbol VX --num-contracts 2
```

Features:
- Reads expiry rules and other settings from `config/market_symbols.yaml`.
- Calculates expiry dates based on rules (e.g., VIX rule: Wednesday before the 3rd Friday, adjusted for holidays).
- Handles rollovers correctly on the calculated expiry date.
- Stores generated contracts in the `continuous_contracts` table, including the `underlying_symbol` for each data point.

### Continuous Contract Verification

A verification script helps ensure the quality and consistency of the generated continuous contracts.

```bash
# Verify all VX continuous contracts (VXc1, VXc2, etc.)
python -m src.scripts.analysis.verify_vx_continuous --symbol-prefix VXc
```

Checks Performed:
- **Sunday Data:** Identifies any data points incorrectly recorded on a Sunday.
- **Price Gaps:** Detects large day-over-day percentage changes in the closing price.
- **Date Gaps:** Finds missing trading days (excluding weekends and known holidays).
- **Rollover Consistency:** Compares actual rollover dates against the expected expiry dates.

### Database Backup

The system includes a backup utility to protect your financial data.

#### Manual Backup

```
python -m src.scripts.database.backup_database
```

Options:
- `-d, --database PATH`: Path to the database file (default: ./data/financial_data.duckdb)
- `-o, --output DIR`: Output directory for backups (default: ./backups)
- `-r, --retention DAYS`: Number of days to keep backups (default: 30)

## Documentation

Full documentation is available in the `docs/` directory:

- [SCRIPTS.md](docs/SCRIPTS.md): Detailed information about all scripts and their usage
- [DATABASE.md](docs/DATABASE.md): Database schema and organization
- [SETUP.md](docs/SETUP.md): Detailed setup instructions

## Recent Changes

- Improved project organization with clear directory structure
- Updated documentation to reflect current structure and usage
- Added script for reorganizing project files
- Added natural language interface for interacting with the system
- Integrated LLM services for improved command understanding
- Improved continuous contract generation with configurable rollover logic
- Enhanced data quality checks and validation
- Added database backup functionality with retention policy
- Added price discrepancy detection for continuous contracts
- Implemented multiple rollover methods for futures contracts

## License

This project is licensed under the MIT License - see the LICENSE file for details. 