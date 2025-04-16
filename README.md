# Financial Data System

A comprehensive system for managing and analyzing financial data, including market data, economic data, and account data.

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
├── docs/                   # Documentation
│   ├── README.md           # Main documentation
│   └── SCRIPTS.md          # Script documentation
├── logs/                   # Log files
├── output/                 # Output files
├── scripts/                # Command-line scripts
│   ├── ai                  # AI interface wrapper for Unix/Linux/Mac
│   ├── ai.bat              # AI interface wrapper for Windows
│   ├── ai-llm              # AI interface LLM wrapper for Unix/Linux/Mac
│   ├── ai-llm.bat          # AI interface LLM wrapper for Windows
│   ├── backup              # Database backup wrapper for Unix/Linux/Mac
│   └── backup.bat          # Database backup wrapper for Windows
├── src/                    # Source code
│   ├── ai/                 # AI interface
│   │   ├── ai_interface.py         # Basic AI interface
│   │   └── ai_interface_llm.py     # LLM-powered AI interface
│   ├── agents/             # Agent modules
│   ├── api/                # API modules
│   ├── database/           # Database modules
│   ├── scripts/            # Python scripts
│   │   ├── backup_database.py      # Database backup script
│   │   ├── generate_continuous_contract.py  # Continuous contract generation
│   │   └── scheduled_backup.py     # Scheduled backup script
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

The system provides a natural language interface for interacting with the various tools and agents. You can use it with simple English commands:

On Unix/Linux/Mac:
```
./scripts/ai "plot SPY for the last 30 days"
./scripts/ai-llm "plot SPY for the last 30 days"
```

On Windows:
```
scripts\ai.bat "plot SPY for the last 30 days"
scripts\ai-llm.bat "plot SPY for the last 30 days"
```

To list available tools:
```
./scripts/ai --list
./scripts/ai-llm --list
```

### Continuous Contract Generation

The system supports continuous futures contract generation based on configurable expiry rules defined in `config/market_symbols.yaml`. Rollovers occur *on* the expiry date of the front contract.

```bash
# Generate VXc1 and VXc2 continuous contracts based on config
python src/scripts/market_data/generate_continuous_futures.py --root-symbol VX --num-contracts 2

# Generate only VXc1
python src/scripts/market_data/generate_continuous_futures.py --root-symbol VX --num-contracts 1
```

Features:
- Reads expiry rules and other settings from `config/market_symbols.yaml`.
- Calculates expiry dates based on rules (e.g., VIX rule: Wednesday before the 3rd Friday, adjusted for holidays).
- Handles rollovers correctly on the calculated expiry date.
- Stores generated contracts in the `continuous_contracts` table, including the `underlying_symbol` for each data point.
- Comprehensive logging of generation and rollover events.

### Continuous Contract Verification

A verification script helps ensure the quality and consistency of the generated continuous contracts.

```bash
# Verify all VX continuous contracts (VXc1, VXc2, etc.)
python src/scripts/analysis/verify_vx_continuous.py --symbol-prefix VXc

# Verify with a custom price gap threshold (e.g., 15%)
python src/scripts/analysis/verify_vx_continuous.py --symbol-prefix VXc --gap-threshold 0.15
```

Checks Performed:
- **Sunday Data:** Identifies any data points incorrectly recorded on a Sunday.
- **Price Gaps:** Detects large day-over-day percentage changes in the closing price (configurable threshold).
- **Date Gaps:** Finds missing trading days (excluding weekends and known holidays based on `config/market_symbols.yaml`).
- **Rollover Consistency:** Compares actual rollover dates (when `underlying_symbol` changes) against the expected expiry dates calculated using the same logic as the generation script.

### Database Backup

The system includes a backup utility to protect your financial data. You can run it manually or set it up as a scheduled task.

#### Manual Backup

On Unix/Linux/Mac:
```
./scripts/backup [options]
```

On Windows:
```
scripts\backup.bat [options]
```

Options:
- `-d, --database PATH`: Path to the database file (default: ./data/financial_data.duckdb)
- `-o, --output DIR`: Output directory for backups (default: ./backups)
- `-r, --retention DAYS`: Number of days to keep backups (default: 30)
- `-v, --verbose`: Enable verbose output

#### Scheduled Backup

For automated backups, you can set up a cron job (Unix/Linux/Mac) or a scheduled task (Windows):

**Unix/Linux/Mac (cron):**
```
# Run backup daily at 2 AM
0 2 * * * cd /path/to/financial-data-system && ./venv/bin/python src/scripts/scheduled_backup.py >> logs/backup.log 2>&1
```

**Windows (Task Scheduler):**
Create a scheduled task that runs:
```
cmd /c "cd C:\path\to\financial-data-system && venv\Scripts\python.exe src\scripts\scheduled_backup.py >> logs\backup.log 2>&1"
```

The scheduled backup script reads configuration from environment variables:
- `BACKUP_DIR`: Directory to store backups (default: ./backups)
- `RETENTION_DAYS`: Number of days to keep backups (default: 30)
- `DATABASE_PATH`: Path to the database file (default: ./data/financial_data.duckdb)

### Available Commands

The AI interface supports a wide range of commands, including:

- Plotting data: `plot SPY for the last 30 days`
- Generating continuous contracts: `generate continuous contract for ES futures`
- Checking database quality: `check the database for missing data`
- Fetching market data: `fetch SPY data for the last month`
- Analyzing data: `analyze performance of AAPL over the last 6 months`
- And many more...

## Recent Changes

- Added natural language interface for interacting with the system
- Integrated LLM services for improved command understanding
- Improved continuous contract generation with configurable rollover logic
- Enhanced data quality checks and validation
- Reorganized project structure for better maintainability
- Added database backup functionality with retention policy
- Added price discrepancy detection for continuous contracts
- Implemented multiple rollover methods for futures contracts
- Added comprehensive script documentation
- Refactored continuous futures generation (`generate_continuous_futures.py`)
- Added continuous futures verification script (`verify_vx_continuous.py`)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 