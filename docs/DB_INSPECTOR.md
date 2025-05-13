# Enhanced DB Inspector

The Enhanced DB Inspector is a comprehensive tool for analyzing, visualizing, and managing financial market data in the DuckDB database. It provides an intuitive interface for exploring database schema, analyzing data quality, and performing market structure analysis.

## Features

### Core Inspection Capabilities
- Interactive SQL query execution with syntax highlighting, history, and auto-completion
- Data exploration by asset class, symbol, date range, and interval
- Schema browser with relationship visualization
- Performance monitoring and query optimization

### Data Quality Analysis
- Missing data detection with heatmaps and gap analysis
- Outlier identification with configurable thresholds
- Consistency checks across data sources (OHLC validation, etc.)
- Statistical summaries for OHLCV data

### Market Structure Tools
- Roll analysis for futures contracts
- Continuous contract inspection with adjustment visualization
- Correlation matrices and cross-asset analysis
- Volatility and volume profile analysis

### Visualization
- Interactive time series charts
- Candlestick charts with indicators
- Volume and liquidity visualization
- Cross-instrument comparison

### Data Management
- Manual data correction interface with audit tracking
- Import/export tools for various formats
- Backup and restore functionality
- Data quality metrics tracking

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required packages (will be checked and optionally installed by launcher):
  - Core dependencies: duckdb, pandas, numpy, pyyaml, rich
  - Visualization dependencies: matplotlib, seaborn
  - Schema visualization: networkx
  - Interactive SQL: prompt_toolkit, pygments

### Installation

#### Automatic Installation (Recommended)
Both launcher scripts (`DB_inspect_enhanced.bat` for Windows and `db_inspect_enhanced` for Linux/Mac) will:
1. Check for required dependencies
2. Prompt to install any missing dependencies
3. Check for optional dependencies that improve functionality
4. Prompt to install any missing optional dependencies

When running the launcher for the first time, you'll be guided through the installation process.

#### Manual Installation
If you prefer to install dependencies manually, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### Minimal Installation
If you only need core functionality and want to minimize dependencies:

```bash
pip install pandas>=2.0.0 rich>=13.7.0 duckdb>=1.2.1 numpy>=1.24.0 pyyaml>=6.0.1
```

Note: With minimal installation, some features will be disabled or have limited functionality.

#### Full Installation (All Features)
For the complete experience with all features:

```bash
pip install pandas>=2.0.0 rich>=13.7.0 duckdb>=1.2.1 numpy>=1.24.0 pyyaml>=6.0.1 networkx>=3.0 prompt_toolkit>=3.0.33 pygments>=2.15.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

### Launching the DB Inspector

#### Windows
Run the `DB_inspect_enhanced.bat` script in the root directory:

```
DB_inspect_enhanced.bat
```

#### Linux/Mac
Run the `db_inspect_enhanced` script in the root directory:

```bash
./db_inspect_enhanced
```

### Command Line Options

```
usage: python -m src.inspector [-h] [--database DATABASE] [--write] [--query QUERY] [--file FILE] 
                              [--schema] [--analyze ANALYZE] [--verbose]

DB Inspector - Financial Market Data Inspection Tool

options:
  -h, --help            show this help message and exit
  --database DATABASE, -d DATABASE
                        Path to database file
  --write, -w           Open database in write mode
  --query QUERY, -q QUERY
                        Execute SQL query and exit
  --file FILE, -f FILE  Execute SQL file and exit
  --schema, -s          Show database schema and exit
  --analyze ANALYZE, -a ANALYZE
                        Analyze data quality for symbol and exit
  --verbose, -v         Enable verbose logging
```

## Using the Inspector

### Main Menu
The main menu provides access to all DB Inspector functionality:

1. Execute SQL Query - Interactive SQL query execution
2. Browse Schema - Explore database schema and relationships
3. Analyze Data Quality - Analyze data quality for symbols
4. Market Structure Tools - Analyze futures contracts and continuous contracts
5. Data Management - Manage, correct, and clean data
6. Database Information - View database statistics
7. Backup/Restore - Backup and restore database
8. Exit - Exit the application

### SQL Executor
- F2: Execute query
- F3: Format query
- F4: Show tables

### Schema Browser
Commands:
- `tables` - List all tables
- `views` - List all views
- `table <name>` - Show details for a specific table
- `view <name>` - Show details for a specific view
- `visualize` - Visualize full database schema
- `visualize <table>` - Visualize schema around a specific table
- `related <table>` - List tables related to a specific table
- `join <table1> <table2>` - Show sample join query between two tables
- `filter <pattern>` - Filter tables by name pattern
- `help` - Show this help
- `exit` - Exit schema browser

### Data Quality Analyzer
Commands:
- `analyze [-s SYMBOL] [-d START_DATE END_DATE] [-i INTERVAL]` - Analyze market data quality
- `config <key> <value>` - Set configuration value
- `config` - Show current configuration
- `reset` - Reset configuration to defaults
- `help` - Show help
- `exit` - Exit data quality analyzer

## Configuration

The DB Inspector configuration is stored in `config/inspector_config.yaml`. This file is created automatically on first run with default values, but can be modified to customize behavior.

Key configuration sections:
- `paths`: File paths for database, SQL files, exports, etc.
- `ui`: User interface settings
- `data_quality`: Thresholds for data quality analysis
- `performance`: Cache and performance settings
- `visualization`: Chart and graph settings
- `backups`: Backup and restore settings
- `market_structure`: Settings for market structure analysis

## Examples

### Running a Quick SQL Query
```bash
./db_inspect_enhanced -q "SELECT * FROM market_data WHERE symbol = 'ES' LIMIT 10"
```

### Analyzing Data Quality for a Symbol
```bash
./db_inspect_enhanced -a "ES"
```

### Exporting Schema Information
```bash
./db_inspect_enhanced -s > schema_info.txt
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
```
Error: Required dependencies missing: pandas, rich
```
- **Solution**: Run the launcher scripts which will offer to install dependencies.
- **Alternative**: Manually install dependencies with `pip install -r requirements.txt`

#### Limited Functionality
```
Warning: Optional dependencies missing: networkx, matplotlib
```
- **Solution**: Install optional dependencies for full functionality.
- **Command**: `pip install networkx>=3.0 matplotlib>=3.7.0 seaborn>=0.12.0 prompt_toolkit>=3.0.33 pygments>=2.15.0`

#### Database Connection Errors
```
Error connecting to database: file not found
```
- **Solution**: Check database path in configuration.
- **Default Path**: `data/financial_data.duckdb`
- **Check Config**: `config/inspector_config.yaml`

#### Permission Issues
```
Error: Permission denied
```
- **Solution 1**: Use write mode: `./db_inspect_enhanced -w`
- **Solution 2**: Check file permissions on database file
- **Solution 3**: Run as administrator if needed

#### Visualization Errors
```
Error: No module named 'matplotlib'
```
- **Solution**: Install matplotlib and seaborn:
- **Command**: `pip install matplotlib>=3.7.0 seaborn>=0.12.0`

#### Interactive SQL Issues
```
Interactive SQL mode requires prompt_toolkit
```
- **Solution**: Install prompt_toolkit and pygments:
- **Command**: `pip install prompt_toolkit>=3.0.33 pygments>=2.15.0`

### Debug Mode

For detailed debugging information, add the `--debug` flag:

```bash
./db_inspect_enhanced --debug
```

This will show detailed stack traces and error information.

### Checking Logs

Logs are stored in the `logs` directory and contain detailed information for troubleshooting:

```bash
# View the last 20 lines of the log
tail -n 20 logs/inspector.log
```

### Still Having Problems?

1. Check the full log file in the `logs` directory
2. Try running with minimal functionality (use the command line instead of interactive mode)
3. Verify your Python version with `python --version` (should be 3.8+)
4. Reinstall dependencies with `pip install --upgrade -r requirements.txt`

## Contributing

Contributions to improve the DB Inspector are welcome! Please follow the project's coding standards and submit pull requests for new features or bug fixes.