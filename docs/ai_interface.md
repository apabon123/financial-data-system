# AI Interface for Financial Data System

The AI Interface provides a natural language interface to the Financial Data System. It allows you to interact with the system using simple English commands, without needing to remember specific command-line arguments or tool names.

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make the AI interface script executable (Unix/Linux/Mac):
   ```
   chmod +x ai
   ```

## Usage

### Basic Usage

```bash
# On Unix/Linux/Mac
./ai "plot SPY for the last 30 days"

# On Windows
ai.bat "plot SPY for the last 30 days"
```

### Command-Line Options

```
Usage: ai [OPTIONS] QUERY

  Run a natural language query through the appropriate tool.

Arguments:
  QUERY  Natural language query [required]

Options:
  --verbose, -v       Enable verbose output
  --database, -d      Path to database [default: data/financial_data.duckdb]
  --list, -l          List available tools
  --help              Show this message and exit.
```

### Examples

```bash
# Plot SPY for the last 30 days
./ai "plot SPY for the last 30 days"

# Generate a continuous contract for ES futures
./ai "generate continuous contract for ES futures"

# Check the database for missing data
./ai "check if there's any missing data in the database"

# Download historical data for TSLA
./ai "download historical data for TSLA from 2020 to 2022"

# Calculate RSI for SPY
./ai "calculate RSI for SPY"

# List all available tools
./ai --list
```

## Available Tools

The AI Interface can route your queries to the following tools:

### Agents

- **analysis**: Performs analysis on financial data including technical, portfolio, correlation, and performance metrics
- **data_retrieval**: Retrieves financial data from various sources
- **data_validation**: Validates and cleans financial data
- **derived_indicators**: Calculates technical indicators and derived metrics
- **economic_data**: Retrieves economic data from various APIs
- **market_data**: Retrieves and processes market data from TradeStation
- **account_data**: Retrieves and processes account data from TradeStation
- **web_scraping**: Scrapes financial data from websites
- **schema_management**: Manages database schemas and data structures
- **data_normalization**: Normalizes financial data for consistency
- **duckdb_write**: Writes data to DuckDB database

### Scripts

- **continuous_contract**: Generates continuous contracts for futures
- **check_market_data**: Checks the quality and completeness of market data
- **fetch_market_data**: Fetches market data from various sources
- **cleanup_market_data**: Cleans up market data in the database
- **generate_futures_symbols**: Generates futures symbols for various contracts
- **visualize_data**: Visualizes financial data with charts and graphs
- **check_db**: Checks the database for data quality and completeness

## How It Works

The AI Interface uses a combination of keyword matching and natural language processing to determine which tool should handle your query. It then routes your query to the appropriate tool with the necessary parameters.

In the current implementation, the routing is done using a simple keyword-based approach. In a future version, it will use a more sophisticated LLM-based approach to better understand your queries.

## Extending the AI Interface

To add a new tool to the AI Interface:

1. Add the tool to the `TOOLS` dictionary in `src/ai_interface.py`
2. Update the `route_command` function to handle queries related to your new tool

## Troubleshooting

If you encounter any issues with the AI Interface:

1. Make sure you have installed all the required dependencies
2. Check that the tool you're trying to use exists and is properly configured
3. Try running the command with the `--verbose` flag to see more detailed output
4. Check the logs for any error messages

## Future Enhancements

- **LLM Integration**: Use a more sophisticated LLM to better understand queries
- **Agent Chaining**: Allow multiple agents to work together
- **History & Context**: Maintain conversation context across requests
- **Learning**: Track successful commands and improve routing over time
- **Configuration**: Easy way to add new agents or customize existing ones
- **Natural Error Handling**: Have the LLM translate error messages into user-friendly explanations 