# Financial Data Management System Specification

## Project Overview
- Goal: Build a data management system to retrieve market data from TradeStation APIs, economic data from various sources, and web scraping for additional data
- Storage: Use DuckDB for time-series financial data storage
- Architecture: Implement using single-file agents pattern

## Data Sources
- TradeStation API for market data (OHLCV)
- Economic data APIs (e.g., FRED, BEA)
- Web scraping for additional data

## Data Structure
- Primary focus on time-series data (OHLCV format)
- Additional fields for up/down volume
- Daily writing of data to database
- Minute and daily data timeframes

## System Components
1. **Data Collection Agents**:
   - TradeStation API agent
   - Economic data API agent
   - Web scraping agent

2. **Data Processing Agents**:
   - Data normalization agent
   - Data validation agent

3. **Storage Agents**:
   - DuckDB write agent
   - Schema management agent

4. **Query Agents**:
   - Data retrieval agent
   - Analysis agent

## Technical Requirements
- Built using Python
- Utilize UV package manager for dependencies
- DuckDB for data storage
- Single-file agent pattern for modularity

## Agent CLI Structure

Each agent should support the following command-line interface:

\`\`\`bash
# Basic usage
uv run <agent_name>.py -d ./path/to/database.duckdb -q \"natural language query\"

# Additional options
uv run <agent_name>.py -d ./path/to/database.duckdb -q \"query\" -c 5 -v
\`\`\`

Parameters:
- \`-d, --database\`: Path to DuckDB database file
- \`-q, --query\`: Natural language query to process
- \`-c, --compute_loops\`: Number of reasoning iterations (default: 3)
- \`-v, --verbose\`: Enable verbose output