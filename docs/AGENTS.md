# Agent Specifications

Each agent in this system follows the single-file agent pattern, focusing on a single responsibility.

## Data Collection Agents

### TradeStation Market Data Agent

**Purpose**: Retrieve market data from TradeStation API and prepare it for storage in DuckDB

**Inputs**:
- API credentials (key, secret)
- Symbol list
- Timeframe (minute/daily)
- Interval value (1, 5, 15, etc.)
- Date range (start_date, end_date)
- Optional parameters (adjust for splits/dividends, include up/down volume)

**Outputs**:
- Normalized OHLCV data in a format ready for storage
- Error logs for failed requests
- Rate limit status

**Behavior**:
1. Authenticate with TradeStation API using OAuth
2. Request data for each symbol in batches
3. Convert responses to standardized format matching market_data schema
4. Handle rate limiting with exponential backoff
5. Process errors and retry failed requests
6. Return consolidated data

### TradeStation Account Data Agent

**Purpose**: Retrieve account data from TradeStation API including balances, positions, orders, and trades

**Inputs**:
- API credentials (key, secret)
- Account ID(s)
- Data type (balances, positions, orders, trades)
- Date range for historical data
- Optional filters (symbol, status, etc.)

**Outputs**:
- Normalized account data in a format ready for storage
- Error logs for failed requests
- Rate limit status

**Behavior**:
1. Authenticate with TradeStation API using OAuth
2. Request account data based on specified type
3. Convert responses to standardized format matching corresponding schema tables
4. Handle rate limiting with exponential backoff
5. Process errors and retry failed requests
6. Return consolidated data

### Economic Data API Agent

**Purpose**: Retrieve economic data from sources like FRED and prepare it for storage

**Inputs**:
- API credentials for economic data sources
- Indicator list
- Date range
- Frequency (daily, weekly, monthly, quarterly)

**Outputs**:
- Normalized economic data ready for storage
- Error logs for failed requests
- Last revision information

**Behavior**:
1. Authenticate with economic data APIs
2. Request data for each indicator
3. Convert responses to standardized format
4. Track data revisions and updates
5. Handle rate limiting and errors
6. Return consolidated data

### Web Scraping Agent

**Purpose**: Collect additional financial data from websites not available via APIs

**Inputs**:
- URL list
- Scraping patterns
- Data category
- Update frequency

**Outputs**:
- Structured data ready for storage
- Scraping logs
- Error reports

**Behavior**:
1. Request web pages with appropriate headers and delays
2. Extract data using defined patterns
3. Validate extracted data against expected format
4. Handle timeouts and errors
5. Return structured data

## Data Processing Agents

### Data Normalization Agent

**Purpose**: Transform and standardize data from various sources

**Inputs**:
- Raw data from collection agents
- Transformation rules
- Target schema

**Outputs**:
- Normalized data ready for storage
- Transformation logs
- Data quality metrics

**Behavior**:
1. Apply transformation rules to raw data
2. Handle missing values and outliers
3. Validate against target schema
4. Generate data quality metrics
5. Return normalized data

### Data Validation Agent

**Purpose**: Ensure data quality and integrity

**Inputs**:
- Data to validate
- Validation rules
- Historical reference data

**Outputs**:
- Validation report
- Flagged records
- Quality score

**Behavior**:
1. Apply validation rules to data
2. Compare with historical patterns
3. Detect anomalies and outliers
4. Score data quality
5. Generate validation report

### Derived Indicators Agent

**Purpose**: Calculate technical indicators from market data

**Inputs**:
- Market data
- Indicator specifications
- Parameters
- Timeframes

**Outputs**:
- Calculated indicator values
- Computation logs

**Behavior**:
1. Retrieve required market data
2. Apply indicator calculations
3. Handle edge cases and special conditions
4. Validate results
5. Return calculated indicators

## Storage Agents

### DuckDB Write Agent

**Purpose**: Write data to DuckDB efficiently

**Inputs**:
- Processed data from collection/processing agents
- Target table
- Write mode (insert, upsert, replace)
- Batch size

**Outputs**:
- Write operation status
- Row counts
- Performance metrics

**Behavior**:
1. Connect to DuckDB database
2. Prepare data for writing
3. Execute write operation in batches
4. Handle conflicts and errors
5. Report operation status and performance

### Schema Management Agent

**Purpose**: Manage database schema evolution

**Inputs**:
- Schema change specifications
- Current schema version
- Migration scripts

**Outputs**:
- Migration status
- New schema version
- Backup status

**Behavior**:
1. Determine required schema changes
2. Create backup of current schema
3. Apply migrations in correct order
4. Verify schema integrity
5. Update schema version metadata

## Query Agents

### Data Retrieval Agent

**Purpose**: Retrieve and format data from DuckDB

**Inputs**:
- Query parameters (symbols, date range, timeframe)
- Output format
- Filtering criteria

**Outputs**:
- Formatted query results
- Query performance metrics

**Behavior**:
1. Construct optimized SQL query
2. Execute query with appropriate indices
3. Format results according to specifications
4. Handle large result sets efficiently
5. Return formatted data

### Analysis Agent

**Purpose**: Perform analysis on financial data

**Inputs**:
- Analysis type
- Data parameters
- Configuration options

**Outputs**:
- Analysis results
- Visualization data
- Statistical metrics

**Behavior**:
1. Retrieve required data
2. Apply analytical methods
3. Generate statistics and metrics
4. Prepare visualization data
5. Return comprehensive analysis results

## Agent CLI Structure

Each agent should support the following command-line interface:

```bash
# Basic usage
python <agent_name>.py -d ./path/to/database.duckdb -q "natural language query"

# Additional options
python <agent_name>.py -d ./path/to/database.duckdb -q "query" -c 5 -v
```

Parameters:
- `-d, --database`: Path to DuckDB database file
- `-q, --query`: Natural language query to process
- `-c, --compute_loops`: Number of reasoning iterations (default: 3)
- `-v, --verbose`: Enable verbose output

## Agent Implementation Template

```python
#!/usr/bin/env python
"""
Single-file agent template following the agent pattern.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import duckdb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Agent:
    """Base agent template with core functionality."""
    
    def __init__(
        self, 
        database_path: str,
        verbose: bool = False,
        compute_loops: int = 3
    ):
        """Initialize the agent.
        
        Args:
            database_path: Path to DuckDB database
            verbose: Enable verbose output
            compute_loops: Number of reasoning iterations
        """
        self.database_path = database_path
        self.verbose = verbose
        self.compute_loops = compute_loops
        self.conn = None
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._connect_database()
    
    def _connect_database(self) -> None:
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.database_path)
            logger.debug(f"Connected to database: {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        Args:
            query: Natural language query to process
            
        Returns:
            Dict containing the results and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Implement agent-specific query processing
        results = self._execute_compute_loops(query)
        
        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_compute_loops(self, query: str) -> Any:
        """Execute reasoning iterations.
        
        Args:
            query: Query to process
            
        Returns:
            Processed results
        """
        # Implement agent-specific reasoning process
        result = None
        
        for i in range(self.compute_loops):
            logger.debug(f"Compute loop {i+1}/{self.compute_loops}")
            # Process iteration
            
        return result
    
    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.add_argument(
        "-d", "--database", 
        required=True, 
        help="Path to DuckDB database file"
    )
    parser.add_argument(
        "-q", "--query", 
        required=True, 
        help="Natural language query to process"
    )
    parser.add_argument(
        "-c", "--compute_loops", 
        type=int, 
        default=3, 
        help="Number of reasoning iterations (default: 3)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize agent
        agent = Agent(
            database_path=args.database,
            verbose=args.verbose,
            compute_loops=args.compute_loops
        )
        
        # Process query
        result = agent.process_query(args.query)
        
        # Output result
        print(result)
        
        # Clean up
        agent.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## TradeStation API Agent Implementation Notes

When implementing the TradeStation API agent specifically, consider:

1. **Authentication**: TradeStation uses OAuth 2.0 for API authentication. You'll need to:
   - Store client ID and secret in environment variables
   - Implement OAuth token acquisition and refresh
   - Handle token expiration gracefully

2. **Endpoint structure**: For market data, use the barcharts endpoint:
   ```
   https://api.tradestation.com/v3/marketdata/barcharts/{symbol}?interval={interval}&unit={unit}
   ```

3. **Rate limiting**: TradeStation has the following rate limits:
   - 250 requests per 5-minute interval for accounts, orders, balances, and positions
   - Implement exponential backoff when hitting rate limits

4. **Response handling**: The API returns JSON that needs transformation to match our schema:
   - Convert timestamp format
   - Transform field names (e.g., Open -> open)
   - Handle missing fields with appropriate defaults
   - Validate data quality before storage
