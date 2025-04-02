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
- Built using Python 3.9+
- Utilize UV package manager for dependencies
- DuckDB for data storage
- Single-file agent pattern for modularity

## Agent CLI Structure

Each agent should support the following command-line interface:

```bash
# Basic usage
uv run <agent_name>.py -d ./path/to/database.duckdb -q "natural language query"

# Additional options
uv run <agent_name>.py -d ./path/to/database.duckdb -q "query" -c 5 -v
```

Parameters:
- `-d, --database`: Path to DuckDB database file
- `-q, --query`: Natural language query to process
- `-c, --compute_loops`: Number of reasoning iterations (default: 3)
- `-v, --verbose`: Enable verbose output

## System Requirements
- CPU: Multi-core processor recommended for parallel processing
- Memory: Minimum 8GB RAM, 16GB+ recommended for large datasets
- Storage: SSD recommended for improved database performance
- OS: Cross-platform (Linux, macOS, Windows)
- Network: Stable internet connection for API access
- Python: Version 3.9 or higher

## Error Handling and Logging
- All agents must implement comprehensive error handling
- Errors should be categorized as:
  - Critical: System cannot continue (e.g., database connection failure)
  - Warning: Operation can continue but with limitations
  - Info: Non-critical information
- Exponential backoff for API rate limits and connection failures
- Structured logging with timestamps, agent name, and severity level
- Log rotation configuration for production environments

## Performance Expectations
- Data Collection: Should handle 500+ symbols for daily data within 30 minutes
- Minute Data: 10+ symbols for a trading day within 15 minutes
- Query Performance: Simple queries should return in under 1 second
- Complex operations (e.g., calculating derived indicators) should complete within 5 minutes for 100 symbols
- Database size should remain manageable (< 10GB for a year of data for 1000 symbols)

## Security Considerations
- All API credentials stored in environment variables or secure credential store
- No hardcoded credentials in source code
- Sanitize all inputs, especially for web scraping agents
- Input validation for all CLI parameters
- Rate limiting for database connections
- Consider implementing read-only mode for production queries

## Testing Requirements

All code must be thoroughly tested using the following approach:

### Test Coverage Requirements
- Minimum 85% code coverage for all production code
- 100% coverage for critical components (authentication, data validation, storage)
- All public methods and functions must have unit tests
- All error handling paths must be tested

### Types of Tests Required
1. **Unit Tests**:
   - Each agent must have comprehensive unit tests
   - Test all public methods and functions
   - Test edge cases and error conditions
   - Test parameter validation
   - Mock external dependencies (APIs, database)

2. **Integration Tests**:
   - Test interactions between agents
   - Test complete workflows (collection → processing → storage → query)
   - Test database interactions with a test database
   - Test API rate limiting and retry mechanisms

3. **Mock Tests**:
   - All external API calls must be mockable for testing
   - Create realistic mock responses for all API endpoints
   - Test timeout and error response handling
   - Test authentication workflows

4. **Data Quality Tests**:
   - Validate schema conformance
   - Test data type consistency
   - Test for missing values handling
   - Test data transformation logic
   - Test derived calculations for accuracy

5. **Performance Tests**:
   - Test with large datasets (>10,000 records)
   - Measure and validate query response times
   - Test memory usage with large operations
   - Test database write performance

6. **Regression Tests**:
   - Create tests for any bugs discovered
   - Ensure fixes don't break existing functionality
   - Create baselines for data processing results

### Test Framework and Tools
- Use pytest as the primary testing framework
- Use pytest-cov for coverage reporting
- Use pytest-mock or unittest.mock for mocking
- Use responses library for mocking HTTP requests
- Use in-memory database (:memory:) for test isolation

### Test Data Management
- Create reproducible test fixtures
- Store test data in version control
- Document the purpose and contents of test data
- Include both valid and invalid test cases
- Create test data generators for randomized testing

### Continuous Integration
- All tests must pass before merging code
- Coverage reports must be generated on each build
- Performance tests should run on scheduled intervals
- Test in multiple Python environments (3.9, 3.10, 3.11)

### Test Documentation
- Document test setup procedures
- Include examples of running specific test suites
- Document any external dependencies needed for testing
- Explain the purpose of complex test cases

## Data Backup Strategy
- Daily snapshots of the database
- Consider implementing Write-Ahead Logging (WAL) for transaction safety
- Export critical data to CSV periodically as additional backup
- Define retention policy for backups (e.g., daily for 7 days, weekly for 4 weeks)

## Documentation Standards
- Docstrings for all functions following Google Python Style Guide
- README for each agent explaining purpose and usage
- Schema documentation in markdown and as database views
- Code comments for complex logic
- Type hints for function parameters and return values
- Examples for each agent's common operations

## Interoperability
- Standard JSON format for agent results
- Consistent timestamp format (ISO 8601)
- CSV export functionality for integration with external tools
- Standard error format for consistency

## Monitoring and Maintenance
- Agent execution time tracking
- Database size monitoring
- Data quality metrics
- API rate limit tracking
- Scheduled data validation checks
- Health check endpoint or command