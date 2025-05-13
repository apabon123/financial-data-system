# sql/

This directory contains SQL scripts and schema definitions for the financial data system.

## Subdirectories
- `schema/`: Database schema and table/view definitions.
- `analysis/`: SQL scripts for data analysis and reporting.
- `validation/`: SQL scripts for data validation and quality checks.
- `maintenance/`: SQL scripts for database maintenance and housekeeping.

SQL scripts are organized by their primary function to support modular and maintainable database operations.

## Overview

The SQL files in this directory are designed to be used with DuckDB and provide a suite of tools for:

- Database maintenance and monitoring
- Financial data analysis
- Data validation and integrity checks
- Continuous futures contract management
- Symbol inventory management
- Data gap detection

## File Structure

### Database Maintenance
- **database_maintenance.sql**: Queries for maintaining and monitoring database health, including table statistics, space usage, and performance metrics.

### Data Analysis
- **data_analysis.sql**: Queries for analyzing financial data, including price movements, volume patterns, and market statistics.

### Data Validation
- **data_validation.sql**: Queries for validating data integrity, checking for anomalies, and ensuring data quality.

### Continuous Contracts
- **continuous_contracts.sql**: Queries for analyzing continuous futures contracts, detecting rollover issues, and checking data quality.

### Data Gaps
- **data_gaps.sql**: Queries for identifying and analyzing gaps in financial data.

### Data Quality
- **data_quality.sql.txt**: Queries for assessing and reporting on data quality metrics.

### Symbol Inventories
- **symbol_inventories.sql**: Queries for managing and analyzing symbol inventories.

### Schema
- **init_schema.sql**: SQL script for initializing the database schema.

## Usage

These queries can be executed directly in DuckDB or integrated into Python scripts using the DuckDB Python API. Each query is documented with comments explaining its purpose and expected output.

### Example Usage in Python

```python
import duckdb

# Connect to the database
conn = duckdb.connect('./data/financial_data.duckdb')

# Read and execute a query from a file
with open('sql/data_analysis.sql', 'r') as f:
    query = f.read()
    
# Execute the query
result = conn.execute(query).fetchdf()

# Display the results
print(result)
```

## Query Categories

### Database Maintenance Queries
- Table statistics and space usage
- Index usage and optimization
- Table bloat analysis
- Long-running query monitoring
- Growth rate tracking
- Maintenance recommendations

### Data Analysis Queries
- Price movement analysis
- Volume patterns
- Market statistics
- Price distribution
- Trading session analysis
- Correlation analysis
- Volatility metrics
- Market breadth
- Momentum indicators
- Market depth analysis

### Data Validation Queries
- Data completeness checks
- Price validity validation
- Volume consistency
- Time continuity checks
- Data quality scoring
- Source consistency
- Timestamp validation
- Data freshness monitoring
- Symbol consistency
- Comprehensive completeness reporting

### Continuous Contract Queries
- Contract overview
- Gap detection
- Rollover issue detection
- Rollover date identification

## Best Practices

1. **Parameterization**: When using these queries in scripts, consider parameterizing values like date ranges and symbols.
2. **Performance**: Some queries may be resource-intensive. Consider adding appropriate filters to limit the data processed.
3. **Regular Execution**: Schedule regular execution of maintenance and validation queries to ensure database health.
4. **Customization**: Adapt queries to your specific needs by modifying filters and thresholds.

## Contributing

When adding new queries:
1. Place them in the appropriate category file
2. Add clear comments explaining the purpose and expected output
3. Follow the existing formatting and naming conventions
4. Update this README if adding new categories or significant changes 