# Claude Code Task

I need you to help me build a financial data management system based on the specifications in this repository. This system will retrieve market data from TradeStation APIs, economic data from various sources, and additional data through web scraping, using DuckDB for storage.

## Project Overview
This project implements a data management system using the single-file agents pattern. Each agent focuses on a specific responsibility within the system:
- Data collection from various sources
- Data processing and validation
- Storage management
- Data retrieval and analysis

## Task Priority

1. **Analysis Phase**
   - Analyze the project structure and requirements in PROJECT_SPEC.md
   - Review the agent specifications in AGENTS.md
   - Understand the database schema in SCHEMA.md
   - Study the examples in EXAMPLES.md to understand the expected behavior

2. **Implementation Phase**
   - Implement the DuckDB schema as specified in SCHEMA.md and sql/init_schema.sql
   - Create the TradeStation API agent based on the template in templates/agent_template.py
   - Implement the data validation and normalization agents
   - Implement storage and query agents

3. **Testing Phase**
   - Create unit tests for each agent
   - Create integration tests for agent interactions
   - Implement test fixtures and mock API responses
   - Ensure comprehensive test coverage

4. **Documentation Phase**
   - Update documentation for implemented agents
   - Add comprehensive code comments
   - Create additional examples for complex operations
   - Document any design decisions or trade-offs

## Implementation Guidelines

### General Requirements
- Follow the single-file agents pattern described in AGENTS.md
- Each agent should be a self-contained Python script
- Use the agent_template.py as a starting point for all agents
- Implement proper error handling and logging
- Follow the standardized CLI interface

### Data Collection Agents
- Implement OAuth authentication for TradeStation API
- Handle rate limiting with exponential backoff
- Process API responses into the format defined in the schema
- Validate data quality before storage

### Data Processing Agents
- Implement data validation against schema definitions
- Normalize data from different sources to a consistent format
- Create derived indicators based on raw market data
- Handle missing data and outliers appropriately

### Storage Agents
- Implement efficient writing to DuckDB
- Handle schema updates and migrations
- Implement data partitioning for performance
- Create indexing strategies for query optimization

### Query Agents
- Implement natural language parsing for queries
- Create optimized SQL queries for data retrieval
- Format results for display or export
- Implement analysis capabilities

## Testing Requirements
- Minimum 85% code coverage
- Unit tests for all public methods
- Integration tests for agent interactions
- Mock external API calls for testing
- Test data quality and validation rules
- Performance tests for large datasets

## Deliverables
1. Complete implementation of all agents described in AGENTS.md
2. Comprehensive test suite
3. Schema initialization script
4. Example queries and workflows
5. Documentation on usage and extensibility

## Additional Notes
- Focus on creating modular, reusable components
- Prioritize data quality and validation
- Add comprehensive error handling and logging
- Document all code thoroughly
- Consider performance optimizations for large datasets
- Implement security best practices for API credentials

Please start by analyzing the project requirements and then proceed with implementation following the priority order outlined above.