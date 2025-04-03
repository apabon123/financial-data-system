# Environment Setup

## System Requirements
- Python 3.9 or higher
- 8GB RAM minimum (16GB+ recommended for large datasets)
- SSD storage recommended for database performance
- Stable internet connection for API access

## Dependencies
- Python 3.9+
- UV package manager
- Required Python packages:
  - duckdb
  - pandas
  - requests
  - beautifulsoup4
  - python-dotenv
  - typer
  - rich
  - matplotlib (for visualization)
  - numpy
- Required API keys:
  - TradeStation API (client ID and secret)
  - FRED API key for economic data
  - Optional: Other economic data APIs as needed

## Installation

### Install UV Package Manager
```bash
# Install UV package manager
curl -sSf https://install.pydantic.dev | python3
```

### Clone Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/financial-data-system.git
cd financial-data-system
```

### Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

If you don't have a requirements.txt file, install the dependencies individually:
```bash
uv pip install duckdb pandas requests beautifulsoup4 python-dotenv typer rich matplotlib numpy
```

## API Configuration

### TradeStation API Setup
1. Register for a TradeStation developer account at [TradeStation Developer Portal](https://developer.tradestation.com/)
2. Create a new application to get your client ID and secret
3. Note the permissions required:
   - Market Data access for OHLCV data
   - Read Account access for account/position data

### FRED API Setup
1. Register for a FRED API key at [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Follow the instructions to get your API key

## Environment Variables
Create a `.env` file in the project root with the following:

```
# API Keys
TRADESTATION_API_KEY=your-tradestation-client-id-here
TRADESTATION_API_SECRET=your-tradestation-client-secret-here
FRED_API_KEY=your-fred-api-key-here

# Optional: Additional API keys for other data sources
# ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
# FINNHUB_API_KEY=your-finnhub-key-here

# Configuration
LOG_LEVEL=INFO
DATA_DIR=./data
BACKUP_DIR=./backups
```

## Database Initialization

Initialize the DuckDB database with the required schema:

```bash
# Create the database and initialize schema
python init_database.py -d ./financial_data.duckdb
```

This will create the database file and set up all required tables and indexes as defined in the `init_schema.sql` file.

## Verify Installation

Run a simple test to verify your setup:

```bash
# Test the TradeStation Market Data Agent
python tradestation_market_data_agent.py -d ./financial_data.duckdb -q "fetch daily data for SPY from 2023-01-01 to 2023-01-31" -v
```

If successful, you should see output indicating data was fetched and stored in the database.

## Troubleshooting

### Common Issues

1. **API Authentication Errors**:
   - Verify your API keys in the `.env` file
   - Check that your TradeStation developer account has the necessary permissions
   - Ensure your TradeStation app is approved for the required scopes

2. **Database Connection Errors**:
   - Ensure you have permission to write to the directory where the database is stored
   - Check that DuckDB is properly installed with `uv pip list | grep duckdb`

3. **Missing Dependencies**:
   - Run `uv pip install -r requirements.txt` again to ensure all dependencies are installed
   - If you encounter "module not found" errors, install the specific package: `uv pip install <package_name>`

4. **Rate Limiting**:
   - API requests may be rate-limited. Ensure your code implements proper backoff strategies
   - Consider using smaller batches for data retrieval

### Getting Help

If you encounter issues not covered here:
1. Check the logs in the `logs/` directory
2. Review the TradeStation API documentation for specific error codes
3. Open an issue in the GitHub repository with a detailed description of your problem

## Next Steps

Once your environment is set up:
1. Review the examples in `EXAMPLES.md` to learn how to use the system
2. Explore the agent documentation in `AGENTS.md` to understand each component
3. Study the database schema in `SCHEMA.md` to understand the data structure
4. Try running the demo scripts in the `examples/` directory

## Testing Framework

This project requires comprehensive testing to ensure data quality and system reliability. We use the following testing approach:

### Test Directory Structure
```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_collection/    # Tests for data collection agents
│   ├── test_processing/    # Tests for data processing agents
│   ├── test_storage/       # Tests for storage agents
│   └── test_query/         # Tests for query agents
├── integration/            # Integration tests across components
├── fixtures/               # Test data and mock responses
└── conftest.py             # pytest configuration
```

### Setting Up Testing Environment

Install test dependencies:
```bash
uv pip install pytest pytest-cov pytest-mock responses
```

### Writing Tests

Each agent should have corresponding test files following these guidelines:

1. **Unit Tests**: Test individual functions and methods in isolation
   ```python
   # Example unit test for TradeStation Market Data Agent
   def test_parse_query():
       agent = TradeStationMarketDataAgent(database_path=":memory:")
       params = agent._parse_query("fetch daily data for AAPL from 2023-01-01 to 2023-01-31")
       assert params["symbols"] == ["AAPL"]
       assert params["timeframe"] == "daily"
       assert params["start_date"].isoformat() == "2023-01-01"
       assert params["end_date"].isoformat() == "2023-01-31"
   ```

2. **Mock API Tests**: Test API interactions without making real requests
   ```python
   # Example mock test using responses library
   @responses.activate
   def test_fetch_market_data():
       # Setup mock response
       responses.add(
           responses.GET, 
           "https://api.tradestation.com/v3/marketdata/barcharts/AAPL",
           json={"Bars": [{"Open": 150.0, "High": 152.0, "Low": 149.0, "Close": 151.0, "TimeStamp": "2023-01-01T00:00:00Z"}]},
           status=200
       )
       
       agent = TradeStationMarketDataAgent(database_path=":memory:")
       result = agent.fetch_market_data({"symbols": ["AAPL"], "timeframe": "daily"})
       
       assert len(result) > 0
       assert result.iloc[0]["symbol"] == "AAPL"
       assert result.iloc[0]["open"] == 150.0
   ```

3. **Integration Tests**: Test interactions between multiple agents
   ```python
   def test_data_pipeline():
       # Test the flow from data collection to storage to query
       market_agent = TradeStationMarketDataAgent(database_path=":memory:")
       write_agent = DuckDBWriteAgent(database_path=":memory:")
       query_agent = DataRetrievalAgent(database_path=":memory:")
       
       # Test data flow through the pipeline
       # ...
   ```

4. **Data Quality Tests**: Verify data integrity and consistency
   ```python
   def test_data_quality():
       # Load test data
       df = pd.read_csv("tests/fixtures/sample_market_data.csv")
       
       # Run validation
       validation_agent = DataValidationAgent(database_path=":memory:")
       validation_result = validation_agent.validate_data(df)
       
       assert validation_result["records_valid"] == validation_result["records_checked"]
   ```

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage report:
```bash
pytest --cov=agents tests/
```

Run specific test categories:
```bash
pytest tests/unit/  # Run only unit tests
pytest tests/integration/  # Run only integration tests
```

### Continuous Integration

The project is configured to run tests automatically on each pull request and push to main branch. See `.github/workflows/tests.yml` for CI configuration.

### Test Data

- Sample data for testing is stored in `tests/fixtures/`
- Mock API responses are stored in `tests/fixtures/responses/`
- Database fixtures for testing are regenerated using `tests/fixtures/generate_fixtures.py`

## Maintenance

- Regularly update your Python packages: `uv pip install --upgrade -r requirements.txt`
- Back up your database regularly: `python backup_database.py -d ./financial_data.duckdb -o ./backups/`
- Monitor your API usage to avoid hitting rate limits
- Run tests before and after making changes: `pytest tests/`