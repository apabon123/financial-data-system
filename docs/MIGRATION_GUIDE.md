# Migration Guide

This guide provides instructions for migrating custom scripts and workflows to the new Financial Data System architecture.

## Overview

The Financial Data System has been transformed with a new architecture that includes:

1. Component-based design with clear interfaces
2. Improved continuous futures generation with the Panama method
3. Data cleaning pipeline with raw/cleaned data separation
4. Enhanced configuration and logging systems

This guide helps you update your custom scripts to take advantage of these improvements.

## Step 1: Identify Scripts to Migrate

First, identify custom scripts that need migration. Run the following to generate a migration manifest:

```bash
python src/scripts/utilities/generate_migration_manifest.py
```

This will create a CSV file listing files that need attention, with their migration status and recommended replacements.

## Step 2: Understand Key Architecture Changes

### Database Schema Changes

- Raw data is now stored in `*_raw` tables
- Cleaned data is stored in the original tables
- Cleaning operations are logged in `*_cleaning_log` tables
- Continuous contracts are built using the Panama method

### Component Interface Changes

- Access database through `DatabaseConnector` instead of direct DuckDB connections
- Use the `Application` class to coordinate components
- Access configuration through `ConfigManager` instead of direct YAML loading

## Step 3: Update Imports

Replace old imports with new architecture components:

```python
# Old imports
import duckdb
import yaml
from src.scripts.utilities.database import get_db_engine
from src.scripts.market_data.generate_continuous_futures import generate_continuous

# New imports
from src.core.database import DatabaseConnector
from src.core.config import ConfigManager
from src.core.app import Application
from src.processors.continuous.panama import PanamaContractGenerator
from src.processors.continuous.registry import get_registry
```

## Step 4: Replace Database Access

Replace direct DuckDB connections with the new `DatabaseConnector`:

```python
# Old approach
conn = duckdb.connect(database='data/financial_data.duckdb')
result = conn.execute("SELECT * FROM market_data").fetchdf()
conn.close()

# New approach
db = DatabaseConnector(db_path='data/financial_data.duckdb')
result = db.query("SELECT * FROM market_data")
db.close()
```

## Step 5: Update Continuous Futures Code

Replace calls to the old continuous futures generation with the Panama method:

```python
# Old approach
from src.scripts.market_data.generate_continuous_futures import generate_continuous
generate_continuous(conn, 'VX', config, '2020-01-01', '2022-12-31', 'volume', force=True)

# New approach
from src.processors.continuous.registry import get_registry
registry = get_registry()
generator = registry.create(
    'panama',
    root_symbol='VX',
    position=1,
    roll_strategy='volume',
    db_connector=db,
    ratio_limit=0.75
)
continuous_df = generator.generate(start_date='2020-01-01', end_date='2022-12-31')
```

## Step 6: Update Data Cleaning Code

Replace ad-hoc data cleaning with the new pipeline:

```python
# Old approach
# Custom cleaning logic inline or in one-off scripts

# New approach
from src.processors.cleaners.pipeline import DataCleaningPipeline
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner

pipeline = DataCleaningPipeline(name="my_cleaner", db_connector=db)
pipeline.add_cleaner(VXZeroPricesCleaner(db_connector=db))
pipeline.process_symbol('VXF23', interval_unit='daily', interval_value=1)
```

## Step 7: Update Configuration Handling

Replace direct YAML loading with `ConfigManager`:

```python
# Old approach
with open('config/market_symbols.yaml', 'r') as f:
    config = yaml.safe_load(f)

# New approach
from src.core.config import ConfigManager
config_mgr = ConfigManager('config/market_symbols.yaml')
config = config_mgr.get_config()
```

## Step 8: Use the Application Class

For scripts that orchestrate multiple components, use the `Application` class:

```python
# Old approach
# Multiple setup and coordination steps

# New approach
from src.core.app import Application
app = Application(
    config_path='config/market_symbols.yaml',
    db_path='data/financial_data.duckdb'
)

# Use high-level methods
app.update_vx_futures()
app.update_es_futures()
app.close()
```

## Step 9: Update Command-Line Scripts

For command-line scripts, use the new entry points:

```bash
# Old approach
./update_market_data.bat

# New approach
./update_market_data_v2.bat
```

During the transition period, both entry points are maintained, with the old one redirecting to a compatibility wrapper.

## Step 10: Create Custom Data Cleaners

To create custom data cleaners:

1. Subclass `DataCleanerBase`
2. Implement the `clean()` method
3. Use `log_modification()` to track changes

Example:

```python
from src.processors.cleaners.base import DataCleanerBase

class MyCustomCleaner(DataCleanerBase):
    def __init__(self, db_connector=None, enabled=True):
        super().__init__(
            name="my_custom_cleaner",
            description="Custom cleaner for XYZ",
            db_connector=db_connector,
            fields_to_clean=['open', 'close'],
            enabled=enabled,
            priority=100,
            config={}
        )
    
    def clean(self, df):
        result = df.copy()
        # Custom cleaning logic here
        # ...
        # Track modifications
        self.log_modification(
            timestamp=timestamp,
            symbol=symbol,
            field='close',
            old_value=old_value,
            new_value=new_value,
            reason="Custom cleaning reason",
            details="Additional details"
        )
        return result
```

## Handling Special Cases

### Accessing Raw vs. Cleaned Data

To access raw (uncleaned) data:

```python
raw_data = db.query("SELECT * FROM market_data_raw")
```

To access cleaned data:

```python
cleaned_data = db.query("SELECT * FROM market_data")
```

### Viewing Cleaning Logs

To see what changes were made to a symbol:

```python
cleaning_log = db.query("""
    SELECT * FROM market_data_cleaning_log
    WHERE symbol = ?
    ORDER BY timestamp DESC
""", ["VXF23"])
```

### Comparing Legacy vs. Panama Continuous Contracts

During validation, you may want to compare old and new methods:

```python
# Legacy continuous contract
legacy_data = db.query("""
    SELECT * FROM continuous_contracts
    WHERE symbol = '@VX=101XN'
""")

# Panama continuous contract
panama_data = db.query("""
    SELECT * FROM continuous_contracts
    WHERE symbol = '@VX=1P75V'
""")
```

## Testing Your Migration

After migration, run the test suite to ensure everything works correctly:

```bash
# Run unit tests
python -m unittest discover -s tests/unit

# Run integration tests
python -m unittest discover -s tests/integration

# Run validation tests
python -m unittest discover -s tests/validation
```

## Getting Help

If you encounter issues during migration:

1. Check the `docs/NEW_ARCHITECTURE.md` documentation
2. Review the test cases for examples
3. Examine the implementation of core components

For specific questions, contact the development team.