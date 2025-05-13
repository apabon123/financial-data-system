# Financial Data System - New Architecture

This document describes the new architecture for the Financial Data System, which brings improvements in code organization, data cleaning, continuous contract generation, and component integration.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Components](#core-components)
3. [Data Cleaning Pipeline](#data-cleaning-pipeline)
4. [Continuous Contract Generation](#continuous-contract-generation)
5. [Using the System](#using-the-system)
6. [Transitioning from Legacy Architecture](#transitioning-from-legacy-architecture)

## Introduction

The new architecture focuses on several key improvements:

- **Component-Based Design**: Modular, interchangeable components with clear interfaces
- **Data Cleaning Pipeline**: Systematic approach to cleaning data with full modification tracking
- **Advanced Contract Generation**: Improved continuous futures generation using the Panama method
- **Graceful Fallbacks**: Seamless transition between old and new architectures

## Core Components

### Application Class

The `FinancialDataSystem` class in `src/core/app.py` serves as the central coordinator that ties all components together:

```python
app = FinancialDataSystem(config_path="config/market_symbols.yaml")
app.update_market_data(symbols=["VXF25", "VXG25"], validate=True, clean=True)
```

### Configuration Management

The configuration system in `src/core/config.py` provides a unified way to manage settings:

```python
# Access nested configuration 
panama_ratio = app.config.get('continuous_futures.panama.ratio_limit', 0.75)
```

### Database Connector

The database layer in `src/core/database.py` provides a unified interface to DuckDB:

```python
# Automatic transaction management
with app.db.transaction():
    app.db.execute("INSERT INTO market_data VALUES (?, ?, ?)", params)
```

## Data Cleaning Pipeline

The data cleaning pipeline in `src/processors/cleaners/pipeline.py` provides a framework for cleaning financial data with full modification tracking.

### Key Features

- **Systematic Data Cleaning**: Apply multiple cleaners in sequence
- **Modification Tracking**: Log all changes to raw data with timestamps and reasons
- **Separate Storage**: Keep raw data separate from cleaned data
- **Audit Trail**: Comprehensive logging of all changes for regulatory compliance

### Example Usage

```python
# Create a cleaning pipeline
cleaning_pipeline = DataCleaningPipeline(
    name="market_data_cleaner",
    db_connector=app.db_connector
)

# Add cleaners to the pipeline
cleaning_pipeline.add_cleaner(VXZeroPricesCleaner())
cleaning_pipeline.add_cleaner(PriceSpikeCleaner())

# Process data for a symbol
success, summary = cleaning_pipeline.process_symbol(
    symbol="VXF25",
    interval_unit="daily",
    interval_value=1
)

print(f"Cleaned {summary.get('total_modifications', 0)} data points")
```

### Creating Custom Cleaners

Create custom cleaners by inheriting from `DataCleanerBase`:

```python
class MyCustomCleaner(DataCleanerBase):
    def __init__(self, db_connector=None, config=None):
        super().__init__(
            name="my_custom_cleaner",
            description="Fixes specific issues in my data",
            db_connector=db_connector,
            fields_to_clean=["open", "high", "low", "close"],
            config=config or {}
        )
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your cleaning logic here
        for i, row in df.iterrows():
            if row['close'] < 0:  # Example condition
                # Log the modification
                self.log_modification(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    field='close',
                    old_value=row['close'],
                    new_value=0,
                    reason="Negative price fixed"
                )
                # Fix the value
                df.at[i, 'close'] = 0
                
        return df
```

## Continuous Contract Generation

The new architecture implements the Panama method for continuous futures, providing a balanced approach to contract adjustments.

### Panama Method

The Panama method balances forward and backward adjustment by distributing the adjustment between past and future contracts. This maintains the relative economics of the market while avoiding the distortions that accumulate in traditional methods.

Key benefits:
- Preserves percentage returns at each roll date
- Avoids excessive distortion of price levels in deep history
- Keeps price levels relatively close to current market prices

### Implementation

The Panama method is implemented in `src/processors/continuous/panama.py`, with the core adjustment algorithm:

```python
# The adjustment involves taking the nth root of the price ratio:
# - For old contract: price_ratio^ratio_limit
# - For new contract: price_ratio^(ratio_limit-1)

old_contract_factor = price_ratio ** self.ratio_limit
new_contract_factor = price_ratio ** (self.ratio_limit - 1.0)
```

### Usage Example

```python
# Get the continuous contract registry
registry = get_registry()

# Create a Panama contract generator
generator = registry.create(
    'panama',
    root_symbol="VX",
    position=1,
    ratio_limit=0.75,  # 0 = forward adjustment, 1 = back adjustment, 0.75 = balanced
    db_connector=app.db_connector,
    roll_strategy='volume'
)

# Generate continuous contract
result_df = generator.generate(start_date="2020-01-01")

# Save to database
app.save_continuous_contracts(result_df)
```

## Using the System

### Command Line Interface

The new architecture is available through the `update_market_data_v2.bat` entry point:

```
update_market_data_v2.bat --skip-vix --panama-ratio 0.65
```

### Python API

```python
from src.core.app import FinancialDataSystem

# Initialize the system
app = FinancialDataSystem("config/market_symbols.yaml")

# Update market data
app.update_market_data(
    symbols=["VXF25", "VXG25"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    validate=True,
    clean=True
)

# Close resources
app.close()
```

## Transitioning from Legacy Architecture

The system includes a wrapper that automatically detects available components and gracefully falls back to the legacy system when needed:

```
update_market_data_wrapper.py --force-legacy  # Force use of legacy architecture
update_market_data_wrapper.py --force-new     # Force use of new architecture
```

### Compatibility

Most command line arguments are compatible between the old and new architectures. The wrapper automatically translates arguments when falling back to the legacy system.

### Migration Path

1. Initially, continue using `update_market_data.bat` which uses the legacy system
2. When ready to test the new system, use `update_market_data_v2.bat` 
3. If any issues occur, the system will fall back to the legacy components
4. Once fully migrated, the legacy components can be removed

---

## Contributing

To contribute to the new architecture, please review the code organization and interfaces. Key interfaces are defined in the base classes for each component type.

New components should implement the appropriate base class and register themselves with the relevant registry.

For more information, see the [Contributing Guide](CONTRIBUTING.md).