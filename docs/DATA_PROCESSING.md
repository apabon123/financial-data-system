# Data Processing Guide

This document provides a comprehensive guide to data processing in the system, including both the data cleaning pipeline and the data update process.

## Overview

The system handles two main aspects of data processing:

1. **Data Cleaning**: A systematic approach to cleaning market data while maintaining full traceability of all changes
2. **Data Updates**: Regular updates of market data from various sources and generation of derived data

## Data Cleaning Pipeline

The data cleaning pipeline ensures:

1. **Raw Data Preservation**: Original data is never modified directly
2. **Change Tracking**: All modifications are logged with what, when, and why information
3. **Process Structure**: Cleaners are applied in a controlled, ordered sequence
4. **Extensibility**: New cleaners can be added easily

### Architecture

The data cleaning pipeline consists of:

1. **Base Cleaner Class**: Common interface for all cleaners
2. **Cleaning Pipeline**: Orchestrates multiple cleaners
3. **Specific Cleaners**: Implement cleaning rules for different data issues
4. **Database Schema**: Tables and logs to track changes

#### Database Schema

The pipeline uses the following database structure:

- `<table_name>_raw`: Contains the original, unmodified data
- `<table_name>`: Contains the cleaned data
- `<table_name>_cleaning_log`: Records all modifications

For example:
- `market_data_raw`: Raw market data
- `market_data`: Cleaned market data
- `market_data_cleaning_log`: Log of all cleaning operations

#### Log Structure

Each cleaning operation is logged with:

- `timestamp`: When the data point was created
- `symbol`: The symbol being cleaned
- `raw_timestamp`: When the cleaning occurred
- `cleaning_operation`: Name of the cleaning operation
- `field_name`: Field that was changed
- `old_value`: Original value
- `new_value`: New value after cleaning
- `reason`: Why the change was made
- `cleaned_by`: Which cleaner made the change
- `details`: Additional information

### Data Cleaner Base Class

All data cleaners inherit from `DataCleanerBase` in `src/processors/cleaners/base.py`:

```python
class DataCleanerBase(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        db_connector = None,
        fields_to_clean: List[str] = None,
        enabled: bool = True,
        priority: int = 100,
        config: Dict[str, Any] = None
    ):
        # Initialize cleaner

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement cleaning logic
        pass

    def log_modification(self, 
                       timestamp, symbol, field, 
                       old_value, new_value, reason, details=None):
        # Log a modification to the cleaning log
```

Key points:
- `name` and `description` identify the cleaner
- `fields_to_clean` lists the data fields this cleaner operates on
- `priority` determines execution order (lower number = higher priority)
- `config` provides customization options

### Cleaning Pipeline

The `DataCleaningPipeline` class in `src/processors/cleaners/pipeline.py` orchestrates multiple cleaners:

```python
class DataCleaningPipeline:
    def __init__(
        self,
        name: str,
        db_connector = None,
        cleaners: List[DataCleanerBase] = None,
        config: Dict[str, Any] = None
    ):
        # Initialize pipeline

    def add_cleaner(self, cleaner: DataCleanerBase) -> None:
        # Add a cleaner to the pipeline

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Apply all cleaners to a DataFrame
        
    def process_symbol(self, symbol: str, ...) -> Tuple[bool, Dict[str, Any]]:
        # Process a symbol from the database
        
    def process_table(self, table_name: str, ...) -> Dict[str, Any]:
        # Process all data in a table
```

### Included Cleaners

#### VX Zero Prices Cleaner

`VXZeroPricesCleaner` in `src/processors/cleaners/vx_zero_prices.py` fixes zero or missing prices in VX futures:

```python
class VXZeroPricesCleaner(DataCleanerBase):
    def __init__(
        self,
        db_connector = None,
        enabled: bool = True,
        config: Dict[str, Any] = None
    ):
        # Initialize with specific config for VX cleaning
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fix zero or missing prices using interpolation
```

This cleaner:
- Detects zero or missing prices in VX contracts
- Uses linear interpolation to fill gaps
- Estimates volume and open interest for missing points
- Logs all replacements for transparency

## Data Update Process

The data update process is designed to fetch the latest available data for configured instruments and generate derived data like continuous futures contracts.

### Triggering Updates

The update process is initiated by running:

```bash
update_market_data.bat
```

This script executes the core Python orchestrator script: `src/scripts/market_data/update_all_market_data.py`.

### Update Sequence

The update process follows these steps:

1. **Step 0: Update Symbol Metadata**
   - Updates the `symbol_metadata` table from `config/market_symbols.yaml`
   - Defines which symbols to process and their configurations

2. **Step 1: Fetch/Update Individual Instruments**
   - Updates data for individual futures, indices, and equities from TradeStation
   - Handles active contract determination for futures
   - Stores data in the `market_data` table

3. **Step 2: Update Raw CBOE VX Futures Data**
   - Fetches daily data for active VIX futures contracts
   - Updates the `market_data_cboe` table
   - Maintains the `futures_roll_calendar`

4. **Step 3: Update Continuous Contracts**
   - Builds continuous contracts from underlying data
   - Handles all TradeStation-sourced continuous contracts
   - Stores results in the `continuous_contracts` table

### Key Data Tables

- **`symbol_metadata`**: Central control table for update configuration
- **`market_data`**: TradeStation-sourced individual instruments
- **`market_data_cboe`**: Raw VIX futures data from CBOE
- **`continuous_contracts`**: Generated continuous contract data
- **`futures_roll_calendar`**: VIX contract roll information

## Usage Examples

### Using the Cleaning Pipeline

```python
from src.processors.cleaners.pipeline import DataCleaningPipeline
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner
from src.core.database import DatabaseConnector

# Connect to database
db = DatabaseConnector(db_path='data/financial_data.duckdb')

# Create pipeline
pipeline = DataCleaningPipeline(
    name="market_data_cleaner",
    db_connector=db
)

# Add cleaners
pipeline.add_cleaner(VXZeroPricesCleaner(
    db_connector=db,
    config={'interpolation_method': 'linear'}
))

# Process a symbol
success, summary = pipeline.process_symbol(
    symbol='VXF23',
    interval_unit='daily',
    interval_value=1
)

# Check results
if success:
    print(f"Cleaned {summary['total_modifications']} data points")
```

### Creating Custom Cleaners

```python
class PriceSpikeCleaner(DataCleanerBase):
    def __init__(self, db_connector=None, enabled=True):
        super().__init__(
            name="price_spike_cleaner",
            description="Fixes abnormal price spikes",
            db_connector=db_connector,
            fields_to_clean=['open', 'high', 'low', 'close'],
            enabled=enabled,
            priority=75,  # Run after zero price cleaner
            config={'max_change_percent': 20.0}
        )
    
    def clean(self, df):
        result = df.copy()
        
        # Detect and fix price spikes
        for field in self.fields_to_clean:
            # Calculate percentage changes
            pct_change = result[field].pct_change().abs()
            
            # Find spikes exceeding threshold
            threshold = self.config.get('max_change_percent', 20.0) / 100
            spikes = pct_change > threshold
            
            # Fix each spike
            for idx in result.index[spikes]:
                timestamp = result.loc[idx, 'timestamp']
                symbol = result.loc[idx, 'symbol']
                old_value = result.loc[idx, field]
                
                # Replace with interpolated value
                new_value = (result.loc[idx-1, field] + result.loc[idx+1, field]) / 2
                result.loc[idx, field] = new_value
                
                # Log the modification
                self.log_modification(
                    timestamp=timestamp,
                    symbol=symbol,
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    reason=f"Price spike exceeding {threshold*100}%",
                    details=f"Changed from {old_value} to {new_value}"
                )
        
        return result
```

## Configuration

The system uses `config/market_symbols.yaml` to define:
- Which symbols to process
- Data sources for each symbol
- Target database tables
- Data frequencies
- Specific parameters like expiry rules

Refer to `docs/futures_configuration.md` for details on the configuration file structure. 