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

The data update process is designed to fetch the latest available data for configured instruments and generate derived data like continuous futures contracts. It is primarily managed through the `DB_inspect_enhanced.bat` interface, which calls the main Python orchestrator script `src/scripts/market_data/update_all_market_data_v2.py`.

### Triggering Updates

The update process is typically initiated via options within the `DB_inspect_enhanced.bat` menu.
This batch file, in turn, executes the core Python orchestrator script: `src/scripts/market_data/update_all_market_data_v2.py`.

### Update Sequence

The `update_all_market_data_v2.py` script, guided by the `symbol_metadata` table, generally follows these steps:

1. **Step 0: Ensure Symbol Metadata is Current (Pre-requisite)**
   - The `symbol_metadata` table should be up-to-date. This is typically handled by running the `src/scripts/database/populate_symbol_metadata.py` script (often an option in `DB_inspect_enhanced.bat`) whenever `config/market_symbols.yaml` changes.
   - `symbol_metadata` defines which symbols to process, their data sources, target tables, intervals, and specific scripts for fetching or generation.

2. **Step 1: Fetch/Update Individual Instruments**
   - Iterates through entries in `symbol_metadata` for individual instruments (e.g., specific futures contracts, equities, indices).
   - Based on `symbol_metadata.data_source` and `symbol_metadata.historical_script_path` (or `update_script_path`), it calls appropriate fetchers (e.g., for TradeStation, CBOE data).
   - Data is stored in tables like `market_data` or `market_data_cboe` as specified in `symbol_metadata.data_table`.
   - Handles active contract determination for individual futures if applicable (logic might be within the called scripts).

3. **Step 2: Update/Generate Continuous Contracts**
   - Processes entries in `symbol_metadata` where `asset_type` is `continuous_future`.
   - Calls `src/scripts/market_data/continuous_contract_loader.py` for these symbols.
   - `continuous_contract_loader.py` behavior depends on the `source` and other fields from `symbol_metadata` for each continuous contract symbol:
     - If `source` indicates an external provider (e.g., 'tradestation') that offers pre-formed continuous contracts, these are fetched and stored.
     - If `source` is 'in_house' (or similar indicating local generation):
       - For unadjusted series (e.g., where `symbol_metadata.additional_metadata` might indicate no adjustment or a simple roll), it may perform concatenation or a basic roll.
       - For adjusted series (e.g., where `symbol_metadata.additional_metadata` indicates an adjustment method like 'panama'), it will invoke the appropriate generation logic (e.g., `PanamaContractBuilder` or similar) to calculate and store the adjusted continuous series.
   - Results are stored in the `continuous_contracts` table.

### Key Data Tables

- **`symbol_metadata`**: Central control table defining all symbols, their properties, data sources, intervals, storage tables, and processing scripts. This table is critical for the entire update process.
- **`market_data`**: Stores data for individual instruments, typically from sources like TradeStation.
- **`market_data_cboe`**: Stores raw VIX futures and index data from CBOE.

## Usage Examples

### Using the Cleaning Pipeline

```