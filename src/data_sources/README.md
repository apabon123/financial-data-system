# data_sources/

This directory contains modules for interfacing with external data providers and APIs. Each module is responsible for fetching and standardizing data from a specific source.

## Files
- `cboe.py`: Interface for CBOE data.
- `tradestation.py`: Interface for TradeStation data.
- `base.py`: Base class for data source modules.

Data source modules are designed to be extensible for adding new providers as needed. 