# Continuous Futures Contracts Guide

This document explains how continuous futures contracts are handled within the system. All continuous contracts specified in `symbol_metadata` with `data_source = 'tradestation'` are now built using a unified process that leverages underlying contract data from TradeStation.

## Overview

Continuous futures contracts provide a synthetic, unbroken price series for a futures instrument. In this system, all such contracts (e.g., `@ES=...`, `@NQ=...`, `@VX=...`) are built by the `src/scripts/market_data/continuous_contract_loader.py` script. This script fetches the necessary data for the underlying individual contracts from TradeStation and then stitches them together based on TradeStation's roll logic.

The distinction between "locally generated VIX from CBOE data" and "directly fetched ES/NQ from TradeStation" is no longer accurate. **All continuous contracts sourced from TradeStation are now built by `continuous_contract_loader.py` using TradeStation underlying data.**

All resulting continuous contract data is stored in the `continuous_contracts` table.

## Naming Conventions

The naming convention for continuous contracts is defined in `config/market_symbols.yaml` and reflected in the `symbol_metadata` table (typically in the `data_table` column for `asset_type = 'continuous_future'`). Examples include:

*   `@ES=102XC`: ES continuous contract (e.g., front month, calendar-weighted).
*   `@NQ=202XN`: NQ continuous contract (e.g., second month, nearest).
*   `@VX=101XN`: VIX continuous contract (e.g., front month, nearest).

The exact meaning of the suffices (e.g., `102XC`, `101XN`) is determined by TradeStation's conventions for identifying continuous contract calculation methods and series.

## Unified Continuous Contract Generation Process

The generation of all TradeStation-sourced continuous contracts is managed by `src/scripts/market_data/update_all_market_data.py`, which calls `src/scripts/market_data/continuous_contract_loader.py` for each relevant symbol.

1.  **Identification of Continuous Contracts:**
    *   The `update_all_market_data.py` script queries the `symbol_metadata` table.
    *   It selects `data_table` values where `asset_type = 'continuous_future'` and `data_source = 'tradestation'`.

2.  **Invocation of `continuous_contract_loader.py`:**
    *   For each continuous symbol identified, `update_all_market_data.py` executes `continuous_contract_loader.py` as a Python module, passing the specific continuous symbol name (e.g., `@ES=102XC`) and other necessary parameters (`--db-path`, `--config-path`, `--fetch-mode='auto'`, `--lookback-days`, `--roll-proximity-threshold-days`).

3.  **Logic within `continuous_contract_loader.py`:**
    *   **Initialization:** Initializes its own `MarketDataFetcher` instance.
    *   **Determine Processing Mode:** Based on the `fetch_mode` (defaulting to 'auto'), `force` flag (if passed, though not by the orchestrator in 'auto' mode), and `is_near_roll()` status (for adjusted contracts), it decides whether to fetch full history, a recent chunk, or full history due to roll proximity.
    *   **Fetch Underlying Data:** Calls `fetcher.fetch_data_for_continuous_builder()` method. This crucial method in `MarketDataFetcher` is responsible for:
        *   Interpreting the continuous symbol (e.g., `@ES=102XC`) to understand which series it represents.
        *   Fetching the historical price data for the required *individual underlying contracts* from the TradeStation API for the determined period.
    *   **Build Continuous Series:** Stitches the data from the fetched underlying contracts. The logic for how TradeStation rolls from one contract to the next is implicitly handled by how `fetch_data_for_continuous_builder` retrieves and prepares the data for stitching.
    *   **Store Data:** Upserts the resulting continuous time series data into the `continuous_contracts` table.

## Key Points of the New System:

*   **Unified Builder:** `continuous_contract_loader.py` is the single point of logic for building all continuous contracts that rely on TradeStation underlying data.
*   **TradeStation as Underlying Source:** Even for VIX continuous contracts (e.g., `@VX=101XN`), the underlying individual VIX futures data is now fetched from TradeStation, not CBOE, for the purpose of building these continuous series.
*   **No Local VIX Mapping Table:** The previous `continuous_contract_mapping` table (specific to VIX and CBOE data) is no longer used for generating the primary continuous VIX series stored in `continuous_contracts` if they are configured as `data_source='tradestation'`.
*   **Configuration Driven:** The `symbol_metadata` table (populated from `config/market_symbols.yaml`) dictates which continuous contracts are processed by this mechanism.

## Accessing Continuous Data

Continuous contract data can be queried directly from the `continuous_contracts` table using standard SQL or viewed via the `DB_inspect.bat` tool.

```sql
-- Example: Get recent front-month continuous VIX data (now built from TS underlying)
SELECT *
FROM continuous_contracts
WHERE symbol = '@VX=101XN' -- Or whatever specific @VX symbol is in symbol_metadata
  AND interval_unit = 'daily'
ORDER BY timestamp DESC
LIMIT 10;

-- Example: Get recent front-month continuous ES data (calendar-weighted)
SELECT *
FROM continuous_contracts
WHERE symbol = '@ES=102XC'
  AND interval_unit = 'daily'
ORDER BY timestamp DESC
LIMIT 10;
``` 