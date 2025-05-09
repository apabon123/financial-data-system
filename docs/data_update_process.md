# Data Update Process Guide

This document details the process for updating the market data in the `data/financial_data.duckdb` database.

## Overview

The primary goal of the update process is to fetch the latest available data for configured instruments and generate derived data like continuous futures contracts. The process is designed to be run regularly (e.g., daily via a scheduled task) to keep the database current.

## Triggering the Update

The update process is initiated by running the main batch script from the project root:

```bash
update_market_data.bat
```

This script serves as a simple wrapper that executes the core Python orchestrator script: `src/scripts/market_data/update_all_market_data.py`. It forwards command-line arguments such as `--db-path`, `--config-path`, `--lookback-days`, `--roll-proximity-days`, `--fetch-all-individual-history`, and `--verify` to the Python script.

## Core Orchestrator: `update_all_market_data.py`

This Python script manages the sequence of update steps, handles database connections, and calls specialized sub-scripts or modules for specific tasks.

**Key Responsibilities:**

*   Parses command-line arguments.
*   Initializes and configures logging.
*   Loads the market symbols configuration from `config/market_symbols.yaml`.
*   Initializes a `MarketDataFetcher` instance for TradeStation data.
*   Executes update steps in a defined order, calling other scripts/modules as needed.
*   Logs the process extensively to the `logs/` directory.

## Update Sequence and Data Sources

The `update_all_market_data.py` script executes the following steps:

1.  **Step 0: Update Symbol Metadata**
    *   **Script:** Calls `src.scripts.database.populate_symbol_metadata.main()` as a Python module.
    *   **Data Source:** `config/market_symbols.yaml`.
    *   **Action:** Reads the YAML configuration and populates/updates the `symbol_metadata` table in the database. This table acts as the control center, defining which symbols to process, their asset types, data sources, target database tables, and data frequencies.

2.  **Step 1: Fetch/Update Individual Instruments from TradeStation**
    *   **Mechanism:** Uses the initialized `MarketDataFetcher` instance.
    *   **Data Source:** TradeStation API.
    *   **Action:**
        *   Iterates through symbols defined in `symbol_metadata` for asset types 'futures', 'indices', and 'equities' that are sourced from 'tradestation' and are not continuous groups.
        *   For futures, it first calls `fetcher.get_active_futures_symbols()` to determine the currently active individual contracts based on expiry rules. For other asset types, it uses the `base_symbol`.
        *   For each symbol and its configured frequencies (from `symbol_metadata`), calls `fetcher.process_symbol()`. This function handles fetching new or historical data and upserting it into the relevant table (typically `market_data`).
        *   The `fetcher`'s database connection is closed after this step to allow subsequent steps (run as separate processes) to access the database.

3.  **Step 2: Update Raw CBOE VX Futures Data**
    *   **Script:** Calls `src.scripts.market_data.vix.update_vx_futures.main()` as a Python module.
    *   **Data Source:** CBOE Website (CSV downloads).
    *   **Action:** Fetches the latest daily data for active individual VIX futures contracts (e.g., `VXK25`, `VXM25`) and upserts them into the **`market_data_cboe`** table. This script also handles updating the `futures_roll_calendar` for VIX if necessary.
    *   *Note: The VIX Index (`$VIX.X`) update, previously a separate step, is assumed to be handled within `update_vx_futures.py` if still required, or configured as a separate 'index' entry in `market_symbols.yaml` to be fetched via Step 1 if sourced from TradeStation.*

4.  **Step 3: Update TradeStation-Sourced Continuous Contracts**
    *   **Mechanism:**
        *   Queries the `symbol_metadata` table (using a new temporary DB connection) to get a list of *specific* continuous contract symbols (e.g., `@ES=102XC`, `@VX=101XN`) that are configured with `asset_type = 'continuous_future'` and `data_source = 'tradestation'`. The actual symbol identifier is typically read from the `data_table` column in `symbol_metadata`.
        *   For each identified continuous symbol, it calls `src.scripts.market_data.continuous_contract_loader.main()` as a Python module.
    *   **`continuous_contract_loader.py` Logic:**
        *   **Data Source:** TradeStation API (for underlying contract data).
        *   **Action:**
            *   Receives the specific continuous symbol (e.g., `@ES=102XC`, `@VX=101XN`), database path, config path, and fetch parameters (`fetch_mode='auto'`, `lookback_days`, `roll_proximity_threshold_days`).
            *   Initializes its own `MarketDataFetcher` instance.
            *   Determines the `current_processing_mode` ('full_history', 'latest_chunk', or 'full_history_near_roll') based on the `fetch_mode`, `force` flag (not directly used by orchestrator in 'auto' mode), and whether the contract is adjusted and near a roll (using `is_near_roll()` which considers `roll_proximity_threshold_days`).
            *   Calls `fetcher.fetch_data_for_continuous_builder()` to get the necessary historical data for the underlying individual contracts from TradeStation.
            *   Constructs the continuous contract series by stitching together data from the underlying contracts based on TradeStation's roll logic (which `fetch_data_for_continuous_builder` and the subsequent processing in `continuous_contract_loader` respect).
            *   Upserts the generated continuous contract data into the `continuous_contracts` table.
        *   *Important Change:* This step now handles **all** continuous contracts that are built from TradeStation underlying data, including VIX continuous contracts (e.g., `@VX=101XN`) which were previously generated using CBOE data and local roll mapping.

5.  **Verification (Placeholder)**
    *   **Trigger:** If `--verify` argument is passed to `update_market_data.bat`.
    *   **Action:** The `update_all_market_data.py` script logs that it received the flag but currently ignores it. Any actual verification would be performed by other scripts if `update_market_data.bat` calls them (e.g., `check_db.py`).

6.  **Completion:**
    *   Logs final summary information.

## Configuration (`config/market_symbols.yaml`)

This file is crucial for defining which symbols are processed, their data sources, target tables, frequencies, and specific parameters like expiry rules (used by `MarketDataFetcher`'s `get_active_futures_symbols` and `_calculate_expiry_date_from_config` methods). Refer to `docs/futures_configuration.md` for details on this file's structure.

## Continuous Contracts

*   **All Continuous Contracts (e.g., `@ES=...`, `@NQ=...`, `@VX=...`):**
    *   If `data_source` in `symbol_metadata` is 'tradestation' and `asset_type` is 'continuous_future'.
    *   Are now built by `src/scripts/market_data/continuous_contract_loader.py`.
    *   This script fetches underlying contract data from **TradeStation API** using `MarketDataFetcher`.
    *   It then constructs the continuous series based on the fetched underlying data.
    *   The resulting data is stored in the `continuous_contracts` table.
    *   The distinction between "TradeStation pre-calculated continuous" and "locally generated from CBOE" is no longer accurate. All continuous contracts listed in `symbol_metadata` as 'tradestation' sourced are built by `continuous_contract_loader.py` using underlying data from TradeStation.

## Key Data Tables

*   **`symbol_metadata`**: Central control table, populated from `config/market_symbols.yaml`. Defines what and how to update.
*   **`market_data`**: Stores data for TradeStation-sourced individual futures, equities, and indices.
*   **`market_data_cboe`**: Stores raw VIX futures data (and potentially VIX Index) directly from CBOE.
*   **`continuous_contracts`**: Stores all continuous contract data built by `continuous_contract_loader.py`.
*   **`futures_roll_calendar`**: Primarily used by `src/scripts/market_data/vix/update_vx_futures.py` for determining active CBOE VIX contracts. 