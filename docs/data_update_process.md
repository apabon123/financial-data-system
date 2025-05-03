# Data Update Process Guide

This document details the process for updating the market data in the `data/financial_data.duckdb` database.

## Overview

The primary goal of the update process is to fetch the latest available data for configured instruments and generate derived data like continuous futures contracts. The process is designed to be run regularly (e.g., daily via a scheduled task) to keep the database current.

## Triggering the Update

The update process is initiated by running the main batch script from the project root:

```bash
update_market_data.bat
```

This script serves as a simple wrapper that executes the core Python orchestrator script: `src/scripts/market_data/update_all_market_data.py`. Any command-line arguments passed to `update_market_data.bat` (like `--verify`) are forwarded to the Python script.

## Core Orchestrator: `update_all_market_data.py`

This Python script manages the sequence of update steps, handles database connections, and calls specialized sub-scripts or modules for specific tasks.

**Key Responsibilities:**

*   Establishes and manages a main DuckDB connection passed to most sub-tasks.
*   Loads configuration from `config/market_symbols.yaml`.
*   Initializes data fetching agents (e.g., `MarketDataFetcher` for TradeStation).
*   Logs the process extensively to the `logs/` directory.
*   Executes update steps in a defined order.

## Update Sequence and Data Sources

The `update_all_market_data.py` script executes the following steps:

1.  **Initialization:**
    *   Connects to the database (`data/financial_data.duckdb`).
    *   Loads `config/market_symbols.yaml`.
    *   Authenticates with TradeStation API (if ES/NQ are configured).

2.  **VIX Index Update:**
    *   **Script:** Calls `src.scripts.market_data.vix.update_vix_index.main()`
    *   **Data Source:** CBOE Website (CSV download)
    *   **Action:** Fetches the latest daily data for `$VIX.X` and upserts it into the **`market_data_cboe`** table.

3.  **Individual VIX Futures Update:**
    *   **Script:** Calls `src.scripts.market_data.vix.update_vx_futures.main()`
    *   **Data Source:** CBOE Website (CSV downloads for each active contract)
    *   **Action:**
        *   Identifies active VX contracts based on the `futures_roll_calendar` table.
        *   Downloads the historical CSV for each active contract.
        *   Prepares and cleans the data.
        *   Performs a `DELETE` and `INSERT` for the daily data of each contract (e.g., `VXK25`, `VXM25`) into the **`market_data_cboe`** table. This table is specifically for CBOE-sourced VIX data.

4.  **VIX Continuous Contract Mapping Update:**
    *   **Script:** Calls `src.scripts.market_data.vix.create_continuous_contract_mapping.main()`
    *   **Data Source:** `futures_roll_calendar` and `market_data_cboe` tables (for trading days).
    *   **Action:** Rebuilds the `continuous_contract_mapping` table for the `VX` root symbol. This table maps each trading date to the specific underlying contract (e.g., `VXK25`) that should be used for each continuous contract series (e.g., `@VX=101XN`, `@VX=201XN`, etc.) based on roll rules.

5.  **VIX Continuous Contract Generation:**
    *   **Script:** Calls `src.scripts.market_data.generate_continuous_futures.main()` specifically for `root_symbol='VX'`. 
    *   **Data Source:** `market_data_cboe` (underlying VX contracts) and `continuous_contract_mapping` table.
    *   **Action:**
        *   Generates daily price series for continuous VIX contracts (e.g., `@VX=101XN`, `@VX=201XN`, etc.) by stitching together data from the appropriate underlying contracts identified in the mapping table.
        *   By default, updates the last ~90 days. A full regeneration can be triggered via arguments.
        *   Upserts the generated continuous data into the `continuous_contracts` table.

6.  **Individual ES & NQ Futures Update:**
    *   **Mechanism:** Calls `fetch_market_data.py`'s `update_futures_contracts` function (likely via `MarketDataFetcher` instance). Note: The older method of running a separate script via subprocess might still exist but this is the intended path.
    *   **Data Source:** TradeStation API
    *   **Action:**
        *   Identifies active ES and NQ contracts based on configuration (`config/market_symbols.yaml`).
        *   Fetches latest daily and intraday (1min, 15min) data for these active contracts (e.g., `ESM25`, `ESH25`, `NQM25`, `NQH25`).
        *   Upserts the data into the `market_data` table.

7.  **ES & NQ Continuous Contract Update:**
    *   **Mechanism:** Calls `fetch_market_data.py`'s `update_continuous_contracts` function (likely via `MarketDataFetcher` instance).
    *   **Data Source:** TradeStation API (provides pre-calculated continuous contracts)
    *   **Action:**
        *   Fetches latest *daily* data for TradeStation's continuous contract symbols (e.g., `@ES=102XC`, `@NQ=102XN`).
        *   Upserts this data into the `continuous_contracts` table.
        *   *Important Distinction:* Unlike VX, the continuous contracts for ES and NQ are currently *fetched directly* from the source (TradeStation) rather than generated locally from underlying contracts.

8.  **Verification (Optional):**
    *   **Trigger:** If `--verify` argument is passed to `update_market_data.bat`.
    *   **Script:** Calls internal verification methods within `update_all_market_data.py` or potentially dedicated verification scripts.
    *   **Action:** Performs checks, such as:
        *   Ensuring recent data exists for key continuous symbols.
        *   Comparing row counts.
        *   Checking for gaps in recent data.

9.  **Completion:**
    *   Logs final summary information.
    *   Closes the database connection.

## Configuration (`config/market_symbols.yaml`)

This file is crucial for defining which symbols are processed, their data sources, frequencies, and specific parameters like expiry rules used for roll calculations (relevant for VIX continuous mapping). Refer to `docs/futures_configuration.md` for details on this file's structure.

## Continuous Contracts

*   **VIX (@VX=...):** Generated *locally* by `generate_continuous_futures.py` using the `continuous_contract_mapping` table which determines rolls based on expiry rules defined in the config and the `futures_roll_calendar`. Data is stored in the `continuous_contracts` table.
*   **ES/NQ (@ES=..., @NQ=...):** *Fetched directly* from TradeStation as pre-calculated continuous series. Data is stored in the `continuous_contracts` table. Local generation logic for ES/NQ from underlying contracts is not currently implemented or used in the main update flow. 