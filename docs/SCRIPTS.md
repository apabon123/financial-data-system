# Scripts and Core Modules Documentation

This document provides information about the key scripts, modules, and entry points in the Financial Data System.

## Main Entry Points (User Facing)

These are the primary ways to interact with the system.

*   **`update_market_data.bat`**: (Located in project root)
    *   **Purpose:** The main script for triggering a full market data update. Typically run via a scheduled task.
    *   **Action:** Executes `src/scripts/market_data/update_all_market_data.py` and passes along any command-line arguments (e.g., `--verify`).
    *   **Usage:** `update_market_data.bat [--verify]`

*   **`DB_inspect.bat`**: (Located in project root)
    *   **Purpose:** Provides an interactive command-line menu for inspecting database contents, viewing contract details, and running basic data checks.
    *   **Action:** Calls various underlying Python analysis and viewing scripts based on user menu selections.
    *   **Usage:** `DB_inspect.bat`

## Core Update Orchestration & Fetching

Located in `src/scripts/market_data/`:

*   **`update_all_market_data.py`**:
    *   **Purpose:** The central Python script that orchestrates the entire data update process. Called by `update_market_data.bat`.
    *   **Action:** Manages database connections and sequences calls to:
        1.  Update `symbol_metadata` (via `populate_symbol_metadata.py`).
        2.  Update individual instruments (futures, equities, indices) from TradeStation (via `MarketDataFetcher`).
        3.  Update raw VIX futures from CBOE (via `update_vx_futures.py`).
        4.  Update TradeStation-sourced continuous contracts, including VIX (via `continuous_contract_loader.py`).
    *   See `docs/data_update_process.md` for the detailed flow.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.update_all_market_data [--db-path PATH] [--config-path PATH] [--lookback-days DAYS] [--roll-proximity-days DAYS] [--fetch-all-individual-history] ...` (See script argparse for all options).

*   **`fetch_market_data.py`** (module containing `MarketDataFetcher` class):
    *   **Purpose:** Module containing the `MarketDataFetcher` class responsible for interacting with the TradeStation API and handling expiry logic for determining active contracts.
    *   **Action:** Handles authentication, fetching of historical bars (daily, intraday) for individual contracts, and providing underlying data for continuous contract construction. Used by `update_all_market_data.py` and `continuous_contract_loader.py`.
    *   **Direct Usage:** Not typically run directly; used programmatically via `MarketDataFetcher`.

*   **`continuous_contract_loader.py`** (in `src/scripts/market_data/`):
    *   **Purpose:** Builds continuous contract time series using underlying individual contract data from TradeStation.
    *   **Action:** Called by `update_all_market_data.py` for each TradeStation-sourced continuous contract (e.g., `@ES=...`, `@VX=...`) defined in `symbol_metadata`.
        *   Initializes its own `MarketDataFetcher`.
        *   Determines fetch period (full history, latest chunk, or full history near roll).
        *   Uses `MarketDataFetcher` to get underlying contract data from TradeStation.
        *   Stitches the data and upserts it into the `continuous_contracts` table.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.continuous_contract_loader SYMBOL [--db-path PATH] [--config-path PATH] [--fetch-mode MODE] [--lookback-days DAYS] [--roll-proximity-threshold-days DAYS] [--force]`

## VIX Specific Data Processing

Located in `src/scripts/market_data/vix/`:

*   **`update_vix_index.py`**:
    *   **Purpose:** Fetches and updates daily VIX Index (`$VIX.X`) data if configured and called.
    *   **Data Source:** CBOE Website (CSV).
    *   **Called By:** Potentially `update_all_market_data.py` if specific VIX Index logic is retained there, or could be configured as a standard 'index' in `symbol_metadata` to be fetched by `MarketDataFetcher` if sourced from TradeStation.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.update_vix_index [--db-path PATH]`

*   **`update_vx_futures.py`**:
    *   **Purpose:** Fetches and updates daily data for individual, active VIX futures contracts from CBOE.
    *   **Data Source:** CBOE Website (CSV per contract).
    *   **Called By:** `update_all_market_data.py` (for raw CBOE VIX data to `market_data_cboe` table).
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.update_vx_futures [--db-path PATH] [--config-path PATH]`

*   **`create_continuous_contract_mapping.py`**:
    *   **Purpose:** (Largely historical/deprecated for primary continuous VIX series if they are built from TradeStation data) Generates/updates the `continuous_contract_mapping` table for local VIX continuous contract generation *from CBOE data*.
    *   **Inputs:** `futures_roll_calendar`, `config/market_symbols.yaml`.
    *   **Called By:** Historically by `update_all_market_data.py`. Its current role depends on whether any CBOE-based local VIX continuous generation is still active.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.create_continuous_contract_mapping [--root-symbol VX] [--num-contracts N] ...`

## Continuous Contract Generation (VIX) -- DEPRECATED SECTION TITLE
## (See `continuous_contract_loader.py` above for current TS-based generation)

Located in `src/scripts/market_data/`:

*   **`generate_continuous_futures.py`**:
    *   **Purpose:** (Largely historical/deprecated for primary continuous series if `continuous_contract_loader.py` is used) Generates continuous contract time series *locally* by stitching underlying contract data based on a `continuous_contract_mapping` table. Previously used for VIX from CBOE data.
    *   **Inputs:** `market_data_cboe` table, `continuous_contract_mapping` table.
    *   **Called By:** Historically by `update_all_market_data.py`. Its current role is minimal if VIX continuous contracts are built by `continuous_contract_loader.py`.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.generate_continuous_futures --root-symbol VX [--db-path PATH] [--start-date DATE] [--end-date DATE] [--force]`

## Inspection / Analysis Support Scripts

Located in `src/scripts/analysis/` or `src/scripts/market_data/`:

*   **`view_futures_contracts.py`** (in `src/scripts/market_data/`):
    *   **Purpose:** Script called by `DB_inspect.bat` (Option I1) to display a summary table of individual futures contracts for a given base symbol and interval (e.g., list all VX daily contracts with row counts, date ranges).
    *   **Direct Usage:** `python -m src.scripts.market_data.view_futures_contracts --base-symbol SYMBOL [--interval-value V] [--interval-unit U]`

*   *(Other analysis scripts may exist in `src/scripts/analysis/` and might be called by `DB_inspect.bat` for specific checks or views).*

## Database Administration & Population

Located in `src/scripts/database/`:

*   **`populate_symbol_metadata.py`**:
    *   **Purpose:** Reads `config/market_symbols.yaml` and populates/updates the `symbol_metadata` table in the database.
    *   **Action:** This script translates the human-readable YAML configuration into a structured table that drives the data update processes.
    *   **Called By:** `update_all_market_data.py` (as Step 0).
    *   **Direct Usage (Advanced):** `python -m src.scripts.database.populate_symbol_metadata [--db-path PATH] [--config-path PATH]`

*   **`backup_database.py`**:
    *   **Purpose:** Performs a backup of the DuckDB database file.
    *   **Action:** Creates a timestamped copy of the `.duckdb` file in the specified backup directory. Can implement retention logic.
    *   **Usage:** `python -m src.scripts.database.backup_database [--db-path PATH] [--output DIR] [--retention DAYS]` (Likely called by an external scheduled task).

## Utilities

Located in `src/scripts/utility/` or `src/utils/`:

*   *(Various utility scripts or modules might exist for common tasks like date calculations, logging setup, database connections, etc. List key ones if applicable).*
*   **`duckdb_launcher.py`** (if still present/used): Launches an interactive DuckDB CLI session connected to the database.

## Deprecated / Removed Scripts

The following scripts were previously part of the project but have been removed or are considered deprecated:
*   `check_*.py` (various temporary check scripts)
*   `fix_vx_continuous.bat`
*   `regenerate_vx_continuous.py`
*   `fill_vx_continuous_gaps.py`
*   `fill_vx_zero_prices.py`
*   `reorganize_project.py`
*   *(Add others as identified)*