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
    *   **Action:** Manages database connections and sequences calls to update VIX Index, individual VIX/ES/NQ futures, and continuous contracts from different sources (CBOE, TradeStation). See `docs/data_update_process.md` for the detailed flow.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.update_all_market_data [--verify] [--db-path PATH] [--config-path PATH] ...` (See script argparse for all options).

*   **`fetch_market_data.py`** (within `src/data_fetching` or similar, contains `MarketDataFetcher` class):
    *   **Purpose:** Module containing the `MarketDataFetcher` class responsible for interacting with the TradeStation API.
    *   **Action:** Handles authentication, fetching of historical bars (daily, intraday) for individual contracts (ES, NQ), fetching continuous contracts (@ES, @NQ), and potentially account data. Used by `update_all_market_data.py`.
    *   **Direct Usage:** Not typically run directly; used programmatically via `MarketDataFetcher`.

## VIX Specific Data Processing

Located in `src/scripts/market_data/vix/`:

*   **`update_vix_index.py`**:
    *   **Purpose:** Fetches and updates daily VIX Index (`$VIX.X`) data.
    *   **Data Source:** CBOE Website (CSV).
    *   **Called By:** `update_all_market_data.py`.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.update_vix_index [--db-path PATH]`

*   **`update_vx_futures.py`**:
    *   **Purpose:** Fetches and updates daily data for individual, active VIX futures contracts.
    *   **Data Source:** CBOE Website (CSV per contract).
    *   **Called By:** `update_all_market_data.py`.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.update_vx_futures [--db-path PATH] [--config-path PATH]`

*   **`create_continuous_contract_mapping.py`**:
    *   **Purpose:** Generates/updates the `continuous_contract_mapping` table used for local VIX continuous contract generation.
    *   **Inputs:** `futures_roll_calendar`, `config/market_symbols.yaml`.
    *   **Called By:** `update_all_market_data.py`.
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.vix.create_continuous_contract_mapping [--root-symbol VX] [--num-contracts N] ...`

## Continuous Contract Generation (VIX)

Located in `src/scripts/market_data/`:

*   **`generate_continuous_futures.py`**:
    *   **Purpose:** Generates continuous contract time series *locally* by stitching underlying contract data based on the `continuous_contract_mapping` table. Currently used only for VIX (`@VX=...`).
    *   **Inputs:** `market_data` table, `continuous_contract_mapping` table.
    *   **Called By:** `update_all_market_data.py` (specifically for `root_symbol='VX'`).
    *   **Direct Usage (Advanced):** `python -m src.scripts.market_data.generate_continuous_futures --root-symbol VX [--db-path PATH] [--start-date DATE] [--end-date DATE] [--force]`

## Inspection / Analysis Support Scripts

Located in `src/scripts/analysis/` or `src/scripts/market_data/`:

*   **`view_futures_contracts.py`** (in `src/scripts/market_data/`):
    *   **Purpose:** Script called by `DB_inspect.bat` (Option I1) to display a summary table of individual futures contracts for a given base symbol and interval (e.g., list all VX daily contracts with row counts, date ranges).
    *   **Direct Usage:** `python -m src.scripts.market_data.view_futures_contracts --base-symbol SYMBOL [--interval-value V] [--interval-unit U]`

*   *(Other analysis scripts may exist in `src/scripts/analysis/` and might be called by `DB_inspect.bat` for specific checks or views).*

## Database Administration

Located in `src/scripts/database/`:

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