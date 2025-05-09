# Financial Data System

A system for downloading, updating, managing, and inspecting financial market data, primarily focusing on futures contracts (VIX, ES, NQ) using DuckDB.

## Overview

This system automates key tasks related to maintaining a financial market database:

1.  **Database Backup:** Regularly backs up the DuckDB database via scheduled tasks (configured outside this project, assumed working).
2.  **Database Update:** Updates market data (VIX, ES, NQ futures, VIX Index) via a scheduled task (`update_market_data.bat`). This script orchestrates calls to various Python scripts within the `src/scripts/market_data/` directory.
3.  **Database Inspection:** Provides an interactive command-line interface (`DB_inspect.bat`) to view market data inventory, contract details, and perform basic data checks using scripts in `src/scripts/analysis/` and `src/scripts/market_data/`.

## Core Components

*   **Data Storage:** `data/financial_data.duckdb` - The central DuckDB database. This database contains multiple tables:
    *   `market_data`: Holds data for most futures contracts (e.g., ES, NQ) and potentially other non-CBOE instruments.
    *   `market_data_cboe`: Specifically holds daily VIX futures (`VX%` symbols) and VIX Index (`$VIX.X`) data downloaded directly from the CBOE website.
    *   Other tables for configuration, continuous contracts, etc.
*   **Configuration:** `config/market_symbols.yaml` - Defines symbols, exchanges, update sources, and other parameters.
*   **Update Orchestrator:** `src/scripts/market_data/update_all_market_data.py` - The main Python script called by `update_market_data.bat` to handle the update process for different data sources (CBOE for VIX, TradeStation for ES/NQ, continuous contracts).
*   **Data Fetchers/Processors:** Various scripts within `src/scripts/market_data/` handle specific download, processing, and database update tasks for each data type (e.g., `update_vx_futures.py` which populates `market_data_cboe`, `fetch_market_data.py`, `generate_continuous_futures.py`).
*   **Inspection/Analysis Scripts:** Scripts within `src/scripts/analysis/` and `src/scripts/market_data/view_futures_contracts.py` support the `DB_inspect.bat` interface. `view_futures_contracts.py` dynamically queries either `market_data` or `market_data_cboe` based on the requested symbol.

## Usage

### 1. Update Market Data

Run the batch script. This is typically done via a scheduled task.

```bash
update_market_data.bat
```

This script executes `src/scripts/market_data/update_all_market_data.py`, which orchestrates the following major steps:

*   **Symbol Metadata Update:** Populates/updates the `symbol_metadata` table from `config/market_symbols.yaml`. This table guides the subsequent update processes.
*   **Individual Instruments Update (TradeStation):**
    *   Fetches/updates data for active individual futures (e.g., ESM25, NQM25), indices (e.g., SPX), and equities (e.g., SPY) as defined in `symbol_metadata` to be sourced from TradeStation.
    *   Data is retrieved using `MarketDataFetcher` and stored in the `market_data` table.
*   **Raw VIX Futures Update (CBOE):**
    *   Fetches/updates daily data for active individual VIX futures contracts (e.g., VXK25, VXM25) directly from the CBOE website.
    *   This data is stored in the `market_data_cboe` table.
*   **Continuous Contracts Update (TradeStation Sourced):**
    *   Identifies continuous contract symbols (e.g., `@ES=102XC`, `@NQ=102XN`, `@VX=101XN`) from `symbol_metadata` that are designated with `data_source = 'tradestation'`.
    *   For each, `continuous_contract_loader.py` is called. This script:
        *   Uses `MarketDataFetcher` to retrieve data for the underlying individual contracts from TradeStation.
        *   Builds the continuous contract series.
        *   Stores the resulting data in the `continuous_contracts` table.
    *   This now includes VIX continuous contracts, which are built from TradeStation underlying VIX futures data.
*   **Intraday Data:** The process also updates intraday (e.g., 1-minute, 15-minute) data for individual contracts from TradeStation as configured in `symbol_metadata`.
*   **(Optional) Verification:** The `--verify` flag can be passed, though its primary handling within `update_all_market_data.py` is to log its presence. The batch script might separately call other verification scripts.

Logs for the update process are stored in the `logs/` directory.

### 2. Inspect Database

Run the interactive inspection tool:

```bash
DB_inspect.bat
```

This tool provides a menu to:
*   View inventory summaries.
*   List details for specific futures contracts (individual and continuous). The script intelligently queries the correct table (`market_data` or `market_data_cboe`) based on the symbol.
*   Perform data quality checks.
*   Export data.

### 3. Manual Script Execution (Advanced)

Individual Python scripts within `src/scripts/` can be run manually if needed, but the primary intended workflows are via the `.bat` files.

## Project Structure

```
financial-data-system/
├── backups/                # Database backup files
├── config/                 # Configuration files (market_symbols.yaml)
├── data/                   # Data storage (financial_data.duckdb)
├── docs/                   # Documentation
├── logs/                   # Log files from updates and scripts
├── output/                 # Generated output files (e.g., CSV exports)
├── src/                    # Source code
│   ├── scripts/            # Core Python scripts
│   │   ├── analysis/       # Data analysis and inspection scripts
│   │   ├── database/       # Database utility scripts
│   │   ├── market_data/    # Market data fetching and processing scripts
│   │   └── utility/        # General utility scripts
│   ├── api/                # API interaction modules (e.g., TradeStation)
│   ├── utils/              # Shared utility functions
│   ├── sql/                # Reusable SQL query files
│   └── templates/          # Template files (if any)
├── tasks/                  # Scheduled task configurations/definitions
├── tests/                  # Unit and integration tests
│   └── Test Framework/     # Specific test framework components
├── venv/                   # Python virtual environment
├── .gitignore              # Git ignore file
├── DB_inspect.bat          # Interactive database inspection tool
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── update_market_data.bat  # Main data update script
```

## Installation / Setup

1.  **Clone:** `git clone <repository_url>`
2.  **Environment:** Create and activate a Python virtual environment (e.g., using `venv`).
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Dependencies:** Install required packages.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration:**
    *   Review and potentially adjust `config/market_symbols.yaml`.
    *   Ensure TradeStation API credentials are correctly configured if using TradeStation data sources (likely managed via environment variables or a secure configuration method not stored in Git).
5.  **Database:** The database `data/financial_data.duckdb` will be created automatically on the first run if it doesn't exist.
6.  **Scheduled Tasks:** Set up scheduled tasks to run `update_market_data.bat` and handle database backups according to your desired frequency.