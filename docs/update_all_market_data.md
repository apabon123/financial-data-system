# Documentation for `update_all_market_data.py`

## Overview

`update_all_market_data.py` is the main orchestrator script responsible for updating various types of market data within the financial data system. It sequences calls to specialized scripts to fetch, process, and store data for indices, futures contracts (individual and continuous), and performs verification checks.

The script manages a single database connection that is passed down to the various sub-scripts it calls, ensuring efficient and consistent database access.

## Execution

The script is typically run from the root directory of the project:

```bash
python src/scripts/main/update_all_market_data.py [arguments]
```

Common arguments include:

*   `--verify`: Runs verification checks after updating data.
*   `--full-update`: Performs a more comprehensive update, potentially including historical gap filling.
*   `--symbols SYMBOLS`: Specifies which base symbols (e.g., "ES NQ VX") to update. Defaults to symbols defined in the configuration.
*   `--start-date YYYY-MM-DD`: Specifies the start date for continuous contract generation/verification.
*   `--end-date YYYY-MM-DD`: Specifies the end date for continuous contract generation/verification.
*   `--config-path PATH`: Path to the market symbols configuration file (defaults to `config/market_symbols.yaml`).
*   `--db-path PATH`: Path to the DuckDB database file (defaults to `data/financial_data.duckdb`).

## Update Process Flow

The script executes the following steps in sequence:

1.  **Initialization:** Connects to the DuckDB database.
2.  **VIX Index Update:** Calls `update_vix_index.py` to download and update the daily VIX index data (`$VIX.X`) from CBOE.
3.  **VX Futures Update:** Calls `update_vx_futures.py` to download and update data for individual VIX futures contracts (e.g., VXK25, VXM25) from CBOE.
4.  **ES/NQ Futures Update:** Executes `update_tradestation_futures.py` (via a temporary wrapper) as a subprocess to fetch and update daily and intraday data for ES and NQ futures contracts from the TradeStation API. The main database connection is temporarily closed during this step to avoid conflicts.
5.  **Database Reconnect:** Re-establishes the main database connection.
6.  **ES/NQ Continuous Update (TradeStation):** Uses the `MarketDataFetcher` (which utilizes `continuous_contract_loader.py` logic internally) to update *daily* continuous contracts for ES and NQ (`@ES=102XC`, `@ES=102XN`, etc.) fetched from TradeStation. *Note: Generation of continuous contracts from underlying futures is handled separately.*
7.  **VX Continuous Generation:** Calls `generate_continuous_futures.py` to generate daily continuous VIX futures contracts (e.g., VXc1, VXc2) based on the previously updated individual contracts and the roll calendar.
8.  **Historical Gap Filling (Conditional):** If the `--full-update` flag is provided, calls `fill_historical_gaps.py` to identify and potentially fill missing historical data points.
9.  **Verification (Conditional):** If the `--verify` flag is provided, calls `improved_verify_continuous.py` for each continuous contract generated (e.g., VXc1, VXc2) to check for consistency and potential errors against the underlying contract data and roll logic.
10. **Data Counts:** Logs counts of key data points (e.g., VIX rows, continuous contract rows).
11. **Completion:** Closes the database connection and exits.

## Script Dependencies

`update_all_market_data.py` depends on the following scripts:

*   `src/scripts/market_data/vix/update_vix_index.py`
*   `src/scripts/market_data/vix/update_vx_futures.py`
*   `src/scripts/market_data/tradestation/update_tradestation_futures.py` (Called indirectly via subprocess)
*   `src/scripts/market_data/generate_continuous_futures.py`
*   `src/scripts/analysis/verify/improved_verify_continuous.py`
*   `src/scripts/market_data/fill_historical_gaps.py` (Conditional based on `--full-update`)
*   `src/scripts/market_data/tradestation/continuous_contract_loader.py` (Logic used internally by `MarketDataFetcher`)
*   `src/data_fetching/fetch_market_data.py` (Contains `MarketDataFetcher`)

**Important:** Ensure these scripts are present and functional. Do not remove or rename them without updating the calls within `update_all_market_data.py`. 