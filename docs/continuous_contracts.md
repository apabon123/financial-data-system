# Continuous Futures Contracts Guide

This document explains how continuous futures contracts are handled within the system, covering both locally generated contracts (VIX) and source-provided contracts (ES, NQ).

## Overview

Continuous futures contracts provide a synthetic, unbroken price series for a futures instrument by stitching together individual monthly or quarterly contracts based on predefined roll rules. This system uses two distinct methods depending on the root symbol:

1.  **Local Generation (VIX):** Continuous VIX contracts (`@VX=...`) are generated locally using data from individual VIX contracts (`VXK25`, `VXM25`, etc.) stored in the `market_data` table. The stitching logic relies on a pre-calculated mapping table (`continuous_contract_mapping`).
2.  **Direct Fetching (ES, NQ):** Continuous ES (`@ES=...`) and NQ (`@NQ=...`) contracts are fetched *directly* from the TradeStation API, which provides its own pre-calculated continuous series. These are *not* generated locally by stitching individual contracts.

All continuous contract data, regardless of origin, is stored in the `continuous_contracts` table.

## Naming Conventions

*   **VIX (Generated):** `@VX=N01XN`
    *   `@VX`: Indicates the VIX root symbol.
    *   `N`: The contract number (1-9, representing front month, second month, etc.).
    *   `01XN`: A fixed suffix used for locally generated VIX contracts.
    *   *Example:* `@VX=101XN` (Front month continuous VIX), `@VX=201XN` (Second month continuous VIX).

*   **ES/NQ (TradeStation):** `@ROOT=N02XT` (where T is C or N)
    *   `@ROOT`: Indicates the root symbol (ES or NQ).
    *   `N`: The contract number (typically 1 or 2 for front/second month provided by TradeStation).
    *   `02`: A prefix indicating TradeStation's continuous contract type.
    *   `X`: Separator.
    *   `T`: Calculation type (`C` for Calendar-Weighted, `N` for Nearest).
    *   *Example:* `@ES=102XC` (ES front month, calendar-weighted), `@NQ=202XN` (NQ second month, nearest).

## VIX: Local Generation Process

The generation of continuous VIX contracts (`@VX=...`) is a two-step process integrated into the main data update (`update_all_market_data.py`):

1.  **Update Mapping Table (`continuous_contract_mapping`):**
    *   **Script:** `src.scripts.market_data.vix.create_continuous_contract_mapping.main()`
    *   **Purpose:** Creates/updates a table that maps every trading date to the specific underlying VIX contract (e.g., `VXK25`) that represents the 1st, 2nd, ..., 9th continuous contract on that day.
    *   **Inputs:** Uses the `futures_roll_calendar` table (for expiry dates) and trading dates derived from `market_data`. Relies on expiry rules defined in `config/market_symbols.yaml`.
    *   **Frequency:** Run during every `update_all_market_data.py` execution to ensure the mapping is current.

2.  **Generate Continuous Data:**
    *   **Script:** `src.scripts.market_data.generate_continuous_futures.main()` (called with `root_symbol='VX'`)
    *   **Purpose:** Uses the `continuous_contract_mapping` table to identify the correct underlying contract for each date and stitches their price data (`market_data` table) together to form the continuous series (`@VX=101XN`, `@VX=201XN`, etc.).
    *   **Output:** Upserts the resulting continuous time series data into the `continuous_contracts` table.
    *   **Frequency:** Run during every `update_all_market_data.py` execution. By default, it updates the last ~90 days, but can perform a full regeneration if specified.

## ES/NQ: Direct Fetching Process

Continuous contracts for ES and NQ are updated as part of the main data update (`update_all_market_data.py`) using the `MarketDataFetcher`:

1.  **Fetch Continuous Data:**
    *   **Mechanism:** `fetch_market_data.py`'s `update_continuous_contracts` function.
    *   **Data Source:** TradeStation API.
    *   **Action:** Fetches the latest *daily* data points for the configured continuous symbols (e.g., `@ES=102XC`, `@NQ=102XN`).

2.  **Store Data:**
    *   **Action:** Upserts the fetched data directly into the `continuous_contracts` table.

*Note:* There is no local stitching or mapping table involved for ES/NQ continuous contracts in the current workflow.

## Accessing Continuous Data

Continuous contract data can be queried directly from the `continuous_contracts` table using standard SQL or viewed via the `DB_inspect.bat` tool.

```sql
-- Example: Get recent front-month continuous VIX data
SELECT *
FROM continuous_contracts
WHERE symbol = '@VX=101XN'
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