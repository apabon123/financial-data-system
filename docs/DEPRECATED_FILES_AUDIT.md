# Deprecated Files Audit

This document provides a comprehensive audit of deprecated files in the Financial Data System, including their original functionality and replacements in the new architecture.

## Root Directory Files

| File | Functionality | Replacement | Status |
|------|--------------|-------------|--------|
| `DB_inspect.bat` | Command-line interface for database inspection | `src/scripts/scripts/db_inspect.bat` | Move to scripts directory |
| `update_market_data.bat` | Legacy market data update script | `update_market_data_v2.bat` | Replace with new version |
| `CLAUDE.md` | Temporary file for Claude sessions | N/A | Remove |

## Deprecated Directory

| File | Functionality | Replacement | Status |
|------|--------------|-------------|--------|
| `build_continuous_contracts.py` | Back-adjusted continuous futures contracts generator | `src/processors/continuous/panama.py` | Replaced with Panama method |
| `check_cboe_table.py` | Validates CBOE data table structure | `src/processors/cleaners/vx_zero_prices.py` | Functionality integrated into data cleaning pipeline |
| `find_missing_vx_continuous.py` | Identifies gaps in VX continuous contracts | `src/processors/cleaners/pipeline.py` | Functionality integrated into data cleaning pipeline |
| `fix_cboe_intervals.py` | Fixes inconsistent interval data in CBOE tables | `src/processors/cleaners/pipeline.py` | Functionality integrated into data cleaning pipeline |
| `mktschematest.py` | Tests market data schema | `src/core/database.py` | Schema validation integrated into database connector |
| `print_function.py` | Utility for printing formatted data | `src/core/logging.py` | Logging functionality in core module |
| `setup_tasks.bat` | Windows task scheduler setup | `src/scripts/scripts/` | Move to scripts directory |
| `temp_fetch_vxu25.py` | Temporary script for fetching VX futures | `src/scripts/market_data/fetch_market_data.py` | Consolidated into main fetching script |
| `temp_verify_vx_latest.py` | Temporary script for verifying VX data | `src/processors/cleaners/vx_zero_prices.py` | Verification integrated into data cleaning pipeline |
| `view_futures_contracts_full.txt` | Text output of futures contracts | `src/scripts/market_data/list_market_symbols.py` | Replaced with dynamic listing script |

## Duplicate/Temp Directories

| Directory | Description | Action |
|-----------|-------------|--------|
| `Projects/data-management/financial-data-system/` | Duplicate project structure | Remove - redundant |
| `financial_data_system_new/` | Temporary directory for new structure | Remove after verifying all files are properly migrated |

## Temp Database Scripts

The `src/scripts/database/temp_*.py` scripts were temporary utilities created for one-time database operations. All of their functionality has been replaced by the new data cleaning pipeline and continuous futures processors.

| File | Functionality | Replacement | Status |
|------|--------------|-------------|--------|
| `temp_check_continuous_es_variants.py` | Checks variants of ES continuous contracts | `src/processors/continuous/registry.py` | Replaced by continuous contract registry |
| `temp_check_populator_output.py` | Checks symbol metadata population | `src/core/app.py` | Integrated into application class |
| `temp_cleanup_generic_ats.py` | Cleans up generic @-symbol contracts | `src/processors/cleaners/pipeline.py` | Integrated into data cleaning pipeline |
| `temp_cleanup_generic_cont.py` | Cleans up generic continuous contracts | `src/processors/cleaners/pipeline.py` | Integrated into data cleaning pipeline |
| `temp_correct_es_mislabeled.py` | Fixes mislabeled ES contracts | `src/processors/cleaners/pipeline.py` | Integrated into data cleaning pipeline |
| `temp_debug_ccl_metadata_lookup.py` | Debugs contract lookup | `src/processors/continuous/roll_calendar.py` | Replaced by roll calendar handling |
| `temp_debug_o1_es_counts.py` | Debugs ES contract counts | `src/processors/continuous/registry.py` | Replaced by continuous contract registry |
| `temp_delete_es_continuous_data.py` | Deletes ES continuous data | `src/processors/continuous/roll_calendar.py` | Handled by roll calendar component |
| `temp_delete_specific_es_from_market_data.py` | Deletes specific ES contracts | `src/processors/cleaners/pipeline.py` | Integrated into data cleaning pipeline |
| `temp_find_specific_es_variants.py` | Finds ES contract variants | `src/processors/continuous/roll_calendar.py` | Handled by roll calendar component |
| `temp_verify_at_symbol_counts.py` | Verifies @-symbol counts | `src/processors/continuous/registry.py` | Replaced by continuous contract registry |

## Legacy Test Scripts

The test scripts in the root of the `tests/` directory have been replaced by the structured test framework in `tests/unit/`, `tests/integration/`, and `tests/validation/`.

| File | Functionality | Replacement | Status |
|------|--------------|-------------|--------|
| `tests/check_all_intervals.py` | Checks all interval data | `tests/unit/test_data_cleaning_pipeline.py` | Move to tests/legacy |
| `tests/check_intervals.py` | Simple interval checker | `tests/unit/test_data_cleaning_pipeline.py` | Move to tests/legacy |
| `tests/test_fetch_vxf25.py` | Tests VX future fetching | `tests/unit/test_panama_method.py` | Move to tests/legacy |
| `tests/test_fetch_vxk20.py` | Tests VX future fetching | `tests/unit/test_panama_method.py` | Move to tests/legacy |
| `tests/test_fetch_vxz24.py` | Tests VX future fetching | `tests/unit/test_panama_method.py` | Move to tests/legacy |
| `tests/test_market_data_agent.py` | Tests market data agent | `tests/integration/test_continuous_contract_flow.py` | Move to tests/legacy |
| `tests/test_market_data_fetcher.py` | Tests market data fetcher | `tests/integration/test_continuous_contract_flow.py` | Move to tests/legacy |
| `tests/test_schema.py` | Tests database schema | `tests/unit/test_data_cleaning_pipeline.py` | Move to tests/legacy |

## Tasks Directory

The `tasks/` directory contains Windows scheduled task configuration files that should be moved to the scripts directory for better organization.

| File | Functionality | Replacement | Status |
|------|--------------|-------------|--------|
| `tasks/import_tasks.bat` | Imports scheduled tasks | `src/scripts/scripts/` | Move to scripts directory |
| `tasks/setup_tasks_simple.bat` | Sets up simple scheduled tasks | `src/scripts/scripts/` | Move to scripts directory |
| `tasks/setup_user_tasks.bat` | Sets up user scheduled tasks | `src/scripts/scripts/` | Move to scripts directory |
| `tasks/setup_vix_data_task.bat` | Sets up VIX data scheduled tasks | `src/scripts/scripts/` | Move to scripts directory |
| `tasks/setup_vix_data_task.ps1` | PowerShell script for VIX data tasks | `src/scripts/scripts/` | Move to scripts directory |
| `tasks/vix_update_evening.xml` | Task XML for evening VIX update | `src/scripts/resources/` | Move to resources directory |
| `tasks/vix_update_morning.xml` | Task XML for morning VIX update | `src/scripts/resources/` | Move to resources directory |