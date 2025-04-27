@echo off
:: Futures Database Inspection and Maintenance Batch File
:: This script provides a menu to view data and run maintenance tasks

:: Set coding to UTF-8
chcp 65001 > nul

echo.
echo =========================================
echo   Futures Database Inspection ^& Maint.
echo =========================================
echo.

:menu
echo Choose an option:
echo.
echo  --- Individual Contracts ---
echo    I1. View Specific Contracts (e.g., ESH24) by Base/Interval
echo    I2. Fetch NEW data for Specific Symbol (Append Only)
echo    I3. FORCE Fetch for Specific Symbol (Overwrite Existing)
echo    I4. Check for Missing Futures Contracts by Base/Interval
echo    I5. Update active ES and NQ contracts (Python)
echo.
echo  --- Continuous Contracts ---
echo    C1. View Continuous Contract Inventory Summary
echo    C2. View Continuous Contract Intervals by Base (e.g., @ES)
echo    C3. Fetch NEW data for Specific Continuous Symbol (@ES=...)
echo    C4. FORCE Fetch for Specific Continuous Symbol (@ES=...)
echo.
echo  --- Overall Inventory ---
echo    O1. View Market Data Inventory Summary (Base Symbols)
echo    O2. Data Quality Checks (SQL)
echo    O3. Data Gap Analysis (SQL)
echo.
echo  --- System Maintenance ---
echo    S1. Run FULL Market Data Update (Calls update_market_data.bat)
echo    S2. Run Database Maintenance (SQL - VACUUM/ANALYZE)
echo.
echo  E. Exit
echo.

set /p choice="Enter your choice: "
echo.

:: Convert choice to uppercase
for /f "delims=" %%A in ('echo %choice%^| find /v ""') do set choice=%%A

if /i "%choice%"=="I1" goto view_specific_contracts
if /i "%choice%"=="I2" goto fetch_specific_symbol
if /i "%choice%"=="I3" goto force_fetch_specific_symbol
if /i "%choice%"=="I4" goto check_missing_contracts
if /i "%choice%"=="I5" goto update_active

if /i "%choice%"=="C1" goto continuous_summary
if /i "%choice%"=="C2" goto view_continuous_details
if /i "%choice%"=="C3" goto fetch_continuous
if /i "%choice%"=="C4" goto force_fetch_continuous

if /i "%choice%"=="O1" goto symbol_summary_py
if /i "%choice%"=="O2" goto sql_data_quality
if /i "%choice%"=="O3" goto sql_data_gaps

if /i "%choice%"=="S1" goto run_market_update
if /i "%choice%"=="S2" goto sql_db_maintenance

if /i "%choice%"=="E" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:: --- Implementation Sections --- 

:view_specific_contracts
set base_symbol=
set interval_str=
set interval_unit=daily
set interval_value=1

set /p base_symbol="Enter Base Symbol (e.g., ES, NQ, VX): "
if not defined base_symbol (
    echo No base symbol entered. Aborting.
    goto pause_menu
)

set /p interval_str="Enter Interval [1min, 15min, daily] (leave blank for daily): "

if /i "%interval_str%"=="1min" ( 
    set interval_unit=minute
    set interval_value=1
) else if /i "%interval_str%"=="15min" ( 
    set interval_unit=minute
    set interval_value=15
) else if /i "%interval_str%"=="daily" ( 
    set interval_unit=daily
    set interval_value=1
) else if not defined interval_str ( 
    echo Using default interval: 1 daily
    set interval_unit=daily
    set interval_value=1
) else ( 
    echo Invalid interval entered. Using default: 1 daily
    set interval_unit=daily
    set interval_value=1
)

echo Viewing %base_symbol% contracts for %interval_value% %interval_unit%...
python src/scripts/market_data/view_futures_contracts.py %base_symbol% --interval-unit %interval_unit% --interval-value %interval_value%
goto pause_menu

:check_missing_contracts
set base_symbol=
set interval_str=
set interval_unit=daily
set interval_value=1
set fetch_flag=

set /p base_symbol="Enter Base Symbol to Check (e.g., ES, NQ, VX): "
if not defined base_symbol (
    echo No base symbol entered. Aborting.
    goto pause_menu
)

set /p interval_str="Enter Interval to Check [1min, 15min, daily] (leave blank for daily): "

if /i "%interval_str%"=="1min" ( 
    set interval_unit=minute
    set interval_value=1
) else if /i "%interval_str%"=="15min" ( 
    set interval_unit=minute
    set interval_value=15
) else if /i "%interval_str%"=="daily" ( 
    set interval_unit=daily
    set interval_value=1
) else if not defined interval_str ( 
    echo Using default interval: 1 daily
    set interval_unit=daily
    set interval_value=1
) else ( 
    echo Invalid interval entered. Using default: 1 daily
    set interval_unit=daily
    set interval_value=1
)

:: Prompt for fetching missing data
set /p fetch_choice="Attempt to fetch missing data? (Y/N, default N): "
if /i "%fetch_choice%"=="Y" set fetch_flag=--fetch-missing

echo Checking for missing %base_symbol% contracts for %interval_value% %interval_unit%...
echo Running: python src/scripts/analysis/check_futures_contracts.py %base_symbol% --interval-unit %interval_unit% --interval-value %interval_value% %fetch_flag%
python src/scripts/analysis/check_futures_contracts.py %base_symbol% --interval-unit %interval_unit% --interval-value %interval_value% %fetch_flag%
rem Add --missing-only to show just the list, or --fetch-missing to try and get them
goto pause_menu

:symbol_summary_py
echo Running Market Data Inventory Summary (Python)...
python src/scripts/market_data/summarize_symbol_inventory.py
goto pause_menu

:continuous_summary
echo Running Continuous Contract Inventory Summary (Python)...
python src/scripts/market_data/summarize_symbol_inventory.py --continuous-only
goto pause_menu

:view_continuous_details
set cont_base_symbol=
set /p cont_base_symbol="Enter Continuous Base Symbol (e.g., @ES, @VX): "
if not defined cont_base_symbol (
    echo No continuous base symbol entered. Aborting.
    goto pause_menu
)
echo Viewing Interval Details for %cont_base_symbol%...
python src/scripts/market_data/summarize_symbol_inventory.py --base-symbol %cont_base_symbol%
goto pause_menu

:sql_data_quality
echo Running Data Quality Checks SQL...
python src/scripts/db_inspector.py data_quality.sql
goto pause_menu

:sql_data_gaps
echo Running Data Gap Analysis SQL...
python src/scripts/db_inspector.py data_gaps.sql
goto pause_menu

:fetch_specific_symbol
set /p symbol_to_fetch="Enter the symbol to fetch (e.g., SPY, $VIX.X, ES, ESH25): "
echo Fetching NEW data for %symbol_to_fetch%...
python src/scripts/market_data/fetch_market_data.py --symbol %symbol_to_fetch%
goto pause_menu

:force_fetch_specific_symbol
set /p symbol_to_fetch="Enter the symbol to force fetch (e.g., SPY, $VIX.X, ES, ESH25): "
echo FORCE Fetching data for %symbol_to_fetch% (OVERWRITING EXISTING)...
python src/scripts/market_data/fetch_market_data.py --symbol %symbol_to_fetch% --force
goto pause_menu

:fetch_continuous
set cont_symbol_to_fetch=
set /p cont_symbol_to_fetch="Enter the specific continuous symbol to fetch (e.g., @ES=102XC, @NQ=102XN): "
if not defined cont_symbol_to_fetch (
    echo No symbol entered. Aborting.
    goto pause_menu
)
echo Fetching NEW data for %cont_symbol_to_fetch%...
python -m src.scripts.market_data.continuous_contract_loader --symbol %cont_symbol_to_fetch%
goto pause_menu

:force_fetch_continuous
set cont_symbol_to_fetch=
set /p cont_symbol_to_fetch="Enter the specific continuous symbol to FORCE fetch (e.g., @ES=102XC): "
if not defined cont_symbol_to_fetch (
    echo No symbol entered. Aborting.
    goto pause_menu
)
echo FORCE Fetching data for %cont_symbol_to_fetch% (OVERWRITING HISTORY)...
python -m src.scripts.market_data.continuous_contract_loader --symbol %cont_symbol_to_fetch% --force
goto pause_menu

:update_active
echo Updating active ES and NQ futures contracts...
python src/scripts/market_data/update_active_es_nq_futures.py
goto pause_menu

:sql_db_maintenance
echo Running Database Maintenance SQL (VACUUM/ANALYZE)...
python src/scripts/db_inspector.py database_maintenance.sql
goto pause_menu

:run_market_update
echo Calling main market data update script (may take a while)...
call update_market_data.bat
goto pause_menu

:pause_menu
echo.
echo Press any key to return to menu...
pause > nul
goto menu

:end
echo Exiting...
exit /b 0 