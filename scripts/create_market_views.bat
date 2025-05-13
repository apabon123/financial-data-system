@echo off
setlocal enabledelayedexpansion

:: Find Python executable
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_PATH=python
) else (
    where python3 >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_PATH=python3
    ) else (
        echo Python not found in PATH
        exit /b 1
    )
)

:: Run the script
echo Creating market data views...
%PYTHON_PATH% src/scripts/database/create_market_views.py

if %ERRORLEVEL% EQU 0 (
    echo Market views created successfully
) else (
    echo Error creating market views
    exit /b 1
)

exit /b 0 