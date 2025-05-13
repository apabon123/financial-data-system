@echo off
REM DB Inspector Dependency Checker
REM This script checks all required and optional dependencies for the DB Inspector tool

SETLOCAL EnableDelayedExpansion

echo ===============================================
echo DB Inspector Dependency Checker
echo ===============================================
echo.

REM Check if Python is installed
echo Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo Using Python %PYTHON_VERSION%

REM Check if version is at least 3.8
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.8 or higher is required. Found version %PYTHON_VERSION%
    pause
    exit /b 1
)

echo.
echo Checking core dependencies...
echo.

REM Check core dependencies
set CORE_DEPS=pandas duckdb numpy pyyaml rich
set CORE_MISSING=
set CORE_INSTALLED=

for %%d in (%CORE_DEPS%) do (
    if "%%d"=="pyyaml" (
        python -c "import yaml" 2>nul
    ) else (
        python -c "import %%d" 2>nul
    )
    
    if !ERRORLEVEL! NEQ 0 (
        set CORE_MISSING=!CORE_MISSING! %%d
    ) else (
        set CORE_INSTALLED=!CORE_INSTALLED! %%d
    )
)

if "!CORE_MISSING!"=="" (
    echo [OK] All core dependencies are installed:
    
    REM Display versions for installed packages
    for %%d in (%CORE_DEPS%) do (
        if "%%d"=="pyyaml" (
            for /f "tokens=*" %%v in ('python -c "import yaml; print(getattr(yaml, '__version__', 'unknown'))" 2^>nul') do (
                echo    - yaml: %%v
            )
        ) else (
            for /f "tokens=*" %%v in ('python -c "import %%d; print(getattr(%%d, '__version__', 'unknown'))" 2^>nul') do (
                echo    - %%d: %%v
            )
        )
    )
) else (
    echo [MISSING] Core dependencies:!CORE_MISSING!
    echo.
    echo These dependencies are required for DB Inspector to function.
    echo Install them with:
    echo.
    echo    pip install!CORE_MISSING!
    echo.
    echo Or install all dependencies with:
    echo.
    echo    pip install -r requirements.txt
)

echo.
echo Checking optional dependencies...
echo.

REM Check visualization dependencies
set VIZ_DEPS=matplotlib seaborn
set VIZ_MISSING=

for %%d in (%VIZ_DEPS%) do (
    python -c "import %%d" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        set VIZ_MISSING=!VIZ_MISSING! %%d
    )
)

if "!VIZ_MISSING!"=="" (
    echo [OK] All visualization dependencies are installed.
) else (
    echo [LIMITED] Missing visualization dependencies:!VIZ_MISSING!
    echo    These are used for data plots and charts.
)

REM Check schema visualization dependencies
set SCHEMA_DEPS=networkx
set SCHEMA_MISSING=

for %%d in (%SCHEMA_DEPS%) do (
    python -c "import %%d" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        set SCHEMA_MISSING=!SCHEMA_MISSING! %%d
    )
)

if "!SCHEMA_MISSING!"=="" (
    echo [OK] All schema visualization dependencies are installed.
) else (
    echo [LIMITED] Missing schema visualization dependencies:!SCHEMA_MISSING!
    echo    These are used for database schema visualization.
)

REM Check SQL interface dependencies
set SQL_DEPS=prompt_toolkit pygments
set SQL_MISSING=

for %%d in (%SQL_DEPS%) do (
    python -c "import %%d" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        set SQL_MISSING=!SQL_MISSING! %%d
    )
)

if "!SQL_MISSING!"=="" (
    echo [OK] All SQL interface dependencies are installed.
) else (
    echo [LIMITED] Missing SQL interface dependencies:!SQL_MISSING!
    echo    These are used for the interactive SQL interface.
)

REM Check other optional dependencies
set OTHER_DEPS=tabulate
set OTHER_MISSING=

for %%d in (%OTHER_DEPS%) do (
    python -c "import %%d" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        set OTHER_MISSING=!OTHER_MISSING! %%d
    )
)

if "!OTHER_MISSING!"=="" (
    echo [OK] All other optional dependencies are installed.
) else (
    echo [LIMITED] Missing other optional dependencies:!OTHER_MISSING!
    echo    These are used for enhanced formatting and display.
)

REM Combine all missing optional dependencies
set ALL_OPT_MISSING=!VIZ_MISSING!!SCHEMA_MISSING!!SQL_MISSING!!OTHER_MISSING!

if "!ALL_OPT_MISSING!" NEQ "" (
    echo.
    echo To install all missing optional dependencies, run:
    echo.
    echo    pip install!ALL_OPT_MISSING!
    echo.
)

REM Summary
echo.
echo Summary:
echo ========

if "!CORE_MISSING!"=="" (
    if "!ALL_OPT_MISSING!"=="" (
        echo [OK] All dependencies are installed. DB Inspector should work with full functionality.
        echo    Run DB_inspect_enhanced.bat to start the application.
    ) else (
        echo [LIMITED] Core dependencies are installed, but some optional features will be limited.
        echo    Run DB_inspect_enhanced.bat to start with limited functionality.
        echo    Run DB_inspect_enhanced.bat --fallback for a more basic interface if you encounter issues.
    )
) else (
    echo [ERROR] Core dependencies are missing. DB Inspector may not function correctly.
    echo    Install required dependencies before running the application.
    echo    Alternatively, run DB_inspect_enhanced.bat --fallback for a basic interface using only standard library.
)

echo.
echo For more information, see the documentation or run:
echo    DB_inspect_enhanced.bat --help

pause
ENDLOCAL