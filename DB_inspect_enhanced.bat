@echo off
REM Enhanced DB Inspector for Financial Data System
REM This script launches the new DB Inspector with improved functionality

SETLOCAL EnableDelayedExpansion

REM Determine if we're in verbose mode
set VERBOSE=0
for %%a in (%*) do (
    if "%%a"=="--verbose" set VERBOSE=1
)

echo ====================================================
echo DB Inspector - Financial Market Data Inspection Tool
echo ====================================================
echo.

REM Set Python path to use virtual environment
set PYTHON_PATH=.\venv\Scripts\python.exe

REM Check if Python is installed
echo [1/4] Checking Python installation...
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Virtual environment not found.
    echo.
    echo Please activate the virtual environment first:
    echo .\venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=*" %%i in ('%PYTHON_PATH% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo Found Python version %PYTHON_VERSION%

REM Check if version is at least 3.8
%PYTHON_PATH% -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.8 or higher is required. Found version %PYTHON_VERSION%
    pause
    exit /b 1
)

REM Check for requirements.txt
if not exist "requirements.txt" (
    echo [WARNING] requirements.txt not found. Will use hardcoded dependency list.
)

REM Check required dependencies
echo [2/4] Checking required dependencies...
echo.
echo Checking for required packages:
echo - pandas
echo - duckdb
echo - numpy
echo - pyyaml
echo - rich
echo.

REM Check required dependencies directly using Python
echo Running dependency check...
for /f "tokens=*" %%i in ('%PYTHON_PATH% -c "import sys; missing = []; [missing.append(pkg) for pkg in ['pandas', 'duckdb', 'numpy', 'yaml', 'rich'] if __import__(pkg, fromlist=['']) is None]; print(' '.join(missing))"') do set MISSING_DEPS=%%i

if "!MISSING_DEPS!"=="" (
    echo All required dependencies are installed.
    goto :check_optional
) else (
    echo Missing dependencies found: !MISSING_DEPS!
)

REM Set installation strings
set INSTALL_CMD=
set INSTALL_DEPS=
if "!MISSING_DEPS!" NEQ "" (
    for %%d in (!MISSING_DEPS!) do (
        if "%%d"=="pandas" set INSTALL_DEPS=!INSTALL_DEPS! pandas>=2.0.0
        if "%%d"=="duckdb" set INSTALL_DEPS=!INSTALL_DEPS! duckdb>=1.2.1
        if "%%d"=="numpy" set INSTALL_DEPS=!INSTALL_DEPS! numpy>=1.24.0
        if "%%d"=="pyyaml" set INSTALL_DEPS=!INSTALL_DEPS! pyyaml>=6.0.1
        if "%%d"=="rich" set INSTALL_DEPS=!INSTALL_DEPS! rich>=13.7.0
    )
    if exist "requirements.txt" (
        set INSTALL_CMD=%PYTHON_PATH% -m pip install -r requirements.txt
    ) else (
        set INSTALL_CMD=%PYTHON_PATH% -m pip install!INSTALL_DEPS!
    )
)

REM Check if there are missing dependencies and prompt for installation
if "!MISSING_DEPS!" NEQ "" (
    if "!INSTALL_DEPS!" == "" (
        set INSTALL_DEPS=pandas>=2.0.0 duckdb>=1.2.1 numpy>=1.24.0 pyyaml>=6.0.1 rich>=13.7.0
    )

    echo [MISSING] Required dependencies: !MISSING_DEPS!
    echo.
    echo These core dependencies are required for DB Inspector to function.
    echo.
    echo Would you like to install the missing dependencies now? (Y/N)
    set /p INSTALL=

    if /i "!INSTALL!"=="Y" (
        echo.
        echo Installing required dependencies...
        echo Running: !INSTALL_CMD!
        echo.

        REM Redirect output to a log file and console
        !INSTALL_CMD! > logs\pip_install.log 2>&1

        if !ERRORLEVEL! NEQ 0 (
            echo.
            echo [ERROR] Failed to install dependencies.
            echo.
            echo Please try manually with:
            echo pip install!INSTALL_DEPS!
            echo.
            echo Or install all dependencies with:
            echo pip install -r requirements.txt
            echo.
            echo Check logs\pip_install.log for details.
            if !VERBOSE!==1 (
                echo.
                echo Installation log:
                type logs\pip_install.log
            )
            pause
            exit /b 1
        ) else (
            echo [SUCCESS] Required dependencies installed successfully.
            goto :check_optional
        )
    ) else (
        echo.
        echo Installation aborted. DB Inspector requires these dependencies to function.
        echo.
        echo You can install them manually with:
        echo pip install!INSTALL_DEPS!
        echo.
        echo Or install all dependencies with:
        echo pip install -r requirements.txt
        pause
        exit /b 1
    )
)

:check_optional
REM Check optional dependencies
echo.
echo [3/4] Checking optional dependencies...
echo.
echo Checking for optional packages:
echo - networkx
echo - prompt_toolkit
echo - pygments
echo - matplotlib
echo - seaborn
echo - tabulate
echo.

REM Check optional dependencies directly using Python
echo Running optional dependency check...
for /f "tokens=*" %%i in ('%PYTHON_PATH% -c "import sys; missing = []; [missing.append(pkg) for pkg in ['networkx', 'prompt_toolkit', 'pygments', 'matplotlib', 'seaborn', 'tabulate'] if __import__(pkg, fromlist=['']) is None]; print(' '.join(missing))"') do set MISSING_OPT=%%i

if "!MISSING_OPT!"=="" (
    echo All optional dependencies are installed.
    goto :launch_inspector
) else (
    echo Missing optional dependencies found: !MISSING_OPT!
)

REM Set installation strings for optional deps
set INSTALL_OPT_CMD=
set INSTALL_OPT_DEPS=
if "!MISSING_OPT!" NEQ "" (
    for %%d in (!MISSING_OPT!) do (
        if "%%d"=="networkx" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! networkx>=3.0
        if "%%d"=="prompt_toolkit" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! prompt_toolkit>=3.0.33
        if "%%d"=="pygments" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! pygments>=2.15.0
        if "%%d"=="matplotlib" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! matplotlib>=3.7.0
        if "%%d"=="seaborn" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! seaborn>=0.12.0
        if "%%d"=="tabulate" set INSTALL_OPT_DEPS=!INSTALL_OPT_DEPS! tabulate>=0.9.0
    )
    set INSTALL_OPT_CMD=%PYTHON_PATH% -m pip install!INSTALL_OPT_DEPS!
)

REM Check if there are missing optional dependencies and prompt for installation
if "!MISSING_OPT!" NEQ "" (
    echo [LIMITED] Missing optional dependencies: !MISSING_OPT!
    echo.
    echo These dependencies enable enhanced features but are not required:
    echo - networkx: Schema visualization
    echo - prompt_toolkit, pygments: Interactive SQL
    echo - matplotlib, seaborn: Data visualization
    echo - tabulate: Table formatting
    echo.
    echo Would you like to install these optional dependencies for enhanced functionality? (Y/N)
    set /p INSTALL_OPT=

    if /i "!INSTALL_OPT!"=="Y" (
        echo.
        echo Installing optional dependencies...
        echo Running: !INSTALL_OPT_CMD!
        echo.

        REM Redirect output to a log file and console
        mkdir -p logs 2>nul
        !INSTALL_OPT_CMD! > logs\pip_install_opt.log 2>&1

        if !ERRORLEVEL! NEQ 0 (
            echo [WARNING] Some optional dependencies could not be installed.
            echo.
            echo You can try manually with:
            echo pip install!INSTALL_OPT_DEPS!
            echo.
            echo Check logs\pip_install_opt.log for details.
            echo.
            echo Continuing with limited functionality...
            if !VERBOSE!==1 (
                echo.
                echo Installation log:
                type logs\pip_install_opt.log
            )
        ) else (
            echo [SUCCESS] Optional dependencies installed successfully.
        )
    ) else (
        echo Continuing with limited functionality...
    )
)

:launch_inspector
echo.
echo [4/4] Launching DB Inspector...
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Launch the DB Inspector using the virtual environment's Python
%PYTHON_PATH% src/inspector/__main__.py %*

REM After the Python application exits, we should exit the batch script too
echo.
echo DB Inspector has exited.
goto :end_script

:end_script
ENDLOCAL
exit /b 0

REM The following data management functionality is incomplete and causes flow issues
REM Commenting out for now until it's properly implemented
REM
REM :data_management_menu
REM echo.
REM echo Data Management Menu:
REM echo ---------------------
REM echo 1. Manage Individual Contracts (Not Implemented Yet)
REM echo 2. Manage Continuous Contracts
REM echo 3. Populate Roll Calendar
REM echo 4. Calculate Volume Roll Dates
REM echo G. Generate Continuous Futures Series (NEW)
REM echo B. Back to Main Menu
REM echo.
REM set /p DM_CHOICE=Enter your choice: 
REM if /i "!DM_CHOICE!"=="1" goto :manage_individual_contracts
REM if /i "!DM_CHOICE!"=="2" goto :manage_continuous_contracts
REM if /i "!DM_CHOICE!"=="3" goto :populate_roll_calendar
REM if /i "!DM_CHOICE!"=="4" goto :calculate_volume_roll_dates
REM if /i "!DM_CHOICE!"=="G" goto :generate_continuous_futures
REM if /i "!DM_CHOICE!"=="B" goto :main_menu
REM echo Invalid choice. Please try again.
REM goto :data_management_menu

REM :generate_continuous_futures
REM cls
REM echo Generate Continuous Futures Series
REM echo ------------------------------------
REM echo This will run the src/scripts/scripts/generate_back_adjusted_futures.py script.
REM echo Please provide the following parameters:
REM echo.

REM :get_root_symbol
REM set /p ROOT_SYMBOL=Enter Root Symbol (e.g., ES, NQ): 
REM if "!ROOT_SYMBOL!"=="" (
REM     echo Root Symbol cannot be empty.
REM     goto :get_root_symbol
REM )

REM :get_roll_type
REM set /p ROLL_TYPE=Enter Roll Type (e.g., 01X, volume): 
REM if "!ROLL_TYPE!"=="" (
REM     echo Roll Type cannot be empty.
REM     goto :get_roll_type
REM )

REM :get_contract_position
REM set /p CONTRACT_POSITION=Enter Contract Position (e.g., 1, 2, default: 1): 
REM if "!CONTRACT_POSITION!"=="" set CONTRACT_POSITION=1
REM REM Basic numeric check
REM echo !CONTRACT_POSITION! | findstr /r /c:"^[1-9][0-9]*$" >nul
REM if errorlevel 1 (
REM     echo Invalid Contract Position. Must be a positive integer.
REM     set CONTRACT_POSITION=
REM     goto :get_contract_position
REM )

REM :get_interval_value
REM set /p INTERVAL_VALUE=Enter Interval Value (e.g., 1, 15): 
REM if "!INTERVAL_VALUE!"=="" (
REM     echo Interval Value cannot be empty.
REM     goto :get_interval_value
REM )
REM echo !INTERVAL_VALUE! | findstr /r /c:"^[1-9][0-9]*$" >nul
REM if errorlevel 1 (
REM     echo Invalid Interval Value. Must be a positive integer.
REM     set INTERVAL_VALUE=
REM     goto :get_interval_value
REM )

REM :get_interval_unit
REM set /p INTERVAL_UNIT=Enter Interval Unit (minute, daily, hour): 
REM if "!INTERVAL_UNIT!"=="" (
REM     echo Interval Unit cannot be empty.
REM     goto :get_interval_unit
REM )
REM if /i not "!INTERVAL_UNIT!"=="minute" if /i not "!INTERVAL_UNIT!"=="daily" if /i not "!INTERVAL_UNIT!"=="hour" (
REM     echo Invalid Interval Unit. Must be 'minute', 'daily', or 'hour'.
REM     set INTERVAL_UNIT=
REM     goto :get_interval_unit
REM )

REM :get_adjustment_type
REM set /p ADJUSTMENT_TYPE=Enter Adjustment Type (C for constant, N for none, default: C): 
REM if /i "!ADJUSTMENT_TYPE!"=="C" set ADJUSTMENT_TYPE_ARG=C
REM if /i "!ADJUSTMENT_TYPE!"=="N" set ADJUSTMENT_TYPE_ARG=N
REM if "!ADJUSTMENT_TYPE!"=="" set ADJUSTMENT_TYPE_ARG=C
REM if not defined ADJUSTMENT_TYPE_ARG (
REM     echo Invalid Adjustment Type. Enter C or N.
REM     set ADJUSTMENT_TYPE=
REM     goto :get_adjustment_type
REM )


REM set OUTPUT_SYMBOL_SUFFIX=_d
REM set SOURCE_IDENTIFIER_BASE=inhouse_backadj

REM set CMD_FLAGS=
REM :get_force_delete
REM set /p FORCE_DELETE_CHOICE=Force delete existing data for this symbol? (Y/N, default: N): 
REM if /i "!FORCE_DELETE_CHOICE!"=="Y" set CMD_FLAGS=!CMD_FLAGS! --force-delete
REM if /i "!FORCE_DELETE_CHOICE!"=="" set FORCE_DELETE_CHOICE=N

REM :get_recreate_table
REM set /p RECREATE_TABLE_CHOICE=Recreate table on PK mismatch (DEV ONLY)? (Y/N, default: N): 
REM if /i "!RECREATE_TABLE_CHOICE!"=="Y" set CMD_FLAGS=!CMD_FLAGS! --recreate-table-on-pk-mismatch
REM if /i "!RECREATE_TABLE_CHOICE!"=="" set RECREATE_TABLE_CHOICE=N

REM REM Construct the command
REM set SCRIPT_PATH=src\\scripts\\scripts\\generate_back_adjusted_futures.py
REM set COMMAND_TO_RUN=!PYTHON_PATH! %SCRIPT_PATH% ^
REM     --root-symbol !ROOT_SYMBOL! ^
REM     --roll-type !ROLL_TYPE! ^
REM     --contract-position !CONTRACT_POSITION! ^
REM     --interval-value !INTERVAL_VALUE! ^
REM     --interval-unit !INTERVAL_UNIT! ^
REM     --adjustment-type !ADJUSTMENT_TYPE_ARG! ^
REM     --output-symbol-suffix %OUTPUT_SYMBOL_SUFFIX% ^
REM     --source-identifier %SOURCE_IDENTIFIER_BASE% ^
REM     !CMD_FLAGS!

REM echo.
REM echo The following command will be executed:
REM echo -----------------------------------------
REM echo !COMMAND_TO_RUN!
REM echo -----------------------------------------
REM echo.
REM :confirm_run_generate
REM set /p CONFIRM_RUN=Run this command? (Y/N): 
REM if /i "!CONFIRM_RUN!"=="Y" (
REM     echo Executing...
REM     !COMMAND_TO_RUN!
REM     echo.
REM     echo Script execution finished.
REM ) else if /i "!CONFIRM_RUN!"=="N" (
REM     echo Command cancelled.
REM ) else (
REM     echo Invalid choice.
REM     goto :confirm_run_generate
REM )
REM pause
REM goto :data_management_menu

REM ENDLOCAL