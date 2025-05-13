@echo off
REM Ensure date and time variables are properly set for log files
set logdate=%date:~-4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set logdate=%logdate: =0%

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Log start of execution
echo Starting market data update at %date% %time% > logs\update_%logdate%.log

REM Set working directory explicitly - critical for scheduled tasks
cd /d "C:\Users\alexp\OneDrive\Gdrive\Trading\GitHub Projects\data-management\financial-data-system"
echo Changed directory to: %CD% >> logs\update_%logdate%.log

REM Process command line arguments
set SCRIPT_ARGS=--verify
set SKIP_VIX=
set SKIP_VX=
set SKIP_ES_NQ=
set DRY_RUN=
set UPDATE_ACTIVE_ES_15MIN=
set UPDATE_ACTIVE_ES_1MIN=

:parse_args
if "%~1"=="" goto run_script
if /i "%~1"=="--skip-vix" set SKIP_VIX=--skip-vix& shift & goto parse_args
if /i "%~1"=="--skip-vx" set SKIP_VX=--skip-futures& shift & goto parse_args
if /i "%~1"=="--skip-es-nq" set SKIP_ES_NQ=--skip-es-nq& shift & goto parse_args
if /i "%~1"=="--dry-run" set DRY_RUN=--dry-run& shift & goto parse_args
if /i "%~1"=="--full-update" set SCRIPT_ARGS=%SCRIPT_ARGS% --full-update& shift & goto parse_args
if /i "%~1"=="/active_es15min" set UPDATE_ACTIVE_ES_15MIN=--update-active-es-15min& shift & goto parse_args
if /i "%~1"=="/active_es1min" set UPDATE_ACTIVE_ES_1MIN=--update-active-es-1min& shift & goto parse_args

REM Handle unknown arguments (optional: log or ignore)
echo Ignoring unknown argument: %1 >> logs\update_%logdate%.log
shift
goto parse_args

:run_script
REM Run the Python script with full paths and command line arguments
echo Running Python script with arguments: %SCRIPT_ARGS% %SKIP_VIX% %SKIP_VX% %SKIP_ES_NQ% %DRY_RUN% %UPDATE_ACTIVE_ES_15MIN% %UPDATE_ACTIVE_ES_1MIN% >> logs\update_%logdate%.log
"C:\Users\alexp\OneDrive\Gdrive\Trading\GitHub Projects\data-management\financial-data-system\venv\Scripts\python.exe" -m src.scripts.market_data.update_all_market_data %SCRIPT_ARGS% %SKIP_VIX% %SKIP_VX% %SKIP_ES_NQ% %DRY_RUN% %UPDATE_ACTIVE_ES_15MIN% %UPDATE_ACTIVE_ES_1MIN% >> logs\update_%logdate%.log 2>&1

REM The specific SPY update call that was here has been removed as the main script now handles it.

REM Log completion status
set EXIT_CODE=%ERRORLEVEL%
echo Script finished at %date% %time% with exit code %EXIT_CODE% >> logs\update_%logdate%.log

REM Create a "last_run.log" file that's easy to identify
copy logs\update_%logdate%.log logs\last_run.log > nul

exit /b %EXIT_CODE% 