@echo off
REM Database inspector script for Financial Data System
REM This is the new version using the updated architecture

REM Set up variables
set DB_PATH=data\financial_data.duckdb
set PYTHON_PATH=venv\Scripts\python.exe
set MODULE_PATH=src.scripts.db_inspector

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Get current date and time for log file
set timestamp=%date:~-4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set timestamp=%timestamp: =0%

REM Log the start
echo Starting DB inspector at %date% %time% > logs\db_inspect_%timestamp%.log

REM Set working directory explicitly
cd /d "%~dp0\..\..\.."
echo Working directory: %CD% >> logs\db_inspect_%timestamp%.log

REM Run the DB inspector module
"%PYTHON_PATH%" -m %MODULE_PATH% --db-path=%DB_PATH% %* >> logs\db_inspect_%timestamp%.log 2>&1

REM Check exit code
set EXIT_CODE=%ERRORLEVEL%
echo DB inspector completed with exit code %EXIT_CODE% >> logs\db_inspect_%timestamp%.log

REM Copy to last_inspect.log for easy access
copy logs\db_inspect_%timestamp%.log logs\last_inspect.log > nul

exit /b %EXIT_CODE%