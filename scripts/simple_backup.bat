@echo off
REM Simple Database Backup Script for Windows

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Set database and logs directories with absolute paths
set DATABASE_PATH=%PROJECT_ROOT%\data\financial_data.duckdb
set BACKUP_DIR=%PROJECT_ROOT%\backups
set LOGS_DIR=%PROJECT_ROOT%\logs

REM Create directories if they don't exist
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM Run the backup script with full paths
python "%SCRIPT_DIR%simple_backup.py" --database "%DATABASE_PATH%" --output "%BACKUP_DIR%" --log-dir "%LOGS_DIR%" %*

REM Echo the exit code to help with troubleshooting
echo Backup script exited with code %ERRORLEVEL% 