@echo off
setlocal enabledelayedexpansion

REM Set explicit paths with double quotes to handle spaces in paths
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=!TIMESTAMP: =0!"
set "LOG_FILE=%LOG_DIR%\db_backup_%TIMESTAMP%.log"

REM Create log directory if it doesn't exist
mkdir "%LOG_DIR%" 2>nul

REM Start logging
echo ===== Database Backup Started at %date% %time% ===== > "%LOG_FILE%"

REM Log system info
echo Computer Name: %COMPUTERNAME% >> "%LOG_FILE%"
echo Working Directory: %CD% >> "%LOG_FILE%"
echo Batch File: %~f0 >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Run in explicit paths
cd /d "%PROJECT_ROOT%" >> "%LOG_FILE%" 2>&1

REM Check if Python is in PATH
where python >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH >> "%LOG_FILE%"
    echo Trying with explicit Python path >> "%LOG_FILE%"
    set "PYTHON_CMD=C:\Users\alexp\AppData\Local\Programs\Python\Python310\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Using Python: %PYTHON_CMD% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Set database and backup paths
set "DATABASE_PATH=%PROJECT_ROOT%\data\financial_data.duckdb"
set "BACKUP_DIR=%PROJECT_ROOT%\backups"

REM Make directories if they don't exist
mkdir "%BACKUP_DIR%" 2>nul

REM Check if database exists
if not exist "%DATABASE_PATH%" (
    echo ERROR: Database file not found: %DATABASE_PATH% >> "%LOG_FILE%"
    echo ===== Backup FAILED at %date% %time% ===== >> "%LOG_FILE%"
    exit /b 1
)

REM Run backup script with full paths
echo Running backup script... >> "%LOG_FILE%"
echo Command: %PYTHON_CMD% "%SCRIPT_DIR%simple_backup.py" --database "%DATABASE_PATH%" --output "%BACKUP_DIR%" --log-dir "%LOG_DIR%" >> "%LOG_FILE%"

%PYTHON_CMD% "%SCRIPT_DIR%simple_backup.py" --database "%DATABASE_PATH%" --output "%BACKUP_DIR%" --log-dir "%LOG_DIR%" >> "%LOG_FILE%" 2>&1

set BACKUP_RESULT=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo Backup script exited with code: %BACKUP_RESULT% >> "%LOG_FILE%"

if %BACKUP_RESULT% NEQ 0 (
    echo ERROR: Backup failed with exit code %BACKUP_RESULT% >> "%LOG_FILE%"
    echo ===== Backup FAILED at %date% %time% ===== >> "%LOG_FILE%"
    exit /b %BACKUP_RESULT%
) else (
    echo Backup completed successfully >> "%LOG_FILE%"
    echo ===== Backup COMPLETED at %date% %time% ===== >> "%LOG_FILE%"
    exit /b 0
) 