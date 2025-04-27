@echo off
REM Database Backup Wrapper Script for Windows

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Run the backup script with the provided arguments
python "%SCRIPT_DIR%..\src\scripts\database\backup_database.py" %* 