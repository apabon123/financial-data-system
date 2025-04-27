@echo off
rem This script simply calls the main update script located in the tasks folder.
call tasks\update_market_data.bat %*
exit /b %ERRORLEVEL% 