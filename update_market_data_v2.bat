@echo off
REM Update market data using the new architecture with graceful fallback
REM This batch file uses the wrapper script that automatically selects the appropriate architecture

SET LOGFILE=logs\update_market_data_v2.log

echo Starting market data update with the new architecture... >> %LOGFILE%
echo. >> %LOGFILE%

python -m src.scripts.market_data.update_all_market_data_wrapper --skip-panama %* >> %LOGFILE% 2>>&1

if %ERRORLEVEL% NEQ 0 (
  echo. >> %LOGFILE%
  echo Market data update failed with error code %ERRORLEVEL% >> %LOGFILE%
  exit /b %ERRORLEVEL%
) else (
  echo. >> %LOGFILE%
  echo Market data update completed successfully. >> %LOGFILE%
)

exit /b 0