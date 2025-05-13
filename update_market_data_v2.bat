@echo off
REM Update market data using the new architecture with graceful fallback
REM This batch file uses the wrapper script that automatically selects the appropriate architecture

echo Starting market data update with the new architecture...
echo.

python -m src.scripts.market_data.update_all_market_data_wrapper %*

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Market data update failed with error code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
) else (
  echo.
  echo Market data update completed successfully.
)

exit /b 0