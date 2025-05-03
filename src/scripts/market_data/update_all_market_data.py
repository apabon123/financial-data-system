#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Market Data Update Script

This script orchestrates the full update process for financial market data:
1. Updates the VIX Index
2. Updates active VX futures contracts
3. Updates continuous VX contracts
4. Fills historical gaps in VXc1 and VXc2 for 2004-2005 if full update requested

Run this script daily to keep the database up-to-date.
"""

import os
import sys
from pathlib import Path

# Add project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent) # Go up four levels from the script's location
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import logging
import subprocess
from datetime import datetime, timedelta, date
import duckdb
# Import necessary components for continuous contract loading
from src.scripts.market_data.continuous_contract_loader import load_continuous_data as load_ts_continuous_data, load_market_symbols_config
from src.scripts.market_data.fetch_market_data import MarketDataFetcher # Assuming fetch_market_data is in the same directory
# Import main functions from other scripts
from src.scripts.market_data.vix.update_vix_index import main as update_vix_main
from src.scripts.market_data.vix.update_vx_futures import main as update_vx_futures_main
from src.scripts.market_data.generate_continuous_futures import main as generate_vx_continuous_main # Import VX continuous generator
# Note: update_es_nq_futures still uses subprocess due to its internal complexity
import re
import yaml
import pandas as pd
from src.utils.database import get_db_engine # Corrected import path
# MarketSymbolManager likely doesn't exist either based on previous errors, removing for now
# from src.utils.market_symbol_manager import MarketSymbolManager # Add import for verification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "data/financial_data.duckdb"
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"

def run_script(script_name, args=None):
    """Run a Python module with arguments."""
    cmd = [sys.executable, "-m", script_name]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully completed {script_name}")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def check_db_exists(db_path):
    """Check if the database file exists, create parent directories if needed."""
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        logger.info(f"Database file not found. Will be created at: {db_path}")
        return False
    return True

def update_vix_index(conn):
    """Update the VIX Index data."""
    logger.info("=== Updating VIX Index ($VIX.X) ===")
    args_dict = {'db_conn': conn}
    try:
        update_vix_main(args_dict=args_dict)
        return True
    except Exception as e:
        logger.error(f"Error in update_vix_index: {e}", exc_info=True)
        return False

def update_vx_futures(conn, config_path, full_regen=False):
    """Update active VX futures contracts."""
    logger.info("=== Updating VX Futures Contracts ===")
    args_dict = {
        'config_path': config_path,
        'full_regen': full_regen,
        'db_conn': conn
    }
    try:
        update_vx_futures_main(args_dict=args_dict)
        return True
    except Exception as e:
        logger.error(f"Error in update_vx_futures: {e}", exc_info=True)
        return False

def update_es_nq_futures(db_path, dry_run=False):
    """Update active ES and NQ futures contracts."""
    logger.info("=== Updating ES and NQ Futures Contracts ===")
    
    # Since update_active_es_nq_futures.py doesn't handle command line arguments,
    # we need to create a temporary modified version that uses the correct DB_PATH
    
    # Create a modified version of the script with the updated DB_PATH
    temp_script_path = "temp_update_es_nq.py"
    
    try:
        # Read the original script
        with open(os.path.join("src", "scripts", "market_data", "update_active_es_nq_futures.py"), "r") as f:
            script_content = f.read()
        
        # Replace the DB_PATH with our parameter
        script_content = script_content.replace(
            'DB_PATH = "./data/financial_data.duckdb"',
            f'DB_PATH = "{db_path}"'
        )

        # Remove the FileHandler to prevent stray log file creation
        script_content = re.sub(
            r"(\s*logging\.FileHandler\([^)]*\),?)\n",
            "",
            script_content
        )
        
        # Add dry-run functionality if needed
        if dry_run:
            # Find the main function
            main_func_start = script_content.find("def main():")
            if main_func_start != -1:
                # Add DRY_RUN flag after the function definition
                insert_pos = script_content.find("\n", main_func_start) + 1
                script_content = (
                    script_content[:insert_pos] + 
                    "    # Dry run mode - don't save data\n" +
                    "    DRY_RUN = True\n" +
                    script_content[insert_pos:]
                )
                
                # Modify the fetch_single_contract function call to not save in dry run mode
                script_content = script_content.replace(
                    "success, message = fetch_single_contract(symbol, start_date, end_date, DB_PATH)",
                    "if 'DRY_RUN' in locals() and DRY_RUN:\n" +
                    "                logger.info(f\"DRY RUN: Would fetch {symbol} from {start_date} to {end_date}\")\n" +
                    "                success, message = True, f\"DRY RUN - Would fetch {symbol}\"\n" +
                    "            else:\n" +
                    "                success, message = fetch_single_contract(symbol, start_date, end_date, DB_PATH)"
                )
        
        # Write the modified script to a temporary file
        with open(temp_script_path, "w") as f:
            f.write(script_content)
        
        # Run the modified script
        cmd = [sys.executable, temp_script_path]
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Successfully completed ES and NQ futures update")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ES and NQ futures update: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error preparing ES and NQ futures update script: {e}")
        return False
    
    finally:
        # Clean up the temporary script
        if os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary script {temp_script_path}: {e}")
        
        # Clean up the stray log file if it exists
        stray_log_file = "@update_active_es_nq_futures.log"
        if os.path.exists(stray_log_file):
            try:
                os.remove(stray_log_file)
                logger.debug(f"Removed stray log file: {stray_log_file}")
            except Exception as e:
                logger.warning(f"Failed to remove stray log file {stray_log_file}: {e}")

def update_tradestation_continuous(fetcher: MarketDataFetcher, config_path: str):
    """Load DAILY continuous contract data for ES and NQ using TradeStation loader."""
    logger.info("=== Updating ES, NQ Continuous Contracts (@ES, @NQ) - Daily ===")

    # Load config once
    # try:
    #     with open(config_path, 'r') as f:
    #         config = yaml.safe_load(f)

    # Load DAILY data for each symbol - REMOVED VX
    try:
        # Check if fetcher has config loaded
        if not fetcher or not hasattr(fetcher, 'config') or not fetcher.config:
             logger.error("Fetcher does not have configuration loaded. Cannot update continuous data.")
             return False
        
        config = fetcher.config # Use config from fetcher

        for symbol in ['ES', 'NQ']:
            logger.info(f"--- Processing Daily Continuous Data for {symbol} ---")
            load_ts_continuous_data(
                fetcher=fetcher,
                symbol_root=symbol,
                config=config,
                interval_value=1,
                interval_unit='daily'
            )
            logger.info(f"Successfully updated DAILY continuous data for {symbol}")
    except Exception as e:
        logger.error(f"Error updating DAILY continuous data: {e}", exc_info=True)
        return False

    return True

def update_active_es_15min(fetcher: MarketDataFetcher, config_path: str):
    """Update 15-minute data for the 2 closest ACTIVE ES contracts."""
    logger.info("=== Updating ACTIVE ES Contracts - 15 Minute ===")
    success = True
    
    # --- Logic adapted from update_active_es_nq_futures.py --- 
    def get_active_es_contracts(num_active=2):
        base_symbol = 'ES'
        today = datetime.now().date()
        current_month = today.month
        current_year = today.year
        quarterly_months = [3, 6, 9, 12] # H, M, U, Z
        month_codes = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
        contracts = []
        
        next_quarterly_idx = 0
        for i, month in enumerate(quarterly_months):
            if month >= current_month:
                next_quarterly_idx = i
                break
        else: # If current_month > 12 (shouldn't happen) or > last quarterly month
             next_quarterly_idx = 0 # Wrap around to next year's first quarter

        contract_count = 0
        idx = next_quarterly_idx
        year = current_year
        first_contract_checked = False

        while contract_count < num_active:
            if idx >= len(quarterly_months):
                idx = 0
                year += 1
            
            month = quarterly_months[idx]
            month_code = month_codes[month]
            year_to_check = year

            # Adjust year if we wrapped around
            if month < current_month and idx < next_quarterly_idx:
                year_to_check = current_year + 1
            elif month < current_month and next_quarterly_idx == 0: # Handling Dec -> March wrap
                year_to_check = current_year + 1
            else:
                year_to_check = current_year
            
            # Check 3rd Friday rule only for the *first* potential contract
            if not first_contract_checked:
                first_contract_checked = True # Mark as checked
                try:
                    # Calculate the 3rd Friday of the month
                    first_day = date(year_to_check, month, 1)
                    days_until_friday = (4 - first_day.weekday() + 7) % 7 # 0=Mon, 4=Fri
                    first_friday = first_day + timedelta(days=days_until_friday)
                    third_friday = first_friday + timedelta(weeks=2)
                    
                    # If today is past the 3rd Friday, advance the index
                    if today > third_friday:
                        logger.debug(f"Today ({today}) is past 3rd Friday ({third_friday}) of {month}/{year_to_check}. Skipping to next contract.")
                        idx += 1
                        continue # Re-evaluate the while loop with the new index
                except ValueError as e:
                    logger.error(f"Date calculation error for {month}/{year_to_check}: {e}")
                    # Decide how to handle - skip or error out? For now, log and continue loop might be risky.
                    return [] # Return empty list on date error

            # Construct symbol if check passed or wasn't needed
            year_str = str(year_to_check)[-2:]
            contract = f"{base_symbol}{month_code}{year_str}"
            if contract not in contracts: # Avoid duplicates if logic loops unexpectedly
                 contracts.append(contract)
                 contract_count += 1
            idx += 1 # Move to next potential contract month index
        
        return contracts
    # --- End of adapted logic ---

    try:
        active_es_symbols = get_active_es_contracts(num_active=2)
        logger.info(f"Identified active ES contracts for 15-min update: {active_es_symbols}")

        if not active_es_symbols:
             logger.warning("Could not determine active ES symbols. Skipping 15-min update.")
             return True # Not necessarily an error if no contracts found yet

        lookback_days = 7 # How far back to fetch 15-min data
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in active_es_symbols:
            logger.info(f"Fetching 15-minute data for active contract: {symbol} from {start_date} to {end_date}")
            df = fetcher.fetch_data_since(
                symbol=symbol,
                interval=15,
                unit='minute',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                 # Add required interval columns before saving
                 df['interval_value'] = 15
                 df['interval_unit'] = 'minute'
                 # Add other columns if fetch_data_since doesn't provide them (e.g., source)
                 if 'source' not in df.columns:
                      df['source'] = 'TradeStation' # Assuming source
                 if 'adjusted' not in df.columns:
                      df['adjusted'] = False # Assuming non-adjusted for active contracts
                 if 'quality' not in df.columns:
                      df['quality'] = 100 # Assuming direct source quality
                 
                 logger.info(f"Saving {len(df)} rows of 15-minute data for {symbol} to 'market_data' table...")
                 fetcher.save_to_db(df) # Use the fetcher's save method
                 logger.info(f"Successfully saved 15-minute data for {symbol}.")
            else:
                 logger.warning(f"No 15-minute data returned for active contract: {symbol}")
                 # Not necessarily an error, could be market closed etc.

    except Exception as e:
        logger.error(f"Error updating 15-minute active ES data: {e}", exc_info=True)
        success = False
        
    return success

def update_active_es_1min(fetcher: MarketDataFetcher, config_path: str):
    """Update 1-minute data for the 2 closest ACTIVE ES contracts."""
    logger.info("=== Updating ACTIVE ES Contracts - 1 Minute ===")
    success = True
    
    # --- Reusing the same active contract logic --- 
    def get_active_es_contracts(num_active=2):
        base_symbol = 'ES'
        today = datetime.now().date()
        current_month = today.month
        current_year = today.year
        quarterly_months = [3, 6, 9, 12] # H, M, U, Z
        month_codes = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
        contracts = []
        next_quarterly_idx = 0
        for i, month in enumerate(quarterly_months):
            if month >= current_month:
                next_quarterly_idx = i
                break
        else: 
             next_quarterly_idx = 0 
        contract_count = 0
        idx = next_quarterly_idx
        year = current_year
        first_contract_checked = False
        while contract_count < num_active:
            if idx >= len(quarterly_months):
                idx = 0
                year += 1
            month = quarterly_months[idx]
            month_code = month_codes[month]
            year_to_check = year
            if month < current_month and idx < next_quarterly_idx:
                year_to_check = current_year + 1
            elif month < current_month and next_quarterly_idx == 0: 
                year_to_check = current_year + 1
            else:
                year_to_check = current_year
            if not first_contract_checked:
                first_contract_checked = True 
                try:
                    first_day = date(year_to_check, month, 1)
                    days_until_friday = (4 - first_day.weekday() + 7) % 7
                    first_friday = first_day + timedelta(days=days_until_friday)
                    third_friday = first_friday + timedelta(weeks=2)
                    if today > third_friday:
                        logger.debug(f"Today ({today}) is past 3rd Friday ({third_friday}) of {month}/{year_to_check}. Skipping to next contract.")
                        idx += 1
                        continue 
                except ValueError as e:
                    logger.error(f"Date calculation error for {month}/{year_to_check}: {e}")
                    return [] 
            year_str = str(year_to_check)[-2:]
            contract = f"{base_symbol}{month_code}{year_str}"
            if contract not in contracts: 
                 contracts.append(contract)
                 contract_count += 1
            idx += 1 
        return contracts
    # --- End of adapted logic ---

    try:
        active_es_symbols = get_active_es_contracts(num_active=2)
        logger.info(f"Identified active ES contracts for 1-min update: {active_es_symbols}")

        if not active_es_symbols:
             logger.warning("Could not determine active ES symbols. Skipping 1-min update.")
             return True 

        # Use a shorter lookback for 1-min data due to potential volume/API limits
        lookback_days = 3 
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in active_es_symbols:
            logger.info(f"Fetching 1-minute data for active contract: {symbol} from {start_date} to {end_date}")
            df = fetcher.fetch_data_since(
                symbol=symbol,
                interval=1,       # <<< Fetch 1-minute interval
                unit='minute',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                 # Add required interval columns before saving
                 df['interval_value'] = 1 # <<< Set interval_value to 1
                 df['interval_unit'] = 'minute'
                 if 'source' not in df.columns:
                      df['source'] = 'TradeStation' 
                 if 'adjusted' not in df.columns:
                      df['adjusted'] = False 
                 if 'quality' not in df.columns:
                      df['quality'] = 100 
                 
                 logger.info(f"Saving {len(df)} rows of 1-minute data for {symbol} to 'market_data' table...")
                 fetcher.save_to_db(df) 
                 logger.info(f"Successfully saved 1-minute data for {symbol}.")
            else:
                 logger.warning(f"No 1-minute data returned for active contract: {symbol}")

    except Exception as e:
        logger.error(f"Error updating 1-minute active ES data: {e}", exc_info=True)
        success = False
        
    return success

def verify_continuous_data(conn, config_path: str, start_date_str=None, end_date_str=None):
    """Verify data presence and basic counts for continuous symbols in the 'continuous_contracts' table.
       Checks VX (@VX=...) generated locally and ES/NQ (@ES=..., @NQ=...) fetched from TradeStation.
    """
    logger.info("=== Verifying Continuous Contract Data in 'continuous_contracts' table ===")
    all_verified = True
    target_table = "continuous_contracts"
    MAX_VX_CONTRACTS = 8 # Max sequence number usually generated for VX

    if not conn:
         logger.error("Database connection is not valid for verification.")
         return False

    # --- Determine symbols to verify from config ---
    es_nq_symbols_to_verify = []
    vx_symbols_to_verify = []
    try:
        with open(config_path, 'r') as f:
             config = yaml.safe_load(f)
             futures_config = config.get('futures', [])
             for fc in futures_config:
                 root = fc.get('base_symbol')
                 is_cont = fc.get('is_continuous', False)
                 source = fc.get('source', '').lower()
                 if root and is_cont:
                     if root == 'VX' and source == 'cboe': # VX is generated locally from CBOE
                         vx_symbols_to_verify.extend([f"@{root}={i}01XN" for i in range(1, MAX_VX_CONTRACTS + 1)])
                         logger.debug(f"Identified locally generated VX continuous symbols for verification: {vx_symbols_to_verify}")
                     elif root in ['ES', 'NQ'] and source == 'tradestation': # ES/NQ are fetched from TS
                         es_nq_symbols_to_verify.extend([f"@{root}=102XC", f"@{root}=102XN"])
                         logger.debug(f"Identified TradeStation continuous symbols for verification: {es_nq_symbols_to_verify}")
                     # Add handling for other potential continuous symbols if needed
             
             if not es_nq_symbols_to_verify and not vx_symbols_to_verify:
                 logger.warning("No continuous symbols marked for verification in config. Verification cannot proceed.")
                 return True # Return True as no checks were specified

    except Exception as e:
        logger.error(f"Could not load or parse config {config_path} for verification roots: {e}. Verification skipped.")
        return False # Error state

    # --- Define verification period ---
    try:
        start_date = pd.to_datetime(start_date_str or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = pd.to_datetime(end_date_str or datetime.now().strftime('%Y-%m-%d'))
    except Exception as date_e:
        logger.error(f"Invalid start/end date for verification: {date_e}. Using default 7-day lookback.")
        start_date = pd.to_datetime((datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))

    # --- Verification Helper Function ---
    def _verify_symbol_group(symbols: list, group_name: str):
        nonlocal all_verified # Allow modification of the outer variable
        group_verified = True
        if not symbols:
            logger.debug(f"No {group_name} symbols configured for verification.")
            return

        logger.info(f"--- Verifying {group_name} symbols: {symbols} ---")
        placeholders = ','.join(['?'] * len(symbols))
        
        # 1. Check Presence
        logger.info(f"Verifying DAILY data presence for {group_name} in '{target_table}' between {start_date.date()} and {end_date.date()}")
        query_presence = f"""
        SELECT DISTINCT symbol 
        FROM {target_table} 
        WHERE symbol IN ({placeholders}) 
          AND interval_unit = 'daily' 
          AND Timestamp::DATE >= ? 
          AND Timestamp::DATE <= ?
        """
        try:
            result_df = conn.execute(query_presence, symbols + [start_date.date(), end_date.date()]).df()
            if result_df.empty:
                logger.warning(f"No DAILY data found for any {group_name} symbols in '{target_table}' within the verification period.")
                group_verified = False
            else:
                present_symbols = result_df['symbol'].tolist()
                missing_symbols = [s for s in symbols if s not in present_symbols]
                if missing_symbols:
                    logger.warning(f"Missing DAILY data for {group_name} symbols in the period: {missing_symbols}")
                    # Optional: Decide if missing symbols is a failure
                    # group_verified = False 
                else:
                    logger.info(f"DAILY data found for all expected {group_name} symbols in the period.")
        except Exception as pres_e:
            logger.error(f"Error checking presence for {group_name} symbols: {pres_e}", exc_info=True)
            group_verified = False
        
        # 2. Check Counts
        logger.info(f"Verifying DAILY row counts for {group_name} symbols in '{target_table}'")
        query_counts = f"""
        SELECT symbol, COUNT(*) as count 
        FROM {target_table} 
        WHERE symbol IN ({placeholders}) 
          AND interval_unit = 'daily'
        GROUP BY symbol
        ORDER BY symbol
        """
        try:
            counts_df = conn.execute(query_counts, symbols).df()
            if not counts_df.empty:
                logger.info(f"Total DAILY row counts per {group_name} symbol:")
                for _, row in counts_df.iterrows():
                     logger.info(f"  {row['symbol']}: {row['count']} rows")
                # Add more checks here, e.g., minimum expected count based on date range
            else:
                logger.warning(f"No DAILY rows found for any {group_name} symbols during count verification.")
                group_verified = False
        except Exception as count_e:
            logger.error(f"Error checking counts for {group_name} symbols: {count_e}", exc_info=True)
            group_verified = False
            
        # Update overall verification status
        if not group_verified:
            all_verified = False

    # --- Run Verification for Each Group ---
    try:
        _verify_symbol_group(es_nq_symbols_to_verify, "ES/NQ (TradeStation)")
        _verify_symbol_group(vx_symbols_to_verify, "VX (Generated)")

        # 3. Check active VX futures counts (in 'market_data_cboe') - Kept separate
        if vx_symbols_to_verify: # Only check this if VX was supposed to be processed
            logger.info("--- Verifying Active VX Futures Count (in market_data_cboe) ---")
            try:
                result = conn.execute(
                    """SELECT COUNT(DISTINCT symbol) 
                       FROM market_data_cboe 
                       WHERE symbol LIKE 'VX%' AND symbol NOT LIKE '@VX=%'"""
                ).fetchone()
                if result:
                     logger.info(f"Active VX futures contracts count: {result[0]} distinct symbols")
                else:
                     logger.warning("Could not get active VX futures count from 'market_data_cboe'.")
                     all_verified = False # Consider this a verification failure
            except Exception as vx_count_e:
                 logger.error(f"Error verifying active VX futures count: {vx_count_e}", exc_info=True)
                 all_verified = False

    except Exception as e:
        logger.error(f"Error during overall data verification: {e}", exc_info=True)
        all_verified = False
        
    # --- Final Result ---
    if all_verified:
        logger.info("=== Continuous data verification passed. ===")
    else:
        logger.warning("=== Continuous data verification completed with issues. ===")
        
    return all_verified

def main():
    parser = argparse.ArgumentParser(description='Complete update workflow for financial market data.')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Specify the path to the DuckDB database file.')
    parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH, help='Specify the path to the market symbols configuration YAML file.')
    parser.add_argument('--full-update', action='store_true', help='Perform a comprehensive update, including regenerating historical VX continuous contracts and filling gaps. Note: Historical fill is deprecated.')
    parser.add_argument('--start-date', type=str, help='Set the start date (YYYY-MM-DD) for updating continuous contracts (default: 90 days ago).')
    parser.add_argument('--end-date', type=str, help='Set the end date (YYYY-MM-DD) for updating continuous contracts (default: today).')
    parser.add_argument('--verify', action='store_true', help='Run verification checks on continuous contracts and data counts after the update.')
    parser.add_argument('--skip-vix', action='store_true', help='Exclude the VIX Index ($VIX.X) data update from the process.')
    parser.add_argument('--skip-vx', action='store_true', help='Exclude the update of active VX futures contracts and VX continuous generation.') 
    parser.add_argument('--skip-es-nq', action='store_true', help='Exclude the update of active ES and NQ futures contracts from the process.')
    parser.add_argument('--skip-continuous', action='store_true', help='Exclude the update of daily TradeStation continuous contracts (@ES, @NQ).')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run for ES/NQ futures update without saving data to the database.')
    parser.add_argument('--update-active-es-15min', action='store_true', help='Include update for 15-minute data for the two closest active ES contracts.') 
    parser.add_argument('--update-active-es-1min', action='store_true', help='Include update for 1-minute data for the two closest active ES contracts.')
    parser.add_argument('--skip-intraday', action='store_true', help='Skip updating 15-min and 1-min data for active ES contracts.')
    
    args = parser.parse_args()
    
    # Ensure database directory exists
    check_db_exists(args.db_path)

    # --- Workflow Initialization ---
    # Establish the main database connection used by most steps
    conn = None
    try:
        logger.info(f"Connecting to database: {args.db_path}")
        conn = get_db_engine(db_path=args.db_path) # Returns a connection-like object
        if not conn:
             raise ConnectionError(f"Failed to establish database connection to {args.db_path}.")
        logger.info("Database connection established.")
        
    except Exception as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        sys.exit(1) # Critical failure if DB connection fails

    # Initialize MarketDataFetcher (for TradeStation API interaction)
    # This is done early but only if needed, as authentication takes time.
    # It manages its own connection internally if needed but can reuse the main one.
    fetcher = None
    fetcher_initialized = False
    if not args.skip_continuous or not args.skip_intraday:
        logger.info("Initializing MarketDataFetcher...")
        try:
            # Pass the existing connection to the fetcher
            fetcher = MarketDataFetcher(existing_conn=conn) 
            if not fetcher.ts_agent or not fetcher.ts_agent.authenticate():
                logger.warning("Failed to initialize or authenticate MarketDataFetcher. Skipping TradeStation steps.")
                fetcher = None # Ensure fetcher is None if auth fails
            else:
                fetcher_initialized = True
                logger.info("MarketDataFetcher initialized and authenticated.")
                # --- ADDED: Load config into fetcher --- 
                try:
                    with open(args.config_path, 'r') as f:
                         fetcher.config = yaml.safe_load(f)
                         logger.info(f"Configuration loaded into MarketDataFetcher from {args.config_path}")
                except Exception as cfg_load_err:
                    logger.error(f"Failed to load config '{args.config_path}' into MarketDataFetcher: {cfg_load_err}")
                    fetcher = None # Mark fetcher as unusable if config load fails
                    fetcher_initialized = False
                # --- END ADDED --- 
        except Exception as fetcher_init_e:
            logger.error(f"Error initializing MarketDataFetcher: {fetcher_init_e}", exc_info=True)
            fetcher = None # Ensure fetcher is None on error

    # --- Data Update Sequence ---
    # Track overall success, step by step
    success = True

    # Step 1: Update VIX Index Data (from CBOE URL)
    if not args.skip_vix:
        logger.info("--- Step 1: Updating VIX Index ---")
        success &= update_vix_index(conn)
    else:
        logger.info("--- Step 1: Skipping VIX Index update ---")

    # Step 2: Update Individual VX Futures Contracts (from CBOE URLs)
    vx_futures_updated = False # Flag needed for step 3
    if not args.skip_vx:
        logger.info("--- Step 2: Updating Individual VX Futures (CBOE) ---")
        # Uses the main connection (conn)
        vx_futures_updated = update_vx_futures(conn, args.config_path, args.full_update)
        success &= vx_futures_updated
    else:
        logger.info("--- Step 2: Skipping VX Futures update ---")

    # Step 3a: Update the VX continuous contract mapping table
    # This creates/updates the lookup table that maps each date to the
    # appropriate contract for each continuous contract (@VX=101XN, etc.)
    if not args.skip_vx:
        if vx_futures_updated:
            logger.info("--- Step 3a: Updating VX Continuous Contract Mapping Table ---")
            try:
                # Import the create_continuous_mapping function
                from src.utils.continuous_contracts import create_continuous_mapping
                
                # Load config to get the start date
                with open(args.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    vx_config = next((item for item in config.get('futures', []) if item['base_symbol'] == 'VX'), None)
                    if not vx_config:
                        raise ValueError("VX configuration not found in market_symbols.yaml")
                    mapping_start_date = vx_config.get('start_date', '2004-01-01')
                
                # Set end date to 25 years in the future
                mapping_end_date = (date.today() + timedelta(days=365*25)).strftime('%Y-%m-%d')
                
                logger.info(f"Updating continuous contract mapping for VX from {mapping_start_date} to {mapping_end_date}")
                
                # Create the mapping
                mapping_success = create_continuous_mapping(
                    root_symbol='VX',
                    start_date=mapping_start_date,
                    end_date=mapping_end_date,
                    num_contracts=9,
                    db_path=args.db_path
                )
                
                if mapping_success:
                    logger.info("Successfully updated VX continuous contract mapping table.")
                else:
                    logger.error("Failed to update VX continuous contract mapping table.")
                    success = False
            except Exception as mapping_e:
                logger.error(f"Error updating VX continuous contract mapping table: {mapping_e}", exc_info=True)
                success = False
        else:
            logger.warning("--- Step 3a: Skipping VX continuous contract mapping update (VX futures update failed) ---")
    else:
        logger.info("--- Step 3a: Skipping VX continuous contract mapping update (related to --skip-vx) ---")

    # Step 3b: Generate Continuous VX Contracts (@VX=...) (from CBOE data in DB)
    # This step *builds* the continuous contracts by stitching together the individual
    # futures data downloaded in Step 2, using roll rules.
    # It MUST run after Step 2 and 3a.
    if not args.skip_vx: # Also depends on not skipping VX futures
        if vx_futures_updated:
            logger.info("--- Step 3b: Generating Continuous VX Contracts (@VX=...) ---")
            try:
                # Determine date range and force flag based on --full-update
                gen_end_date = date.today().strftime('%Y-%m-%d')
                force_flag = False
                gen_start_date = None # Initialize
                if args.full_update: # Use the overall full_update flag
                    logger.info("Full VX continuous regeneration requested.")
                    gen_start_date = None # generate_vx_continuous_main will use config default
                    force_flag = True # Force delete for a clean full regeneration
                else:
                    # Default: ~95 day lookback for generation
                    gen_start_date = (date.today() - timedelta(days=95)).strftime('%Y-%m-%d')

                log_start = "config default" if gen_start_date is None else gen_start_date
                logger.info(f"Generating VX continuous contracts from {log_start} to {gen_end_date} (force={force_flag})")

                # Prepare args dictionary for generate_vx_continuous_main
                # Note: It needs db_path, not connection object directly
                gen_args = {
                    'db_path': args.db_path,
                    'config_path': args.config_path,
                    'root_symbol': 'VX',
                    'start_date': gen_start_date,
                    'end_date': gen_end_date,
                    'force': force_flag
                }
                generate_vx_continuous_main(args_dict=gen_args)
                logger.info("Successfully completed VX continuous contract generation.")
            except Exception as gen_e:
                logger.error(f"Error generating VX continuous contracts: {gen_e}", exc_info=True)
                success = False
        else:
            logger.warning("--- Step 3b: Skipping VX continuous contract generation (VX futures update failed) ---")
    else:
        logger.info("--- Step 3b: Skipping VX Continuous generation (related to --skip-vx) ---")

    # Step 4: Update Individual ES & NQ Futures Contracts (via TradeStation)
    # This uses a subprocess because the target script has complex internal logic
    # and doesn't accept a connection object easily. Requires closing the main connection.
    # Temporarily close main connection to avoid DB lock by subprocess
    if conn:
        try:
            conn.close()
            logger.debug("Temporarily closed main DB connection for ES/NQ subprocess.")
            conn = None # Mark as closed
        except Exception as close_err:
            logger.warning(f"Error closing connection before ES/NQ subprocess: {close_err}")
            success = False # Cannot proceed reliably

    # Execute subprocess only if connection was closed and previous steps were successful
    if success and not args.skip_es_nq:
        logger.info("--- Step 4: Updating Individual ES/NQ Futures (TradeStation via Subprocess) ---")
        if conn is not None:
             logger.warning("Main connection was not closed before ES/NQ subprocess. Skipping step to avoid lock.")
             success = False
        else:
             # Subprocess needs db_path, not connection object
             success &= update_es_nq_futures(args.db_path, args.dry_run)
    elif args.skip_es_nq:
        logger.info("--- Step 4: Skipping ES/NQ Futures update ---")
    elif not success:
         logger.warning("--- Step 4: Skipping ES/NQ Futures update due to previous errors ---")

    # --- Re-establish main connection after subprocess --- 
    if conn is None: # Reopen if we closed it
        try:
            logger.debug(f"Re-connecting to main database: {args.db_path}")
            conn = get_db_engine(db_path=args.db_path) # Get new connection object
            if not conn:
                 raise ConnectionError("Failed to re-establish main database connection.")
            logger.debug("Main database connection re-established.")
            
            # If connection reopened successfully, update the fetcher's connection
            if fetcher:
                fetcher.set_connection(conn)
        except Exception as reopen_err:
            logger.error(f"Error re-connecting to main database: {reopen_err}", exc_info=True)
            success = False
            conn = None # Ensure conn is None on failure

    # --- Proceed with TradeStation steps only if connection is valid ---
    if not conn:
         logger.error("Main database connection not available. Skipping subsequent TradeStation steps.")
         success = False

    # Step 5: Update Daily Continuous ES/NQ Contracts (@ES, @NQ via TradeStation)
    # This step *fetches* pre-constructed continuous data directly from TradeStation API.
    # It relies on the initialized 'fetcher'.
    if success and fetcher_initialized and not args.skip_continuous:
        logger.info("--- Step 5: Updating Daily Continuous ES/NQ (@ES, @NQ via TradeStation) ---")
        success &= update_tradestation_continuous(fetcher, args.config_path)
    elif args.skip_continuous:
        logger.info("--- Step 5: Skipping Daily Continuous ES/NQ update ---")
    elif not fetcher_initialized and not args.skip_continuous:
        logger.warning("--- Step 5: Skipping Daily Continuous ES/NQ update (Fetcher not initialized) ---")
    elif not success:
        logger.warning("--- Step 5: Skipping Daily Continuous ES/NQ update due to previous errors ---")

    # Step 6: Update Intraday Data for Active ES Contracts (via TradeStation)
    # Fetches 15-min and 1-min data for the nearest active ES contracts.
    if success and fetcher_initialized and not args.skip_intraday:
        logger.info("--- Step 6: Updating Intraday Active ES (15min & 1min via TradeStation) ---")
        # Both 15min and 1min updates are performed if not skipped
        success &= update_active_es_15min(fetcher, args.config_path)
        success &= update_active_es_1min(fetcher, args.config_path)
    elif args.skip_intraday:
        logger.info("--- Step 6: Skipping Intraday Active ES update ---")
    elif not fetcher_initialized and not args.skip_intraday:
        logger.warning("--- Step 6: Skipping Intraday Active ES update (Fetcher not initialized) ---")
    elif not success:
        logger.warning("--- Step 6: Skipping Intraday Active ES update due to previous errors ---")

    # Step 7: Verify Data Integrity (Optional)
    # Checks counts and presence of data in key tables.
    if args.verify:
        logger.info("--- Step 7: Verifying Data ---")
        if not conn:
             logger.error("Cannot verify data, database connection lost.")
             success = False
        elif not success:
             logger.warning("Running verification, but previous steps had errors.")
             # Run verification but don't let it change the overall 'success' status
             verify_continuous_data(conn, args.config_path, args.start_date, args.end_date)
        else:
             # If all previous steps were successful, verification result contributes to overall success
             success &= verify_continuous_data(conn, args.config_path, args.start_date, args.end_date)
    else:
         logger.info("--- Step 7: Skipping Verification ---")

    # --- Final Cleanup --- 
    # Ensure the main connection is closed if it exists
    if conn:
        try:
            conn.close()
            logger.debug("Main database connection closed.")
        except Exception as conn_close_e:
            logger.warning(f"Error closing main connection: {conn_close_e}")

    # Final Status Report and Exit Code
    exit_code = 0
    if success:
        logger.info("=== Market Data Update Workflow Completed Successfully ===")
    else:
        logger.error("=== Market Data Update Workflow Completed with Errors ===")
        exit_code = 1 # Signal error to calling process (e.g., batch script)

    sys.exit(exit_code)


if __name__ == "__main__":
    main() # Let main handle sys.exit 