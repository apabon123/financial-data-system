import duckdb
import os
import sys
import logging
import argparse
import yaml
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import pandas_market_calendars as mcal
from src.scripts.market_data.fetch_market_data import MarketDataFetcher, get_trading_calendar
import re
from typing import Optional

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_continuous_db():
    """Create the continuous contracts table if it doesn't exist"""
    con = duckdb.connect('data/financial_data.duckdb')
    
    # First drop the existing table
    con.execute("""
    DROP TABLE IF EXISTS continuous_contracts;
    """)
    
    # Create the table with proper constraints
    create_table_sql = """
    CREATE TABLE continuous_contracts (
        timestamp TIMESTAMP,      -- The date of the data point
        symbol VARCHAR,           -- The continuous contract symbol (e.g., @ES=202XC)
        underlying_symbol VARCHAR,-- Specific contract used for this row
        open DOUBLE,             -- Opening price
        high DOUBLE,             -- Highest price
        low DOUBLE,             -- Lowest price
        close DOUBLE,            -- Closing price
        volume BIGINT,           -- Trading volume
        open_interest BIGINT,    -- Open Interest
        up_volume BIGINT,        -- Optional up volume
        down_volume BIGINT,      -- Optional down volume
        source VARCHAR,          -- Data source (origin of underlying data)
        built_by VARCHAR,        -- How the continuous row was built ('local_generator', 'tradestation')
        interval_value INTEGER,  -- Interval length (e.g., 1, 15)
        interval_unit VARCHAR,   -- Interval unit (e.g., 'day', 'minute')
        adjusted BOOLEAN,        -- Whether the price is adjusted
        quality INTEGER,         -- Data quality indicator
        settle DOUBLE,           -- Settlement price
        CONSTRAINT continuous_contracts_pkey PRIMARY KEY (symbol, timestamp, interval_value, interval_unit)
    );
    """
    
    con.execute(create_table_sql)
    
    # Create an index on the primary key columns
    con.execute("""
    CREATE INDEX IF NOT EXISTS continuous_contracts_idx 
    ON continuous_contracts(symbol, timestamp, interval_value, interval_unit);
    """)
    
    con.close()

def load_market_symbols_config():
    """Load market symbols configuration from YAML file."""
    config_path = os.path.join('config', 'market_symbols.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_next_roll_date(symbol_root: str, config: dict, fetcher: MarketDataFetcher) -> pd.Timestamp:
    """
    Calculate the next roll date for the given futures symbol based on its configured cycle 
    and expiry rule from the YAML config.
    Roll date is calculated as 2 trading days prior to the calculated expiration date.
    
    Args:
        symbol_root: The root symbol (ES, NQ, VX, etc.)
        config: Market symbols configuration dictionary
        fetcher: An initialized MarketDataFetcher instance (needed for expiry calc)
        
    Returns:
        pd.Timestamp of the next roll date
    """
    if not fetcher:
         raise ValueError("MarketDataFetcher instance is required to calculate expiration dates.")
         
    # Get the futures configuration for this symbol
    futures_config = next(
        (f for f in config.get('futures', []) if f['base_symbol'] == symbol_root),
        None
    )
    if not futures_config:
        # Raise specific error for missing symbol config
        raise ValueError(f"No 'futures' configuration found for '{symbol_root}' in market_symbols.yaml. Please add or verify the entry.")
        
    # --- Determine Next Contract Month/Year --- 
    # Get the contract months patterns specific to this symbol from config
    if 'historical_contracts' not in futures_config or 'patterns' not in futures_config['historical_contracts']:
         # Raise specific error for missing patterns
         raise ValueError(f"No 'historical_contracts.patterns' defined for futures symbol '{symbol_root}' in market_symbols.yaml. Please add the contract month codes (e.g., [H, M, U, Z] or [F, G, ...]).")
    contract_months_codes = futures_config['historical_contracts']['patterns']
    
    full_month_map = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    reverse_month_map = {v: k for k, v in full_month_map.items()}
    
    contract_month_numbers = sorted([full_month_map[code] for code in contract_months_codes if code in full_month_map])
    if not contract_month_numbers:
         # Raise specific error for invalid/missing month codes in patterns
         raise ValueError(f"No valid month codes found in patterns for {symbol_root}: {contract_months_codes}. Check market_symbols.yaml.")

    today = pd.Timestamp.now().normalize()
    current_month = today.month
    current_year = today.year
    
    next_contract_month = None
    next_contract_year = current_year
    
    found_next = False
    temp_idx = 0 # Loop counter for safety break
    # Need mutable state for year and current_month_start within the loop
    check_year = current_year 
    check_month = current_month

    # Find the first contract month >= current month whose roll hasn't passed
    while not found_next:
         # Determine the next month in the symbol's cycle to check
         current_cycle_month_num = None
         for month_num in contract_month_numbers:
             if month_num >= check_month:
                  current_cycle_month_num = month_num
                  break
         # If no month found this year in the cycle, wrap to next year's first cycle month
         if current_cycle_month_num is None:
              current_cycle_month_num = contract_month_numbers[0]
              check_year += 1
              check_month = 1 # Start checking from Jan next year
              
         # Construct the specific contract symbol for expiry calculation
         next_contract_month_code = reverse_month_map.get(current_cycle_month_num)
         if not next_contract_month_code:
              logger.error(f"Logic error: Could not find month code for month number {current_cycle_month_num}")
              # This should not happen if contract_month_numbers is valid
              raise RuntimeError("Internal logic error in roll date calculation.")
              
         next_contract_symbol = f"{symbol_root}{next_contract_month_code}{str(check_year)[-2:]}"
         logger.debug(f"Checking potential next contract: {next_contract_symbol}")
         
         # Calculate expiry for this potential next contract using fetcher's method
         expiry_date = fetcher.calculate_expiration_date(next_contract_symbol)
         
         if expiry_date:
             # Calculate roll date (2 trading days before expiry)
             calendar_name = futures_config.get('calendar', 'NYSE') # Get calendar name from config
             calendar = get_trading_calendar(calendar_name) # Use fetcher's helper
             schedule_around_expiry = calendar.schedule(start_date=expiry_date - pd.Timedelta(days=10), end_date=expiry_date)
             
             if not schedule_around_expiry.empty:
                 # Ensure expiry_date is naive for comparison with naive index
                 naive_expiry_date = expiry_date.tz_localize(None) if expiry_date.tz is not None else expiry_date
                 trading_days_before_expiry = schedule_around_expiry.index[schedule_around_expiry.index < naive_expiry_date] # Strictly before expiry
                 if len(trading_days_before_expiry) >= 2:
                     potential_roll_date = trading_days_before_expiry[-2] # 2nd day back is 2 trading days before expiry
                     
                     # Check if today is strictly before this potential roll date
                     if today < potential_roll_date:
                          # Found the next contract whose roll date hasn't passed
                          next_contract_month = current_cycle_month_num
                          next_contract_year = check_year
                          found_next = True
                          break # Exit while loop
                     else:
                          logger.debug(f"Roll date {potential_roll_date.date()} for {next_contract_symbol} has passed today ({today.date()}). Checking next contract month.")
                 else:
                     # Handle case where expiry is very soon, fewer than 2 trading days before it
                     logger.warning(f"Could not find 2 trading days before expiry {expiry_date.date()} for {next_contract_symbol}. Assuming roll has passed or is today. Checking next contract month.")
             else:
                  # Handle case where calendar schedule might be empty (unlikely for near dates)
                  logger.warning(f"Could not find trading days around expiry {expiry_date.date()} for {next_contract_symbol}. Checking next contract month.")
         else:
             # Expiry calculation failed for this potential symbol
             logger.warning(f"Could not calculate expiry date for potential next contract {next_contract_symbol}. Checking next contract month.")

         # --- Advance to the next month in the cycle for the next iteration --- 
         current_month_index_in_cycle = contract_month_numbers.index(current_cycle_month_num)
         next_month_index_in_cycle = (current_month_index_in_cycle + 1) % len(contract_month_numbers)
         check_month = contract_month_numbers[next_month_index_in_cycle]
         # If we wrapped around the cycle (e.g., from Z back to H), increment the check_year
         if next_month_index_in_cycle == 0:
              check_year += 1 
              
         # Safety break to prevent infinite loops in case of unexpected logic error or config issue
         temp_idx += 1
         if temp_idx > len(contract_month_numbers) * 2: # Allow checking up to two full cycles
             logger.error(f"Potential infinite loop detected while determining next roll date for {symbol_root}. Check config and logic.")
             raise RuntimeError(f"Could not determine next roll date for {symbol_root} after checking multiple cycles.")
             
    # --- End While Loop --- 
    
    if not found_next:
         # This should ideally not be reached if the loop logic is correct
         logger.error(f"Failed to find the next contract roll date for {symbol_root}. Loop completed without success.")
         raise ValueError(f"Unable to determine next roll date for {symbol_root}")

    # --- Now we have the correct next_contract_month/year, calculate its final expiry and roll date --- 
    final_next_month_code = reverse_month_map.get(next_contract_month)
    final_next_symbol = f"{symbol_root}{final_next_month_code}{str(next_contract_year)[-2:]}"
    
    logger.info(f"Determined next contract for roll calculation: {final_next_symbol}")
    
    final_expiry_date = fetcher.calculate_expiration_date(final_next_symbol)
    if not final_expiry_date:
         # This is more serious, calculation failed for the identified next contract
         raise ValueError(f"Failed to calculate final expiration date for identified next contract: {final_next_symbol}")

    # Calculate final roll date (2 trading days prior)
    calendar_name = futures_config.get('calendar', 'NYSE')
    calendar = get_trading_calendar(calendar_name)
    schedule = calendar.schedule(start_date=final_expiry_date - pd.Timedelta(days=10), end_date=final_expiry_date)
    
    # Check if schedule is valid and contains enough days before expiry
    if schedule.empty:
        logger.warning(f"Could not find trading days around expiry {final_expiry_date.date()} for {final_next_symbol}. Returning expiry date minus 2 calendar days as fallback.")
        roll_date = final_expiry_date - pd.Timedelta(days=2) # Fallback
    else:
        # Ensure final_expiry_date is naive for comparison
        naive_final_expiry_date = final_expiry_date.tz_localize(None) if final_expiry_date.tz is not None else final_expiry_date
        trading_days_before_expiry = schedule.index[schedule.index < naive_final_expiry_date] # <-- NEW
        if len(trading_days_before_expiry) < 2:
            logger.warning(f"Could not find 2 trading days before expiry {final_expiry_date.date()} for {final_next_symbol}. Returning expiry date minus 2 calendar days as fallback.")
            roll_date = final_expiry_date - pd.Timedelta(days=2) # Fallback
        else:
             roll_date = trading_days_before_expiry[-2] # 2nd day back = 2 trading days before
    
    logger.info(f"Calculated expiry for {final_next_symbol}: {final_expiry_date.date()}, Roll Date: {roll_date.date()}")
    return roll_date.normalize() # Normalize to remove time component

def is_near_roll(symbol_root: str, config: dict, fetcher: MarketDataFetcher) -> bool:
    """
    Check if we're near a roll date for the given symbol using its specific cycle.
    We'll consider "near roll" as within 5 trading days of the roll.
    """
    try:
        # Pass fetcher to get_next_roll_date
        next_roll = get_next_roll_date(symbol_root, config, fetcher) 
        today = pd.Timestamp.now().normalize()
        
        futures_config = next(
            (f for f in config.get('futures', []) if f['base_symbol'] == symbol_root),
            None
        )
        if not futures_config:
             logger.warning(f"No futures config found for {symbol_root} during roll check.")
             return False 
        if 'calendar' not in futures_config:
             logger.warning(f"No 'calendar' key found for {symbol_root} in config during roll check.")
             return False 
             
        # Use fetcher to get calendar instance
        calendar = get_trading_calendar(futures_config['calendar'])
        
        # Ensure next_roll is timezone-naive if today is
        if today.tz is None and next_roll.tz is not None:
             next_roll = next_roll.tz_localize(None)
             
        trading_days = calendar.valid_days(start_date=today, end_date=next_roll)
        
        # Check if roll date is today or in the future, and <= 5 trading days away
        is_near = (next_roll >= today) and (len(trading_days) <= 5)
        logger.debug(f"Roll check for {symbol_root}: Today={today.date()}, NextRoll={next_roll.date()}, TradingDays={len(trading_days)}, IsNear={is_near}")
        return is_near
        
    except Exception as e:
        # Log error but don't crash the whole update if roll check fails for one symbol
        logger.error(f"Error checking roll date for {symbol_root}: {e}", exc_info=True)
        return False # Default to False if error occurs

def get_start_date(symbol_root: str, config: dict, force: bool = False, interval_unit: str = 'daily') -> str:
    """
    Determine the start date for data fetching based on force flag, symbol type, and interval.
    
    Args:
        symbol_root: The root symbol (e.g., 'ES' or 'NQ')
        config: Market symbols configuration
        force: Whether to force update the entire series
        interval_unit: The interval unit (e.g., 'daily' or 'minute')
    
    Returns:
        start_date: The date to start fetching data from
    """
    # Get the futures configuration for this symbol
    futures_config = next(
        (f for f in config.get('futures', []) if f['base_symbol'] == symbol_root),
        None
    )
    if not futures_config:
        # Use a default start date if config not found, maybe log warning
        logger.warning(f"No futures config found for {symbol_root} in get_start_date. Using default lookbacks.")
        default_start_daily = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        default_start_intraday = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        return default_start_daily if interval_unit == 'daily' else default_start_intraday

    if force:
        # Use the start_date from config for daily data
        config_start = futures_config.get('start_date')
        if not config_start:
             logger.warning(f"'start_date' not found in config for {symbol_root} during force update. Using default.")
             config_start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d') # Default 5 years
             
        # For intraday data, limit to 30 days even when forced
        if interval_unit == 'daily':
            return config_start
        else:
            return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # For normal updates, use different lookback periods based on interval
    if interval_unit == 'daily':
        return (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        # For intraday data, use a shorter lookback to avoid API limitations
        return (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

def load_continuous_data(fetcher: MarketDataFetcher, symbol_root: str, config: dict, force: bool = False, interval_value: int = 1, interval_unit: str = 'daily', specific_symbol: Optional[str] = None):
    """
    Load continuous contract data for a given symbol root using the existing MarketDataFetcher.
    If specific_symbol is provided, only that symbol is fetched.
    
    Args:
        fetcher: Instance of MarketDataFetcher
        symbol_root: The root symbol (e.g., 'ES' or 'NQ') used for config lookups.
        config: Market symbols configuration
        force: Whether to force update the entire series
        interval_value: The interval value (e.g., 1 for daily, 15 for 15-minute)
        interval_unit: The interval unit (e.g., 'daily' or 'minute')
        specific_symbol: If set, fetch only this exact continuous symbol.
    """
    # Check if we're near a roll date, passing the fetcher
    # Only relevant if processing based on root, not specific symbol
    near_roll = False
    if not specific_symbol:
        near_roll = is_near_roll(symbol_root, config, fetcher)
        if near_roll:
            logger.info(f"{symbol_root} is near roll date - will update entire constant-adjusted series")
    
    # Define the symbols to process
    if specific_symbol:
        symbols_to_process = [specific_symbol]
        logger.info(f"Processing specific continuous symbol: {specific_symbol}")
    else:
        # Default: process both constant-adjusted and unadjusted continuous contract symbols
        symbols_to_process = [
            f"@{symbol_root}=102XC",  # First month, 02 days before expiry, constant adjustment
            f"@{symbol_root}=102XN"   # First month, 02 days before expiry, no adjustment
        ]
        logger.info(f"Processing standard continuous symbols for root: {symbol_root}")
    
    for continuous_symbol in symbols_to_process:
        # Determine adjusted flag based on the actual symbol being processed
        is_adjusted = 'XC' in continuous_symbol
        
        try:
            # Determine start date based on type and conditions
            # Pass the correct interval_unit to get_start_date
            if force:
                # Force mode: Use base symbol's configured start date
                start_date = get_start_date(symbol_root, config, True, interval_unit) 
            elif is_adjusted and near_roll and not specific_symbol:
                # For constant-adjusted contracts, update entire series if forced or near roll
                # Only apply near_roll logic if NOT processing a specific symbol
                start_date = get_start_date(symbol_root, config, True, interval_unit)
            else:
                # For unadjusted or normal updates, just get recent data
                start_date = get_start_date(symbol_root, config, False, interval_unit)
            
            logger.info(f"Fetching {interval_value}-{interval_unit} data for {continuous_symbol} from {start_date}")
            
            # Get data using the existing fetcher
            df = fetcher.fetch_data_since(
                symbol=continuous_symbol,
                interval=interval_value,
                unit=interval_unit,
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df is None or df.empty:
                logger.error(f"No data returned for {continuous_symbol}. This could be due to:")
                logger.error("1. Symbol not supporting this interval")
                logger.error("2. No trading activity in the specified period")
                logger.error("3. API limitations on historical data")
                continue
            
            # Add metadata columns to match the existing schema
            df['symbol'] = continuous_symbol
            df['underlying_symbol'] = None  # Will be filled by TradeStation's data
            df['source'] = 'tradestation'
            df['interval_value'] = interval_value
            df['interval_unit'] = interval_unit
            df['adjusted'] = is_adjusted
            df['quality'] = 100  # Direct from source
            df['built_by'] = 'tradestation' # Added BuiltBy field
            df['open_interest'] = None  # Not available in continuous contracts
            df['up_volume'] = None
            df['down_volume'] = None
            df['settle'] = df['close']
            
            # Save to database using the fetcher's connection
            # Ensure fetcher.conn is valid before using
            if not hasattr(fetcher, 'conn') or not fetcher.conn:
                 logger.error(f"Fetcher database connection not available. Cannot save data for {continuous_symbol}.")
                 continue # Skip saving if connection invalid
            try:
                rows_before_query = "SELECT COUNT(*) FROM continuous_contracts"
                rows_before_result = fetcher.conn.execute(rows_before_query).fetchone()
                rows_before = rows_before_result[0] if rows_before_result else 0
            except Exception as count_e:
                logger.error(f"Error getting row count before save for {continuous_symbol}: {count_e}")
                rows_before = -1 # Indicate error
            
            # Use DuckDB's efficient UPSERT capability via temp view registration
            temp_view_name = f"temp_cont_{symbol_root}_{interval_value}{interval_unit}_view"
            try:
                 fetcher.conn.register(temp_view_name, df)
                 
                 # Define columns to insert/update dynamically
                 insert_columns = list(df.columns) # Use columns present in the dataframe
                 if 'built_by' not in insert_columns: insert_columns.append('built_by') # Ensure built_by is included
                 insert_columns_str = ", ".join(insert_columns)
                 update_setters_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in insert_columns if col not in ['timestamp', 'symbol', 'interval_value', 'interval_unit', 'adjusted']]) # Don't update PK cols or adjusted

                 upsert_sql = f"""
                 INSERT INTO continuous_contracts ({insert_columns_str})
                 SELECT 
                     {insert_columns_str}
                 FROM {temp_view_name}
                 ON CONFLICT (symbol, timestamp, interval_value, interval_unit) DO UPDATE SET 
                     {update_setters_str}
                 """
                 fetcher.conn.execute(upsert_sql)
                 fetcher.conn.commit() # Commit after each symbol pair (XC/XN) processing
                 
                 try:
                     rows_after_query = "SELECT COUNT(*) FROM continuous_contracts"
                     rows_after_result = fetcher.conn.execute(rows_after_query).fetchone()
                     rows_after = rows_after_result[0] if rows_after_result else 0
                     rows_added = rows_after - rows_before if rows_before != -1 else -1
                 except Exception as count_e:
                     logger.error(f"Error getting row count after save for {continuous_symbol}: {count_e}")
                     rows_after = -1
                     rows_added = -1
                     
                 logger.info(f"Successfully loaded continuous contract data for {continuous_symbol}")
                 if rows_added != -1:
                      logger.info(f"Upserted {len(df)} rows (approx {rows_added} new) to the database")
                 else:
                      logger.info(f"Upserted {len(df)} rows to the database (count change unavailable)")
                      
            except Exception as save_e:
                 logger.error(f"Error saving data to database for {continuous_symbol}: {save_e}", exc_info=True)
                 try:
                     fetcher.conn.rollback() # Attempt rollback on error
                 except Exception as rb_e:
                     logger.error(f"Rollback failed: {rb_e}")
                 continue # Continue to next symbol even if save fails
            finally:
                 try:
                      fetcher.conn.unregister(temp_view_name) # Clean up the view
                 except Exception as unreg_e:
                      logger.warning(f"Could not unregister temp view {temp_view_name}: {unreg_e}")
                      
        except Exception as e:
            logger.error(f"Error processing {continuous_symbol}: {str(e)}", exc_info=True)
            continue # Continue to the next symbol in XC/XN pair

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load continuous contract data')
    parser.add_argument('--force', action='store_true', help='Force update entire series')
    parser.add_argument('--interval-value', type=int, default=1, help='Interval value (e.g., 1 for daily, 15 for 15-minute)')
    parser.add_argument('--interval-unit', type=str, default='daily', help='Interval unit (e.g., daily or minute)')
    parser.add_argument('--symbol', type=str, default=None, help='Fetch a specific continuous contract symbol (e.g., @ES=102XC)')
    args = parser.parse_args()
    
    # Setup database table
    # setup_continuous_db() # Removed setup call - should be handled by main orchestrator
    
    # Load market symbols configuration
    try:
        config = load_market_symbols_config()
    except Exception as e:
        logger.error(f"Failed to load market symbols configuration: {e}")
        sys.exit(1)
    
    # Initialize the fetcher and authenticate with TradeStation
    fetcher = None
    try:
        fetcher = MarketDataFetcher() # Manages its own connection if none passed
        if not fetcher.ts_agent.authenticate():
            logger.error("Failed to authenticate with TradeStation")
            sys.exit(1)
        # Load config into fetcher if needed for its internal methods
        fetcher.config = config 
    except Exception as e:
        logger.error(f"Failed to initialize or authenticate TradeStation fetcher: {e}")
        sys.exit(1)
    
    if args.symbol:
        # Handle specific symbol fetch
        logger.info(f"Processing specific continuous contract symbol: {args.symbol}...")
        
        # Extract root symbol from the continuous symbol for config lookup
        match = re.match(r"^@([A-Z]{1,3})=[0-9]+[A-Z]{2}$", args.symbol)
        root_symbol_for_config = match.group(1) if match else None
        
        if not root_symbol_for_config:
            logger.error(f"Could not extract base symbol from {args.symbol}. Cannot proceed.")
            sys.exit(1)
            
        # Find the config for this root symbol
        symbol_config = next((f for f in config.get('futures', []) if f.get('base_symbol') == root_symbol_for_config), None)
        
        if not symbol_config:
            logger.error(f"No configuration found for base symbol '{root_symbol_for_config}' derived from {args.symbol}.")
            sys.exit(1)
            
        # Check the source defined in the config
        source = symbol_config.get('source', '').lower()
        
        if source == 'tradestation':
            logger.info(f"Symbol {args.symbol} uses 'tradestation' source. Proceeding with fetch/force fetch.")
            load_continuous_data(fetcher, root_symbol_for_config, config, args.force, args.interval_value, args.interval_unit, specific_symbol=args.symbol)
            logger.info(f"Completed processing {args.symbol}")
        else:
            logger.error(f"Symbol {args.symbol} (base: {root_symbol_for_config}) is configured with source '{source}', not 'tradestation'.")
            logger.error("This script only fetches/force-fetches continuous contracts directly from TradeStation.")
            logger.error(f"To update locally generated symbols like {args.symbol}, use M3 (Full Update) or run generate_continuous_futures.py directly.")
            sys.exit(1) # Exit as we cannot process this type of symbol here
            
    else:
        # Default behavior: Load data for configured continuous symbols (e.g., ES, NQ from config)
        logger.info("No specific symbol provided, loading configured continuous symbols sourced from TradeStation...")
        loaded_count = 0
        for future_config in config.get('futures', []):
            root_symbol = future_config.get('base_symbol')
            is_cont = future_config.get('is_continuous')
            source = future_config.get('source', '').lower()
            
            # Load only if marked continuous and sourced from tradestation
            if root_symbol and is_cont and source == 'tradestation': 
                logger.info(f"Loading continuous contract data for base symbol {root_symbol} (source: {source})...")
                load_continuous_data(fetcher, root_symbol, config, args.force, args.interval_value, args.interval_unit)
                logger.info(f"Completed loading {root_symbol} data")
                loaded_count += 1
        if loaded_count == 0:
            logger.warning("No continuous contracts configured with source 'tradestation' found in the config file.")

    # Close fetcher connection if opened
    if fetcher and hasattr(fetcher, 'conn') and fetcher.conn: # Check if fetcher manages its own conn
         try:
             fetcher.conn.close()
             logger.info("Fetcher database connection closed.")
         except Exception as e:
             logger.warning(f"Error closing fetcher connection: {e}")
             
if __name__ == '__main__':
    main() 