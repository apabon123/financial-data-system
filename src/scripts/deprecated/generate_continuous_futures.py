#!/usr/bin/env python
"""
Generate Continuous Futures Script

This script generates continuous futures contracts by combining individual futures contracts
and handling rollovers on expiry days. For example, for VX futures:
- VXJ25 (VXc1) expires on a Wednesday
- VXK25 (VXc2) becomes VXc1 on the expiry day of VXJ25
- The rollover happens on the expiry day, not the next day

Database Structure:
- market_data: Main table containing all raw market data with various intervals
- daily_bars: VIEW on market_data that filters for daily interval data
- continuous_contracts: Table storing generated continuous futures contracts

The script reads raw futures data from the market_data table (filtered for daily data)
and writes the generated continuous contracts to the continuous_contracts table.
"""

import os
import sys
import yaml
import logging
import duckdb
import pandas as pd
import calendar # Keep calendar for month_map backup if needed
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, date, timedelta # Added date, timedelta
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU # Added relativedelta
import argparse
import re
import numpy as np # Added for NaN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[3] # Keep definition for paths
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "market_symbols.yaml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"

# --- Date Helper Functions (Defined locally below) ---
def load_config(config_path: Path) -> Optional[Dict]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config file: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return None

def get_holidays(config: Dict, calendar_name: str = "NYSE") -> Set[date]:
    """Extracts holidays for a given calendar from the config."""
    holidays = set()
    try:
        holiday_config = config.get('settings', {}).get('holidays', {}).get(calendar_name, {})
        fixed_dates = holiday_config.get('fixed_dates', [])
        relative_dates = holiday_config.get('relative_dates', [])
        current_year = date.today().year # Use a reasonable range if needed

        # Process fixed date holidays
        for year in range(current_year - 30, current_year + 5): # Expanded range
            for fixed_date in fixed_dates:
                month, day = map(int, fixed_date.split('-'))
                try:
                    holidays.add(date(year, month, day))
                except ValueError:
                    pass

        # Process relative date holidays
        weekday_map = {"monday": MO, "tuesday": TU, "wednesday": WE, "thursday": TH, "friday": FR, "saturday": SA, "sunday": SU}
        for year in range(current_year - 30, current_year + 5): # Expanded range
            for rel_date in relative_dates:
                month = rel_date['month']
                day_type = rel_date['day_type'].lower()
                occurrence = rel_date['occurrence']
                weekday = weekday_map.get(day_type)

                if not weekday:
                    if rel_date.get('name') == 'Good Friday':
                        # Requires Easter calculation - skipping for simplicity
                        pass
                    continue
                try:
                    if occurrence > 0:
                        dt = date(year, month, 1) + relativedelta(weekday=weekday(occurrence))
                    else:
                        next_month_start = date(year, month, 1) + relativedelta(months=1)
                        last_day_of_month = next_month_start - timedelta(days=1)
                        dt = last_day_of_month + relativedelta(weekday=weekday(occurrence))

                    if dt.month == month:
                        holidays.add(dt)
                except ValueError: # Handle invalid date combinations
                     pass

        logger.info(f"Loaded {len(holidays)} holidays for calendar '{calendar_name}'")
        return holidays

    except Exception as e:
        logger.error(f"Error processing holidays from config: {e}")
        return set()

def get_nth_weekday_of_month(year: int, month: int, weekday_int: int, n: int) -> date:
    """
    Gets the date of the nth occurrence of a specific weekday in a given month and year.
    weekday_int: 0=Mon, 1=Tue, ..., 6=Sun
    n: 1 for 1st, 2 for 2nd, etc.
    """
    first_day_of_month = date(year, month, 1)
    day_of_week_first = first_day_of_month.weekday()
    days_to_add = (weekday_int - day_of_week_first + 7) % 7
    first_occurrence_date = first_day_of_month + timedelta(days=days_to_add)
    nth_occurrence_date = first_occurrence_date + timedelta(weeks=n - 1)
    if nth_occurrence_date.month != month:
        raise ValueError(f"The {n}th weekday {weekday_int} does not exist in {year}-{month:02d}")
    return nth_occurrence_date

def get_previous_weekday(target_date: date, weekday_int: int) -> date:
    """Gets the date of the most recent specified weekday before or on the target_date."""
    days_to_subtract = (target_date.weekday() - weekday_int + 7) % 7
    return target_date - timedelta(days=days_to_subtract)

def adjust_for_holiday(expiry_date: date, holidays: Set[date], direction: int = -1) -> date:
    """Adjusts a date backward (default) or forward if it falls on a holiday or weekend."""
    adjusted_date = expiry_date
    # Ensure adjustment only happens if the original date IS a holiday/weekend
    is_holiday_or_weekend = adjusted_date in holidays or adjusted_date.weekday() >= 5 
    
    while adjusted_date in holidays or adjusted_date.weekday() >= 5:
        adjusted_date += timedelta(days=direction)
    
    # Log if adjustment occurred
    if is_holiday_or_weekend and adjusted_date != expiry_date:
        logger.debug(f"Adjusted expiry date {expiry_date} -> {adjusted_date} due to holiday/weekend.")
        
    return adjusted_date


# --- ContinuousFuturesGenerator Class ---
class ContinuousFuturesGenerator:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, db_path: str = DEFAULT_DB_PATH):
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)
        self.config = load_config(self.config_path)
        if not self.config:
            raise ValueError(f"Could not load configuration from {self.config_path}")
        self.conn = self._connect_db()
        self.holidays = {} # Cache holidays per calendar
        # TODO: Make adjustment method configurable per root symbol
        self.adjustment_method = 'none' # Hardcoded to 'none' for now, specifically for VX

    def _connect_db(self):
        # ... existing connect logic ...
        return duckdb.connect(database=str(self.db_path), read_only=False)
        
    def _get_futures_config(self, root_symbol: str) -> Optional[Dict[str, Any]]:
        """Get the configuration for a specific futures root symbol."""
        if not self.config or 'futures' not in self.config:
            logger.error("Futures configuration not found in loaded config.")
            return None
            
        futures_list = self.config.get('futures', [])
        for future_config in futures_list:
            if future_config.get('base_symbol') == root_symbol:
                logger.debug(f"Found config for {root_symbol}")
                return future_config
                
        logger.error(f"Configuration for root symbol '{root_symbol}' not found in futures list.")
        return None
        
    def _get_contract_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get daily contract data for a specific symbol from the market_data table."""
        try:
            query = """
                SELECT 
                    timestamp::DATE as date, -- Cast to date for joining/indexing
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    settle, -- Make sure settle is selected
                    volume,
                    open_interest -- Include open interest
                    -- Add other necessary columns like up_volume, down_volume if needed
                FROM market_data
                WHERE symbol = ?
                AND interval_value = 1 
                AND interval_unit = 'day'
                AND settle > 0 -- Exclude rows with zero or negative settle
            """
            params = [symbol]

            # Optional date filtering
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp"

            df = self.conn.execute(query, params).fetchdf()
            if df.empty:
                # This is now expected if a contract has no data, log as debug or remove
                logger.debug(f"No data found for {symbol} within the specified date range.") 
                # logger.warning(f"No data found for {symbol}") 
            
            # Ensure 'date' column is of date type after fetching
            if not df.empty:
                 df['date'] = pd.to_datetime(df['date']).dt.date
                 
            return df

        except Exception as e:
            logger.error(f"Error getting contract data for {symbol}: {str(e)}")
            return pd.DataFrame()
        
    def _get_available_contracts(self, root_symbol: str) -> List[str]:
        """Get all available individual contracts for a root symbol from the database."""
        try:
            query = """
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE ? 
                  AND symbol NOT LIKE ? -- Exclude continuous contracts like VXc1
                  AND interval_value = 1 
                  AND interval_unit = 'day'
                ORDER BY symbol
            """
            # Parameters: root_symbol%, root_symbolc%
            params = [f"{root_symbol}%", f"{root_symbol}c%"]
            df = self.conn.execute(query, params).fetchdf()
            
            if df.empty:
                logger.warning(f"No contracts found for root symbol {root_symbol} in market_data table.")
                return []

            return df['symbol'].tolist()

        except Exception as e:
            logger.error(f"Error getting available contracts for {root_symbol}: {str(e)}")
            return []

    def _get_expiry_date(self, contract: str, future_config: dict) -> Optional[date]:
        """Get the expiry date for a contract based on config rules."""
        try:
            if 'c' in contract[-3:]:
                logger.debug(f"Skipping continuous contract {contract}")
                return None

            match = re.match(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$", contract)
            if not match:
                logger.error(f"Could not parse contract symbol format: {contract}")
                return None
            
            _, month_code, year_str = match.groups()

            # Handle year conversion (YY -> YYYY)
            year_int = int(year_str)
            if year_int < 70: # Pivot year 70 (e.g., 24 -> 2024, 99 -> 1999)
                year = 2000 + year_int
            else:
                year = 1900 + year_int

            month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                         'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
            month = month_map.get(month_code)
            if not month:
                logger.error(f"Invalid month code {month_code} in contract {contract}")
                return None

            expiry_rule = future_config.get('expiry_rule')
            if not expiry_rule:
                logger.error(f"No expiry rule found for {contract}")
                return None

            # Load holidays if not cached for this calendar
            holiday_calendar = expiry_rule.get('holiday_calendar', 'NYSE') # Default to NYSE
            if holiday_calendar not in self.holidays:
                 self.holidays[holiday_calendar] = get_holidays(self.config, holiday_calendar)
            holidays_set = self.holidays[holiday_calendar]

            day_type = expiry_rule.get('day_type', '').lower()
            day_number = expiry_rule.get('day_number', 0)
            adjust_flag = expiry_rule.get('adjust_for_holiday', False)
            days_before = expiry_rule.get('days_before') # For rules like 'N days before X'
            
            calculated_expiry = None

            # --- Implement different expiry rule types --- 
            # Example: 3rd Friday
            if day_type == 'friday' and day_number == 3:
                calculated_expiry = get_nth_weekday_of_month(year, month, 4, 3) # Friday is 4
            
            # Example: 3rd Wednesday (Simplified - matches old logic if needed)
            elif day_type == 'wednesday' and day_number == 3 and not days_before: # Specific rule for VX
                 # Decide whether to use OLD simple logic or NEW robust logic
                 # For now, implementing the NEW robust logic as requested
                 try:
                    # Standard VIX rule: Wednesday preceding the 3rd Friday
                    third_friday = get_nth_weekday_of_month(year, month, 4, 3) # Friday=4
                    # Find Wednesday before that Friday
                    calculated_expiry = get_previous_weekday(third_friday - timedelta(days=1), 2) # Wednesday=2
                    logger.debug(f"[{contract}] Rule: Wed before 3rd Fri. 3rd Fri={third_friday}, Calculated Exp={calculated_expiry}")
                 except ValueError as e:
                     logger.error(f"[{contract}] Error applying 'Wed before 3rd Fri' rule: {e}")
                     return None

            # Add other rule types as needed (e.g., days_before, specific date)
            # Example: days_before rule (like CL)
            # elif day_type == 'business_day' and days_before is not None:
            #     # Complex logic needed here involving finding a specific date (e.g., 25th)
            #     # and then counting business days backward, considering holidays.
            #     # Placeholder - requires more implementation.
            #     logger.warning(f"'days_before' rule not fully implemented for {contract}")
            #     return None 
                 
            else:
                logger.error(f"Unsupported or ambiguous expiry rule for {contract}: {expiry_rule}")
                return None

            if not calculated_expiry:
                logger.error(f"Could not calculate expiry date for {contract} with rule {expiry_rule}")
                return None

            # Adjust for holidays if specified in config
            final_expiry = calculated_expiry
            if adjust_flag:
                # VIX rule adjusts *backward* from the calculated Wednesday if it's a holiday
                final_expiry = adjust_for_holiday(calculated_expiry, holidays_set, direction=-1)
                if final_expiry != calculated_expiry:
                     logger.info(f"[{contract}] Adjusted expiry from {calculated_expiry} to {final_expiry} due to holiday/weekend.")
            else:
                logger.debug(f"[{contract}] Holiday adjustment not configured or not needed. Final expiry: {final_expiry}")

            return final_expiry # Return date object

        except Exception as e:
            logger.error(f"Error getting expiry date for {contract}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ... (keep _sort_contracts_by_expiry)
    def _sort_contracts_by_expiry(self, contracts: List[str], future_config: dict) -> List[Tuple[str, date]]:
        # ... existing logic, ensure it returns List[Tuple[str, date]] ...
        contract_dates = []
        for contract in contracts:
            expiry_date = self._get_expiry_date(contract, future_config)
            if expiry_date:
                contract_dates.append((contract, expiry_date))
        
        contract_dates.sort(key=lambda x: x[1])
        return contract_dates # Return list of tuples

    # --- Helper to find active contract for a date --- 
    def _get_active_contract_for_date(self, current_date: date, sorted_contracts_with_expiry: List[Tuple[str, date]], contract_number: int) -> Optional[str]:
        """Determine which contract should be active on a specific date for a given contract number.
        
        Args:
            current_date: The date to check
            sorted_contracts_with_expiry: List of (contract, expiry_date) sorted by expiry
            contract_number: Which continuous contract we're building (1 for front month, etc.)
            
        Returns:
            Symbol of the contract that should be active, or None if no appropriate contract
        """
        # Find all contracts whose expiry date is on or after the current date
        # Ensure expiry is a date object for comparison
        active_candidates = [(contract, expiry) for contract, expiry in sorted_contracts_with_expiry 
                             if isinstance(expiry, date) and expiry > current_date]
        
        # If we don't have enough contracts active for the desired number, return None
        if len(active_candidates) < contract_number:
            # logger.debug(f"Not enough active candidates ({len(active_candidates)}) for contract #{contract_number} on {current_date}")
            return None
            
        # Return the nth active contract (0-indexed, so subtract 1)
        return active_candidates[contract_number - 1][0]

    # --- _generate_continuous_future Method --- 
    def _generate_continuous_future(self, root_symbol: str, contract_number: int, generation_start_date_str: Optional[str] = None, generation_end_date_str: Optional[str] = None) -> bool:
        """Generate a continuous future contract for a root symbol using rollover logic.
        
        Args:
            root_symbol: The base symbol (e.g., 'VX').
            contract_number: The contract number (1 for front month, 2 for second, etc.).
            generation_start_date_str: Optional start date (YYYY-MM-DD).
            generation_end_date_str: Optional end date (YYYY-MM-DD).
            
        Returns:
            True if generation was successful, False otherwise.
        """
        continuous_symbol = f"{root_symbol}c{contract_number}"
        logger.info(f"Generating continuous contract {continuous_symbol} (Contract Number: {contract_number})")

        future_config = self._get_futures_config(root_symbol)
        if not future_config:
            logger.error(f"No configuration found for {root_symbol}")
            return False

        # 1. Get all relevant contracts and sort them by expiry
        available_contracts = self._get_available_contracts(root_symbol)
        if not available_contracts:
            logger.warning(f"No individual contracts found for {root_symbol}")
            return False
        
        sorted_contracts_with_expiry = self._sort_contracts_by_expiry(available_contracts, future_config)

        if not sorted_contracts_with_expiry:
            logger.error(f"No contracts with valid expiry dates found for {root_symbol}.")
            return False
            
        logger.info(f"Found {len(sorted_contracts_with_expiry)} contracts with expiry dates for {root_symbol}.")

        # 2. Determine Date Range
        # Get the earliest start date and latest end date from the underlying data
        try:
            date_range_query = """
                SELECT MIN(timestamp::DATE), MAX(timestamp::DATE)
                FROM market_data WHERE symbol LIKE ? AND symbol NOT LIKE ? AND interval_value = 1 AND interval_unit = 'day'
            """
            params = [f"{root_symbol}%", f"{root_symbol}c%"]
            min_max_dates = self.conn.execute(date_range_query, params).fetchone()
            if not min_max_dates or min_max_dates[0] is None:
                logger.error(f"Could not determine date range from market_data for {root_symbol}")
                return False
            data_start_date, data_end_date = min_max_dates[0], min_max_dates[1]
        except Exception as e:
             logger.error(f"Error getting date range for {root_symbol}: {e}")
             return False

        # Parse user-provided start/end dates and override if necessary
        loop_start_date = data_start_date
        if generation_start_date_str:
            try:
                user_start_date = datetime.strptime(generation_start_date_str, '%Y-%m-%d').date()
                loop_start_date = max(data_start_date, user_start_date)
            except ValueError:
                logger.warning(f"Invalid --start-date format: {generation_start_date_str}. Using data start date {data_start_date}.")

        loop_end_date = data_end_date
        if generation_end_date_str:
            try:
                user_end_date = datetime.strptime(generation_end_date_str, '%Y-%m-%d').date()
                loop_end_date = min(data_end_date, user_end_date)
            except ValueError:
                logger.warning(f"Invalid --end-date format: {generation_end_date_str}. Using data end date {data_end_date}.")

        logger.info(f"Generating data for {continuous_symbol} from {loop_start_date} to {loop_end_date}")

        # 3. Iterate through dates and generate data
        all_continuous_data = []
        current_date = loop_start_date
        while current_date <= loop_end_date:
            theoretical_symbol = self._get_active_contract_for_date(
                current_date, 
                sorted_contracts_with_expiry, 
                contract_number
            )

            record_data = None
            underlying_for_record = None

            if theoretical_symbol:
                query_single = "SELECT timestamp::DATE, symbol, open, high, low, close, settle, volume, open_interest FROM market_data WHERE symbol = ? AND timestamp::DATE = ? AND interval_value = 1 AND interval_unit = 'day' AND settle > 0 LIMIT 1"
                try:
                    result = self.conn.execute(query_single, [theoretical_symbol, current_date]).fetchone()
                    if result:
                        record_data = result
                        underlying_for_record = theoretical_symbol
                except Exception as e:
                    logger.error(f"DB error querying {theoretical_symbol} on {current_date}: {e}")
                    result = None # Treat DB error as missing data

            # Append data (either found or None/NaN)
            all_continuous_data.append({
                'timestamp': current_date,
                'symbol': continuous_symbol,
                'open': record_data[2] if record_data else None,
                'high': record_data[3] if record_data else None,
                'low': record_data[4] if record_data else None,
                'close': record_data[5] if record_data else None,
                'volume': record_data[7] if record_data else None,
                'settle': record_data[6] if record_data else None,
                'open_interest': record_data[8] if record_data else None,
                'underlying_symbol': underlying_for_record
            })

            current_date += timedelta(days=1)
            
        # 4. Prepare DataFrame and Insert into database using DuckDB connection
        if not all_continuous_data:
            logger.warning(f"No data generated for {continuous_symbol} in the date range.")
            return True # Not an error, just no data

        continuous_df = pd.DataFrame(all_continuous_data)
        continuous_df['timestamp'] = pd.to_datetime(continuous_df['timestamp'])
        logger.info(f"Generated {len(continuous_df)} data points for {continuous_symbol}.")

        try:
            # Use DuckDB's capability to insert from a Pandas DataFrame
            # The deletion is handled in the run method based on force flag
            logger.info(f"Inserting data for {continuous_symbol} into continuous_contracts table...")
            # Ensure DataFrame columns match table schema or adjust as needed
            # For simplicity, assuming direct insertion works if columns are named correctly
            
            # Construct column names string for insertion query
            # Filter df columns to only those expected in the table to avoid errors
            expected_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest', 'underlying_symbol']
            cols_to_insert = [col for col in expected_cols if col in continuous_df.columns]
            col_names_str = ", ".join([f'"{col}"' for col in cols_to_insert])
            
            # Register DataFrame as a temporary view
            temp_view_name = f"temp_cont_insert_{continuous_symbol}" 
            self.conn.register(temp_view_name, continuous_df[cols_to_insert])
            
            # Insert data from the view
            insert_sql = f"INSERT INTO continuous_contracts ({col_names_str}) SELECT {col_names_str} FROM {temp_view_name}"
            self.conn.execute(insert_sql)
            self.conn.unregister(temp_view_name) # Clean up view

            logger.info(f"Successfully wrote {len(continuous_df)} rows for {continuous_symbol}.")
            return True
        except Exception as e:
            logger.error(f"Failed to write data for {continuous_symbol} to database: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace for debugging
            return False
            

    def run(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, force: bool = False):
        # ... (rest of run method remains similar, but ensure it uses the correct end_date) ...
        # Parse root and contract number
        if 'c' not in symbol or not symbol[-1].isdigit():
            logger.error(f"Invalid continuous symbol format: {symbol}. Expected format like 'VXc1'.")
            return
        
        root_symbol = symbol.split('c')[0]
        try:
            contract_number = int(symbol[-1])
        except ValueError:
            logger.error(f"Could not parse contract number from {symbol}")
            return
            
        logger.info(f"Running generation for {symbol} (Root: {root_symbol}, Number: {contract_number})")
        
        # Handle force update - Delete existing data for this continuous symbol
        if force:
            logger.warning(f"--force specified. Deleting existing data for {symbol} from continuous_contracts.")
            try:
                delete_query = "DELETE FROM continuous_contracts WHERE symbol = ?"
                self.conn.execute(delete_query, [symbol])
                logger.info(f"Successfully deleted existing data for {symbol}.")
            except Exception as e:
                logger.error(f"Failed to delete existing data for {symbol}: {e}")
                # Optionally, decide whether to proceed or stop if deletion fails
                # return 
                
        # Generate the data using the refactored logic
        success = self._generate_continuous_future(root_symbol, contract_number, start_date, end_date)
        
        if success:
            logger.info(f"Continuous contract generation for {symbol} completed successfully.")
        else:
            logger.error(f"Continuous contract generation for {symbol} failed.")

    def close(self):
        # ... existing close logic ...
        pass

# --- Main Execution --- 
def main():
    parser = argparse.ArgumentParser(description='Generate continuous futures contracts based on underlying market data.')
    parser.add_argument('--root-symbol', required=True, help='Root symbol of the futures contract (e.g., VX, ES)')
    parser.add_argument('--config', default='config/market_symbols.yaml', help='Path to the configuration file.')
    parser.add_argument('--db-path', default='data/financial_data.duckdb', help='Path to the DuckDB database file.')
    parser.add_argument('--num-contracts', type=int, default=None, help='Number of continuous contracts to generate (overrides config if specified).')
    parser.add_argument('--start-date', type=str, default=None, help='Optional start date (YYYY-MM-DD) for generation period.')
    parser.add_argument('--end-date', type=str, default=None, help='Optional end date (YYYY-MM-DD) for generation period.') # Added end-date argument
    parser.add_argument('--force', action='store_true', help='Force overwrite of existing continuous contract data.') # Added force argument

    args = parser.parse_args()

    logger.info("Starting continuous futures generation script...")
    generator = ContinuousFuturesGenerator(args.config, args.db_path)
    
    # Determine number of contracts to generate
    num_to_generate = args.num_contracts
    if num_to_generate is None:
        future_config = generator._get_futures_config(args.root_symbol)
        if future_config:
            num_to_generate = future_config.get('num_active_contracts', 1)
        else:
            num_to_generate = 1 # Default to 1 if config not found
            logger.warning(f"Config not found for {args.root_symbol}, defaulting to generating 1 contract.")
            
    logger.info(f"Generating {num_to_generate} continuous contract(s) for {args.root_symbol}.")

    # Loop to generate each required continuous contract
    for i in range(1, num_to_generate + 1):
        continuous_symbol = f"{args.root_symbol}c{i}"
        # Pass the parsed arguments to the run method
        generator.run(continuous_symbol, args.start_date, args.end_date, args.force)
    
    generator.close()
    logger.info("Continuous futures generation script finished.")

if __name__ == "__main__":
    # Setup logging
    log_level_str = os.environ.get('LOGGING_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Silence noisy loggers if necessary
    # logging.getLogger('duckdb').setLevel(logging.WARNING)
    
    main() 