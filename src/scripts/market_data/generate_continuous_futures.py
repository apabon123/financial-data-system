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

# Add project root to Python path
# ... (existing path logic)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "market_symbols.yaml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "financial_data.duckdb"

# --- Date Helper Functions (Copied/Adapted from verify_vx_continuous.py) ---
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
                    volume,
                    open_interest -- Include open interest
                    -- Add other necessary columns like up_volume, down_volume if needed
                FROM market_data
                WHERE symbol = ?
                AND interval_value = 1 
                AND interval_unit = 'day'
            """
            params = [symbol]

            # Optional date filtering (though usually we fetch all data for a contract)
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

    # --- _generate_continuous_future Method --- 
    def _generate_continuous_future(self, root_symbol: str, contract_number: int) -> pd.DataFrame:
        """Generate a continuous future contract for a root symbol using rollover logic."""
        continuous_symbol = f"{root_symbol}c{contract_number}" # Define continuous_symbol here
        logger.info(f"Generating continuous contract {continuous_symbol} (Contract Number: {contract_number})")

        future_config = self._get_futures_config(root_symbol)
        if not future_config:
            logger.error(f"No configuration found for {root_symbol}")
            return pd.DataFrame()

        # 1. Get all relevant contracts and sort them by expiry
        available_contracts = self._get_available_contracts(root_symbol) # Ensure this line exists and is active
        if not available_contracts:
            logger.warning(f"No individual contracts found for {root_symbol}")
            return pd.DataFrame()
        
        sorted_contracts_with_expiry = self._sort_contracts_by_expiry(available_contracts, future_config)

        if not sorted_contracts_with_expiry:
            logger.error(f"No contracts with valid expiry dates found for {root_symbol}.")
            return pd.DataFrame()
            
        logger.info(f"Found {len(sorted_contracts_with_expiry)} contracts with expiry dates for {root_symbol}.")

        # 2. Get all unique trading dates for the root symbol from daily data
        query_dates = """ 
            SELECT DISTINCT timestamp::DATE as date
            FROM market_data
            WHERE symbol LIKE ?
              AND interval_value = 1 AND interval_unit = 'day'
              AND symbol NOT LIKE '%c%' -- Exclude continuous symbols themselves
            ORDER BY date
        """
        try:
            all_dates_df = self.conn.execute(query_dates, [f"{root_symbol}%"]).fetchdf()
            if all_dates_df.empty:
                logger.warning(f"No trading dates found for {root_symbol}")
                return pd.DataFrame()
            all_dates = all_dates_df['date'].tolist()
            logger.info(f"Found {len(all_dates)} unique trading dates for {root_symbol}.")
        except Exception as e:
            logger.error(f"Error fetching trading dates for {root_symbol}: {e}")
            return pd.DataFrame()
            
        # Ensure all_dates contains date objects
        all_dates = [d.date() if isinstance(d, pd.Timestamp) else d for d in all_dates]

        # 3. Build the continuous series
        continuous_data = []
        if contract_number > len(sorted_contracts_with_expiry):
            logger.warning(f"Contract number {contract_number} exceeds the number of available contracts ({len(sorted_contracts_with_expiry)}). Cannot generate {continuous_symbol}.")
            return pd.DataFrame()

        # Determine the initial contract (contract_number'th contract in the sorted list)
        initial_contract_index = contract_number - 1
        active_contract, expiry_date = sorted_contracts_with_expiry[initial_contract_index]
        logger.info(f"Starting {continuous_symbol} generation using initial contract {active_contract} (expires {expiry_date})")

        # Get data for the initial contract
        current_contract_data_df = self._get_contract_data(active_contract)
        if current_contract_data_df.empty:
            logger.error(f"No data found for initial contract {active_contract}. Cannot generate {continuous_symbol}.")
            return pd.DataFrame()
            
        # Ensure index is date
        if not isinstance(current_contract_data_df.index, pd.DatetimeIndex):
             current_contract_data_df.index = pd.to_datetime(current_contract_data_df.index)
        current_contract_data_df.index = current_contract_data_df.index.date
            
        # Find the start date for the continuous series
        series_start_date = current_contract_data_df.index.min()
        if not series_start_date or not isinstance(series_start_date, date):
             logger.error(f"Could not determine valid start date (type: {type(series_start_date)}) for {active_contract}. Cannot generate {continuous_symbol}.")
             return pd.DataFrame()
             
        logger.info(f"Starting {continuous_symbol} generation from {series_start_date} (first data for initial contract {active_contract})")

        # Iterate through all trading dates starting from the series start date
        # Ensure comparison is between date objects
        relevant_dates = [d for d in all_dates if isinstance(d, date) and d >= series_start_date]
        if len(relevant_dates) != len([d for d in all_dates if d >= series_start_date]):
            logger.warning("Some dates were filtered out during type checking before comparison.")
        
        next_contract_index = initial_contract_index + 1 # Prepare for the first potential rollover

        for current_date in relevant_dates:
            # Check if it's time to roll over (ON the expiry date of the current contract)
            if current_date == expiry_date: # Roll on expiry day (compare date objects)
                if next_contract_index < len(sorted_contracts_with_expiry):
                    prev_contract = active_contract
                    prev_expiry = expiry_date
                    # Switch to the next contract
                    active_contract, expiry_date = sorted_contracts_with_expiry[next_contract_index]
                    logger.info(f"{continuous_symbol}: Rolled over on {current_date}. Previous contract {prev_contract} expired {prev_expiry}. New active contract: {active_contract} expiring {expiry_date}")
                    
                    # Fetch data for the new active contract
                    current_contract_data_df = self._get_contract_data(active_contract)
                    if not current_contract_data_df.empty:
                         current_contract_data_df['date'] = pd.to_datetime(current_contract_data_df['date']).dt.date
                         current_contract_data_df = current_contract_data_df.set_index('date')
                    else:
                         logger.warning(f"No data found for new active contract {active_contract} after rollover on {current_date}. Gaps may occur.")
                         # Keep using old data? No, better to represent the gap.
                         current_contract_data_df = pd.DataFrame() # Ensure loop below handles empty df
                         
                    next_contract_index += 1
                else:
                    logger.info(f"Reached end of available contracts for {continuous_symbol} after expiry of {active_contract} on {current_date}. Stopping generation.")
                    break # Stop processing dates

            # Get data for the current date from the currently active contract's data
            if not current_contract_data_df.empty and current_date in current_contract_data_df.index:
                row_data = current_contract_data_df.loc[current_date].to_dict()
                row_data['date'] = current_date # Ensure date is included
                row_data['underlying_symbol'] = active_contract # Add the underlying symbol
                continuous_data.append(row_data)
            else:
                # Handle missing data for the active contract on this date
                logger.warning(f"No data found for active contract {active_contract} on {current_date}. Leaving gap in {continuous_symbol}.")
                # Optional: Add a row with NaNs or skip?
                # For now, we skip, creating a gap in the continuous series.
                pass
                
        if not continuous_data:
            logger.warning(f"No data generated for {continuous_symbol}. Returning empty DataFrame.")
            return pd.DataFrame()

        # Create DataFrame
        result_df = pd.DataFrame(continuous_data)
        
        # Ensure correct column order and add missing columns with defaults
        final_columns = [
            'date', 'symbol', 'underlying_symbol', 'open', 'high', 
            'low', 'close', 'volume', 'open_interest', 'up_volume', 
            'down_volume', 'source', 'interval_value', 'interval_unit', 
            'adjusted', 'quality'
        ]
        result_df['symbol'] = continuous_symbol
        result_df['source'] = 'continuous'
        result_df['interval_value'] = 1
        result_df['interval_unit'] = 'day'
        result_df['adjusted'] = True
        result_df['quality'] = 100
        result_df['timestamp'] = pd.to_datetime(result_df['date']) # Add timestamp column from date
        result_df = result_df.drop(columns=['date']) # Drop the temporary date column
        
        # Add any missing columns expected by the schema with default None/NaN
        for col in final_columns:
             if col not in result_df.columns and col != 'date': # date was handled via timestamp
                 result_df[col] = None # Or appropriate default (e.g., 0 for volume? Check schema)
        
        # Reorder columns to match final_columns, handling potential missing ones added above
        # Filter final_columns to only those present in result_df to avoid errors
        ordered_cols = [col for col in final_columns if col in result_df.columns and col != 'date'] + [col for col in result_df.columns if col not in final_columns]
        result_df = result_df[ordered_cols]
        
        # Convert types to match schema as best as possible before insertion
        # (DuckDB often handles this, but good practice)
        # Example: result_df['volume'] = result_df['volume'].astype('Int64') # Use nullable integer

        return result_df

    # ... (keep insert_continuous_contract, generate_for_symbol, close)
    def insert_continuous_contract(self, df: pd.DataFrame, continuous_symbol: str):
        # ... existing logic ...
        pass
        
    def generate_for_symbol(self, root_symbol: str, num_contracts: int):
        """Generate the specified number of continuous contracts for a root symbol."""
        logger.info(f"Starting generation for {root_symbol}, num_contracts={num_contracts}") # Added log
        future_config = self._get_futures_config(root_symbol)
        if not future_config:
            logger.error(f"No configuration found for {root_symbol}. Skipping generation.")
            return
            
        configured_num = future_config.get('num_active_contracts', 1)
        if num_contracts > configured_num:
             logger.warning(f"Requested {num_contracts} contracts, but config only specifies {configured_num} active contracts for {root_symbol}. Generating {configured_num}.")
             num_to_generate = configured_num
        else:
             num_to_generate = num_contracts

        for i in range(1, num_to_generate + 1):
            df = self._generate_continuous_future(root_symbol, i)
            if not df.empty:
                continuous_symbol = f"{root_symbol}c{i}"
                self.insert_continuous_contract(df, continuous_symbol)
            else:
                 logger.warning(f"Generation returned empty DataFrame for {root_symbol}c{i}. Skipping insertion.")

    def close(self):
        # ... existing logic ...
        pass

# --- Main Execution --- 
if __name__ == "__main__":
    logger.info("Starting continuous futures generation script...") # Added log
    parser = argparse.ArgumentParser(description='Generate continuous futures contracts')
    parser.add_argument('--root-symbol', type=str, required=True, help='Root symbol (e.g. VX for VIX futures)')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='Path to market symbols config file')
    parser.add_argument('--db-path', type=str, default=str(DEFAULT_DB_PATH), help='Path to database file')
    parser.add_argument('--num-contracts', type=int, default=2, help='Number of continuous contracts to generate (e.g., 2 for c1, c2)')

    args = parser.parse_args()
    logger.info(f"Arguments: {args}") # Added log

    try:
        generator = ContinuousFuturesGenerator(config_path=args.config, db_path=args.db_path)
        logger.info("Generator initialized.") # Added log
        generator.generate_for_symbol(args.root_symbol, args.num_contracts)
        generator.close()
        logger.info("Continuous futures generation completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during generation: {e}")
        import traceback
        traceback.print_exc()

# ... (keep class methods) ...

    def generate_for_symbol(self, root_symbol: str, num_contracts: int):
        """Generate the specified number of continuous contracts for a root symbol."""
        logger.info(f"Starting generation for {root_symbol}, num_contracts={num_contracts}") # Added log
        future_config = self._get_futures_config(root_symbol)
        if not future_config:
            logger.error(f"No configuration found for {root_symbol}. Skipping generation.")
            return
            
        configured_num = future_config.get('num_active_contracts', 1)
        if num_contracts > configured_num:
             logger.warning(f"Requested {num_contracts} contracts, but config only specifies {configured_num} active contracts for {root_symbol}. Generating {configured_num}.")
             num_to_generate = configured_num
        else:
             num_to_generate = num_contracts

        for i in range(1, num_to_generate + 1):
            df = self._generate_continuous_future(root_symbol, i)
            if not df.empty:
                continuous_symbol = f"{root_symbol}c{i}"
                self.insert_continuous_contract(df, continuous_symbol)
            else:
                 logger.warning(f"Generation returned empty DataFrame for {root_symbol}c{i}. Skipping insertion.")

# ... (keep class methods) ... 