#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates the VIX Futures Roll Calendar based on CBOE rules.

Calculates the Final Settlement Date and Last Trading Day for VIX futures
for a specified year range and outputs to a CSV file.
"""

import os
import sys
import pandas as pd
import yaml
import logging
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import argparse
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay # Removed Juneteenth, IndependenceDay, ChristmasDay

# --- Configuration ---
DEFAULT_CONFIG_PATH = "config/market_symbols.yaml"
DEFAULT_OUTPUT_DIR = "data/roll_calendars"
DEFAULT_START_YEAR = 2004
DEFAULT_END_YEAR = date.today().year + 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Holiday Loading ---
def load_holidays(config_path: str, start_year: int, end_year: int) -> set:
    """Loads US holidays based on rules defined in the YAML config file for a given year range."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        us_holidays_set = set()
        if 'settings' in config and 'holidays' in config['settings'] and 'NYSE' in config['settings']['holidays']:
            holiday_rules = config['settings']['holidays']['NYSE']
            logger.info(f"Loading holiday rules from settings.holidays.NYSE in {config_path}")

            # Define a basic holiday calendar using pandas rules for standard observances
            # We'll primarily use this for the observance logic implicit in pandas Holidays
            class RuleBasedUSHolidayCalendar(AbstractHolidayCalendar):
                rules = [
                    Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
                    USMartinLutherKingJr,
                    USPresidentsDay,
                    GoodFriday,
                    USMemorialDay,
                    USLaborDay,
                    USThanksgivingDay,
                    # USChristmasDay removed due to import issues
                ]

            cal = RuleBasedUSHolidayCalendar()
            # Generate holidays for the required range using the pandas calendar
            pd_holidays = cal.holidays(start=date(start_year, 1, 1), end=date(end_year, 12, 31))
            # Convert pandas Timestamps to datetime.date objects
            us_holidays_set = set(pd_holidays.date)

            # Manually add Juneteenth (observed) from 2021 onwards
            juneteenth_start_year = 2021
            for year in range(max(start_year, juneteenth_start_year), end_year + 1):
                juneteenth_date = date(year, 6, 19)
                # Observe nearest weekday (Friday if Sat, Monday if Sun)
                weekday = juneteenth_date.weekday()
                if weekday == 5: # Saturday
                    observed_date = juneteenth_date - timedelta(days=1)
                elif weekday == 6: # Sunday
                    observed_date = juneteenth_date + timedelta(days=1)
                else: # Weekday
                    observed_date = juneteenth_date
                
                # Basic check
                if observed_date.year == year:
                   us_holidays_set.add(observed_date)
                   logger.debug(f"Added manually calculated observed Juneteenth for {year}: {observed_date}")

            # Manually add Independence Day (observed)
            for year in range(start_year, end_year + 1):
                independence_day_date = date(year, 7, 4)
                weekday = independence_day_date.weekday()
                if weekday == 5: # Saturday
                    observed_date = independence_day_date - timedelta(days=1) # Observed Friday
                elif weekday == 6: # Sunday
                    observed_date = independence_day_date + timedelta(days=1) # Observed Monday
                else: # Weekday
                    observed_date = independence_day_date
                
                # Basic check
                if observed_date.year == year: 
                    us_holidays_set.add(observed_date)
                    logger.debug(f"Added manually calculated observed Independence Day for {year}: {observed_date}")

            # Manually add Christmas Day (observed)
            for year in range(start_year, end_year + 1):
                christmas_date = date(year, 12, 25)
                weekday = christmas_date.weekday()
                if weekday == 5: # Saturday
                    observed_date = christmas_date - timedelta(days=1) # Observed Friday
                elif weekday == 6: # Sunday
                    observed_date = christmas_date + timedelta(days=1) # Observed Monday
                else: # Weekday
                    observed_date = christmas_date
                
                # Basic check to avoid adding if it pushed observance out of the current year
                if observed_date.year == year:
                    us_holidays_set.add(observed_date)
                    logger.debug(f"Added manually calculated observed Christmas Day for {year}: {observed_date}")

            # Note: The YAML structure with fixed/relative dates isn't directly used here anymore.
            # We rely on pandas built-in holiday rules which cover standard US holidays.
            # If the YAML rules were significantly different or custom, we'd need custom parsing logic here.
            logger.info(f"Generated {len(us_holidays_set)} US holidays from {start_year} to {end_year} using pandas rules and manual Juneteenth/Independence/Christmas Day.")
            return us_holidays_set

        else:
            logger.warning(f"'settings.holidays.NYSE' section not found in {config_path}. Proceeding without holidays.")
            return set()
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing config from {config_path}: {e}")
        sys.exit(1)

# --- Date Calculation Helpers ---

def is_business_day(target_date: date, holidays: set) -> bool:
    """Checks if a date is a business day (Mon-Fri and not a holiday)."""
    return target_date.weekday() < 5 and target_date not in holidays

def get_previous_business_day(target_date: date, holidays: set) -> date:
    """Finds the business day immediately preceding the target_date."""
    prev_day = target_date - timedelta(days=1)
    while not is_business_day(prev_day, holidays):
        prev_day -= timedelta(days=1)
    return prev_day

def get_nth_weekday_of_month(year: int, month: int, weekday_int: int, n: int) -> date:
    """
    Gets the date of the nth occurrence of a specific weekday in a given month and year.
    weekday_int: 0=Mon, 1=Tue, ..., 6=Sun
    n: 1 for 1st, 2 for 2nd, etc.
    """
    if not (0 <= weekday_int <= 6):
        raise ValueError("weekday_int must be between 0 and 6")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    first_day_of_month = date(year, month, 1)
    day_of_week_first = first_day_of_month.weekday() # 0=Mon, 6=Sun

    # Calculate days needed to reach the first occurrence of the target weekday
    days_to_add = (weekday_int - day_of_week_first + 7) % 7
    
    first_occurrence_date = first_day_of_month + timedelta(days=days_to_add)
    
    # Add (n-1) weeks to get the nth occurrence
    nth_occurrence_date = first_occurrence_date + timedelta(weeks=n - 1)
    
    # Check if the calculated date is still in the same month
    if nth_occurrence_date.month != month:
        raise ValueError(f"The {n}th weekday {weekday_int} does not exist in {year}-{month:02d}")
        
    return nth_occurrence_date

def get_third_friday(year: int, month: int) -> date:
    """Return the date of the third Friday of the given month and year."""
    return get_nth_weekday_of_month(year, month, 4, 3) # Friday is 4

# --- VIX Expiration Logic ---

# Mapping from month number to futures code
FUTURES_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

def get_contract_code(symbol: str, year: int, month: int) -> str:
    """Generates the standard futures contract code (e.g., VXF05)."""
    if month not in FUTURES_MONTH_CODES:
        raise ValueError(f"Invalid month number: {month}")
    month_code = FUTURES_MONTH_CODES[month]
    year_short = str(year)[-2:] # Get last two digits of the year
    return f"{symbol.upper()}{month_code}{year_short}"

def calculate_vix_final_settlement(contract_year: int, contract_month: int, holidays: set) -> date:
    """
    Calculates the Final Settlement Date for a VIX contract expiring in a given month.
    Rule: Wednesday 30 days prior to the 3rd Friday of the *following* month.
    Holiday Adjustment: If calculated Wed OR the 3rd Fri (30 days later) is a holiday,
                      settlement moves to the business day preceding the calculated Wed.
    """
    contract_date = date(contract_year, contract_month, 1)
    
    # Determine the month *following* the contract expiry month
    next_month_date = contract_date + relativedelta(months=1)
    next_month_year = next_month_date.year
    next_month_month = next_month_date.month
    
    # Find the third Friday of that following month
    third_friday_next_month = get_third_friday(next_month_year, next_month_month)
    
    # Calculate the date 30 calendar days prior
    calculated_settlement_date = third_friday_next_month - timedelta(days=30)

    # Check the holiday rule
    # If the calculated settlement date (typically a Wed) OR the 3rd Friday are holidays
    if not is_business_day(calculated_settlement_date, holidays) or not is_business_day(third_friday_next_month, holidays):
        # Move settlement to the business day *before* the calculated settlement date
        final_settlement_date = get_previous_business_day(calculated_settlement_date, holidays)
        logger.debug(f"Contract {contract_year}-{contract_month:02d}: Adjusted settlement from {calculated_settlement_date} to {final_settlement_date} due to holiday rule.")
    else:
        # No adjustment needed
        final_settlement_date = calculated_settlement_date
        logger.debug(f"Contract {contract_year}-{contract_month:02d}: Calculated settlement {final_settlement_date} (no holiday adjustment).")

    return final_settlement_date

def calculate_last_trading_day(final_settlement_date: date, holidays: set) -> date:
    """
    Calculates the Last Trading Day.
    Rule: Business day immediately preceding the Final Settlement Date.
    """
    return get_previous_business_day(final_settlement_date, holidays)

# --- Main Generation ---

def generate_calendar(start_year: int, end_year: int, holidays: set) -> pd.DataFrame:
    """Generates the VIX roll calendar DataFrame."""
    calendar_data = []
    root_symbol = "VX" # Assuming VX for this script specificially
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            try:
                contract_month_str = f"{year}-{month:02d}"
                logger.debug(f"Calculating for contract month: {contract_month_str}")
                
                final_settlement = calculate_vix_final_settlement(year, month, holidays)
                last_trading = calculate_last_trading_day(final_settlement, holidays)
                contract_code = get_contract_code(root_symbol, year, month)
                
                calendar_data.append({
                    "ContractMonth": contract_month_str,
                    "ContractCode": contract_code,
                    "FinalSettlementDate": final_settlement,
                    "LastTradingDay": last_trading
                })
            except Exception as e:
                logger.error(f"Error calculating dates for {year}-{month:02d}: {e}")

    df = pd.DataFrame(calendar_data)
    df['FinalSettlementDate'] = pd.to_datetime(df['FinalSettlementDate'])
    df['LastTradingDay'] = pd.to_datetime(df['LastTradingDay'])
    # Ensure correct column order
    df = df[['ContractMonth', 'ContractCode', 'FinalSettlementDate', 'LastTradingDay']]
    df = df.sort_values(by="FinalSettlementDate").reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate VIX Futures Roll Calendar.')
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR, help='Start year for the calendar.')
    parser.add_argument('--end-year', type=int, default=DEFAULT_END_YEAR, help='End year for the calendar.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the market symbols YAML config file.')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save the output CSV file.')
    parser.add_argument('--output-file', type=str, default='vix_roll_calendar.csv', help='Name of the output CSV file.')
    
    args = parser.parse_args()

    logger.info(f"Generating VIX roll calendar from {args.start_year} to {args.end_year}")
    
    # Load holidays for the specified range
    holidays = load_holidays(args.config, args.start_year, args.end_year)
    
    # Generate calendar
    calendar_df = generate_calendar(args.start_year, args.end_year, holidays)
    
    # Prepare output
    output_path = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save to CSV
    try:
        calendar_df.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        logger.info(f"Successfully saved VIX roll calendar to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save calendar to {output_path}: {e}")

if __name__ == "__main__":
    main() 