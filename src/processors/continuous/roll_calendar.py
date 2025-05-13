"""
Roll calendar handling logic for continuous futures contracts.

This module provides functions and classes for managing roll calendars, including:
- Loading and validating roll calendar data from the database
- Generating roll dates based on different methodologies (volume, open interest, calendar)
- Optimizing roll decisions based on liquidity and market conditions
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Set, Any

# Logger for this module
logger = logging.getLogger(__name__)

class RollCalendar:
    """Roll calendar management for continuous futures contracts."""
    
    def __init__(
        self,
        root_symbol: str,
        db_connector = None,
        roll_strategy: str = 'volume',
        roll_days_offset: int = 0,
        roll_time: str = 'close',
        expiration_threshold: int = 5,
        min_volume_ratio: float = 2.0,
        min_open_interest_ratio: float = 1.5
    ):
        """
        Initialize the roll calendar manager.
        
        Args:
            root_symbol: Root symbol for the futures contract (e.g., 'ES', 'VX')
            db_connector: Database connector instance
            roll_strategy: Roll strategy ('volume', 'calendar', 'oi', 'custom')
            roll_days_offset: Days to offset the roll from the detected optimal date
            roll_time: Roll timing ('close', 'open', 'auto')
            expiration_threshold: Days before expiration to limit roll consideration
            min_volume_ratio: Minimum volume ratio threshold for volume-based rolls
            min_open_interest_ratio: Minimum OI ratio threshold for OI-based rolls
        """
        self.root_symbol = root_symbol
        self.db_connector = db_connector
        self.roll_strategy = roll_strategy
        self.roll_days_offset = roll_days_offset
        self.roll_time = roll_time
        self.expiration_threshold = expiration_threshold
        self.min_volume_ratio = min_volume_ratio
        self.min_open_interest_ratio = min_open_interest_ratio
        
        # Internal storage
        self._calendar_df = None
        self._roll_pairs = []
        self._roll_history = {}
        
    def load_from_database(self) -> bool:
        """
        Load roll calendar from the database.
        
        Returns:
            True if calendar was loaded successfully, False otherwise
        """
        if not self.db_connector:
            logger.error("No database connector provided. Cannot load roll calendar.")
            return False
        
        try:
            # Query the roll calendar table
            query = """
                SELECT contract_code, last_trading_day, roll_date, expiration_date, 
                       roll_method, first_notice_day, active_trading_start
                FROM futures_roll_calendar
                WHERE root_symbol = ?
                ORDER BY expiration_date ASC
            """
            calendar_df = self.db_connector.query(query, [self.root_symbol])
            
            if calendar_df.empty:
                logger.warning(f"No roll calendar entries found for {self.root_symbol}.")
                return False
            
            # Process date columns
            date_columns = ['last_trading_day', 'roll_date', 'expiration_date', 
                           'first_notice_day', 'active_trading_start']
            
            for col in date_columns:
                if col in calendar_df.columns:
                    calendar_df[col] = pd.to_datetime(calendar_df[col])
            
            # Store the calendar
            self._calendar_df = calendar_df
            
            logger.info(f"Loaded roll calendar with {len(calendar_df)} entries for {self.root_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading roll calendar for {self.root_symbol}: {e}", exc_info=True)
            return False
    
    def create_from_market_data(self, 
                              start_date: Optional[Union[str, datetime, date]] = None,
                              end_date: Optional[Union[str, datetime, date]] = None) -> bool:
        """
        Create roll calendar from market data based on the specified strategy.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            True if calendar was created successfully, False otherwise
        """
        if not self.db_connector:
            logger.error("No database connector provided. Cannot create roll calendar.")
            return False
        
        try:
            # Convert dates to standard format
            start_date_str = None
            if start_date:
                if isinstance(start_date, (datetime, date)):
                    start_date_str = start_date.strftime('%Y-%m-%d')
                else:
                    start_date_str = start_date
            
            end_date_str = None
            if end_date:
                if isinstance(end_date, (datetime, date)):
                    end_date_str = end_date.strftime('%Y-%m-%d')
                else:
                    end_date_str = end_date
            
            # Load all contracts for this root symbol
            contracts = self._get_contracts_for_symbol(start_date_str, end_date_str)
            
            if not contracts:
                logger.warning(f"No contracts found for {self.root_symbol} to generate roll calendar.")
                return False
            
            # Load market data for these contracts
            market_data = self._load_contract_data(contracts, start_date_str, end_date_str)
            
            if not market_data or all(df.empty for df in market_data.values()):
                logger.warning(f"No market data loaded for {self.root_symbol} contracts.")
                return False
            
            # Determine roll dates based on the specified strategy
            if self.roll_strategy == 'volume':
                calendar_entries = self._generate_volume_based_rolls(market_data)
            elif self.roll_strategy == 'oi':
                calendar_entries = self._generate_oi_based_rolls(market_data)
            elif self.roll_strategy == 'calendar':
                calendar_entries = self._generate_calendar_based_rolls(contracts)
            else:
                logger.error(f"Unsupported roll strategy: {self.roll_strategy}")
                return False
            
            if not calendar_entries:
                logger.warning(f"Failed to generate roll calendar entries for {self.root_symbol}.")
                return False
            
            # Create DataFrame from entries
            self._calendar_df = pd.DataFrame(calendar_entries)
            
            # Process date columns
            date_columns = ['last_trading_day', 'roll_date', 'expiration_date', 
                           'first_notice_day', 'active_trading_start']
            
            for col in date_columns:
                if col in self._calendar_df.columns:
                    self._calendar_df[col] = pd.to_datetime(self._calendar_df[col])
            
            logger.info(f"Created roll calendar with {len(self._calendar_df)} entries for {self.root_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating roll calendar for {self.root_symbol}: {e}", exc_info=True)
            return False
    
    def save_to_database(self) -> bool:
        """
        Save the roll calendar to the database.
        
        Returns:
            True if calendar was saved successfully, False otherwise
        """
        if not self.db_connector:
            logger.error("No database connector provided. Cannot save roll calendar.")
            return False
        
        if self._calendar_df is None or self._calendar_df.empty:
            logger.error("No roll calendar data to save.")
            return False
        
        try:
            # Prepare data for insertion
            insert_data = self._calendar_df.copy()
            
            # Ensure root_symbol column exists
            if 'root_symbol' not in insert_data.columns:
                insert_data['root_symbol'] = self.root_symbol
            
            # Format date columns
            date_columns = ['last_trading_day', 'roll_date', 'expiration_date', 
                           'first_notice_day', 'active_trading_start']
            
            for col in date_columns:
                if col in insert_data.columns:
                    insert_data[col] = insert_data[col].dt.strftime('%Y-%m-%d')
            
            # Get column names and values
            columns = insert_data.columns.tolist()
            columns_str = ', '.join([f'"{col}"' for col in columns])
            placeholders = ', '.join(['?' for _ in columns])
            
            # First delete existing entries
            delete_query = """
                DELETE FROM futures_roll_calendar
                WHERE root_symbol = ?
            """
            self.db_connector.execute(delete_query, [self.root_symbol])
            
            # Insert new entries
            insert_query = f"""
                INSERT INTO futures_roll_calendar ({columns_str})
                VALUES ({placeholders})
            """
            
            # Execute insert for each row
            for _, row in insert_data.iterrows():
                values = [row[col] for col in columns]
                self.db_connector.execute(insert_query, values)
            
            # Commit changes
            self.db_connector.commit()
            
            logger.info(f"Saved {len(insert_data)} roll calendar entries for {self.root_symbol} to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving roll calendar for {self.root_symbol}: {e}", exc_info=True)
            return False
    
    def get_active_contract_for_date(self, 
                                   target_date: Union[str, datetime, date],
                                   position: int = 1) -> Optional[str]:
        """
        Get the active contract for a specific date and position.
        
        Args:
            target_date: Date to check
            position: Contract position (1 for front month, 2 for second month, etc.)
            
        Returns:
            Contract code or None if not found
        """
        if self._calendar_df is None or self._calendar_df.empty:
            logger.error("No roll calendar data available. Cannot determine active contract.")
            return None
        
        try:
            # Convert target_date to datetime
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
            elif isinstance(target_date, date):
                target_date = datetime.combine(target_date, datetime.min.time())
            
            # Ensure timezone-naive datetime for comparison
            target_date = pd.Timestamp(target_date).normalize()
            
            # Ensure the calendar has roll_date column
            if 'roll_date' not in self._calendar_df.columns:
                logger.warning("Roll calendar missing 'roll_date' column. Using 'last_trading_day' as fallback.")
                date_column = 'last_trading_day'
            else:
                date_column = 'roll_date'
            
            # Find the active contracts at the target date
            active_contracts = self._calendar_df[self._calendar_df[date_column] >= target_date].sort_values(date_column)
            
            if active_contracts.empty:
                # No active contracts found for the target date
                # This could happen if the date is after the last roll in our calendar
                logger.debug(f"No active contracts found for {target_date} (position {position})")
                return None
            
            if position <= 0:
                logger.warning(f"Invalid position {position}. Must be >= 1.")
                return None
            
            if position > len(active_contracts):
                logger.debug(f"Position {position} exceeds available contracts ({len(active_contracts)}) for {target_date}")
                return None
            
            # Return the contract code for the requested position
            return active_contracts.iloc[position-1]['contract_code']
            
        except Exception as e:
            logger.error(f"Error getting active contract for {target_date} (position {position}): {e}")
            return None
    
    def get_roll_pairs(self, position: int = 1) -> List[Tuple[datetime, str, str]]:
        """
        Get roll pairs for a specific contract position.
        
        Args:
            position: Contract position (1 for front month, 2 for second month, etc.)
            
        Returns:
            List of tuples (roll_date, from_contract, to_contract)
        """
        if self._calendar_df is None or self._calendar_df.empty:
            logger.error("No roll calendar data available. Cannot determine roll pairs.")
            return []
        
        try:
            # Ensure the calendar has roll_date column
            if 'roll_date' not in self._calendar_df.columns:
                logger.warning("Roll calendar missing 'roll_date' column. Using 'last_trading_day' as fallback.")
                date_column = 'last_trading_day'
            else:
                date_column = 'roll_date'
            
            # Get ordered contracts from the calendar
            contracts = self._calendar_df.sort_values(date_column)['contract_code'].tolist()
            roll_dates = self._calendar_df.sort_values(date_column)[date_column].tolist()
            
            # Adjust for position
            if position <= 0:
                logger.warning(f"Invalid position {position}. Must be >= 1.")
                return []
            
            if position > len(contracts) - 1:
                logger.warning(f"Position {position} requires more contracts than available ({len(contracts)})")
                return []
            
            # Create roll pairs
            roll_pairs = []
            
            # Offset index based on position
            offset = position - 1
            
            for i in range(len(contracts) - offset - 1):
                from_contract = contracts[i + offset]
                to_contract = contracts[i + offset + 1]
                roll_date = roll_dates[i + offset]
                
                roll_pairs.append((roll_date, from_contract, to_contract))
            
            return roll_pairs
            
        except Exception as e:
            logger.error(f"Error getting roll pairs for position {position}: {e}")
            return []
    
    def estimate_rolls_from_data(self, 
                              contracts_data: Dict[str, pd.DataFrame],
                              method: str = 'auto') -> List[Dict[str, Any]]:
        """
        Estimate roll dates from historical market data.
        
        Args:
            contracts_data: Dictionary mapping contract codes to their market data DataFrame
            method: Method to use for roll detection ('volume', 'oi', 'auto')
            
        Returns:
            List of dictionaries with roll information
        """
        if not contracts_data:
            logger.warning("No contract data provided for roll estimation.")
            return []
        
        try:
            # If method is 'auto', determine the best method based on data
            if method == 'auto':
                # Check data completeness for volume and open interest
                has_volume = all('volume' in df.columns for df in contracts_data.values())
                has_oi = all('open_interest' in df.columns for df in contracts_data.values())
                
                if has_volume:
                    method = 'volume'
                elif has_oi:
                    method = 'oi'
                else:
                    method = 'calendar'
                
                logger.info(f"Auto-selected '{method}' method for roll estimation based on data availability")
            
            # Handle based on selected method
            if method == 'volume':
                return self._generate_volume_based_rolls(contracts_data)
            elif method == 'oi':
                return self._generate_oi_based_rolls(contracts_data)
            elif method == 'calendar':
                # Get contract codes
                contracts = list(contracts_data.keys())
                return self._generate_calendar_based_rolls(contracts)
            else:
                logger.error(f"Unsupported roll estimation method: {method}")
                return []
                
        except Exception as e:
            logger.error(f"Error estimating rolls from data: {e}", exc_info=True)
            return []
    
    def _get_contracts_for_symbol(self, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> List[str]:
        """
        Get all contracts for the specified root symbol in a date range.
        
        Args:
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            List of contract codes
        """
        try:
            # Query to get distinct contracts
            query = """
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE ?
                  AND (? IS NULL OR timestamp >= ?)
                  AND (? IS NULL OR timestamp <= ?)
                ORDER BY symbol
            """
            
            pattern = f"{self.root_symbol}%"
            params = [pattern, start_date, start_date, end_date, end_date]
            
            # Execute the query
            contracts_df = self.db_connector.query(query, params)
            
            if contracts_df.empty:
                logger.warning(f"No contracts found for {self.root_symbol} in the specified date range.")
                return []
            
            # Extract and return contract symbols
            return contracts_df['symbol'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting contracts for {self.root_symbol}: {e}")
            return []
    
    def _load_contract_data(self, 
                         contracts: List[str],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load market data for the specified contracts.
        
        Args:
            contracts: List of contract codes
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping contract codes to DataFrame with market data
        """
        if not contracts:
            return {}
        
        result = {}
        
        try:
            for contract in contracts:
                # Query to get data for a specific contract
                query = """
                    SELECT timestamp, symbol, open, high, low, close, settle, 
                           volume, open_interest, source
                    FROM market_data
                    WHERE symbol = ?
                      AND interval_unit = 'daily'
                      AND (? IS NULL OR timestamp >= ?)
                      AND (? IS NULL OR timestamp <= ?)
                    ORDER BY timestamp ASC
                """
                
                params = [contract, start_date, start_date, end_date, end_date]
                
                # Execute the query
                contract_df = self.db_connector.query(query, params)
                
                if contract_df.empty:
                    logger.debug(f"No data found for contract {contract} in the specified date range.")
                    continue
                
                # Process the data
                contract_df['timestamp'] = pd.to_datetime(contract_df['timestamp'])
                
                # Ensure numeric values
                numeric_columns = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']
                for col in numeric_columns:
                    if col in contract_df.columns:
                        contract_df[col] = pd.to_numeric(contract_df[col], errors='coerce')
                
                # Store in result dictionary
                result[contract] = contract_df
                
                logger.debug(f"Loaded {len(contract_df)} rows for contract {contract}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading contract data: {e}", exc_info=True)
            return {}
    
    def _sort_contracts_by_expiry(self, contracts: List[str]) -> List[str]:
        """
        Sort contracts by their expiration date.
        
        Args:
            contracts: List of contract codes
            
        Returns:
            Sorted list of contract codes
        """
        # Extract month code and year from contract symbols
        # This assumes a standard format like 'ESZ23' or 'VXF24'
        contract_info = []
        
        month_map = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }
        
        for contract in contracts:
            try:
                if len(contract) >= 4:
                    # Extract month and year
                    month_code = contract[-3]
                    year_str = contract[-2:]
                    
                    # Convert to numeric values
                    month = month_map.get(month_code, 0)
                    year = 2000 + int(year_str)  # Assumes 21st century
                    
                    # Create sortable value (year * 100 + month)
                    sort_value = year * 100 + month
                    
                    contract_info.append((contract, sort_value))
                else:
                    # Invalid contract format, add with a high sort value
                    logger.warning(f"Invalid contract format: {contract}")
                    contract_info.append((contract, 999999))
            except Exception as e:
                # Handle parsing errors
                logger.warning(f"Error parsing contract {contract}: {e}")
                contract_info.append((contract, 999999))
        
        # Sort by the calculated sort value
        contract_info.sort(key=lambda x: x[1])
        
        # Return the sorted contract codes
        return [info[0] for info in contract_info]
    
    def _generate_volume_based_rolls(self, contracts_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate roll calendar based on volume crossover between contracts.
        
        Args:
            contracts_data: Dictionary mapping contract codes to their market data
            
        Returns:
            List of dictionaries with roll calendar entries
        """
        if not contracts_data:
            return []
        
        try:
            # Sort contracts by expiry
            contracts = list(contracts_data.keys())
            sorted_contracts = self._sort_contracts_by_expiry(contracts)
            
            # Generate roll calendar entries
            calendar_entries = []
            
            for i in range(len(sorted_contracts) - 1):
                curr_contract = sorted_contracts[i]
                next_contract = sorted_contracts[i + 1]
                
                # Skip if we don't have data for either contract
                if curr_contract not in contracts_data or next_contract not in contracts_data:
                    continue
                
                curr_data = contracts_data[curr_contract]
                next_data = contracts_data[next_contract]
                
                # Get overlapping date range
                curr_dates = set(curr_data['timestamp'])
                next_dates = set(next_data['timestamp'])
                overlap_dates = sorted(list(curr_dates.intersection(next_dates)))
                
                if not overlap_dates:
                    logger.warning(f"No overlapping dates found between {curr_contract} and {next_contract}")
                    continue
                
                # Find the volume crossover date
                roll_date = None
                last_trading_day = max(curr_data['timestamp'])
                
                for overlap_date in overlap_dates:
                    curr_vol = curr_data[curr_data['timestamp'] == overlap_date]['volume'].iloc[0]
                    next_vol = next_data[next_data['timestamp'] == overlap_date]['volume'].iloc[0]
                    
                    # Check if next contract volume exceeds current by the threshold
                    if pd.notna(curr_vol) and pd.notna(next_vol) and next_vol >= curr_vol * self.min_volume_ratio:
                        roll_date = overlap_date
                        
                        # Apply roll_days_offset if needed
                        if self.roll_days_offset != 0:
                            roll_date += timedelta(days=self.roll_days_offset)
                        
                        break
                
                # If no volume crossover found, use a date N days before last trading day
                if roll_date is None:
                    days_before = min(self.expiration_threshold, 10)  # Default to 10 if threshold is larger
                    roll_date = last_trading_day - timedelta(days=days_before)
                    logger.warning(f"No volume crossover found between {curr_contract} and {next_contract}. "
                                 f"Using date {days_before} days before last trading day.")
                
                # Calculate approximate expiration date (for display/reference)
                # In a real system, this would come from exchange calendars or contract specs
                expiration_date = last_trading_day + timedelta(days=1)
                
                # Create calendar entry
                entry = {
                    'root_symbol': self.root_symbol,
                    'contract_code': curr_contract,
                    'roll_date': roll_date,
                    'last_trading_day': last_trading_day,
                    'expiration_date': expiration_date,
                    'roll_method': 'volume',
                    'next_contract': next_contract
                }
                
                calendar_entries.append(entry)
            
            return calendar_entries
            
        except Exception as e:
            logger.error(f"Error generating volume-based rolls: {e}", exc_info=True)
            return []
    
    def _generate_oi_based_rolls(self, contracts_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate roll calendar based on open interest crossover between contracts.
        
        Args:
            contracts_data: Dictionary mapping contract codes to their market data
            
        Returns:
            List of dictionaries with roll calendar entries
        """
        if not contracts_data:
            return []
        
        try:
            # Sort contracts by expiry
            contracts = list(contracts_data.keys())
            sorted_contracts = self._sort_contracts_by_expiry(contracts)
            
            # Generate roll calendar entries
            calendar_entries = []
            
            for i in range(len(sorted_contracts) - 1):
                curr_contract = sorted_contracts[i]
                next_contract = sorted_contracts[i + 1]
                
                # Skip if we don't have data for either contract
                if curr_contract not in contracts_data or next_contract not in contracts_data:
                    continue
                
                curr_data = contracts_data[curr_contract]
                next_data = contracts_data[next_contract]
                
                # Get overlapping date range
                curr_dates = set(curr_data['timestamp'])
                next_dates = set(next_data['timestamp'])
                overlap_dates = sorted(list(curr_dates.intersection(next_dates)))
                
                if not overlap_dates:
                    logger.warning(f"No overlapping dates found between {curr_contract} and {next_contract}")
                    continue
                
                # Find the open interest crossover date
                roll_date = None
                last_trading_day = max(curr_data['timestamp'])
                
                for overlap_date in overlap_dates:
                    curr_oi = curr_data[curr_data['timestamp'] == overlap_date]['open_interest'].iloc[0]
                    next_oi = next_data[next_data['timestamp'] == overlap_date]['open_interest'].iloc[0]
                    
                    # Check if next contract OI exceeds current by the threshold
                    if pd.notna(curr_oi) and pd.notna(next_oi) and next_oi >= curr_oi * self.min_open_interest_ratio:
                        roll_date = overlap_date
                        
                        # Apply roll_days_offset if needed
                        if self.roll_days_offset != 0:
                            roll_date += timedelta(days=self.roll_days_offset)
                        
                        break
                
                # If no OI crossover found, use a date N days before last trading day
                if roll_date is None:
                    days_before = min(self.expiration_threshold, 10)  # Default to 10 if threshold is larger
                    roll_date = last_trading_day - timedelta(days=days_before)
                    logger.warning(f"No open interest crossover found between {curr_contract} and {next_contract}. "
                                 f"Using date {days_before} days before last trading day.")
                
                # Calculate approximate expiration date (for display/reference)
                expiration_date = last_trading_day + timedelta(days=1)
                
                # Create calendar entry
                entry = {
                    'root_symbol': self.root_symbol,
                    'contract_code': curr_contract,
                    'roll_date': roll_date,
                    'last_trading_day': last_trading_day,
                    'expiration_date': expiration_date,
                    'roll_method': 'open_interest',
                    'next_contract': next_contract
                }
                
                calendar_entries.append(entry)
            
            return calendar_entries
            
        except Exception as e:
            logger.error(f"Error generating OI-based rolls: {e}", exc_info=True)
            return []
    
    def _generate_calendar_based_rolls(self, contracts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate roll calendar based on calendar dates (no market data required).
        
        Args:
            contracts: List of contract codes
            
        Returns:
            List of dictionaries with roll calendar entries
        """
        if not contracts:
            return []
        
        try:
            # Sort contracts by expiry
            sorted_contracts = self._sort_contracts_by_expiry(contracts)
            
            # Generate roll calendar entries
            calendar_entries = []
            
            for i in range(len(sorted_contracts) - 1):
                curr_contract = sorted_contracts[i]
                next_contract = sorted_contracts[i + 1]
                
                # Extract month and year info for current contract
                # This is a simplified calculation and would be more precise with real exchange calendars
                try:
                    month_code = curr_contract[-3]
                    year_str = curr_contract[-2:]
                    
                    month_map = {
                        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
                    }
                    
                    month = month_map.get(month_code, 1)
                    year = 2000 + int(year_str)  # Assumes 21st century
                    
                    # Approximate last trading day as the third Friday of the contract month
                    # This is true for many financial futures but varies by product
                    # A real implementation would use exchange calendars
                    
                    # Start with first day of month
                    day = 1
                    first_date = datetime(year, month, day)
                    
                    # Find the first Friday
                    while first_date.weekday() != 4:  # 4 = Friday
                        first_date += timedelta(days=1)
                    
                    # Add two weeks to get the third Friday
                    last_trading_day = first_date + timedelta(days=14)
                    
                    # Calculate roll date (typically N days before expiration)
                    roll_date = last_trading_day - timedelta(days=self.expiration_threshold)
                    
                    # Ensure roll date is not a weekend
                    while roll_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                        roll_date -= timedelta(days=1)
                    
                    # Calculate expiration date (typically the day after last trading day)
                    expiration_date = last_trading_day + timedelta(days=1)
                    
                    # Create calendar entry
                    entry = {
                        'root_symbol': self.root_symbol,
                        'contract_code': curr_contract,
                        'roll_date': roll_date,
                        'last_trading_day': last_trading_day,
                        'expiration_date': expiration_date,
                        'roll_method': 'calendar',
                        'next_contract': next_contract
                    }
                    
                    calendar_entries.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Error calculating calendar dates for {curr_contract}: {e}")
                    continue
            
            return calendar_entries
            
        except Exception as e:
            logger.error(f"Error generating calendar-based rolls: {e}", exc_info=True)
            return []