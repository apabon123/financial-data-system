#!/usr/bin/env python
"""
Continuous Futures Base Module

This module defines the base class for continuous futures generation.
It provides the foundation for different continuous contract strategies.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from ...core.database import Database

logger = logging.getLogger(__name__)

class ContractRollover:
    """Represents a contract rollover point."""
    
    def __init__(self, date: pd.Timestamp, from_contract: str, to_contract: str,
                from_price: float = None, to_price: float = None):
        """
        Initialize a contract rollover.
        
        Args:
            date: Rollover date
            from_contract: Contract rolling out of
            to_contract: Contract rolling into
            from_price: Price of the outgoing contract (None if not known)
            to_price: Price of the incoming contract (None if not known)
        """
        self.date = pd.Timestamp(date)  # Ensure it's always a Timestamp
        self.from_contract = from_contract
        self.to_contract = to_contract
        self.from_price = from_price
        self.to_price = to_price
        
        # Calculate the adjustment ratio if prices are available
        self.ratio = None
        if from_price is not None and to_price is not None and from_price != 0:
            self.ratio = to_price / from_price
    
    def __repr__(self):
        return (f"ContractRollover(date={self.date.date()}, "
                f"from={self.from_contract}, to={self.to_contract}, "
                f"ratio={self.ratio:.4f} if self.ratio else 'None')")


class ContinuousContractError(Exception):
    """Exception raised for errors in continuous contract generation."""
    pass


class ContinuousContractBuilder(ABC):
    """Base class for continuous contract generation strategies."""
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        """
        Initialize the continuous contract builder.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        
        # Default settings
        self.roll_calendar_table = config.get('roll_calendar_table', 'futures_roll_dates')
        self.market_data_table = config.get('market_data_table', 'market_data')
        self.continuous_data_table = config.get('continuous_data_table', 'continuous_contracts')
        
        # Initialize a list to track roll dates during generation
        self.roll_dates = []
    
    @staticmethod
    def _normalize_date(date_input: Union[str, pd.Timestamp, datetime, date, None]) -> Optional[pd.Timestamp]:
        """
        Normalize various date inputs to pd.Timestamp for consistent comparisons.
        
        Args:
            date_input: Date in various formats
            
        Returns:
            pd.Timestamp or None if input is None
        """
        if date_input is None:
            return None
        
        if isinstance(date_input, str):
            return pd.Timestamp(date_input)
        elif isinstance(date_input, (datetime, date)):
            return pd.Timestamp(date_input)
        elif isinstance(date_input, pd.Timestamp):
            return date_input
        else:
            # Try to convert anything else
            return pd.Timestamp(date_input)
    
    @staticmethod
    def _date_to_string(date_input: Union[str, pd.Timestamp, datetime, date, None]) -> Optional[str]:
        """
        Convert various date inputs to string format (YYYY-MM-DD) for database queries.
        
        Args:
            date_input: Date in various formats
            
        Returns:
            Date string in YYYY-MM-DD format or None if input is None
        """
        if date_input is None:
            return None
            
        normalized = ContinuousContractBuilder._normalize_date(date_input)
        return normalized.strftime('%Y-%m-%d') if normalized is not None else None
    
    @abstractmethod
    def build_continuous_series(self, root_symbol: str, continuous_symbol: str,
                               interval_unit: str = 'daily', interval_value: int = 1,
                               start_date: str = None, end_date: str = None,
                               force: bool = False) -> pd.DataFrame:
        """
        Build a continuous series for the given parameters.
        
        Args:
            root_symbol: The root symbol (e.g., ES, VX)
            continuous_symbol: Target continuous symbol (e.g., @ES=102XC)
            interval_unit: Time interval unit
            interval_value: Time interval value
            start_date: Start date for the series (None for all available)
            end_date: End date for the series (None for current date)
            force: Whether to force rebuild the entire series
            
        Returns:
            DataFrame with the continuous series data
            
        Raises:
            ContinuousContractError: If continuous series generation fails
        """
        pass
    
    def get_roll_dates(self, root_symbol: str, start_date: str = None, 
                      end_date: str = None) -> List[ContractRollover]:
        """
        Get roll dates for a specific root symbol within or affecting a date range.
        
        This method finds all roll dates that are relevant for building continuous
        contracts in the specified period, including:
        1. The most recent roll before the start date (to determine active contract)
        2. All rolls within the date range
        3. Optionally the first roll after the end date (for context)
        
        Args:
            root_symbol: The root symbol (e.g., ES, VX)
            start_date: Start date for the range (None for all available)
            end_date: End date for the range (None for current date)
            
        Returns:
            List of ContractRollover objects
            
        Raises:
            ContinuousContractError: If roll dates cannot be determined
        """
        try:
            # Normalize date parameters to strings for database queries
            start_date_str = self._date_to_string(start_date)
            end_date_str = self._date_to_string(end_date)
            
            # Build query to get relevant roll dates
            # We need to get rolls that affect the requested period, including the next roll for proper transitions
            if start_date_str is None and end_date_str is None:
                # No date filtering - get all rolls
                date_filter = ""
                params = [root_symbol]
            elif start_date_str is None:
                # Only end date specified - get all rolls up to end date plus next roll for transitions
                date_filter = """
                AND (r.RollDate <= ? OR r.RollDate = (
                    SELECT MIN(RollDate) 
                    FROM {roll_table} r2 
                    WHERE r2.SymbolRoot = r.SymbolRoot 
                    AND r2.RollDate > ?
                ))
                """.format(roll_table=self.roll_calendar_table)
                params = [root_symbol, end_date_str, end_date_str]
            elif end_date_str is None:
                # Only start date specified - get last roll before start + all rolls after start
                date_filter = """
                AND (r.RollDate >= ? OR r.RollDate = (
                    SELECT MAX(RollDate) 
                    FROM {roll_table} r2 
                    WHERE r2.SymbolRoot = r.SymbolRoot 
                    AND r2.RollDate < ?
                ))
                """.format(roll_table=self.roll_calendar_table)
                params = [root_symbol, start_date_str, start_date_str]
            else:
                # Both dates specified - get last roll before start + rolls within range + next roll after end
                date_filter = """
                AND (
                    (r.RollDate >= ? AND r.RollDate <= ?) OR 
                    r.RollDate = (
                        SELECT MAX(RollDate) 
                        FROM {roll_table} r2 
                        WHERE r2.SymbolRoot = r.SymbolRoot 
                        AND r2.RollDate < ?
                    ) OR
                    r.RollDate = (
                        SELECT MIN(RollDate) 
                        FROM {roll_table} r2 
                        WHERE r2.SymbolRoot = r.SymbolRoot 
                        AND r2.RollDate > ?
                    )
                )
                """.format(roll_table=self.roll_calendar_table)
                params = [root_symbol, start_date_str, end_date_str, start_date_str, end_date_str]
                
            # Query roll dates from the database
            query = f"""
            WITH FilteredRolls AS (
                SELECT
                    r.SymbolRoot,
                    r.Contract,
                    r.RollDate,
                    r.RollType,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.SymbolRoot, r.Contract, r.RollDate 
                        ORDER BY CASE WHEN r.RollType = 'volume' THEN 1 ELSE 2 END, r.RollType
                    ) as rn
                FROM {self.roll_calendar_table} r
                WHERE r.SymbolRoot = ?
                  {date_filter}
            ),
            OrderedRolls AS (
                SELECT
                    r.RollDate AS roll_date,
                    CONCAT(r.SymbolRoot, r.Contract) AS from_contract_symbol,
                    CONCAT(r.SymbolRoot, LEAD(r.Contract, 1) OVER (PARTITION BY r.SymbolRoot ORDER BY r.RollDate)) AS to_contract_symbol,
                    r.SymbolRoot
                FROM FilteredRolls r
                WHERE r.rn = 1  -- Only take the first (preferred) roll for each date/contract
            )
            SELECT
                o.roll_date,
                o.from_contract_symbol AS from_contract,
                o.to_contract_symbol AS to_contract,
                f_close.close AS from_price,
                t_close.close AS to_price
            FROM OrderedRolls o
            LEFT JOIN {self.market_data_table} f_close
                ON o.from_contract_symbol = f_close.symbol
                AND o.roll_date = f_close.timestamp::DATE
                AND f_close.interval_unit = 'daily'
            LEFT JOIN {self.market_data_table} t_close
                ON o.to_contract_symbol = t_close.symbol
                AND o.roll_date = t_close.timestamp::DATE 
                AND t_close.interval_unit = 'daily'
            WHERE o.to_contract_symbol IS NOT NULL -- Only include rolls where there is a next contract
            ORDER BY o.roll_date
            """
            
            df = self.db.query_to_df(query, params)
            
            if df.empty:
                logger.warning(f"No roll dates found for {root_symbol} in the specified range")
                return []
                
            # Convert to ContractRollover objects with proper date normalization
            rollovers = []
            for _, row in df.iterrows():
                rollover = ContractRollover(
                    date=self._normalize_date(row['roll_date']),
                    from_contract=row['from_contract'],
                    to_contract=row['to_contract'],
                    from_price=row['from_price'],
                    to_price=row['to_price']
                )
                rollovers.append(rollover)
                
            return rollovers
            
        except Exception as e:
            logger.error(f"Error getting roll dates for {root_symbol}: {e}")
            raise ContinuousContractError(f"Failed to get roll dates: {e}")
    
    def load_contract_data(self, symbol: str, start_date: str = None,
                          end_date: str = None, interval_unit: str = 'daily',
                          interval_value: int = 1) -> pd.DataFrame:
        """
        Load individual contract data for a symbol.
        
        Args:
            symbol: Contract symbol (e.g., ESH25)
            start_date: Start date for the data (None for all available)
            end_date: End date for the data (None for all available)
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            DataFrame with contract data
            
        Raises:
            ContinuousContractError: If data cannot be loaded
        """
        try:
            # Normalize date parameters to strings for database queries
            start_date_str = self._date_to_string(start_date)
            end_date_str = self._date_to_string(end_date)
            
            # Process date parameters
            if start_date_str is None:
                start_clause = ""
                start_params = []
            else:
                start_clause = "AND timestamp::DATE >= ?"
                start_params = [start_date_str]
                
            if end_date_str is None:
                end_clause = ""
                end_params = []
            else:
                end_clause = "AND timestamp::DATE <= ?"
                end_params = [end_date_str]
                
            # Query contract data from the database
            query = f"""
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                open_interest,
                source,
                interval_unit,
                interval_value
            FROM {self.market_data_table}
            WHERE symbol = ?
            AND interval_unit = ?
            AND interval_value = ?
            {start_clause}
            {end_clause}
            ORDER BY timestamp
            """
            
            params = [symbol, interval_unit, interval_value] + start_params + end_params
            df = self.db.query_to_df(query, params)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} in the specified range")
                return pd.DataFrame()
                
            # Ensure timestamp is datetime with proper normalization
            df['timestamp'] = df['timestamp'].apply(self._normalize_date)
            
            # Ensure numeric columns are numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
            
        except Exception as e:
            logger.error(f"Error loading contract data for {symbol}: {e}")
            raise ContinuousContractError(f"Failed to load contract data: {e}")
    
    def get_active_contract_chain(self, root_symbol: str, date_str: str,
                                 interval_unit: str = 'daily', 
                                 interval_value: int = 1) -> List[Dict[str, Any]]:
        """
        Get the chain of active contracts for a given date.
        
        Args:
            root_symbol: The root symbol (e.g., ES, VX)
            date_str: Date string (YYYY-MM-DD) or any date format
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            List of dictionaries with contract info, sorted by expiry
            
        Raises:
            ContinuousContractError: If contract chain cannot be determined
        """
        try:
            # Normalize date parameter to string for database query
            date_str_normalized = self._date_to_string(date_str)
            
            # Query contracts with data on the given date, ordered by expiration
            query = f"""
            WITH contracts AS (
                SELECT DISTINCT 
                    m.symbol,
                    m.timestamp::DATE AS date,
                    e.expiry_date
                FROM {self.market_data_table} m
                JOIN futures_contracts e
                    ON m.symbol = e.contract_symbol
                WHERE m.symbol LIKE '{root_symbol}%'
                AND m.interval_unit = ?
                AND m.interval_value = ?
                AND m.timestamp::DATE = ?
                AND e.root_symbol = ?
            )
            SELECT 
                c.*,
                m.open,
                m.high,
                m.low,
                m.close,
                m.volume,
                m.open_interest
            FROM contracts c
            JOIN {self.market_data_table} m
                ON c.symbol = m.symbol
                AND c.date = m.timestamp::DATE
                AND m.interval_unit = ?
                AND m.interval_value = ?
            ORDER BY c.expiry_date
            """
            
            params = [interval_unit, interval_value, date_str_normalized, root_symbol, 
                     interval_unit, interval_value]
            df = self.db.query_to_df(query, params)
            
            if df.empty:
                logger.warning(f"No active contracts found for {root_symbol} on {date_str_normalized}")
                return []
                
            # Convert to list of dictionaries with proper date normalization
            contracts = []
            for _, row in df.iterrows():
                contract = {
                    'symbol': row['symbol'],
                    'date': self._normalize_date(row['date']),
                    'expiry_date': self._normalize_date(row['expiry_date']),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'open_interest': row['open_interest']
                }
                contracts.append(contract)
                
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting active contract chain for {root_symbol} on {date_str}: {e}")
            raise ContinuousContractError(f"Failed to get active contract chain: {e}")
    
    def store_continuous_data(self, continuous_symbol: str, continuous_data: pd.DataFrame) -> int:
        """
        Store continuous contract data in the database.
        
        Args:
            continuous_symbol: Continuous contract symbol (e.g., @ES=102XC)
            continuous_data: DataFrame with continuous contract data
            
        Returns:
            Number of rows stored
            
        Raises:
            ContinuousContractError: If data cannot be stored
        """
        if continuous_data.empty:
            logger.warning(f"No data to store for {continuous_symbol}")
            return 0
            
        try:
            # Ensure built_by column is set
            if 'built_by' not in continuous_data.columns:
                continuous_data['built_by'] = self.__class__.__name__
                
            # Ensure adjusted flag is set
            if 'adjusted' not in continuous_data.columns:
                # Determine if this is an adjusted series based on the symbol
                is_adjusted = 'C' in continuous_symbol if '=' in continuous_symbol else False
                continuous_data['adjusted'] = is_adjusted
                
            # Add quality column if missing
            if 'quality' not in continuous_data.columns:
                continuous_data['quality'] = 100
                
            # Store the data
            key_columns = ['timestamp', 'symbol', 'interval_value', 'interval_unit']
            rows = self.db.upsert_dataframe(continuous_data, self.continuous_data_table, key_columns)
            logger.info(f"Stored {rows} rows for {continuous_symbol}")
            
            return rows
            
        except Exception as e:
            logger.error(f"Error storing continuous data for {continuous_symbol}: {e}")
            raise ContinuousContractError(f"Failed to store continuous data: {e}")
    
    def delete_continuous_data(self, continuous_symbol: str, 
                              interval_unit: str = None, 
                              interval_value: int = None) -> int:
        """
        Delete existing continuous contract data.
        
        Args:
            continuous_symbol: Continuous contract symbol (e.g., @ES=102XC)
            interval_unit: Time interval unit (None for all)
            interval_value: Time interval value (None for all)
            
        Returns:
            Number of rows deleted
            
        Raises:
            ContinuousContractError: If data cannot be deleted
        """
        try:
            # Build the WHERE clause
            where_clauses = ["symbol = ?"]
            params = [continuous_symbol]
            
            if interval_unit is not None:
                where_clauses.append("interval_unit = ?")
                params.append(interval_unit)
                
            if interval_value is not None:
                where_clauses.append("interval_value = ?")
                params.append(interval_value)
                
            where_clause = " AND ".join(where_clauses)
            
            # Execute the delete query
            query = f"DELETE FROM {self.continuous_data_table} WHERE {where_clause}"
            result = self.db.execute(query, params)
            
            # Get the number of deleted rows
            # In DuckDB, this returns a DuckDBPyRelation with a deleted_rows field
            if hasattr(result, 'fetchone'):
                rows_deleted = result.fetchone()[0]
            else:
                rows_deleted = 0
                
            logger.info(f"Deleted {rows_deleted} rows for {continuous_symbol}")
            return rows_deleted
            
        except Exception as e:
            logger.error(f"Error deleting continuous data for {continuous_symbol}: {e}")
            raise ContinuousContractError(f"Failed to delete continuous data: {e}")
    
    def get_existing_continuous_data(self, continuous_symbol: str,
                                    interval_unit: str = 'daily',
                                    interval_value: int = 1,
                                    start_date: str = None,
                                    end_date: str = None) -> pd.DataFrame:
        """
        Get existing continuous contract data from the database.
        
        Args:
            continuous_symbol: Continuous contract symbol (e.g., @ES=102XC)
            interval_unit: Time interval unit
            interval_value: Time interval value
            start_date: Start date for the data (None for all available)
            end_date: End date for the data (None for all available)
            
        Returns:
            DataFrame with continuous contract data
            
        Raises:
            ContinuousContractError: If data cannot be loaded
        """
        try:
            # Try to check if columns exist (DuckDB specific approach)
            try:
                column_check = self.db.query_to_df(f"DESCRIBE {self.continuous_data_table}")
                available_columns = set(column_check['column_name'].values)
            except:
                # Fallback: assume all columns exist and handle error gracefully
                available_columns = {'timestamp', 'symbol', 'underlying_symbol', 'open', 'high', 'low', 'close', 
                                   'volume', 'open_interest', 'source', 'built_by', 'interval_unit', 
                                   'interval_value', 'adjusted', 'quality'}
            
            # Build the SELECT clause conditionally based on available columns
            select_clauses = [
                "timestamp",
                "symbol",
                "underlying_symbol" if "underlying_symbol" in available_columns else "symbol as underlying_symbol",
                "open",
                "high", 
                "low",
                "close",
                "volume",
                "open_interest"
            ]
            
            # Add optional columns with defaults if they don't exist
            if "source" in available_columns:
                select_clauses.append("source")
            else:
                select_clauses.append("'unknown' as source")
                
            if "built_by" in available_columns:
                select_clauses.append("built_by")
            else:
                select_clauses.append("'legacy' as built_by")
                
            select_clauses.extend(["interval_unit", "interval_value"])
            
            if "adjusted" in available_columns:
                select_clauses.append("adjusted")
            else:
                select_clauses.append("false as adjusted")
                
            if "quality" in available_columns:
                select_clauses.append("quality")
            else:
                select_clauses.append("'unknown' as quality")
                
            # Normalize date parameters to strings for database queries
            start_date_str = self._date_to_string(start_date)
            end_date_str = self._date_to_string(end_date)
            
            # Process date parameters
            if start_date_str is None:
                start_clause = ""
                start_params = []
            else:
                start_clause = "AND timestamp::DATE >= ?"
                start_params = [start_date_str]
                
            if end_date_str is None:
                end_clause = ""
                end_params = []
            else:
                end_clause = "AND timestamp::DATE <= ?"
                end_params = [end_date_str]
                
            # Query continuous data from the database
            query = f"""
            SELECT 
                {', '.join(select_clauses)}
            FROM {self.continuous_data_table}
            WHERE symbol = ?
            AND interval_unit = ?
            AND interval_value = ?
            {start_clause}
            {end_clause}
            ORDER BY timestamp
            """
            
            params = [continuous_symbol, interval_unit, interval_value] + start_params + end_params
            df = self.db.query_to_df(query, params)
            
            if df.empty:
                logger.info(f"No existing data found for {continuous_symbol} in the specified range")
                return pd.DataFrame()
                
            # Ensure timestamp is datetime with proper normalization
            df['timestamp'] = df['timestamp'].apply(self._normalize_date)
            
            # Ensure numeric columns are numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
            
        except Exception as e:
            logger.error(f"Error loading existing continuous data for {continuous_symbol}: {e}")
            raise ContinuousContractError(f"Failed to load existing continuous data: {e}")
    
    def parse_continuous_symbol(self, continuous_symbol: str) -> Dict[str, Any]:
        """
        Parse a continuous contract symbol into its components.
        
        Args:
            continuous_symbol: Continuous contract symbol (e.g., @ES=102XC, @ES=102XN_d)
            
        Returns:
            Dictionary with parsed components
            
        Raises:
            ContinuousContractError: If symbol cannot be parsed
        """
        try:
            # Example formats:
            # @ES=102XC: Adjusted (C), roll 2 days before expiry (X)
            # @VX=101XN: Unadjusted (N), roll 1 day before expiry (X)
            # @ES=102XN_d: Unadjusted (N), with custom suffix (_d)
            
            # Check if symbol starts with @
            if not continuous_symbol.startswith('@'):
                raise ContinuousContractError(f"Invalid continuous symbol format: {continuous_symbol}")
                
            # Extract root symbol
            parts = continuous_symbol.split('=')
            if len(parts) != 2:
                raise ContinuousContractError(f"Invalid continuous symbol format: {continuous_symbol}")
                
            root_symbol = parts[0][1:]  # Remove @ prefix
            settings = parts[1]
            
            # Handle custom suffixes like '_d' by extracting and storing them
            custom_suffix = ""
            if '_' in settings:
                # Split on the first underscore to separate the standard part from custom suffix
                settings_parts = settings.split('_', 1)
                settings = settings_parts[0]  # Standard part (e.g., "102XN")
                custom_suffix = "_" + settings_parts[1]  # Custom suffix (e.g., "_d")
            
            # Parse settings
            if len(settings) < 3:
                raise ContinuousContractError(f"Invalid settings format: {settings}")
                
            # Extract month number (1-9)
            try:
                month_number = int(settings[0])
            except ValueError:
                raise ContinuousContractError(f"Invalid month number in settings: {settings}")
                
            # Extract roll days
            try:
                roll_days = int(settings[1:settings.find('X')])
            except ValueError:
                raise ContinuousContractError(f"Invalid roll days in settings: {settings}")
                
            # Determine roll type (X = days before expiry)
            roll_type = 'days_before_expiry'  # Currently only one type supported
            
            # Determine adjustment type (C = Panama, N = unadjusted)
            if settings.endswith('C'):
                adjustment = 'panama'
            elif settings.endswith('N'):
                adjustment = 'unadjusted'
            else:
                raise ContinuousContractError(f"Invalid adjustment type in settings: {settings}")
                
            return {
                'continuous_symbol': continuous_symbol,
                'root_symbol': root_symbol,
                'month_number': month_number,
                'roll_days': roll_days,
                'roll_type': roll_type,
                'adjustment': adjustment,
                'custom_suffix': custom_suffix  # Store any custom suffix for reference
            }
            
        except Exception as e:
            if isinstance(e, ContinuousContractError):
                raise
            logger.error(f"Error parsing continuous symbol {continuous_symbol}: {e}")
            raise ContinuousContractError(f"Failed to parse continuous symbol: {e}")


class ContinuousContractRegistry:
    """Registry for continuous contract builder classes."""
    
    _builders = {}
    
    @classmethod
    def register(cls, name: str, builder_class: type) -> None:
        """
        Register a continuous contract builder class.
        
        Args:
            name: Name to register the builder under
            builder_class: ContinuousContractBuilder subclass
        """
        if not issubclass(builder_class, ContinuousContractBuilder):
            raise TypeError(f"{builder_class.__name__} is not a subclass of ContinuousContractBuilder")
            
        cls._builders[name] = builder_class
        logger.debug(f"Registered continuous contract builder: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """
        Get a continuous contract builder class by name.
        
        Args:
            name: Name of the builder
            
        Returns:
            ContinuousContractBuilder subclass or None if not found
        """
        return cls._builders.get(name)
    
    @classmethod
    def list_builders(cls) -> List[str]:
        """
        Get a list of all registered builder names.
        
        Returns:
            List of builder names
        """
        return list(cls._builders.keys())
    
    @classmethod
    def create_builder(cls, name: str, db: Database, config: Dict[str, Any]) -> ContinuousContractBuilder:
        """
        Create a continuous contract builder instance.
        
        Args:
            name: Name of the builder
            db: Database instance
            config: Configuration dictionary
            
        Returns:
            ContinuousContractBuilder instance
            
        Raises:
            ValueError: If the builder is not registered
        """
        builder_class = cls.get(name)
        if builder_class is None:
            raise ValueError(f"Continuous contract builder '{name}' is not registered")
            
        return builder_class(db, config)
    
    @classmethod
    def create_builder_for_symbol(cls, continuous_symbol: str, db: Database, 
                               config: Dict[str, Any]) -> ContinuousContractBuilder:
        """
        Create a builder instance appropriate for the continuous symbol.
        
        Args:
            continuous_symbol: Continuous contract symbol (e.g., @ES=102XC)
            db: Database instance
            config: Configuration dictionary
            
        Returns:
            ContinuousContractBuilder instance
            
        Raises:
            ValueError: If no appropriate builder is found
        """
        dummy_builder = ContinuousContractBuilder.__subclasses__()[0](db, config)
        parsed = dummy_builder.parse_continuous_symbol(continuous_symbol)
        
        # Select builder based on adjustment type
        if parsed['adjustment'] == 'panama':
            builder_name = 'panama'
        elif parsed['adjustment'] == 'unadjusted':
            builder_name = 'unadjusted'
        else:
            builder_name = 'unadjusted'  # Default
            
        return cls.create_builder(builder_name, db, config)