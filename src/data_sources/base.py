#!/usr/bin/env python
"""
Data Source Plugin Base Module

This module defines the base class and interfaces for data source plugins
in the Financial Data System. All data sources should inherit from the
DataSourcePlugin class and implement its abstract methods.
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Union, Tuple

from ..core.database import Database

logger = logging.getLogger(__name__)

class DataSourceError(Exception):
    """Exception raised for errors in data source operations."""
    pass

class DataSourcePlugin(ABC):
    """Base class for all data source plugins."""
    
    def __init__(self, config: Dict[str, Any], db: Database):
        """
        Initialize the data source plugin.
        
        Args:
            config: Configuration dictionary for this data source
            db: Database instance for storage
        """
        self.config = config
        self.db = db
        self.name = config.get('name', self.__class__.__name__)
        
        # Validate configuration
        self.validate_config()
        
        logger.info(f"Initialized data source: {self.name}")
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration for this data source.
        
        Raises:
            DataSourceError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: Union[str, date, datetime], 
                  end_date: Union[str, date, datetime],
                  interval_unit: str = 'daily', interval_value: int = 1) -> pd.DataFrame:
        """
        Fetch data for the specified symbol and date range.
        
        Args:
            symbol: Symbol to fetch data for
            start_date: Start date for data range
            end_date: End date for data range
            interval_unit: Time interval unit (e.g., 'daily', 'minute')
            interval_value: Time interval value (e.g., 1, 5, 15)
            
        Returns:
            DataFrame with fetched data
            
        Raises:
            DataSourceError: If data fetching fails
        """
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: pd.DataFrame, symbol: str, 
                      interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Transform raw data to the standard format.
        
        Args:
            raw_data: Raw data from fetch_data
            symbol: Symbol the data is for
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            Transformed DataFrame in standard format
            
        Raises:
            DataSourceError: If transformation fails
        """
        pass
    
    def get_data(self, symbol: str, start_date: Union[str, date, datetime], 
                end_date: Union[str, date, datetime],
                interval_unit: str = 'daily', interval_value: int = 1) -> pd.DataFrame:
        """
        Main method to get data through the complete pipeline.
        
        Args:
            symbol: Symbol to fetch data for
            start_date: Start date for data range
            end_date: End date for data range
            interval_unit: Time interval unit (e.g., 'daily', 'minute')
            interval_value: Time interval value (e.g., 1, 5, 15)
            
        Returns:
            DataFrame with processed data
            
        Raises:
            DataSourceError: If any part of the pipeline fails
        """
        try:
            # Convert dates to standard format
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)
            
            logger.info(f"Fetching data for {symbol} from {start_date_str} to {end_date_str} "
                        f"({interval_value} {interval_unit})")
            
            # Fetch raw data
            raw_data = self.fetch_data(
                symbol, 
                start_date_str, 
                end_date_str, 
                interval_unit, 
                interval_value
            )
            
            if raw_data is None or raw_data.empty:
                logger.warning(f"No data found for {symbol} from {start_date_str} to {end_date_str}")
                return pd.DataFrame()
            
            # Transform to standard format
            transformed_data = self.transform_data(
                raw_data, 
                symbol, 
                interval_unit, 
                interval_value
            )
            
            logger.info(f"Retrieved {len(transformed_data)} rows for {symbol}")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            raise DataSourceError(f"Failed to get data for {symbol}: {e}")
    
    def save_data(self, data: pd.DataFrame, table_name: str) -> int:
        """
        Save data to the database.
        
        Args:
            data: DataFrame to save
            table_name: Target table name
            
        Returns:
            Number of rows saved
            
        Raises:
            DataSourceError: If saving fails
        """
        if data is None or data.empty:
            logger.warning(f"No data to save to {table_name}")
            return 0
            
        try:
            # Check for required columns
            required_columns = [
                'timestamp', 'symbol', 'open', 'high', 'low', 'close',
                'volume', 'interval_value', 'interval_unit'
            ]
            
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Required column '{col}' missing from data")
                    raise DataSourceError(f"Required column '{col}' missing from data")
            
            # Upsert data using timestamp, symbol, interval as the key
            key_columns = ['timestamp', 'symbol', 'interval_value', 'interval_unit']
            
            # Add source column if not present
            if 'source' not in data.columns:
                data['source'] = self.name
                
            # Set data quality column if not present
            if 'quality' not in data.columns:
                data['quality'] = 100  # Default quality
                
            rows = self.db.upsert_dataframe(data, table_name, key_columns)
            logger.info(f"Saved {rows} rows to {table_name}")
            return rows
            
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise DataSourceError(f"Failed to save data: {e}")
    
    @staticmethod
    def _format_date(date_obj: Union[str, date, datetime]) -> str:
        """
        Format a date object as a string in YYYY-MM-DD format.
        
        Args:
            date_obj: Date to format
            
        Returns:
            Formatted date string
        """
        if isinstance(date_obj, str):
            # Assume it's already in the right format
            return date_obj
        elif isinstance(date_obj, (date, datetime)):
            return date_obj.strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Unsupported date type: {type(date_obj)}")
    
    def get_latest_data_date(self, symbol: str, table_name: str,
                           interval_unit: str = 'daily', 
                           interval_value: int = 1) -> Optional[str]:
        """
        Get the date of the latest data for a symbol.
        
        Args:
            symbol: Symbol to check
            table_name: Table to query
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            Date string in YYYY-MM-DD format, or None if no data
        """
        try:
            query = f"""
            SELECT MAX(timestamp)::DATE AS last_date
            FROM {table_name}
            WHERE symbol = ?
            AND interval_unit = ?
            AND interval_value = ?
            """
            params = [symbol, interval_unit, interval_value]
            
            result = self.db.query_to_df(query, params)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return None
                
            return str(result.iloc[0, 0])
            
        except Exception as e:
            logger.error(f"Error getting latest data date for {symbol}: {e}")
            return None
    
    def calculate_start_date(self, symbol: str, table_name: str,
                            interval_unit: str = 'daily', 
                            interval_value: int = 1,
                            default_lookback_days: int = 90) -> str:
        """
        Calculate the start date for data fetching based on existing data.
        
        Args:
            symbol: Symbol to fetch
            table_name: Table to check for existing data
            interval_unit: Time interval unit
            interval_value: Time interval value
            default_lookback_days: Days to look back if no existing data
            
        Returns:
            Start date string in YYYY-MM-DD format
        """
        # Get the latest date we have data for
        latest_date = self.get_latest_data_date(
            symbol, table_name, interval_unit, interval_value
        )
        
        if latest_date:
            # We have data, start from the day after the latest date
            latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
            # Add one day to avoid duplicating the last day
            start_dt = latest_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = start_dt + pd.Timedelta(days=1)
            return start_dt.strftime('%Y-%m-%d')
        else:
            # No existing data, use default lookback
            start_dt = datetime.now() - pd.Timedelta(days=default_lookback_days)
            return start_dt.strftime('%Y-%m-%d')


class DataSourceRegistry:
    """Registry for data source plugins."""
    
    _sources = {}
    
    @classmethod
    def register(cls, name: str, source_class: type) -> None:
        """
        Register a data source plugin class.
        
        Args:
            name: Name to register the data source under
            source_class: DataSourcePlugin subclass
        """
        if not issubclass(source_class, DataSourcePlugin):
            raise TypeError(f"{source_class.__name__} is not a subclass of DataSourcePlugin")
            
        cls._sources[name] = source_class
        logger.debug(f"Registered data source plugin: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """
        Get a data source plugin class by name.
        
        Args:
            name: Name of the data source
            
        Returns:
            DataSourcePlugin subclass or None if not found
        """
        return cls._sources.get(name)
    
    @classmethod
    def list_sources(cls) -> List[str]:
        """
        Get a list of all registered data source names.
        
        Returns:
            List of data source names
        """
        return list(cls._sources.keys())
    
    @classmethod
    def create_source(cls, name: str, config: Dict[str, Any], db: Database) -> DataSourcePlugin:
        """
        Create a data source plugin instance.
        
        Args:
            name: Name of the data source
            config: Configuration for the data source
            db: Database instance
            
        Returns:
            DataSourcePlugin instance
            
        Raises:
            ValueError: If the data source is not registered
        """
        source_class = cls.get(name)
        if source_class is None:
            raise ValueError(f"Data source '{name}' is not registered")
            
        return source_class(config, db)