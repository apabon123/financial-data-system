#!/usr/bin/env python
"""
Data Cleaner Base Module

This module defines the base class for data cleaners in the Financial Data System.
Data cleaners are responsible for validating and correcting issues in market data.
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from ...core.database import Database

logger = logging.getLogger(__name__)

class DataCleanerError(Exception):
    """Exception raised for errors in data cleaning operations."""
    pass


class DataCleaner(ABC):
    """Base class for all data cleaners."""
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        """
        Initialize the data cleaner.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 100) # Default priority
        self.description = config.get('description', 'Base data cleaner')
        
        # Initialize tracking attributes
        self._modifications_count = 0
        self._cleaned_records_count = 0 # Number of records processed by clean()
        self._recent_modifications = [] # Store details of recent changes
        self._max_recent_modifications = config.get('max_recent_modifications', 100)
        self.log_all_modifications = config.get('log_all_modifications', False)
        
        logger.info(f"Initialized data cleaner: {self.name}")
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the provided data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataCleanerError: If cleaning fails
        """
        pass
    
    @abstractmethod
    def can_clean(self, symbol: str) -> bool:
        """
        Check if this cleaner can handle the specified symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if this cleaner can handle the symbol, False otherwise
        """
        pass
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the data and return validation results.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DataCleanerError: If validation fails
        """
        if data.empty:
            return {'valid': True, 'warnings': [], 'errors': []}
            
        warnings = []
        errors = []
        
        # Check for missing required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for NaN values in critical columns
        if set(required_columns).issubset(set(data.columns)):
            for col in ['open', 'high', 'low', 'close']:
                nan_count = data[col].isna().sum()
                if nan_count > 0:
                    warnings.append(f"{nan_count} NaN values found in {col} column")
        
        # Check for logical inconsistencies in OHLC data
        if set(['open', 'high', 'low', 'close']).issubset(set(data.columns)):
            # High should be >= open, close, and low
            high_lt_open = (data['high'] < data['open']).sum()
            if high_lt_open > 0:
                errors.append(f"{high_lt_open} rows where high < open")
                
            high_lt_close = (data['high'] < data['close']).sum()
            if high_lt_close > 0:
                errors.append(f"{high_lt_close} rows where high < close")
                
            # Low should be <= open, close, and high
            low_gt_open = (data['low'] > data['open']).sum()
            if low_gt_open > 0:
                errors.append(f"{low_gt_open} rows where low > open")
                
            low_gt_close = (data['low'] > data['close']).sum()
            if low_gt_close > 0:
                errors.append(f"{low_gt_close} rows where low > close")
                
            # High should be >= low
            high_lt_low = (data['high'] < data['low']).sum()
            if high_lt_low > 0:
                errors.append(f"{high_lt_low} rows where high < low")
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            dup_timestamps = data.duplicated(subset=['timestamp']).sum()
            if dup_timestamps > 0:
                warnings.append(f"{dup_timestamps} duplicate timestamps found")
        
        # Check for sorted timestamps
        if 'timestamp' in data.columns:
            is_sorted = data['timestamp'].sort_values().equals(data['timestamp'])
            if not is_sorted:
                warnings.append("Timestamps are not sorted")
        
        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors
        }
    
    def fix_ohlc_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fix logical inconsistencies in OHLC data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with fixed OHLC values
            
        Raises:
            DataCleanerError: If fixing fails
        """
        if data.empty:
            return data
            
        # Check if required columns exist
        if not set(['open', 'high', 'low', 'close']).issubset(set(data.columns)):
            raise DataCleanerError("Missing OHLC columns")
            
        # Create a copy to avoid modifying the input
        fixed_data = data.copy()
        
        # Ensure high is the maximum of OHLC
        fixed_data['high'] = fixed_data[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is the minimum of OHLC
        fixed_data['low'] = fixed_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Count how many rows were fixed
        high_fixed = (fixed_data['high'] != data['high']).sum()
        low_fixed = (fixed_data['low'] != data['low']).sum()
        
        if high_fixed > 0 or low_fixed > 0:
            logger.info(f"Fixed {high_fixed} high values and {low_fixed} low values")
            
        return fixed_data
    
    def calculate_quality_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a quality score for each row of data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with quality scores
            
        Raises:
            DataCleanerError: If calculation fails
        """
        if data.empty:
            return data
            
        # Create a copy to avoid modifying the input
        scored_data = data.copy()
        
        # Start with a perfect score
        scored_data['quality'] = 100
        
        # Reduce score for missing values
        for col in ['open', 'high', 'low', 'close']:
            if col in scored_data.columns:
                scored_data.loc[scored_data[col].isna(), 'quality'] -= 20
        
        # Reduce score for OHLC inconsistencies
        if set(['open', 'high', 'low', 'close']).issubset(set(scored_data.columns)):
            # High < open or high < close
            mask = (scored_data['high'] < scored_data['open']) | (scored_data['high'] < scored_data['close'])
            scored_data.loc[mask, 'quality'] -= 15
            
            # Low > open or low > close
            mask = (scored_data['low'] > scored_data['open']) | (scored_data['low'] > scored_data['close'])
            scored_data.loc[mask, 'quality'] -= 15
            
            # High < low (severe issue)
            mask = scored_data['high'] < scored_data['low']
            scored_data.loc[mask, 'quality'] -= 25
        
        # Cap the score range
        scored_data['quality'] = scored_data['quality'].clip(0, 100)
        
        # Calculate the average quality
        avg_quality = scored_data['quality'].mean()
        logger.info(f"Average quality score: {avg_quality:.2f}")
        
        return scored_data
    
    def interpolate_missing_values(self, data: pd.DataFrame, 
                                  columns: List[str] = None,
                                  method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate missing values in specified columns.
        
        Args:
            data: DataFrame with market data
            columns: List of columns to interpolate (None for all numeric columns)
            method: Interpolation method ('linear', 'time', etc.)
            
        Returns:
            DataFrame with interpolated values
            
        Raises:
            DataCleanerError: If interpolation fails
        """
        if data.empty:
            return data
            
        # Create a copy to avoid modifying the input
        interp_data = data.copy()
        
        # Default to OHLC columns if not specified
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            
        # Filter to existing columns
        columns = [col for col in columns if col in interp_data.columns]
        
        if not columns:
            return interp_data
            
        # Count NaN values before interpolation
        nan_counts = {col: interp_data[col].isna().sum() for col in columns}
        
        # Interpolate missing values
        try:
            for col in columns:
                if nan_counts[col] > 0:
                    # Ensure the column is numeric
                    interp_data[col] = pd.to_numeric(interp_data[col], errors='coerce')
                    
                    # Interpolate missing values
                    interp_data[col] = interp_data[col].interpolate(method=method)
                    
            # Count remaining NaN values after interpolation
            remaining_nans = {col: interp_data[col].isna().sum() for col in columns}
            
            # Log interpolation results
            for col in columns:
                if nan_counts[col] > 0:
                    filled = nan_counts[col] - remaining_nans[col]
                    logger.info(f"Interpolated {filled} of {nan_counts[col]} missing values in {col}")
                    
            return interp_data
            
        except Exception as e:
            logger.error(f"Error interpolating missing values: {e}")
            raise DataCleanerError(f"Failed to interpolate missing values: {e}")

    def log_modification(
        self,
        timestamp: datetime,
        symbol: str,
        field: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        details: Optional[str] = None
    ) -> None:
        """
        Log a modification made by the cleaner.
        Increments modification count and stores details.
        """
        self._modifications_count += 1
        modification_details = {
            'timestamp': timestamp,
            'symbol': symbol,
            'field': field,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason,
            'details': details,
            'cleaner': self.name,
            'log_time': datetime.now()
        }
        
        if self.log_all_modifications:
            # Log to central logger or specific modification log if configured
            logger.debug(f"Modification by {self.name}: {modification_details}")
            
        # Keep track of recent modifications
        self._recent_modifications.append(modification_details)
        if len(self._recent_modifications) > self._max_recent_modifications:
            self._recent_modifications.pop(0) # Keep the list size bounded

    def get_recent_modifications(self) -> List[Dict[str, Any]]:
        """Return the list of recent modifications."""
        return self._recent_modifications

    def get_modification_stats(self) -> Dict[str, Any]:
        """
        Return statistics about modifications made by this cleaner.
        """
        return {
            'name': self.name,
            'modifications_count': self._modifications_count,
            'cleaned_records_count': self._cleaned_records_count,
            'enabled': self.enabled,
            'priority': self.priority
        }

    def reset_stats(self) -> None:
        """
        Reset the modification and record counts for this cleaner.
        """
        self._modifications_count = 0
        self._cleaned_records_count = 0
        self._recent_modifications = []
        logger.debug(f"Stats reset for cleaner: {self.name}")


class DataCleanerRegistry:
    """Registry for data cleaner classes."""
    
    _cleaners = {}
    
    @classmethod
    def register(cls, name: str, cleaner_class: type) -> None:
        """
        Register a data cleaner class.
        
        Args:
            name: Name to register the cleaner under
            cleaner_class: DataCleaner subclass
        """
        if not issubclass(cleaner_class, DataCleaner):
            raise TypeError(f"{cleaner_class.__name__} is not a subclass of DataCleaner")
            
        cls._cleaners[name] = cleaner_class
        logger.debug(f"Registered data cleaner: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """
        Get a data cleaner class by name.
        
        Args:
            name: Name of the cleaner
            
        Returns:
            DataCleaner subclass or None if not found
        """
        return cls._cleaners.get(name)
    
    @classmethod
    def list_cleaners(cls) -> List[str]:
        """
        Get a list of all registered cleaner names.
        
        Returns:
            List of cleaner names
        """
        return list(cls._cleaners.keys())
    
    @classmethod
    def create_cleaner(cls, name: str, db: Database, config: Dict[str, Any]) -> DataCleaner:
        """
        Create a data cleaner instance.
        
        Args:
            name: Name of the cleaner
            db: Database instance
            config: Configuration dictionary
            
        Returns:
            DataCleaner instance
            
        Raises:
            ValueError: If the cleaner is not registered
        """
        cleaner_class = cls.get(name)
        if cleaner_class is None:
            raise ValueError(f"Data cleaner '{name}' is not registered")
            
        return cleaner_class(db, config)
    
    @classmethod
    def get_cleaner_for_symbol(cls, symbol: str, db: Database, 
                             config: Dict[str, Any]) -> Optional[DataCleaner]:
        """
        Get a cleaner instance that can handle the specified symbol.
        
        Args:
            symbol: Symbol to find a cleaner for
            db: Database instance
            config: Configuration dictionary
            
        Returns:
            DataCleaner instance or None if no suitable cleaner is found
        """
        for name, cleaner_class in cls._cleaners.items():
            try:
                # Create an instance to check if it can clean this symbol
                cleaner = cleaner_class(db, config)
                if cleaner.can_clean(symbol):
                    return cleaner
            except Exception as e:
                logger.warning(f"Error checking cleaner {name} for symbol {symbol}: {e}")
                
        return None