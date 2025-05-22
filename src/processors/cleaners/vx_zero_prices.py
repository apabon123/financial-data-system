"""
Data cleaner for fixing zero prices in VX futures contracts.

This module provides a cleaner that detects and fixes zero prices in VX futures data.
Zero prices can occur due to data feed issues or other anomalies and need to be
corrected to maintain data quality.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from .base import DataCleaner
from ...core.database import Database

# Logger for this module
logger = logging.getLogger(__name__)

class VXZeroPricesCleaner(DataCleaner):
    """
    Cleaner for fixing zero prices in VX futures contracts.
    
    This cleaner detects zero or missing prices in VX futures and replaces them
    with interpolated or estimated values based on surrounding data points.
    """
    
    def __init__(
        self,
        db_connector = None,
        enabled: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the VX zero prices cleaner.
        
        Args:
            db_connector: Database connector instance
            enabled: Whether this cleaner is enabled
            config: Additional configuration options
        """
        # Default configuration
        default_config = {
            'interpolation_method': 'linear',  # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            'max_gap_days': 5,                 # Maximum gap to interpolate across
            'min_valid_ratio': 0.5,            # Minimum ratio of valid data points required for interpolation
            'price_fields': ['open', 'high', 'low', 'close', 'settle'],
            'volume_fields': ['volume', 'open_interest'],
            'zero_threshold': 1e-6,            # Threshold below which prices are considered zero
            'log_all_modifications': True      # Log all modifications for this cleaner
        }
        
        # Merge with provided config
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)

        # Ensure essential parameters for the base class and this class are in merged_config
        merged_config.setdefault('name', "vx_zero_prices")
        merged_config.setdefault('description', "Fixes zero or missing prices in VX futures contracts")
        merged_config.setdefault('fields_to_clean', default_config['price_fields'] + default_config['volume_fields'])
        merged_config.setdefault('enabled', enabled) # Pass through enabled state
        merged_config.setdefault('priority', 50) # Default priority
        
        # Initialize base class with db_connector as db and the comprehensive merged_config
        super().__init__(
            db=db_connector, # Pass db_connector as 'db' to the base class
            config=merged_config
        )
        
        # Set attributes specific to this cleaner or not handled by base, if necessary
        # self.name, self.config are set by base class
        # self.description = merged_config['description'] # If needed explicitly on self
        self.fields_to_clean = merged_config['fields_to_clean']
        self.enabled = merged_config['enabled']
        self.priority = merged_config['priority']
    
    def can_clean(self, symbol: str, df_columns: List[str] = None) -> bool:
        """
        Check if this cleaner is applicable to the given symbol and data.
        This cleaner specifically targets VX futures and related symbols
        and requires price/volume fields to be present.
        Args:
            symbol: The symbol to check (e.g., 'VXN24', '@VX=101XN', '$VIX.X')
            df_columns: Optional list of column names in the DataFrame to be cleaned.
                        Used to ensure necessary fields are present.

        Returns:
            True if the cleaner can process this symbol, False otherwise.
        """
        if not symbol:
            return False

        # Check if the symbol is a VIX-related symbol
        is_vx_symbol = symbol.startswith('VX') or symbol.startswith('@VX') or symbol == '$VIX.X'
        if not is_vx_symbol:
            logger.debug(f"VXZeroPricesCleaner cannot clean non-VX symbol: {symbol}")
            return False

        # Check if necessary fields are present if df_columns is provided
        if df_columns:
            required_fields = set(self.config.get('price_fields', [])) | set(self.config.get('volume_fields', []))
            if not required_fields.intersection(set(df_columns)):
                logger.debug(f"VXZeroPricesCleaner: Symbol {symbol} does not have required fields for cleaning.")
                return False
        
        logger.debug(f"VXZeroPricesCleaner can clean symbol: {symbol}")
        return True
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by fixing zero or missing prices.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to VXZeroPricesCleaner. Returning unmodified.")
            return df
        
        # Skip if not VX contracts
        if not self._is_vx_data(df):
            logger.debug("Non-VX data provided to VXZeroPricesCleaner. Returning unmodified.")
            return df
        
        # Make a copy of the input data
        cleaned_df = df.copy()
        
        # Process each contract separately
        for symbol, contract_df in cleaned_df.groupby('symbol'):
            if contract_df.empty or len(contract_df) < 2:
                logger.debug(f"Skipping symbol {symbol} - insufficient data points")
                continue
            
            # Process this contract
            contract_cleaned = self._clean_contract(contract_df, symbol)
            
            # Update the main DataFrame
            if contract_cleaned is not None:
                # Get indices in the original DataFrame
                idx = cleaned_df.index[cleaned_df['symbol'] == symbol]
                
                # Update only the rows for this symbol
                if len(idx) == len(contract_cleaned):
                    for field in self.fields_to_clean:
                        if field in contract_cleaned.columns:
                            cleaned_df.loc[idx, field] = contract_cleaned[field].values
                else:
                    logger.warning(f"Length mismatch after cleaning {symbol}. Skipping update.")
        
        # Return the cleaned DataFrame
        self._cleaned_records_count += len(cleaned_df)
        return cleaned_df
    
    def _is_vx_data(self, df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame contains VX futures data.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if the data appears to be VX futures, False otherwise
        """
        if 'symbol' not in df.columns:
            return False
        
        # Check if any symbol looks like VX futures
        symbols = df['symbol'].unique()
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.startswith('VX'):
                return True
        
        return False
    
    def _clean_contract(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Clean a single contract's data.
        
        Args:
            df: DataFrame containing a single contract's data
            symbol: Contract symbol
            
        Returns:
            Cleaned DataFrame for this contract
        """
        if df.empty or len(df) < 2:
            return None
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Process price fields first
        price_fields = [f for f in self.config['price_fields'] if f in df_sorted.columns]
        
        for field in price_fields:
            # Find zero or missing values
            is_zero_or_na = (df_sorted[field].isna() | 
                           (df_sorted[field].abs() < self.config['zero_threshold']))
            
            # Skip if no zeros or NAs
            if not any(is_zero_or_na):
                continue
            
            # Count of zeros/NAs
            zero_count = sum(is_zero_or_na)
            total_count = len(df_sorted)
            zero_ratio = zero_count / total_count
            
            logger.debug(f"Found {zero_count}/{total_count} ({zero_ratio:.1%}) "
                       f"zero/NA values in {symbol} {field}")
            
            # Check if we have enough valid data for interpolation
            if zero_ratio > (1 - self.config['min_valid_ratio']):
                logger.warning(f"Too many zeros/NAs in {symbol} {field} ({zero_ratio:.1%}). Skipping field.")
                continue
            
            # Get valid data points
            valid_df = df_sorted[~is_zero_or_na].copy()
            
            # Process zeros/NAs
            for i in df_sorted.index[is_zero_or_na]:
                timestamp = df_sorted.loc[i, 'timestamp']
                original_value = df_sorted.loc[i, field]
                
                # Try to interpolate
                replacement_value = self._interpolate_value(
                    valid_df, field, timestamp, self.config['interpolation_method'])
                
                if pd.notna(replacement_value) and replacement_value > 0:
                    # Log the modification
                    self.log_modification(
                        timestamp=timestamp,
                        symbol=symbol,
                        field=field,
                        old_value=original_value,
                        new_value=replacement_value,
                        reason=f"Zero or missing {field} value",
                        details=f"Interpolated using {self.config['interpolation_method']} method"
                    )
                    
                    # Update the value
                    df_sorted.loc[i, field] = replacement_value
                    
                    # Add to valid set for future interpolations
                    new_row = df_sorted.loc[i:i].copy()
                    valid_df = pd.concat([valid_df, new_row]).sort_values('timestamp')
        
        # Process volume fields separately
        volume_fields = [f for f in self.config['volume_fields'] if f in df_sorted.columns]
        
        for field in volume_fields:
            # Find zero or missing values
            is_zero_or_na = (df_sorted[field].isna() | 
                           (df_sorted[field].abs() < self.config['zero_threshold']))
            
            # Skip if no zeros or NAs
            if not any(is_zero_or_na):
                continue
            
            # Count of zeros/NAs
            zero_count = sum(is_zero_or_na)
            total_count = len(df_sorted)
            zero_ratio = zero_count / total_count
            
            logger.debug(f"Found {zero_count}/{total_count} ({zero_ratio:.1%}) "
                       f"zero/NA values in {symbol} {field}")
            
            # For volume fields, we typically use a different approach than interpolation
            # We might use a rolling average or just a minimal placeholder value
            
            # Process zeros/NAs
            for i in df_sorted.index[is_zero_or_na]:
                timestamp = df_sorted.loc[i, 'timestamp']
                original_value = df_sorted.loc[i, field]
                
                # For volume/OI, use rolling average or minimal value
                if field == 'volume':
                    replacement_value = self._estimate_volume(df_sorted, i)
                elif field == 'open_interest':
                    replacement_value = self._estimate_open_interest(df_sorted, i)
                else:
                    replacement_value = None
                
                if pd.notna(replacement_value) and replacement_value > 0:
                    # Log the modification
                    self.log_modification(
                        timestamp=timestamp,
                        symbol=symbol,
                        field=field,
                        old_value=original_value,
                        new_value=replacement_value,
                        reason=f"Zero or missing {field} value",
                        details=f"Estimated using rolling average"
                    )
                    
                    # Update the value
                    df_sorted.loc[i, field] = replacement_value
        
        return df_sorted
    
    def _interpolate_value(self, valid_df: pd.DataFrame, field: str, 
                         timestamp: datetime, method: str) -> Optional[float]:
        """
        Interpolate a value for a specific timestamp based on valid data points.
        
        Args:
            valid_df: DataFrame with valid data points
            field: Field to interpolate
            timestamp: Timestamp to interpolate for
            method: Interpolation method
            
        Returns:
            Interpolated value or None if interpolation failed
        """
        try:
            # Check if we have enough data
            if len(valid_df) < 2:
                return None
            
            # Create a Series with timestamp index for interpolation
            valid_series = pd.Series(
                valid_df[field].values,
                index=pd.DatetimeIndex(valid_df['timestamp'])
            )
            
            # Create a new Series with the target timestamp
            target_idx = pd.DatetimeIndex([timestamp])
            target_series = pd.Series([np.nan], index=target_idx)
            
            # Combine and interpolate
            combined = pd.concat([valid_series, target_series]).sort_index()
            interpolated = combined.interpolate(method=method)
            
            # Return the interpolated value
            result = interpolated.loc[timestamp]
            
            # Validate result
            if pd.isna(result) or result <= 0:
                return None
            
            return result
            
        except Exception as e:
            logger.warning(f"Error interpolating {field} value: {e}")
            return None
    
    def _estimate_volume(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """
        Estimate trading volume for a specific row.
        
        Args:
            df: Contract DataFrame
            idx: Index of the row to estimate
            
        Returns:
            Estimated volume or None if estimation failed
        """
        try:
            # Use a rolling average approach
            window_size = min(5, len(df) - 1)  # Use up to 5 days, or less if not enough data
            
            # Get window indices, excluding the current index
            window_indices = list(range(max(0, idx - window_size), idx)) + list(range(idx + 1, min(len(df), idx + window_size + 1)))
            
            if not window_indices:
                return None
            
            # Get valid volumes in the window
            volumes = df.loc[window_indices, 'volume']
            valid_volumes = volumes[volumes > 0]
            
            if len(valid_volumes) < window_size / 2:
                return None  # Not enough valid volumes
            
            # Use the median as a robust estimator
            est_volume = valid_volumes.median()
            
            # Round to nearest 100 for volume
            return round(est_volume / 100) * 100
            
        except Exception as e:
            logger.warning(f"Error estimating volume: {e}")
            return None
    
    def _estimate_open_interest(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """
        Estimate open interest for a specific row.
        
        Args:
            df: Contract DataFrame
            idx: Index of the row to estimate
            
        Returns:
            Estimated open interest or None if estimation failed
        """
        try:
            # For open interest, we might want to use the last valid value
            # rather than an average, since OI tends to have trends
            
            # Check if we have a valid previous value
            if idx > 0 and df.loc[idx-1, 'open_interest'] > 0:
                return df.loc[idx-1, 'open_interest']
            
            # Check if we have a valid next value
            if idx < len(df) - 1 and df.loc[idx+1, 'open_interest'] > 0:
                return df.loc[idx+1, 'open_interest']
            
            # If no immediate neighbors, use a similar approach to volume
            window_size = min(5, len(df) - 1)
            
            # Get window indices, excluding the current index
            window_indices = list(range(max(0, idx - window_size), idx)) + list(range(idx + 1, min(len(df), idx + window_size + 1)))
            
            if not window_indices:
                return None
            
            # Get valid open interest values in the window
            oi_values = df.loc[window_indices, 'open_interest']
            valid_oi = oi_values[oi_values > 0]
            
            if len(valid_oi) < window_size / 2:
                return None  # Not enough valid values
            
            # Use the median as a robust estimator
            est_oi = valid_oi.median()
            
            # Round to nearest 100 for open interest
            return round(est_oi / 100) * 100
            
        except Exception as e:
            logger.warning(f"Error estimating open interest: {e}")
            return None