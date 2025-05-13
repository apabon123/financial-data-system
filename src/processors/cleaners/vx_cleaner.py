#!/usr/bin/env python
"""
VX Futures Data Cleaner

This module implements a data cleaner for VX futures contracts,
addressing specific issues like division factor problems in CBOE data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import re

from .base import DataCleaner, DataCleanerError
from ...core.database import Database

logger = logging.getLogger(__name__)

class VXDataCleaner(DataCleaner):
    """VX futures data cleaner."""
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        """
        Initialize the VX data cleaner.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        super().__init__(db, config)
        
        # VX-specific settings
        self.division_factor_threshold = config.get('vx_division_factor_threshold', 1.0)
        self.price_columns = ['open', 'high', 'low', 'close', 'settle']
        self.max_reasonable_price = config.get('vx_max_reasonable_price', 200.0)
        self.min_reasonable_price = config.get('vx_min_reasonable_price', 5.0)
    
    def can_clean(self, symbol: str) -> bool:
        """
        Check if this cleaner can handle the specified symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if this cleaner can handle the symbol, False otherwise
        """
        # Handle VX futures and VIX index
        if symbol == '$VIX.X':
            return True
            
        # Match VX futures (e.g., VXF25, VXK25)
        if re.match(r'^VX[FGHJKMNQUVXZ]\d{2}$', symbol):
            return True
            
        # Match continuous VX (e.g., @VX=101XN)
        if symbol.startswith('@VX='):
            return True
            
        return False
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate VX futures data.
        
        Args:
            data: DataFrame with VX futures data
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataCleanerError: If cleaning fails
        """
        if data.empty:
            return data
            
        # Create a copy to avoid modifying the input
        cleaned_data = data.copy()
        
        # Ensure we have the symbol column
        if 'symbol' not in cleaned_data.columns:
            raise DataCleanerError("Missing symbol column in data")
            
        # Skip VIX index data (already clean)
        if '$VIX.X' in cleaned_data['symbol'].unique():
            # VIX index should not need scaling fixes
            ix_data = cleaned_data[cleaned_data['symbol'] == '$VIX.X']
            non_ix_data = cleaned_data[cleaned_data['symbol'] != '$VIX.X']
            
            # Process other symbols
            if not non_ix_data.empty:
                non_ix_data = self._clean_vx_futures(non_ix_data)
                
                # Combine back
                cleaned_data = pd.concat([ix_data, non_ix_data], ignore_index=True)
                
            return cleaned_data
            
        # Process VX futures data
        cleaned_data = self._clean_vx_futures(cleaned_data)
        
        # Fix OHLC inconsistencies
        cleaned_data = self.fix_ohlc_inconsistencies(cleaned_data)
        
        # Calculate quality scores
        cleaned_data = self.calculate_quality_score(cleaned_data)
        
        return cleaned_data
    
    def _clean_vx_futures(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean VX futures data.
        
        Args:
            data: DataFrame with VX futures data
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
            
        # Create a copy
        cleaned_data = data.copy()
        
        # Ensure price columns are numeric
        for col in self.price_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        # Fix division factor issues (some prices are divided by 1000)
        scaled_rows = 0
        
        # Group by symbol to analyze each contract separately
        for symbol, group in cleaned_data.groupby('symbol'):
            if self._symbol_needs_scaling(group):
                # Get the indices for this group
                idx = group.index
                
                # Apply the scaling
                for col in self.price_columns:
                    if col in cleaned_data.columns:
                        cleaned_data.loc[idx, col] = cleaned_data.loc[idx, col] * 1000
                        
                scaled_rows += len(idx)
                logger.info(f"Scaled prices for {symbol} ({len(idx)} rows)")
        
        if scaled_rows > 0:
            logger.info(f"Scaled prices for {scaled_rows} rows in total")
        
        # Fix missing values
        for col in self.price_columns:
            if col in cleaned_data.columns:
                # Count missing values
                missing = cleaned_data[col].isna().sum()
                
                if missing > 0:
                    # For close and settle, use the other if one is missing
                    if col == 'close' and 'settle' in cleaned_data.columns:
                        mask = cleaned_data['close'].isna() & cleaned_data['settle'].notna()
                        cleaned_data.loc[mask, 'close'] = cleaned_data.loc[mask, 'settle']
                        logger.info(f"Used settle for {mask.sum()} missing close values")
                        
                    elif col == 'settle' and 'close' in cleaned_data.columns:
                        mask = cleaned_data['settle'].isna() & cleaned_data['close'].notna()
                        cleaned_data.loc[mask, 'settle'] = cleaned_data.loc[mask, 'close']
                        logger.info(f"Used close for {mask.sum()} missing settle values")
        
        # Interpolate remaining missing values
        cleaned_data = self.interpolate_missing_values(cleaned_data, self.price_columns)
        
        return cleaned_data
    
    def _symbol_needs_scaling(self, data: pd.DataFrame) -> bool:
        """
        Check if a symbol's prices need scaling (fixing division factor).
        
        Args:
            data: DataFrame with data for a single symbol
            
        Returns:
            True if prices need scaling, False otherwise
        """
        # Collect median prices for available columns
        medians = []
        for col in self.price_columns:
            if col in data.columns:
                median = data[col].median()
                if pd.notna(median):
                    medians.append(median)
        
        if not medians:
            return False
            
        # Calculate the overall median price
        overall_median = np.median(medians)
        
        # Check if the median is suspiciously low (indicating division factor issue)
        if overall_median < self.division_factor_threshold:
            logger.info(f"Symbol with median price {overall_median:.4f} likely needs scaling")
            return True
            
        return False


# Register the cleaner
from .base import DataCleanerRegistry
DataCleanerRegistry.register("vx", VXDataCleaner)