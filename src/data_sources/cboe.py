#!/usr/bin/env python
"""
CBOE Data Source Plugin

This module implements a data source plugin for fetching market data from the
CBOE website, particularly for VIX futures and index data.
"""

import os
import logging
import requests
import time
import pandas as pd
import io
import re
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import DataSourcePlugin, DataSourceError
from ..core.database import Database

logger = logging.getLogger(__name__)

class CBOEDataSource(DataSourcePlugin):
    """CBOE website data source implementation."""
    
    def __init__(self, config: Dict[str, Any], db: Database):
        """
        Initialize the CBOE data source.
        
        Args:
            config: Configuration dictionary for this data source
            db: Database instance for storage
        """
        super().__init__(config, db)
        
        # Base URL for CBOE data
        self.vix_url_template = self.config.get(
            'vix_url_template',
            "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json"
        )
        
        self.vx_futures_url_template = self.config.get(
            'vx_futures_url_template',
            "https://www.cboe.com/us/futures/market_statistics/historical_data/products/csv/{symbol}/"
        )
        
        # Configure rate limiting
        self.rate_limit = self.config.get('rate_limit', 30)  # Requests per minute
        self.request_interval = 60.0 / self.rate_limit
        self.last_request_time = 0
    
    def validate_config(self) -> None:
        """
        Validate the configuration for this data source.
        
        Raises:
            DataSourceError: If configuration is invalid
        """
        # No required keys for basic CBOE functionality
        pass
    
    def fetch_data(self, symbol: str, start_date: Union[str, date, datetime], 
                  end_date: Union[str, date, datetime],
                  interval_unit: str = 'daily', interval_value: int = 1) -> pd.DataFrame:
        """
        Fetch data for the specified symbol and date range from CBOE.
        
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
        # Currently only supports daily data
        if interval_unit != 'daily':
            raise DataSourceError(f"CBOE data source only supports daily data, not {interval_unit}")
            
        # Format dates
        start_date_str = self._format_date(start_date)
        end_date_str = self._format_date(end_date)
        
        # Determine whether this is VIX index or VX futures
        if symbol == '$VIX.X':
            return self._fetch_vix_index(start_date_str, end_date_str)
        elif symbol.startswith('VX'):
            return self._fetch_vx_futures(symbol, start_date_str, end_date_str)
        else:
            raise DataSourceError(f"Unsupported symbol for CBOE data source: {symbol}")
    
    def _fetch_vix_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VIX index data from CBOE.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with VIX index data
        """
        self._rate_limit()
        
        try:
            # Make the request to the CBOE API
            url = self.vix_url_template.format(symbol="VIX")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Record the request time
            self.last_request_time = time.time()
            
            # Parse the JSON response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Process and filter the data
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CBOE VIX index request failed: {e}")
            raise DataSourceError(f"Failed to fetch VIX index data from CBOE: {e}")
        except Exception as e:
            logger.error(f"Error processing CBOE VIX index data: {e}")
            raise DataSourceError(f"Failed to process VIX index data: {e}")
    
    def _fetch_vx_futures(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VX futures data from CBOE.
        
        Args:
            symbol: VX futures symbol (e.g., VXF25)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with VX futures data
        """
        self._rate_limit()
        
        try:
            # Make the request to the CBOE website
            url = self.vx_futures_url_template.format(symbol=symbol)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Record the request time
            self.last_request_time = time.time()
            
            # Check if the response is CSV
            content_type = response.headers.get('Content-Type', '')
            if 'text/csv' not in content_type and 'application/csv' not in content_type:
                logger.warning(f"Unexpected Content-Type: {content_type}")
                
            # Parse the CSV content
            csv_data = response.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Check if we got any data
            if df.empty:
                logger.warning(f"No data returned for VX futures symbol {symbol}")
                return pd.DataFrame()
                
            # Process and filter the data
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
                    
            if not date_col:
                logger.error(f"Could not find date column in VX futures data for {symbol}")
                raise DataSourceError(f"Date column not found in VX futures data")
                
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CBOE VX futures request failed: {e}")
            raise DataSourceError(f"Failed to fetch VX futures data from CBOE: {e}")
        except Exception as e:
            logger.error(f"Error processing CBOE VX futures data: {e}")
            raise DataSourceError(f"Failed to process VX futures data: {e}")
    
    def transform_data(self, raw_data: pd.DataFrame, symbol: str, 
                      interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Transform raw CBOE data to the standard format.
        
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
        if raw_data.empty:
            return pd.DataFrame()
            
        try:
            # Create a new DataFrame with our standard columns
            df = pd.DataFrame()
            
            # Determine transformation based on symbol
            if symbol == '$VIX.X':
                return self._transform_vix_index(raw_data, symbol, interval_unit, interval_value)
            elif symbol.startswith('VX'):
                return self._transform_vx_futures(raw_data, symbol, interval_unit, interval_value)
            else:
                raise DataSourceError(f"Unsupported symbol for transformation: {symbol}")
                
        except Exception as e:
            logger.error(f"Error transforming CBOE data: {e}")
            raise DataSourceError(f"Failed to transform data: {e}")
    
    def _transform_vix_index(self, raw_data: pd.DataFrame, symbol: str,
                           interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Transform VIX index data to standard format.
        
        Args:
            raw_data: Raw VIX index data
            symbol: Symbol (should be $VIX.X)
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            Transformed DataFrame
        """
        # Create a new DataFrame
        df = pd.DataFrame()
        
        # Map columns
        df['timestamp'] = raw_data['Date']
        df['symbol'] = symbol
        df['open'] = raw_data['Open']
        df['high'] = raw_data['High']
        df['low'] = raw_data['Low']
        df['close'] = raw_data['Close']
        
        # Set other required columns
        df['volume'] = None  # VIX index doesn't have volume
        df['open_interest'] = None
        df['up_volume'] = None
        df['down_volume'] = None
        df['interval_unit'] = interval_unit
        df['interval_value'] = interval_value
        df['source'] = 'cboe'
        df['quality'] = 100
        df['adjusted'] = False
        
        return df
    
    def _transform_vx_futures(self, raw_data: pd.DataFrame, symbol: str,
                            interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Transform VX futures data to standard format.
        
        Args:
            raw_data: Raw VX futures data
            symbol: Symbol (e.g., VXF25)
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            Transformed DataFrame
        """
        # Create a new DataFrame
        df = pd.DataFrame()
        
        # Find column mappings
        date_col = None
        open_col = None
        high_col = None
        low_col = None
        close_col = None
        settle_col = None
        volume_col = None
        oi_col = None
        
        for col in raw_data.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                date_col = col
            elif 'open' in col_lower:
                open_col = col
            elif 'high' in col_lower:
                high_col = col
            elif 'low' in col_lower:
                low_col = col
            elif 'close' in col_lower:
                close_col = col
            elif 'settle' in col_lower:
                settle_col = col
            elif 'volume' in col_lower:
                volume_col = col
            elif 'interest' in col_lower or 'oi' in col_lower:
                oi_col = col
                
        # Check that we found the required columns
        required = [date_col, open_col, high_col, low_col, close_col]
        if None in required:
            missing = [col for col, found in zip(
                ['date', 'open', 'high', 'low', 'close'],
                [date_col, open_col, high_col, low_col, close_col]
            ) if found is None]
            raise DataSourceError(f"Missing required columns in VX futures data: {', '.join(missing)}")
        
        # Map columns to standard format
        df['timestamp'] = raw_data[date_col]
        df['symbol'] = symbol
        df['open'] = raw_data[open_col]
        df['high'] = raw_data[high_col]
        df['low'] = raw_data[low_col]
        df['close'] = raw_data[close_col]
        
        # Map optional columns
        if settle_col and settle_col in raw_data.columns:
            df['settle'] = raw_data[settle_col]
        else:
            df['settle'] = df['close']  # Use close if settle not available
            
        if volume_col and volume_col in raw_data.columns:
            df['volume'] = raw_data[volume_col]
        else:
            df['volume'] = None
            
        if oi_col and oi_col in raw_data.columns:
            df['open_interest'] = raw_data[oi_col]
        else:
            df['open_interest'] = None
        
        # Set other required columns
        df['up_volume'] = None
        df['down_volume'] = None
        df['interval_unit'] = interval_unit
        df['interval_value'] = interval_value
        df['source'] = 'cboe'
        df['quality'] = 100
        df['adjusted'] = False
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fix VX futures price scaling issue
        # VX futures prices are sometimes divided by 1000 in CBOE data
        if symbol.startswith('VX'):
            price_cols = ['open', 'high', 'low', 'close', 'settle']
            for col in price_cols:
                if col in df.columns:
                    # Check if prices are too low (likely divided by 1000)
                    median_price = df[col].median()
                    if median_price is not None and median_price < 1.0:
                        logger.info(f"Adjusting {symbol} {col} prices (median: {median_price})")
                        df[col] = df[col] * 1000
        
        return df
    
    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid overloading the CBOE website."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)


# Register the plugin
from .base import DataSourceRegistry
DataSourceRegistry.register("cboe", CBOEDataSource)