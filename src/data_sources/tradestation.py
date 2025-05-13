#!/usr/bin/env python
"""
TradeStation Data Source Plugin

This module implements a data source plugin for fetching market data from the
TradeStation API.
"""

import os
import logging
import requests
import time
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import DataSourcePlugin, DataSourceError
from ..core.database import Database

logger = logging.getLogger(__name__)

class TradeStationAuth:
    """TradeStation API authentication handler."""
    
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize TradeStation authentication.
        
        Args:
            client_id: TradeStation API client ID
            client_secret: TradeStation API client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.refresh_token = None
        self.expires_at = 0
        
        self.auth_url = "https://signin.tradestation.com/oauth/token"
        self.api_url = "https://api.tradestation.com/v3"
    
    def authenticate(self) -> bool:
        """
        Authenticate with the TradeStation API.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        if self.is_authenticated():
            return True
            
        try:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "MarketData ReadAccount"
            }
            
            response = requests.post(self.auth_url, data=auth_data, timeout=30)
            response.raise_for_status()
            
            auth_response = response.json()
            self.access_token = auth_response.get("access_token")
            self.refresh_token = auth_response.get("refresh_token")
            expiry = auth_response.get("expires_in", 3600)
            self.expires_at = time.time() + expiry - 60  # Buffer of 60 seconds
            
            logger.info("TradeStation authentication successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TradeStation authentication failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """
        Check if the current authentication is valid.
        
        Returns:
            True if authenticated and token is valid, False otherwise
        """
        return (
            self.access_token is not None and
            time.time() < self.expires_at
        )
    
    def refresh(self) -> bool:
        """
        Refresh the authentication token.
        
        Returns:
            True if refresh was successful, False otherwise
        """
        if not self.refresh_token:
            return self.authenticate()
            
        try:
            refresh_data = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token
            }
            
            response = requests.post(self.auth_url, data=refresh_data, timeout=30)
            response.raise_for_status()
            
            auth_response = response.json()
            self.access_token = auth_response.get("access_token")
            self.refresh_token = auth_response.get("refresh_token")
            expiry = auth_response.get("expires_in", 3600)
            self.expires_at = time.time() + expiry - 60  # Buffer of 60 seconds
            
            logger.info("TradeStation token refresh successful")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TradeStation token refresh failed: {e}")
            self.access_token = None
            self.refresh_token = None
            self.expires_at = 0
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get authorization headers for API requests.
        
        Returns:
            Dictionary of request headers
        """
        if not self.is_authenticated():
            self.authenticate()
            
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }


class TradeStationDataSource(DataSourcePlugin):
    """TradeStation API data source implementation."""
    
    def __init__(self, config: Dict[str, Any], db: Database):
        """
        Initialize the TradeStation data source.
        
        Args:
            config: Configuration dictionary for this data source
            db: Database instance for storage
        """
        self.auth = None
        super().__init__(config, db)
        
        # API base URL
        self.api_url = "https://api.tradestation.com/v3"
        
        # Initialize authentication
        client_id = os.environ.get(self.config.get('api_key_env', 'TRADESTATION_API_KEY'))
        client_secret = os.environ.get(self.config.get('api_secret_env', 'TRADESTATION_API_SECRET'))
        
        if not client_id or not client_secret:
            raise DataSourceError("TradeStation API credentials not found in environment variables")
            
        self.auth = TradeStationAuth(client_id, client_secret)
        
        # Configure rate limiting
        self.rate_limit = self.config.get('rate_limit', 60)  # Requests per minute
        self.request_interval = 60.0 / self.rate_limit
        self.last_request_time = 0
    
    def validate_config(self) -> None:
        """
        Validate the configuration for this data source.
        
        Raises:
            DataSourceError: If configuration is invalid
        """
        required_keys = ['api_key_env', 'api_secret_env']
        for key in required_keys:
            if key not in self.config:
                raise DataSourceError(f"Missing required configuration key: {key}")
                
        # Check if environment variables are set
        api_key_env = self.config.get('api_key_env')
        api_secret_env = self.config.get('api_secret_env')
        
        if not os.environ.get(api_key_env):
            raise DataSourceError(f"Environment variable {api_key_env} not set")
            
        if not os.environ.get(api_secret_env):
            raise DataSourceError(f"Environment variable {api_secret_env} not set")
    
    def fetch_data(self, symbol: str, start_date: Union[str, date, datetime], 
                  end_date: Union[str, date, datetime],
                  interval_unit: str = 'daily', interval_value: int = 1) -> pd.DataFrame:
        """
        Fetch data for the specified symbol and date range from TradeStation.
        
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
        if not self.auth.authenticate():
            raise DataSourceError("TradeStation authentication failed")
            
        # Convert interval to TradeStation format
        ts_interval = self._to_ts_interval(interval_unit, interval_value)
        
        # Format dates
        start_date_str = self._format_date(start_date)
        end_date_str = self._format_date(end_date)
        
        # Build request URL
        endpoint = f"{self.api_url}/marketdata/barcharts/{symbol}"
        params = {
            "interval": ts_interval,
            "barsback": 10000,  # Maximum number of bars
            "firstdate": start_date_str,
            "lastdate": end_date_str
        }
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Make the request
            response = requests.get(
                endpoint,
                params=params,
                headers=self.auth.get_headers(),
                timeout=60
            )
            
            # Record the request time
            self.last_request_time = time.time()
            
            # Handle errors
            if response.status_code == 401:
                # Try to refresh the token
                if self.auth.refresh():
                    # Retry the request
                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=self.auth.get_headers(),
                        timeout=60
                    )
                else:
                    raise DataSourceError("Failed to refresh authentication token")
            
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            bars = data.get("Bars", [])
            
            if not bars:
                logger.warning(f"No data returned for {symbol} from {start_date_str} to {end_date_str}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            # Return the raw data for transformation
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TradeStation API request failed: {e}")
            raise DataSourceError(f"Failed to fetch data from TradeStation: {e}")
    
    def transform_data(self, raw_data: pd.DataFrame, symbol: str, 
                      interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Transform raw TradeStation data to the standard format.
        
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
            # Rename columns to match our schema
            column_mapping = {
                "TimeStamp": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "OpenInterest": "open_interest"
            }
            
            # Create a new DataFrame with our standard columns
            df = pd.DataFrame()
            
            # Map columns from raw data
            for our_col, ts_col in column_mapping.items():
                if ts_col in raw_data.columns:
                    df[our_col] = raw_data[ts_col]
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add interval information
            df["interval_unit"] = interval_unit
            df["interval_value"] = interval_value
            
            # Add source
            df["source"] = "tradestation"
            
            # Add quality (default 100)
            df["quality"] = 100
            
            # Add optional columns with NaN values if not present
            if "open_interest" not in df.columns:
                df["open_interest"] = pd.NA
                
            # Add up_volume and down_volume (not provided by TradeStation)
            df["up_volume"] = pd.NA
            df["down_volume"] = pd.NA
            
            # Add adjusted flag (assume not adjusted)
            df["adjusted"] = False
            
            # Ensure numeric columns have the right type
            numeric_cols = ["open", "high", "low", "close", "volume", "open_interest", "up_volume", "down_volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            logger.error(f"Error transforming TradeStation data: {e}")
            raise DataSourceError(f"Failed to transform data: {e}")
    
    def _to_ts_interval(self, interval_unit: str, interval_value: int) -> str:
        """
        Convert interval to TradeStation format.
        
        Args:
            interval_unit: Time interval unit (e.g., 'daily', 'minute')
            interval_value: Time interval value (e.g., 1, 5, 15)
            
        Returns:
            TradeStation interval string
            
        Raises:
            DataSourceError: If the interval is not supported
        """
        if interval_unit == 'daily':
            return f"D{interval_value}"
        elif interval_unit == 'minute':
            return f"M{interval_value}"
        elif interval_unit == 'hour':
            return f"H{interval_value}"
        else:
            raise DataSourceError(f"Unsupported interval unit: {interval_unit}")
    
    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid exceeding API limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)


# Register the plugin
from .base import DataSourceRegistry
DataSourceRegistry.register("tradestation", TradeStationDataSource)