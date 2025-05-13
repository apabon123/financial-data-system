#!/usr/bin/env python
"""
Configuration Management Module

This module provides a unified interface for loading, validating, and accessing
configuration settings for the Financial Data System.

Features:
- YAML configuration loading with validation
- Environment variable integration
- Default value handling
- Configuration inheritance
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import re

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

class ConfigManager:
    """Centralized configuration management for the Financial Data System."""
    
    def __init__(self, config_path: Union[str, Path], env_prefix: str = "FDS_"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the YAML configuration file
            env_prefix: Prefix for environment variables that should override config settings
        """
        self.config_path = Path(config_path)
        self.env_prefix = env_prefix
        self.config: Dict[str, Any] = {}
        
        # Load the configuration
        self._load_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate the configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _load_config(self) -> None:
        """Load the configuration from the YAML file."""
        try:
            if not self.config_path.exists():
                raise ConfigError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            if not isinstance(self.config, dict):
                raise ConfigError("Configuration must be a dictionary")
                
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to the configuration."""
        # Look for environment variables with the specified prefix
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Convert environment variable name to config key path
                # e.g., FDS_DATABASE_PATH -> database.path
                key_path = env_var[len(self.env_prefix):].lower().replace('__', '.').replace('_', '.')
                
                # Apply the override
                self._set_nested_value(self.config, key_path, value)
                logger.debug(f"Applied environment override: {env_var} -> {key_path}")
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: str) -> None:
        """
        Set a value in a nested dictionary using a dot-separated key path.
        
        Args:
            config_dict: The configuration dictionary to modify
            key_path: Dot-separated path to the target setting
            value: Value to set
        """
        keys = key_path.split('.')
        current = config_dict
        
        # Navigate to the containing dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the value, attempting to convert to the appropriate type
        try:
            # Try to parse as JSON for complex types
            current[keys[-1]] = json.loads(value)
        except json.JSONDecodeError:
            # If not valid JSON, use the string value
            current[keys[-1]] = value
    
    def _validate_config(self) -> None:
        """Validate the configuration structure and required fields."""
        # Check for required top-level sections
        required_sections = ['data_sources', 'database', 'logging']
        for section in required_sections:
            if section not in self.config:
                raise ConfigError(f"Missing required configuration section: {section}")
        
        # Validate database configuration
        if 'path' not in self.config.get('database', {}):
            raise ConfigError("Database configuration must include 'path'")
        
        # Validate data sources
        for source_name, source_config in self.config.get('data_sources', {}).items():
            if 'type' not in source_config:
                raise ConfigError(f"Data source '{source_name}' must specify 'type'")
    
    def _validate_frequencies(self) -> None:
        """Validate and standardize frequency representations in market symbols config."""
        for asset_type in ['futures', 'indices', 'equities']:
            if asset_type not in self.config:
                continue
                
            for i, item in enumerate(self.config[asset_type]):
                if 'frequencies' in item:
                    # Standardize frequencies to dictionary format
                    freqs = item['frequencies']
                    standardized_freqs = []
                    
                    if isinstance(freqs, list):
                        for freq in freqs:
                            if isinstance(freq, dict):
                                # Already in dictionary format
                                standardized_freqs.append(freq)
                            elif isinstance(freq, str):
                                # Convert string format (e.g., "1min", "daily")
                                if freq == 'daily':
                                    standardized_freqs.append({
                                        'name': 'daily',
                                        'interval': 1,
                                        'unit': 'daily'
                                    })
                                elif 'min' in freq:
                                    try:
                                        minutes = int(freq.replace('min', ''))
                                        standardized_freqs.append({
                                            'name': f'{minutes}min',
                                            'interval': minutes,
                                            'unit': 'minute'
                                        })
                                    except ValueError:
                                        logger.warning(f"Invalid minute format: {freq}")
                    
                    self.config[asset_type][i]['frequencies'] = standardized_freqs
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated key path.
        
        Args:
            key_path: Dot-separated path to the target setting
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default if not found
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific data source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Data source configuration dictionary
            
        Raises:
            ConfigError: If the data source is not configured
        """
        sources = self.get('data_sources', {})
        if source_name not in sources:
            raise ConfigError(f"Data source not configured: {source_name}")
        
        return sources[source_name]
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific market symbol.
        
        Args:
            symbol: Market symbol (e.g., ES, VX, SPY)
            
        Returns:
            Symbol configuration dictionary, or empty dict if not found
        """
        # Search in all asset types
        for asset_type in ['futures', 'indices', 'equities']:
            assets = self.get(asset_type, [])
            for asset in assets:
                if asset.get('symbol') == symbol or asset.get('base_symbol') == symbol:
                    return asset
        
        # If not found as a direct match, try to match continuous contracts
        if symbol.startswith('@'):
            parts = symbol.split('=')
            if len(parts) >= 1:
                base = parts[0].lstrip('@')
                for asset in self.get('futures', []):
                    if asset.get('base_symbol') == base:
                        return asset
        
        return {}
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create a default configuration structure.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "database": {
                "path": "./data/financial_data.duckdb",
                "backup_dir": "./backups",
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/financial_data_system.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "data_sources": {
                "tradestation": {
                    "type": "tradestation",
                    "api_key_env": "TRADESTATION_API_KEY",
                    "api_secret_env": "TRADESTATION_API_SECRET",
                    "rate_limit": 60
                },
                "cboe": {
                    "type": "cboe",
                    "url_template": "https://www.cboe.com/us/futures/market_statistics/historical_data/products/csv/{symbol}/"
                }
            },
            "settings": {
                "default_start_date": "2004-03-26",
                "default_lookback_days": 90,
                "roll_proximity_days": 7
            },
            "futures": [],
            "indices": [],
            "equities": []
        }
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file (defaults to the original path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: Union[str, Path]) -> ConfigManager:
    """
    Helper function to load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)