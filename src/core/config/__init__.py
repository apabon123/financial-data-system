#!/usr/bin/env python
"""
Configuration module for Financial Data System.

This module provides a centralized configuration management system that supports:
- Loading and merging multiple YAML configuration files
- Cross-file references
- Template inheritance
- Environment-specific overrides
- Schema validation

It serves as the main entry point for accessing configuration settings within the application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .loader import ConfigLoader, ConfigManager

logger = logging.getLogger(__name__)

# Default configuration directories
DEFAULT_CONFIG_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent.parent, 'config')
DEFAULT_SCHEMA_DIR = os.path.join(DEFAULT_CONFIG_DIR, 'schemas')

# Global instance of the configuration manager
_config_manager = None

def init_config(config_dir: str = None, schema_dir: str = None, 
                environment: str = None, legacy_support: bool = True):
    """
    Initialize the global configuration manager.
    
    Args:
        config_dir: Path to the directory containing configuration files
        schema_dir: Path to the directory containing schema files
        environment: Environment name for environment-specific overrides
        legacy_support: Whether to support the legacy single-file configuration
        
    Returns:
        The initialized configuration manager
    """
    global _config_manager
    
    # Use default directories if not specified
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    schema_dir = schema_dir or DEFAULT_SCHEMA_DIR
    
    # Create the configuration manager
    _config_manager = ConfigManager(
        config_dir=config_dir,
        schema_dir=schema_dir,
        environment=environment,
        legacy_support=legacy_support
    )
    
    logger.info(f"Initialized configuration manager with config_dir={config_dir}")
    return _config_manager

def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager.
    
    Returns:
        The global configuration manager
    
    Raises:
        RuntimeError: If the configuration manager has not been initialized
    """
    global _config_manager
    
    if _config_manager is None:
        # Auto-initialize with default settings
        init_config()
    
    return _config_manager

def get_section(section: str) -> Dict[str, Any]:
    """
    Get a specific section from the configuration.
    
    Args:
        section: Section name
        
    Returns:
        Configuration section as a dictionary
    """
    return get_config_manager().get_section(section)

def get_item(section: str, item_name: str) -> Dict[str, Any]:
    """
    Get a specific item from a section.
    
    Args:
        section: Section name
        item_name: Item name
        
    Returns:
        Item as a dictionary
    """
    return get_config_manager().get_item(section, item_name)

def get_value(path: str, default: Any = None) -> Any:
    """
    Get a specific value using a dot-notation path.
    
    Args:
        path: Path to the value (e.g., "futures.ES.contract_info.patterns")
        default: Default value if not found
        
    Returns:
        The value at the specified path, or the default if not found
    """
    return get_config_manager().get_value(path, default)

def reload_config():
    """Reload the configuration."""
    get_config_manager().reload()

def set_environment(environment: str):
    """
    Set the environment.
    
    Args:
        environment: Environment name
    """
    get_config_manager().set_environment(environment)

def get_environment() -> str:
    """
    Get the current environment.
    
    Returns:
        The current environment name
    """
    return get_config_manager().get_environment()

def convert_legacy_to_new(output_dir: str = None):
    """
    Convert legacy configuration to the new structure.
    
    Args:
        output_dir: Directory to save the new configuration files
    """
    get_config_manager().convert_legacy_to_new(output_dir)

# Initialize the configuration manager on module import
# This ensures that config is available whenever the module is imported
init_config()