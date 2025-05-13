"""
Core Module Package

This package contains the core functionality of the Financial Data System,
including configuration, database management, and logging.
"""

from .config import ConfigManager, load_config
from .database import Database, get_database
from .logging import (configure_logging, get_logger, add_log_handler,
                     configure_from_dict, create_timestamped_log_file)
from .app import Application, get_app

__all__ = [
    'ConfigManager', 'load_config',
    'Database', 'get_database',
    'configure_logging', 'get_logger', 'add_log_handler',
    'configure_from_dict', 'create_timestamped_log_file',
    'Application', 'get_app'
]