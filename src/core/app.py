#!/usr/bin/env python
"""
Application Core Module

This module provides the core application class for the Financial Data System,
tying together configuration, database, logging, and plugins.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from .config import ConfigManager
from .database import Database
from .logging import configure_logging, create_timestamped_log_file

logger = logging.getLogger(__name__)

class Application:
    """Core application class for the Financial Data System."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the application with configuration.
        
        Args:
            config_path: Path to configuration file (None for default)
        """
        # Determine project root
        self.project_root = self._find_project_root()
        
        # Determine configuration path
        if config_path is None:
            config_path = self.project_root / 'config' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = None
        self.db = None
        self._init_config()
    
    def _find_project_root(self) -> Path:
        """
        Find the project root directory.
        
        Returns:
            Path to project root
        """
        # Start with the directory of this file
        current_dir = Path(__file__).resolve().parent
        
        # Go up until we find a directory with config/ and src/
        while current_dir != current_dir.parent:
            if (current_dir / 'config').exists() and (current_dir / 'src').exists():
                return current_dir
            current_dir = current_dir.parent
        
        # If not found, use the directory containing src/
        current_dir = Path(__file__).resolve().parent.parent.parent
        return current_dir
    
    def _init_config(self) -> None:
        """Initialize configuration from file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            self.config = ConfigManager(self.config_path)
            
            # Initialize logging
            log_config = self.config.get('logging', {})
            log_file = log_config.get('file')
            
            if log_file:
                # Convert relative path to absolute
                if not Path(log_file).is_absolute():
                    log_file = self.project_root / log_file
                
                # Create directory if it doesn't exist
                os.makedirs(Path(log_file).parent, exist_ok=True)
            
            configure_logging(
                level=log_config.get('level', 'INFO'),
                log_file=log_file,
                console=log_config.get('console', True),
                format_str=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                module_levels=log_config.get('module_levels', {})
            )
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            # Basic logging setup in case of configuration errors
            configure_logging(level='INFO')
            logger.error(f"Error initializing configuration: {e}")
            raise
    
    def init_database(self, read_only: bool = False) -> None:
        """
        Initialize the database connection.
        
        Args:
            read_only: Whether to open the database in read-only mode
        """
        try:
            db_config = self.config.get('database', {})
            db_path = db_config.get('path')
            
            # Convert relative path to absolute
            if not Path(db_path).is_absolute():
                db_path = self.project_root / db_path
            
            # Create database directory if it doesn't exist and not in read-only mode
            if not read_only:
                os.makedirs(Path(db_path).parent, exist_ok=True)
            
            self.db = Database(db_path, read_only=read_only)
            logger.info(f"Database initialized: {db_path} (read_only={read_only})")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def load_plugins(self) -> None:
        """Load and register all plugins."""
        try:
            # Import plugins module to trigger registration
            from ..data_sources import register_all_plugins
            register_all_plugins()
            
            # Import processors
            from ..processors import register_all_processors
            register_all_processors()
            
            # Import validators
            from ..validation import register_all_validators
            register_all_validators()
            
            logger.info("All plugins registered successfully")
            
        except ImportError as e:
            logger.warning(f"Error importing plugin modules: {e}")
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
            raise
    
    def create_data_source(self, source_name: str) -> Any:
        """
        Create a data source plugin instance.
        
        Args:
            source_name: Name of the data source to create
            
        Returns:
            Data source plugin instance
            
        Raises:
            ValueError: If the data source is not configured or registered
        """
        if self.db is None:
            self.init_database()
        
        try:
            # Import here to avoid circular imports
            from ..data_sources.base import DataSourceRegistry
            
            # Get data source configuration
            source_config = self.config.get_data_source_config(source_name)
            
            # Create the data source
            return DataSourceRegistry.create_source(
                source_name, 
                source_config, 
                self.db
            )
            
        except Exception as e:
            logger.error(f"Error creating data source '{source_name}': {e}")
            raise
    
    def backup_database(self) -> Path:
        """
        Create a backup of the database.
        
        Returns:
            Path to the backup file
        """
        if self.db is None:
            self.init_database()
        
        try:
            db_config = self.config.get('database', {})
            backup_dir = db_config.get('backup_dir')
            
            # Convert relative path to absolute
            if not Path(backup_dir).is_absolute():
                backup_dir = self.project_root / backup_dir
            
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Set retention days from config or default
            retention_days = db_config.get('backup_retention_days', 30)
            
            # Create the backup
            backup_path = self.db.create_backup(backup_dir, retention_days)
            logger.info(f"Database backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def close(self) -> None:
        """Close all resources."""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("Database connection closed")

# Singleton instance
_app = None

def get_app(config_path: Optional[Union[str, Path]] = None) -> Application:
    """
    Get the application singleton instance.
    
    Args:
        config_path: Path to configuration file (None for default)
        
    Returns:
        Application instance
    """
    global _app
    if _app is None:
        _app = Application(config_path)
    return _app