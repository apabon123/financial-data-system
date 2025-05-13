#!/usr/bin/env python
"""
Logging Configuration Module

This module provides centralized logging configuration for the Financial Data System.

Features:
- Standardized logging setup
- Console and file output
- Configurable log levels
- Module-specific loggers
"""

import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

# Default format string
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class LoggingManager:
    """Centralized logging configuration for the Financial Data System."""
    
    def __init__(self):
        """Initialize the logging manager."""
        self.root_logger = logging.getLogger()
        self.configured = False
        
    def configure(self, 
                  level: str = 'INFO',
                  log_file: Optional[Union[str, Path]] = None,
                  console: bool = True,
                  format_str: str = DEFAULT_FORMAT,
                  date_format: str = DEFAULT_DATE_FORMAT,
                  module_levels: Dict[str, str] = None) -> None:
        """
        Configure the logging system.
        
        Args:
            level: Default log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_file: Path to log file (None for no file logging)
            console: Whether to log to console
            format_str: Log message format string
            date_format: Date format for log messages
            module_levels: Dictionary of module names and their log levels
        """
        # Reset existing handlers
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
            
        # Set the root logger level
        level_obj = self._get_log_level(level)
        self.root_logger.setLevel(level_obj)
        
        # Create formatter
        formatter = logging.Formatter(format_str, date_format)
        
        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.root_logger.addHandler(console_handler)
            
        # Add file handler if log_file specified
        if log_file:
            log_path = Path(log_file)
            
            # Create directory if it doesn't exist
            os.makedirs(log_path.parent, exist_ok=True)
            
            # Determine if we should use a rotating file handler
            if log_path.suffix == '.log':
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path, maxBytes=10*1024*1024, backupCount=5
                )
            else:
                file_handler = logging.FileHandler(log_path)
                
            file_handler.setFormatter(formatter)
            self.root_logger.addHandler(file_handler)
            
        # Configure module-specific log levels
        if module_levels:
            for module, module_level in module_levels.items():
                level_obj = self._get_log_level(module_level)
                logging.getLogger(module).setLevel(level_obj)
                
        self.configured = True
        logging.info("Logging configured successfully")
        
    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Configure logging from a dictionary (usually from config.yaml).
        
        Args:
            config: Configuration dictionary with logging settings
        """
        level = config.get('level', 'INFO')
        log_file = config.get('file')
        console = config.get('console', True)
        format_str = config.get('format', DEFAULT_FORMAT)
        date_format = config.get('date_format', DEFAULT_DATE_FORMAT)
        module_levels = config.get('module_levels', {})
        
        self.configure(
            level=level,
            log_file=log_file,
            console=console,
            format_str=format_str,
            date_format=date_format,
            module_levels=module_levels
        )
        
    @staticmethod
    def _get_log_level(level: str) -> int:
        """
        Convert a log level string to the corresponding logging level.
        
        Args:
            level: Log level string ('DEBUG', 'INFO', etc.)
            
        Returns:
            Logging level constant
            
        Raises:
            ValueError: If the log level is invalid
        """
        level_str = level.upper()
        if level_str == 'DEBUG':
            return logging.DEBUG
        elif level_str == 'INFO':
            return logging.INFO
        elif level_str == 'WARNING':
            return logging.WARNING
        elif level_str == 'ERROR':
            return logging.ERROR
        elif level_str == 'CRITICAL':
            return logging.CRITICAL
        else:
            raise ValueError(f"Invalid log level: {level}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name (usually __name__ or module path)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    def add_log_handler(self, handler: logging.Handler, 
                        formatter: Optional[logging.Formatter] = None) -> None:
        """
        Add a custom log handler to the root logger.
        
        Args:
            handler: Log handler to add
            formatter: Formatter for the handler (None to use default)
        """
        if formatter:
            handler.setFormatter(formatter)
        self.root_logger.addHandler(handler)


# Singleton instance
_manager = LoggingManager()

def configure_logging(level: str = 'INFO',
                      log_file: Optional[Union[str, Path]] = None,
                      console: bool = True,
                      format_str: str = DEFAULT_FORMAT,
                      date_format: str = DEFAULT_DATE_FORMAT,
                      module_levels: Dict[str, str] = None) -> None:
    """
    Configure the logging system (convenience function).
    
    Args:
        level: Default log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (None for no file logging)
        console: Whether to log to console
        format_str: Log message format string
        date_format: Date format for log messages
        module_levels: Dictionary of module names and their log levels
    """
    _manager.configure(
        level=level,
        log_file=log_file,
        console=console,
        format_str=format_str,
        date_format=date_format,
        module_levels=module_levels
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name (convenience function).
    
    Args:
        name: Logger name (usually __name__ or module path)
        
    Returns:
        Logger instance
    """
    return _manager.get_logger(name)

def add_log_handler(handler: logging.Handler,
                    formatter: Optional[logging.Formatter] = None) -> None:
    """
    Add a custom log handler to the root logger (convenience function).
    
    Args:
        handler: Log handler to add
        formatter: Formatter for the handler (None to use default)
    """
    _manager.add_log_handler(handler, formatter)

def configure_from_dict(config: Dict[str, Any]) -> None:
    """
    Configure logging from a dictionary (convenience function).
    
    Args:
        config: Configuration dictionary with logging settings
    """
    _manager.configure_from_dict(config)

def create_timestamped_log_file(base_dir: Union[str, Path], 
                               prefix: str = 'log', 
                               ext: str = 'log') -> Path:
    """
    Create a log file with a timestamp in the name.
    
    Args:
        base_dir: Base directory for log files
        prefix: Prefix for the log file name
        ext: File extension
        
    Returns:
        Path to the timestamped log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir / f"{prefix}_{timestamp}.{ext}"