#!/usr/bin/env python
"""
Logging Configuration Module

This module provides a standardized logging configuration for the financial data system.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(verbose=False, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
        log_file: Path to the log file (default: logs/app_YYYYMMDD.log)
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set up log file path
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"app_{timestamp}.log"
    
    # Set up logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up specific loggers
    loggers = [
        "backup_database",
        "scheduled_backup",
        "ai_interface",
        "ai_interface_llm"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Log the start of the application
    logging.info(f"Logging initialized with level: {logging.getLevelName(level)}")
    logging.info(f"Log file: {log_file}")
    
    return log_file 