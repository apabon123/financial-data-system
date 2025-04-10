#!/usr/bin/env python
"""
Scheduled Database Backup Script

This script is designed to be run as a scheduled task (cron job on Unix/Linux/Mac
or Task Scheduler on Windows) to automatically back up the database at regular intervals.

It reads configuration from environment variables or uses defaults:
- BACKUP_DIR: Directory to store backups (default: ./backups)
- RETENTION_DAYS: Number of days to keep backups (default: 30)
- DATABASE_PATH: Path to the database file (default: ./data/financial_data.duckdb)

Usage:
    python scheduled_backup.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logging_config import setup_logging
from src.scripts.backup_database import create_backup, cleanup_old_backups

# Set up logging
logger = logging.getLogger("scheduled_backup")

def main():
    """Main function for scheduled backup."""
    # Set up logging
    setup_logging(verbose=False)
    
    # Get configuration from environment variables or use defaults
    database_path = os.getenv("DATABASE_PATH", "./data/financial_data.duckdb")
    backup_dir = os.getenv("BACKUP_DIR", "./backups")
    retention_days = int(os.getenv("RETENTION_DAYS", "30"))
    
    # Log the start of the backup process
    logger.info(f"Starting scheduled backup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Database: {database_path}")
    logger.info(f"Backup directory: {backup_dir}")
    logger.info(f"Retention period: {retention_days} days")
    
    # Check if database exists
    if not os.path.exists(database_path):
        logger.error(f"Database file not found: {database_path}")
        return 1
    
    # Create backup
    backup_path = create_backup(database_path, backup_dir)
    
    if backup_path:
        logger.info(f"Backup created successfully: {backup_path}")
        
        # Clean up old backups
        removed_files = cleanup_old_backups(backup_dir, retention_days)
        
        if removed_files:
            logger.info(f"Removed {len(removed_files)} old backup(s)")
            for file in removed_files:
                logger.info(f"  - {file}")
        else:
            logger.info("No old backups to remove")
        
        return 0
    else:
        logger.error("Backup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 