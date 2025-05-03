#!/usr/bin/env python
"""
Simple Database Backup Script

This script creates a backup of the financial data database and manages the retention policy.
It's a simplified version that doesn't rely on imports from the src package.

Usage:
    python simple_backup.py [options]
"""

import os
import sys
import shutil
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

def setup_logging(log_dir="./logs", verbose=False):
    """Set up logging to both console and file."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"db_backup_{timestamp}.log")
    
    # Set up logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("backup_database")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Backup the financial data database")
    parser.add_argument(
        "-d", "--database",
        default="./data/financial_data.duckdb",
        help="Path to the database file"
    )
    parser.add_argument(
        "-o", "--output",
        default="./backups",
        help="Output directory for backups"
    )
    parser.add_argument(
        "-r", "--retention",
        type=int,
        default=30,
        help="Number of days to keep backups"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-l", "--log-dir",
        default="./logs",
        help="Directory to store log files"
    )
    return parser.parse_args()

def create_backup(database_path, output_dir, logger):
    """Create a backup of the database."""
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = os.path.basename(database_path)
        backup_name = f"{os.path.splitext(db_name)[0]}_{timestamp}.duckdb"
        backup_path = os.path.join(output_dir, backup_name)
        
        # Log the backup start
        logger.info(f"Starting backup of database: {database_path}")
        
        # Copy the database file
        shutil.copy2(database_path, backup_path)
        
        # Log success and backup details
        logger.info(f"Created backup: {backup_path}")
        logger.info(f"Backup size: {os.path.getsize(backup_path) / (1024*1024):.2f} MB")
        
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}", exc_info=True)
        return None

def cleanup_old_backups(output_dir, retention_days, logger):
    """Remove backups older than the retention period."""
    removed_files = []
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    try:
        # Log start of cleanup
        logger.info(f"Starting cleanup of backups older than {retention_days} days")
        
        # Get all backup files
        backup_files = [f for f in os.listdir(output_dir) if f.endswith('.duckdb')]
        logger.info(f"Found {len(backup_files)} backup files in {output_dir}")
        
        for backup_file in backup_files:
            # Extract timestamp from filename (format: name_YYYYMMDD_HHMMSS.duckdb)
            try:
                timestamp_str = backup_file.split('_')[-2] + '_' + backup_file.split('_')[-1].split('.')[0]
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Remove if older than retention period
                if backup_date < cutoff_date:
                    file_path = os.path.join(output_dir, backup_file)
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    os.remove(file_path)
                    removed_files.append(backup_file)
                    logger.info(f"Removed old backup: {backup_file} ({file_size:.2f} MB)")
            except (ValueError, IndexError):
                # Skip files that don't match the expected format
                logger.warning(f"Skipping file with unexpected format: {backup_file}")
                
        logger.info(f"Cleanup completed. Removed {len(removed_files)} old backup(s)")
        return removed_files
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}", exc_info=True)
        return removed_files

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.verbose)
    
    logger.info("=== Database Backup Process Started ===")
    logger.info(f"Database path: {args.database}")
    logger.info(f"Backup directory: {args.output}")
    logger.info(f"Retention period: {args.retention} days")
    
    # Check if database exists
    if not os.path.exists(args.database):
        logger.error(f"Database file not found: {args.database}")
        logger.info("=== Database Backup Process Failed ===")
        return 1
    
    # Create backup
    logger.info("Creating database backup...")
    backup_path = create_backup(args.database, args.output, logger)
    
    if backup_path:
        logger.info(f"Backup created successfully: {backup_path}")
        
        # Clean up old backups
        logger.info(f"Cleaning up backups older than {args.retention} days...")
        removed_files = cleanup_old_backups(args.output, args.retention, logger)
        
        if removed_files:
            logger.info(f"Removed {len(removed_files)} old backup(s)")
            for file in removed_files:
                logger.debug(f"  - {file}")
        else:
            logger.info("No old backups to remove")
        
        logger.info("=== Database Backup Process Completed Successfully ===")
        return 0
    else:
        logger.error("Backup failed")
        logger.info("=== Database Backup Process Failed ===")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 