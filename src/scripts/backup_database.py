#!/usr/bin/env python
"""
Database Backup Script

This script creates a backup of the financial data database and manages the retention policy.
It can be run manually or scheduled as a cron job for regular backups.

Usage:
    python backup_database.py [options]

Options:
    -d, --database PATH    Path to the database file (default: ./data/financial_data.duckdb)
    -o, --output DIR       Output directory for backups (default: ./backups)
    -r, --retention DAYS   Number of days to keep backups (default: 30)
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message
"""

import os
import sys
import shutil
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logging_config import setup_logging
from rich.console import Console
from rich.panel import Panel

# Set up logging
logger = logging.getLogger("backup_database")
console = Console()

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
    return parser.parse_args()

def create_backup(database_path: str, output_dir: str) -> Optional[str]:
    """
    Create a backup of the database.
    
    Args:
        database_path: Path to the database file
        output_dir: Directory to store the backup
        
    Returns:
        Path to the backup file if successful, None otherwise
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = os.path.basename(database_path)
        backup_name = f"{os.path.splitext(db_name)[0]}_{timestamp}.duckdb"
        backup_path = os.path.join(output_dir, backup_name)
        
        # Copy the database file
        shutil.copy2(database_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        return None

def cleanup_old_backups(output_dir: str, retention_days: int) -> List[str]:
    """
    Remove backups older than the retention period.
    
    Args:
        output_dir: Directory containing backups
        retention_days: Number of days to keep backups
        
    Returns:
        List of removed backup files
    """
    removed_files = []
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    try:
        # Get all backup files
        backup_files = [f for f in os.listdir(output_dir) if f.endswith('.duckdb')]
        
        for backup_file in backup_files:
            # Extract timestamp from filename (format: name_YYYYMMDD_HHMMSS.duckdb)
            try:
                timestamp_str = backup_file.split('_')[-2] + '_' + backup_file.split('_')[-1].split('.')[0]
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Remove if older than retention period
                if backup_date < cutoff_date:
                    file_path = os.path.join(output_dir, backup_file)
                    os.remove(file_path)
                    removed_files.append(backup_file)
                    logger.info(f"Removed old backup: {backup_file}")
            except (ValueError, IndexError):
                # Skip files that don't match the expected format
                logger.warning(f"Skipping file with unexpected format: {backup_file}")
                
        return removed_files
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")
        return removed_files

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    
    # Check if database exists
    if not os.path.exists(args.database):
        logger.error(f"Database file not found: {args.database}")
        console.print(f"[red]Error: Database file not found: {args.database}[/red]")
        return 1
    
    # Create backup
    console.print(Panel("Creating database backup...", title="Backup", border_style="blue"))
    backup_path = create_backup(args.database, args.output)
    
    if backup_path:
        console.print(f"[green]Backup created successfully: {backup_path}[/green]")
        
        # Clean up old backups
        console.print(Panel(f"Cleaning up backups older than {args.retention} days...", 
                           title="Cleanup", border_style="blue"))
        removed_files = cleanup_old_backups(args.output, args.retention)
        
        if removed_files:
            console.print(f"[yellow]Removed {len(removed_files)} old backup(s):[/yellow]")
            for file in removed_files:
                console.print(f"  - {file}")
        else:
            console.print("[green]No old backups to remove.[/green]")
        
        return 0
    else:
        console.print("[red]Backup failed.[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 