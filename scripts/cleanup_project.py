#!/usr/bin/env python
"""
Cleanup script for the Financial Data System project.

This script reorganizes the project by:
1. Moving outdated files to the deprecated directory
2. Creating necessary directories for new structure
3. Moving scripts to their proper locations
4. Removing redundant or temporary directories
"""

import os
import sys
import shutil
import logging
import csv
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'cleanup_project_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('cleanup')

# Define project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directories to create if they don't exist
DIRS_TO_CREATE = [
    'src/scripts/resources',
    'src/scripts/scripts',
    'tests/legacy',
]

# Files to move to deprecated
FILES_TO_DEPRECATE = [
    ('CLAUDE.md', 'deprecated/CLAUDE.md'),
    ('DB_inspect.bat', 'deprecated/DB_inspect.bat'),
]

# Task directory files to move
TASK_FILES_TO_MOVE = [
    ('tasks/import_tasks.bat', 'src/scripts/scripts/import_tasks.bat'),
    ('tasks/setup_tasks_simple.bat', 'src/scripts/scripts/setup_tasks_simple.bat'),
    ('tasks/setup_user_tasks.bat', 'src/scripts/scripts/setup_user_tasks.bat'),
    ('tasks/setup_vix_data_task.bat', 'src/scripts/scripts/setup_vix_data_task.bat'),
    ('tasks/setup_vix_data_task.ps1', 'src/scripts/scripts/setup_vix_data_task.ps1'),
    ('tasks/vix_update_evening.xml', 'src/scripts/resources/vix_update_evening.xml'),
    ('tasks/vix_update_morning.xml', 'src/scripts/resources/vix_update_morning.xml'),
]

# Test files to move to legacy
TEST_FILES_TO_MOVE = [
    ('tests/check_all_intervals.py', 'tests/legacy/check_all_intervals.py'),
    ('tests/check_intervals.py', 'tests/legacy/check_intervals.py'),
    ('tests/test_fetch_vxf25.py', 'tests/legacy/test_fetch_vxf25.py'),
    ('tests/test_fetch_vxk20.py', 'tests/legacy/test_fetch_vxk20.py'),
    ('tests/test_fetch_vxz24.py', 'tests/legacy/test_fetch_vxz24.py'),
    ('tests/test_market_data_agent.py', 'tests/legacy/test_market_data_agent.py'),
    ('tests/test_market_data_fetcher.py', 'tests/legacy/test_market_data_fetcher.py'),
    ('tests/test_schema.py', 'tests/legacy/test_schema.py'),
]

# Directories to remove
DIRS_TO_REMOVE = [
    'Projects/data-management/financial-data-system',
    'financial_data_system_new',
]

def create_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in DIRS_TO_CREATE:
        full_path = PROJECT_ROOT / dir_path
        if not full_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {dir_path}")

def move_files(file_mappings):
    """
    Move files according to the provided mappings.
    
    Args:
        file_mappings: List of (source, destination) tuples
    """
    for source, dest in file_mappings:
        source_path = PROJECT_ROOT / source
        dest_path = PROJECT_ROOT / dest
        
        if not source_path.exists():
            logger.warning(f"Source file doesn't exist: {source}")
            continue
        
        # Create parent directory if it doesn't exist
        dest_dir = dest_path.parent
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if destination exists
            if dest_path.exists():
                # Back up the existing file
                backup_path = dest_path.with_suffix(f"{dest_path.suffix}.bak")
                logger.warning(f"Destination already exists: {dest}. Creating backup at {backup_path}")
                shutil.copy2(dest_path, backup_path)
            
            # Move the file
            logger.info(f"Moving {source} to {dest}")
            shutil.copy2(source_path, dest_path)  # Preserve metadata
            
            # After successful copy, remove the original
            source_path.unlink()
            logger.info(f"Removed original: {source}")
        except Exception as e:
            logger.error(f"Error moving {source} to {dest}: {e}")

def remove_directories():
    """Remove directories that are no longer needed."""
    for dir_path in DIRS_TO_REMOVE:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            # Check if it's actually a directory
            if full_path.is_dir():
                try:
                    logger.info(f"Removing directory: {dir_path}")
                    # Use shutil.rmtree with error handling
                    shutil.rmtree(full_path, ignore_errors=False)
                except PermissionError:
                    logger.warning(f"Permission denied when removing {dir_path}. Try manual removal.")
                except OSError as e:
                    logger.error(f"Error removing {dir_path}: {e}")
            else:
                logger.warning(f"{dir_path} exists but is not a directory")
        else:
            logger.info(f"Directory doesn't exist: {dir_path}")

def update_entry_points():
    """Update the entry point batch files to use the new structure."""
    # Update update_market_data.bat to use the wrapper
    update_market_data_path = PROJECT_ROOT / 'update_market_data.bat'
    if update_market_data_path.exists():
        # Read current content
        with open(update_market_data_path, 'r') as f:
            content = f.read()
        
        # Check if it's already updated
        if 'update_all_market_data_wrapper.py' in content:
            logger.info("update_market_data.bat already uses the wrapper")
        else:
            # Create backup
            backup_path = update_market_data_path.with_suffix('.bat.bak')
            shutil.copy2(update_market_data_path, backup_path)
            logger.info(f"Created backup of update_market_data.bat at {backup_path}")
            
            # Update the script to use the wrapper
            # Find the line that has update_all_market_data.py
            lines = content.split('\n')
            updated = False
            for i, line in enumerate(lines):
                if 'update_all_market_data.py' in line and not 'update_all_market_data_wrapper.py' in line:
                    # Replace with wrapper
                    lines[i] = line.replace('update_all_market_data.py', 'update_all_market_data_wrapper.py')
                    updated = True
                    break
            
            if updated:
                # Write the updated content
                with open(update_market_data_path, 'w') as f:
                    f.write('\n'.join(lines))
                logger.info("Updated update_market_data.bat to use the wrapper")
            else:
                logger.warning("Could not find update_all_market_data.py reference in update_market_data.bat")
    else:
        logger.warning("update_market_data.bat not found")

def clean_up_temp_database_scripts():
    """
    Move temporary database scripts to the deprecated directory.
    """
    # Find temp database scripts
    temp_scripts_dir = PROJECT_ROOT / 'src' / 'scripts' / 'database'
    if temp_scripts_dir.exists():
        # Create deprecated database scripts directory if it doesn't exist
        deprecated_dir = PROJECT_ROOT / 'deprecated' / 'database_scripts'
        deprecated_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all temp_*.py files
        temp_scripts = list(temp_scripts_dir.glob('temp_*.py'))
        
        for script in temp_scripts:
            # Destination in deprecated directory
            dest_path = deprecated_dir / script.name
            
            try:
                # Move the file
                logger.info(f"Moving {script.relative_to(PROJECT_ROOT)} to deprecated/database_scripts/")
                shutil.copy2(script, dest_path)  # Preserve metadata
                
                # After successful copy, remove the original
                script.unlink()
                logger.info(f"Removed original: {script.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                logger.error(f"Error moving {script.name}: {e}")

def create_deprecation_notice():
    """
    Create a deprecation notice for files that are deprecated but still needed.
    """
    notice_text = """
# DEPRECATION NOTICE

This file is deprecated and will be removed in a future version.
It is maintained temporarily for backward compatibility.

Please use the new components in the src/ directory instead.
See docs/NEW_ARCHITECTURE.md for details on the new architecture.
"""
    
    # Add deprecation notice to update_market_data.bat
    update_market_data_path = PROJECT_ROOT / 'update_market_data.bat'
    if update_market_data_path.exists():
        with open(update_market_data_path, 'r') as f:
            content = f.read()
        
        # Check if notice is already added
        if 'DEPRECATION NOTICE' not in content:
            # Convert notice to batch file comment format
            batch_notice = '\n'.join([f'REM {line}' for line in notice_text.strip().split('\n')])
            
            # Add notice after the first line
            lines = content.split('\n')
            lines.insert(1, batch_notice)
            
            # Write updated content
            with open(update_market_data_path, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info("Added deprecation notice to update_market_data.bat")

def main():
    """Main function to execute the cleanup process."""
    logger.info("Starting project cleanup")
    
    # Create necessary directories
    create_directories()
    
    # Create deprecation notice
    create_deprecation_notice()
    
    # Move files to deprecated
    logger.info("Moving files to deprecated directory")
    move_files(FILES_TO_DEPRECATE)
    
    # Move task files
    logger.info("Moving task files")
    move_files(TASK_FILES_TO_MOVE)
    
    # Move test files
    logger.info("Moving test files")
    move_files(TEST_FILES_TO_MOVE)
    
    # Clean up temp database scripts
    logger.info("Cleaning up temporary database scripts")
    clean_up_temp_database_scripts()
    
    # Update entry points
    logger.info("Updating entry point batch files")
    update_entry_points()
    
    # Remove directories last (after moving files)
    logger.info("Removing redundant directories")
    remove_directories()
    
    logger.info("Project cleanup completed")

if __name__ == "__main__":
    main()