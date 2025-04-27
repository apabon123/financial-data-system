#!/usr/bin/env python
"""
Update Batch Scripts

This script updates the batch files to point to the new locations of the Python scripts
after reorganization.
"""

import os
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_batch_scripts.log')
    ]
)
logger = logging.getLogger("update_batch_scripts")

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, 'data', 'archive')

# Make sure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Map of moved files and their new locations
MOVED_FILES = {
    'view_futures_contracts.py': 'src/scripts/market_data/view_futures_contracts.py',
    'update_active_es_nq_futures.py': 'src/scripts/market_data/update_active_es_nq_futures.py',
    'update_march_2025_contracts.py': 'src/scripts/market_data/update_march_2025_contracts.py',
    'fetch_es_nq_2025.py': 'src/scripts/market_data/fetch_es_nq_2025.py',
    'fix_futures_expiration.py': 'src/scripts/market_data/fix_futures_expiration.py',
    'delete_incorrect_contracts.py': 'src/scripts/market_data/delete_incorrect_contracts.py',
    'fix_march_contracts.py': 'src/scripts/market_data/fix_march_contracts.py',
    'check_db_schema.py': 'src/scripts/utility/check_db_schema.py',
    'check_futures_contracts.py': 'src/scripts/utility/check_futures_contracts.py',
    'clean_futures_database.py': 'src/scripts/database/clean_futures_database.py',
    'purge_old_futures_contracts.py': 'src/scripts/database/purge_old_futures_contracts.py',
}

def archive_file(file_path):
    """Archive a file by copying it to the archive directory with a timestamp."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Generate archive filename with timestamp
    filename = os.path.basename(file_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(ARCHIVE_DIR, f"{filename}.{timestamp}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as src_file:
            content = src_file.read()
        
        with open(archive_path, 'w', encoding='utf-8') as dest_file:
            dest_file.write(content)
        
        logger.info(f"Archived {filename} to {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Error archiving {filename}: {e}")
        return False

def update_batch_file(file_path):
    """Update a batch file to use the new script locations."""
    if not os.path.exists(file_path):
        logger.warning(f"Batch file not found: {file_path}")
        return
    
    # Archive the original file first
    if not archive_file(file_path):
        logger.error(f"Failed to archive {file_path}, skipping update")
        return
    
    logger.info(f"Updating batch file: {file_path}")
    
    try:
        # Read the batch file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Update python script references
        for old_path, new_path in MOVED_FILES.items():
            # Match "python old_path" or similar patterns
            pattern = rf'(python\s+)({re.escape(old_path)})'
            content = re.sub(pattern, f'\\1{new_path}', content)
        
        # Write the updated content back
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        logger.info(f"Successfully updated {file_path}")
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")

def main():
    """Main function to update all batch files."""
    logger.info("Starting batch file updates")
    
    # Find all batch files in the project root
    batch_files = [
        os.path.join(PROJECT_ROOT, f) 
        for f in os.listdir(PROJECT_ROOT) 
        if f.endswith('.bat') or f.endswith('.cmd')
    ]
    
    if not batch_files:
        logger.info("No batch files found in the project root")
        return
    
    # Update each batch file
    for batch_file in batch_files:
        update_batch_file(batch_file)
    
    logger.info("Batch file updates complete")

if __name__ == "__main__":
    main() 