#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reorganizes project files to maintain a clean directory structure.
Moves files from root to appropriate subdirectories according to project standards.
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/reorganize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
def ensure_directories():
    directories = [
        "output",
        "output/reports",
        "output/data",
        "logs",
        "src/scripts/analysis",
        "src/scripts/utility",
        "src/scripts/market_data",
        "data/archive"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# File movement mappings
file_movements = {
    # Python scripts to src/scripts
    "show_vix_continuous_data.py": "src/scripts/analysis/",
    "view_vix_data_formatted.py": "src/scripts/analysis/",
    "view_vix_data.py": "src/scripts/analysis/",
    "check_data_counts.py": "src/scripts/utility/",
    "check_roll_calendar_date.py": "src/scripts/utility/",
    "check_roll_calendar.py": "src/scripts/utility/",
    "create_valid_trading_days.py": "src/scripts/utility/",
    "duckdb_launcher.py": "src/scripts/utility/",
    "check_table_schema.py": "src/scripts/utility/",
    "reorganize_project.py": "src/scripts/utility/",
    
    # Data files to output or data
    "filled_zero_values.csv": "output/reports/",
    "zero_price_fill_report.csv": "output/reports/",
    "vxc1_vxc2_outliers.csv": "output/reports/",
    "valid_trading_days_dates_only.txt": "output/data/",
    "valid_trading_days.csv": "output/data/",
    "vix_data_apr_2004_transition.csv": "output/data/",
    "vix_data_apr_may_2004.csv": "output/data/",
    "vix_data_2004.csv": "output/data/",
    "vix_data_2004_2006.csv": "output/data/",
    
    # Log files to logs
    "output.txt": "logs/",
    "output2.txt": "logs/",
    "vix_count.txt": "logs/"
}

def move_files():
    """Moves files according to the mapping"""
    for file, target_dir in file_movements.items():
        if os.path.exists(file):
            # Generate target path
            target_path = os.path.join(target_dir, file)
            
            # Check if target already exists
            if os.path.exists(target_path):
                logger.warning(f"Target already exists: {target_path}")
                # Archive the existing file in root
                archive_path = os.path.join("data/archive", f"{file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.move(file, archive_path)
                logger.info(f"Archived original file to: {archive_path}")
            else:
                # Move the file
                shutil.move(file, target_path)
                logger.info(f"Moved {file} to {target_path}")
        else:
            logger.warning(f"Source file not found: {file}")

def update_import_references():
    """
    Updates import statements in moved Python files to reflect new locations.
    This is a basic implementation and may need manual review.
    """
    python_files = []
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Basic replacement patterns - these should be expanded based on project needs
            replacements = [
                ('from src.scripts.utility import check_data_counts', 'from src.scripts.utility from src.scripts.utility import check_data_counts'),
                ('from src.scripts.analysis import show_vix_continuous_data', 'from src.scripts.analysis from src.scripts.analysis import show_vix_continuous_data'),
                ('from src.scripts.utility.check_data_counts', 'from src.scripts.utility.check_data_counts'),
                ('from src.scripts.analysis.show_vix_continuous_data', 'from src.scripts.analysis.show_vix_continuous_data')
            ]
            
            modified = False
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    modified = True
            
            if modified:
                with open(file_path, 'w') as file:
                    file.write(content)
                logger.info(f"Updated import references in {file_path}")
                
        except Exception as e:
            logger.error(f"Error updating imports in {file_path}: {e}")

def main():
    """Main function to run the reorganization"""
    logger.info("Starting project reorganization")
    
    ensure_directories()
    move_files()
    update_import_references()
    
    logger.info("Project reorganization complete")
    logger.info("NOTE: You may need to manually verify import statements in Python files")
    logger.info("      and update any hardcoded file paths in scripts.")

if __name__ == "__main__":
    main() 