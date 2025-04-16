#!/usr/bin/env python

"""
Redownload NQ Contracts Script

This script redownloads all NQ futures contracts at 15-minute intervals.
It uses the force parameter to ensure all data is redownloaded.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get the project root and virtual environment paths
project_root = Path(__file__).resolve().parent.parent.parent.parent
venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'

# Constants
CONFIG_PATH = project_root / 'config' / 'market_symbols.yaml'

# List of NQ contracts to update
NQ_CONTRACTS = [
    'CLM13',  # 2023 contracts
    'CLK13',
    'CLG13',
    'CLN12',
    'CLJ11'
]

def update_contract(contract: str) -> bool:
    """Update a single contract using fetch_market_data.py.
    
    Args:
        contract: The contract symbol to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Updating contract {contract}...")
        
        # Run fetch_market_data.py with the contract using venv Python
        cmd = [
            str(venv_python),
            'src/scripts/market_data/fetch_market_data.py',
            '--symbol', contract,
            '--updatehistory',  # Boolean flag, no value needed
            '--force',  # Boolean flag, no value needed
            '--interval-value', '1',  # 15-minute data
            '--interval-unit', 'minute',
            '--config', str(CONFIG_PATH)
        ]
        subprocess.run(cmd, check=True)
        
        logger.info(f"Successfully updated {contract}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error updating {contract}: {e}")
        return False

def main():
    """Main function to update all NQ contracts."""
    try:
        # Update each contract
        for contract in NQ_CONTRACTS:
            update_contract(contract)
            
        logger.info("Finished updating all NQ contracts")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 