#!/usr/bin/env python

"""
Simple script to update NQ contracts using fetch_market_data.py
"""

import subprocess
import logging
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get the virtual environment Python interpreter path
project_root = Path(__file__).resolve().parent.parent.parent
venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'

# List of contracts to update
CONTRACTS = [
    'NQU10',  # 2010 September contract
    'NQZ10'   # 2010 December contract
]

def update_contract(contract):
    """Update a single contract using fetch_market_data.py"""
    logger.info(f"Updating contract {contract}")
    try:
        # Run fetch_market_data.py with the contract using venv Python
        cmd = [
            str(venv_python),
            'src/scripts/fetch_market_data.py',
            '--symbol', contract,
            '--updatehistory',  # Boolean flag, no value needed
            '--force',  # Boolean flag, no value needed
            '--interval-value', '1',  # 1-minute data
            '--interval-unit', 'minute'
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully updated {contract}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error updating {contract}: {e}")

def main():
    """Main function to update the NQ contracts"""
    logger.info("Starting NQ contracts update")
    for contract in CONTRACTS:
        update_contract(contract)
    logger.info("Finished updating NQ contracts")

if __name__ == "__main__":
    main() 