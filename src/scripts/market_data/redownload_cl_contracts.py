#!/usr/bin/env python

"""
Redownload CL Contracts Script

This script redownloads all CL (Crude Oil) futures contracts at 15-minute intervals.
It uses the force parameter to ensure all data is redownloaded.
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
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
MAIN_DB_PATH = project_root / 'data' / 'financial_data.duckdb'

# Create a temporary directory for our temporary databases
temp_dir = tempfile.mkdtemp(prefix='cl_contracts_')
logger.info(f"Created temporary directory: {temp_dir}")

# List of CL contracts to update (2021-2025)
CL_CONTRACTS = [
    # 2021 contracts
    'CLF21', 'CLG21', 'CLH21', 'CLJ21', 'CLK21', 'CLM21', 'CLN21', 'CLQ21', 'CLU21', 'CLV21', 'CLX21', 'CLZ21',
    # 2022 contracts
    'CLF22', 'CLG22', 'CLH22', 'CLJ22', 'CLK22', 'CLM22', 'CLN22', 'CLQ22', 'CLU22', 'CLV22', 'CLX22', 'CLZ22',
    # 2023 contracts
    'CLF23', 'CLG23', 'CLH23', 'CLJ23', 'CLK23', 'CLM23', 'CLN23', 'CLQ23', 'CLU23', 'CLV23', 'CLX23', 'CLZ23',
    # 2024 contracts
    'CLF24', 'CLG24', 'CLH24', 'CLJ24', 'CLK24', 'CLM24', 'CLN24', 'CLQ24', 'CLU24', 'CLV24', 'CLX24', 'CLZ24',
    # 2025 contracts
    'CLF25', 'CLG25', 'CLH25', 'CLJ25', 'CLK25', 'CLM25', 'CLN25', 'CLQ25', 'CLU25', 'CLV25', 'CLX25', 'CLZ25'
]

def update_contract(contract: str, temp_db_path: Path) -> bool:
    """Update a single contract using fetch_market_data.py.
    
    Args:
        contract: The contract symbol to update
        temp_db_path: Path to temporary database file
        
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
            '--interval-value', '15',  # 15-minute data
            '--interval-unit', 'minute',
            '--config', str(CONFIG_PATH),
            '--db-path', str(temp_db_path)
        ]
        subprocess.run(cmd, check=True)
        
        logger.info(f"Successfully updated {contract}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error updating {contract}: {e}")
        return False

def merge_databases(source_db: Path, target_db: Path):
    """Merge data from source database into target database.
    
    Args:
        source_db: Path to source database
        target_db: Path to target database
    """
    try:
        logger.info(f"Merging data from {source_db} into {target_db}")
        
        # Create a script to merge the databases
        merge_script = f"""
        -- Connect to both databases
        ATTACH '{source_db}' AS source;
        ATTACH '{target_db}' AS target;
        
        -- Copy data from source to target
        INSERT INTO target.market_data 
        SELECT * FROM source.market_data
        ON CONFLICT (timestamp, symbol, interval_value, interval_unit) 
        DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            up_volume = excluded.up_volume,
            down_volume = excluded.down_volume,
            source = excluded.source,
            adjusted = excluded.adjusted,
            quality = excluded.quality;
        
        -- Detach databases
        DETACH source;
        DETACH target;
        """
        
        # Write the script to a file
        script_path = Path(temp_dir) / 'merge_script.sql'
        with open(script_path, 'w') as f:
            f.write(merge_script)
        
        # Run the script
        cmd = [
            str(venv_python),
            '-c',
            f"import duckdb; duckdb.execute(open('{script_path}', 'r').read())"
        ]
        subprocess.run(cmd, check=True)
        
        logger.info(f"Successfully merged data from {source_db} into {target_db}")
    except Exception as e:
        logger.error(f"Error merging databases: {e}")

def main():
    """Main function to update all CL contracts."""
    try:
        # Create a temporary database for this run
        temp_db_path = Path(temp_dir) / 'temp_financial_data.duckdb'
        
        # Update each contract
        for contract in CL_CONTRACTS:
            update_contract(contract, temp_db_path)
        
        # Merge the temporary database into the main database
        merge_databases(temp_db_path, MAIN_DB_PATH)
            
        logger.info("Finished updating all CL contracts")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    main() 