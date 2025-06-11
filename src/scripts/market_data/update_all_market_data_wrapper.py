#!/usr/bin/env python
"""
Wrapper script for market data updates that provides compatibility between old and new architectures.

This script acts as a transition layer between the old update_all_market_data.py script
and the new update_all_market_data_v2.py. It:

1. Attempts to use the new architecture if available
2. Falls back to the legacy architecture if any components are missing
3. Provides compatibility for command-line arguments
4. Ensures consistent behavior regardless of which backend is used

Usage:
    python update_all_market_data_wrapper.py [options]
"""

import sys
import os
import argparse
import logging
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = os.path.join(project_root, "config", "market_symbols.yaml")
DEFAULT_DB_PATH = os.path.join(project_root, "data", "financial_data.duckdb")

def check_module_exists(module_path: str) -> bool:
    """
    Check if a Python module exists without importing it.
    
    Args:
        module_path: Path to the module (e.g., 'src.core.app')
        
    Returns:
        True if the module exists, False otherwise
    """
    try:
        # Convert module path to file path
        parts = module_path.split('.')
        if len(parts) == 1:
            # Single module name - search in sys.path
            for path in sys.path:
                file_path = os.path.join(path, f"{parts[0]}.py")
                if os.path.exists(file_path):
                    return True
        else:
            # Nested module - build path
            file_path = os.path.join(project_root, *parts) + ".py"
            return os.path.exists(file_path)
    except Exception as e:
        logger.debug(f"Error checking for module {module_path}: {e}")
        return False

def parse_arguments():
    """Parse command line arguments with compatibility for both old and new architectures."""
    parser = argparse.ArgumentParser(description="Update market data with automatic architecture selection")
    
    # Basic options (compatible with both architectures)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Path to config YAML")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to DuckDB database")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    # Common update control options
    parser.add_argument("--verify", action="store_true", help="Verify data after update")
    parser.add_argument("--full-update", action="store_true", help="Force full data update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    
    # Component selection (compatible with both architectures)
    parser.add_argument("--skip-metadata", action="store_true", help="Skip symbol metadata update")
    parser.add_argument("--skip-vix", action="store_true", help="Skip VIX index update")
    parser.add_argument("--skip-futures", action="store_true", help="Skip VX futures update")
    parser.add_argument("--skip-es-nq", action="store_true", help="Skip ES/NQ futures update")
    parser.add_argument("--skip-continuous", action="store_true", help="Skip continuous contract generation")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-panama", action="store_true", help="Skip Panama continuous contract generation specifically (new architecture)")
    
    # New architecture options (will be ignored in legacy mode)
    parser.add_argument("--panama-ratio", type=float, default=0.75,
                       help="Panama method ratio (0-1, 0=forward adjustment, 1=back adjustment)")
    
    # Force specific architecture (mainly for testing)
    parser.add_argument("--force-legacy", action="store_true", help="Force use of legacy architecture")
    parser.add_argument("--force-new", action="store_true", help="Force use of new architecture")
    
    return parser.parse_args()

def convert_args_for_legacy(args):
    """
    Convert new architecture arguments to legacy format.
    
    Args:
        args: Arguments from argparse
        
    Returns:
        List of command-line arguments for legacy script
    """
    legacy_args = []
    
    # Convert common arguments
    if args.config_path != DEFAULT_CONFIG_PATH:
        legacy_args.extend(["--config", args.config_path])
    
    if args.db_path != DEFAULT_DB_PATH:
        legacy_args.extend(["--database", args.db_path])
    
    if args.log_level != "INFO":
        legacy_args.extend(["--log-level", args.log_level])
    
    # Convert flags
    if args.verify:
        legacy_args.append("--verify")
    
    if args.full_update:
        legacy_args.append("--full-update")
    
    if args.dry_run:
        legacy_args.append("--dry-run")
    
    # Convert skip options
    if args.skip_metadata:
        legacy_args.append("--skip-metadata")
    
    if args.skip_vix:
        legacy_args.append("--skip-vix")
    
    if args.skip_futures:
        legacy_args.append("--skip-futures")
    
    if args.skip_es_nq:
        legacy_args.append("--skip-es-nq")
    
    if args.skip_continuous:
        legacy_args.append("--skip-continuous")
    
    if args.skip_cleaning:
        legacy_args.append("--skip-cleaning")
    
    return legacy_args

def run_legacy_update(args):
    """
    Run the legacy update script with converted arguments.
    
    Args:
        args: Arguments from argparse
        
    Returns:
        Return code from the legacy script
    """
    try:
        # Import the legacy script
        legacy_path = os.path.join(project_root, "src", "scripts", "market_data", "update_all_market_data.py")
        
        if not os.path.exists(legacy_path):
            logger.error(f"Legacy update script not found at: {legacy_path}")
            return 1
        
        # Convert arguments to legacy format
        legacy_args = convert_args_for_legacy(args)
        
        # Set up sys.argv for the legacy script
        original_argv = sys.argv
        sys.argv = [legacy_path] + legacy_args
        
        # Execute the legacy script
        logger.info("Executing legacy update script")
        with open(legacy_path, 'rb') as f:
            code = compile(f.read(), legacy_path, 'exec')
            exec_globals = {
                '__file__': legacy_path,
                '__name__': '__main__',
                '__package__': None,
                '__cached__': None,
            }
            exec(code, exec_globals)
        
        # Restore original argv
        sys.argv = original_argv
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running legacy update script: {e}", exc_info=True)
        return 1

def run_new_update(args):
    """
    Run the new architecture update script.
    
    Args:
        args: Arguments from argparse
        
    Returns:
        Return code from the new architecture script
    """
    try:
        # Import the new script
        new_path = os.path.join(project_root, "src", "scripts", "market_data", "update_all_market_data_v2.py")
        
        if not os.path.exists(new_path):
            logger.error(f"New architecture update script not found at: {new_path}")
            return 1
        
        # Set up sys.argv for the new script
        original_argv = sys.argv
        sys.argv = [new_path] + sys.argv[1:]  # Pass all arguments directly
        
        # Execute the new script
        logger.info("Executing new architecture update script")
        with open(new_path, 'rb') as f:
            code = compile(f.read(), new_path, 'exec')
            exec_globals = {
                '__file__': new_path,
                '__name__': '__main__',
                '__package__': None,
                '__cached__': None,
            }
            exec(code, exec_globals)
        
        # Restore original argv
        sys.argv = original_argv
        
        return 0
        
    except ModuleNotFoundError as e:
        logger.error(f"Module not found while running new architecture: {e}")
        logger.info("Falling back to legacy architecture")
        return run_legacy_update(args)
        
    except Exception as e:
        logger.error(f"Error running new architecture update script: {e}", exc_info=True)
        return 1

def check_new_architecture_components():
    """
    Check if all required components for the new architecture are available.
    
    Returns:
        True if all components are available, False otherwise
    """
    required_modules = [
        'src.core.app',
        'src.core.config',
        'src.processors.continuous.base',
        'src.processors.continuous.panama',
        'src.processors.continuous.registry',
        'src.processors.cleaners.base',
        'src.processors.cleaners.pipeline',
        'src.processors.cleaners.vx_zero_prices'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        if not check_module_exists(module):
            missing_modules.append(module)
    
    if missing_modules:
        logger.warning("Missing required modules for new architecture:")
        for module in missing_modules:
            logger.warning(f"  - {module}")
        return False
    
    return True

def main():
    """Main function to coordinate the update process."""
    start_time = time.time()
    args = parse_arguments()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("-" * 80)
    logger.info(f"Starting market data update process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which architecture to use
    use_new_architecture = True
    
    if args.force_legacy:
        logger.info("Forced to use legacy architecture")
        use_new_architecture = False
    elif args.force_new:
        logger.info("Forced to use new architecture")
        use_new_architecture = True
    else:
        # Check if new architecture components are available
        use_new_architecture = check_new_architecture_components()
        
        if use_new_architecture:
            logger.info("Using new architecture - all components available")
        else:
            logger.info("Using legacy architecture - some components are missing")
    
    # Run the appropriate update process
    if use_new_architecture:
        return_code = run_new_update(args)
    else:
        return_code = run_legacy_update(args)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Market data update completed in {elapsed_time:.2f} seconds")
    logger.info("-" * 80)
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())