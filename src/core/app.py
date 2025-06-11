#!/usr/bin/env python
"""
Application Core Module

This module provides the core application class for the Financial Data System,
tying together configuration, database, logging, and plugins.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import subprocess # Added for running external scripts
import platform # To determine python executable
import yaml # Added for loading futures.yaml
import re # Added for regular expressions
from datetime import datetime, timedelta
import pandas as pd

from .config import ConfigManager
from .database import Database
from .logging import configure_logging, create_timestamped_log_file
from src.scripts.market_data import MarketDataFetcher, continuous_contract_loader_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attempt to import the main function from the target script
try:
    from ..scripts.market_data.vix.update_vix_index import main as update_vix_index_main
except ImportError as e:
    update_vix_index_main = None
    logger.error(f"Failed to import update_vix_index_main: {e}")

try:
    from ..scripts.market_data.vix.update_vx_futures import main as update_vx_futures_main
except ImportError as e:
    update_vx_futures_main = None
    logger.error(f"Failed to import update_vx_futures_main: {e}")

try:
    from ..scripts.market_data.fetch_market_data import MarketDataFetcher
except ImportError as e:
    MarketDataFetcher = None
    logger.error(f"Failed to import MarketDataFetcher: {e}")

class Application:
    """Core application class for the Financial Data System."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the application with configuration.
        
        Args:
            config_path: Path to configuration file (None for default)
        """
        # Determine project root
        self.project_root = self._find_project_root()
        
        # Determine configuration path
        if config_path is None:
            config_path = self.project_root / 'config' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = None
        self.db = None
        self.futures_config = None # Add new attribute for futures.yaml content
        self._init_config()
    
    def _find_project_root(self) -> Path:
        """
        Find the project root directory.
        
        Returns:
            Path to project root
        """
        # Start with the directory of this file
        current_dir = Path(__file__).resolve().parent
        
        # Go up until we find a directory with config/ and src/
        while current_dir != current_dir.parent:
            if (current_dir / 'config').exists() and (current_dir / 'src').exists():
                return current_dir
            current_dir = current_dir.parent
        
        # If not found, use the directory containing src/
        current_dir = Path(__file__).resolve().parent.parent.parent
        return current_dir
    
    def _init_config(self) -> None:
        """Initialize configuration from file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            self.config = ConfigManager(config_dir=self.config_path.parent)
            
            # Load futures.yaml specifically
            futures_yaml_path = self.project_root / 'config' / 'futures.yaml'
            if futures_yaml_path.exists():
                try:
                    with open(futures_yaml_path, 'r') as f_futures:
                        self.futures_config = yaml.safe_load(f_futures)
                    logger.info(f"Futures-specific configuration loaded from {futures_yaml_path}")
                except Exception as e_futures:
                    logger.error(f"Error loading futures-specific configuration from {futures_yaml_path}: {e_futures}")
                    self.futures_config = {} # Ensure it's a dict even on error
            else:
                logger.warning(f"Futures-specific configuration file not found: {futures_yaml_path}")
                self.futures_config = {} # Ensure it's a dict if not found
            
            # Initialize logging
            log_section_data = self.config.get_section('logging')
            log_config = log_section_data if log_section_data else {}
            log_file = log_config.get('file')
            
            if log_file:
                # Convert relative path to absolute
                if not Path(log_file).is_absolute():
                    log_file = self.project_root / log_file
                
                # Create directory if it doesn't exist
                os.makedirs(Path(log_file).parent, exist_ok=True)
            
            configure_logging(
                level=log_config.get('level', 'INFO'),
                log_file=log_file,
                console=log_config.get('console', True),
                format_str=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                module_levels=log_config.get('module_levels', {})
            )
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            # Basic logging setup in case of configuration errors
            configure_logging(level='INFO')
            logger.error(f"Error initializing configuration: {e}")
            raise
    
    def init_database(self, db_path_override: Optional[Union[str, Path]] = None, read_only: bool = False) -> None:
        """
        Initialize the database connection.
        
        Args:
            db_path_override: Optional path to override database path from config
            read_only: Whether to open the database in read-only mode
        """
        try:
            db_path = db_path_override
            if db_path is None:
                db_section_data = self.config.get_section('database')
                db_config = db_section_data if db_section_data else {}
                db_path = db_config.get('path')

            if db_path is None:
                raise ValueError("Database path not specified. Provide it via command-line argument or in the 'database.path' config entry.")
            
            # Convert relative path to absolute
            # If db_path came from config and is relative, it's relative to project_root
            # If db_path_override is used, it's assumed to be absolute or relative to CWD (Path handles this)
            if not Path(db_path).is_absolute() and db_path_override is None:
                db_path = self.project_root / db_path
            else:
                db_path = Path(db_path) # Ensure it's a Path object if overridden
            
            # Create database directory if it doesn't exist and not in read-only mode
            if not read_only:
                os.makedirs(Path(db_path).parent, exist_ok=True)
            
            self.db = Database(db_path, read_only=read_only)
            logger.info(f"Database connection established: {db_path} (read_only={read_only})")
            
            # Explicitly initialize the schema if not in read-only mode
            if not read_only:
                logger.info("Attempting to initialize database schema...")
                try:
                    self.db.initialize_schema()
                    logger.info("Database schema initialization process completed.")
                except Exception as schema_e:
                    logger.error(f"Error during explicit schema initialization: {schema_e}")
                    # Depending on severity, you might want to re-raise or handle differently
                    # For now, logging the error and continuing, as Database object is created.
            
            logger.info(f"Database fully initialized: {db_path} (read_only={read_only})")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def load_plugins(self) -> None:
        """Load and register all plugins."""
        try:
            # Import plugins module to trigger registration
            from ..data_sources import register_all_plugins
            register_all_plugins()
            
            # Import processors
            from ..processors import register_all_processors
            register_all_processors()
            
            # Import validators
            from ..validation import register_all_validators
            register_all_validators()
            
            logger.info("All plugins registered successfully")
            
        except ImportError as e:
            logger.warning(f"Error importing plugin modules: {e}")
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
            raise
    
    def create_data_source(self, source_name: str) -> Any:
        """
        Create a data source plugin instance.
        
        Args:
            source_name: Name of the data source to create
            
        Returns:
            Data source plugin instance
            
        Raises:
            ValueError: If the data source is not configured or registered
        """
        if self.db is None:
            self.init_database()
        
        try:
            # Import here to avoid circular imports
            from ..data_sources.base import DataSourceRegistry
            
            # Get data source configuration
            source_config = self.config.get_item('data_sources', source_name)
            
            # Create the data source
            return DataSourceRegistry.create_source(
                source_name, 
                source_config, 
                self.db
            )
            
        except Exception as e:
            logger.error(f"Error creating data source '{source_name}': {e}")
            raise
    
    def backup_database(self) -> Path:
        """
        Create a backup of the database.
        
        Returns:
            Path to the backup file
        """
        if self.db is None:
            self.init_database()
        
        try:
            db_section_data = self.config.get_section('database')
            db_config = db_section_data if db_section_data else {}
            backup_dir = db_config.get('backup_dir')
            
            # Convert relative path to absolute
            if not Path(backup_dir).is_absolute():
                backup_dir = self.project_root / backup_dir
            
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Set retention days from config or default
            retention_days = db_config.get('backup_retention_days', 30)
            
            # Create the backup
            backup_path = self.db.create_backup(backup_dir, retention_days)
            logger.info(f"Database backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def update_symbol_metadata(self) -> None:
        """Updates symbol metadata by calling the populate_symbol_metadata.py script."""
        logger.info("Application.update_symbol_metadata() called.")
        
        db_path_for_script = None
        original_read_only_for_reopen = False # Default

        if self.db and hasattr(self.db, 'db_path') and self.db.db_path:
            db_path_for_script = self.db.db_path
            original_read_only_for_reopen = self.db.read_only
            logger.info("Closing main app database connection before running subprocess.")
            self.close() # This sets self.db to None
        else:
            # If DB wasn't open, try to determine path from config for the script
            # This path might be different from what init_database eventually uses if an override is involved
            # We need a definitive path for the script argument.
            # Best to ensure init_database has run at least once in the main script with overrides.
            # For now, let's try to get it from config if db not init.
            temp_db_section = self.config.get_section('database')
            temp_db_config = temp_db_section if temp_db_section else {}
            db_path_from_config = temp_db_config.get('path')
            if db_path_from_config:
                if not Path(db_path_from_config).is_absolute():
                    db_path_for_script = self.project_root / db_path_from_config
                else:
                    db_path_for_script = Path(db_path_from_config)
            # read_only state for a potential reopen would be False by default if db wasn't open.

        if not db_path_for_script:
            logger.error("Database path could not be determined. Cannot update symbol metadata.")
            # If db was closed, attempt to reopen it if we can determine its original state
            # This part is tricky if we don't know the original db_path_override.
            # For now, we assume the main script will handle re-init if needed or script fails gracefully.
            return

        script_path = self.project_root / "src" / "scripts" / "database" / "populate_symbol_metadata.py"
        
        db_path_str = str(db_path_for_script)
        config_path_str = str(self.config_path)

        cmd = [
            sys.executable, 
            str(script_path),
            "--db-path", db_path_str,
            "--config-path", config_path_str
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        process_successful = False
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.project_root)
            logger.info("populate_symbol_metadata.py output:\n" + result.stdout)
            if result.stderr:
                logger.warning("populate_symbol_metadata.py errors (stderr):\n" + result.stderr)
            logger.info("Symbol metadata update script completed successfully.")
            process_successful = True
        except subprocess.CalledProcessError as e:
            logger.error(f"populate_symbol_metadata.py failed with exit code {e.returncode}.")
            logger.error("Stdout:\n" + e.stdout)
            logger.error("Stderr:\n" + e.stderr)
        finally:
            # Reopen the database if it was closed by this method
            # Only if it was open before and we have the path to reopen with its original read_only state
            if db_path_for_script and not self.db: # Check if self.db is None (closed)
                logger.info(f"Reopening main app database connection to {db_path_for_script} after subprocess.")
                # We need to use the db_path_override that was initially passed to the app instance if any.
                # This method doesn't know about args.db_path directly. 
                # The init_database method itself handles db_path_override correctly.
                # If db_path_for_script was derived from self.db, then original_read_only is correct.
                # If db_path_for_script was derived from config (because self.db was None), then 
                # original_read_only_for_reopen (which is False) is a guess.
                # The most robust way is to call init_database and let it use its logic (including any original override).
                self.init_database(db_path_override=db_path_for_script, read_only=original_read_only_for_reopen)

    def update_vix_index(self, force_full: bool = False) -> None:
        """Updates VIX index data by calling the update_vix_index.py script's main function."""
        logger.info(f"Application.update_vix_index(force_full={force_full}) called.")

        if self.db is None or self.db.conn is None:
            logger.error("Database not initialized. Cannot update VIX index.")
            # Or self.init_database() if appropriate, but main script should handle initial init.
            return

        if update_vix_index_main is None:
            logger.error("update_vix_index_main function not imported. Cannot update VIX index.")
            return

        try:
            logger.info(f"Calling update_vix_index_main function from src.scripts.market_data.vix.update_vix_index")
            # The script's main function takes args_dict and existing_conn.
            # It doesn't seem to have a 'force_full' arg, so we pass None for args_dict for now.
            # It will use its internal logic for fetching.
            args_for_script = {
                'db_path': str(self.db.db_path) # Pass db_path in case it needs to open its own if conn fails
            }
            update_vix_index_main(args_dict=args_for_script, existing_conn=self.db.conn)
            logger.info("VIX index update function completed.")
        except Exception as e:
            logger.error(f"Error calling update_vix_index_main: {e}", exc_info=True)
            # Consider re-raising or handling error status

    def update_vx_futures(self, force_full: bool = False) -> None:
        """Updates VIX futures data by calling the update_vx_futures.py script's main function."""
        logger.info(f"Application.update_vx_futures(force_full={force_full}) called.")

        if self.db is None or self.db.conn is None:
            logger.error("Database not initialized. Cannot update VIX futures.")
            return

        if update_vx_futures_main is None:
            logger.error("update_vx_futures_main function not imported. Cannot update VIX futures.")
            return

        try:
            logger.info("Calling update_vx_futures_main function from src.scripts.market_data.vix.update_vx_futures")
            args_for_script = {
                'db_path': str(self.db.db_path), 
                'config_path': str(self.config_path), 
                'full_regen': force_full 
            }
            update_vx_futures_main(args_dict=args_for_script, existing_conn=self.db.conn)
            logger.info("VIX futures update function completed.")
        except Exception as e:
            logger.error(f"Error calling update_vx_futures_main: {e}", exc_info=True)

    def update_future_instrument_raw_data(self, symbol_root: str, interval_unit: str, interval_value: int, force_full: bool = False, fetch_mode: str = "auto", lookback_days: str = "90", roll_proximity_threshold_days: str = "7") -> None:
        """
        Update raw data for a future instrument using the continuous contract loader.
        
        Args:
            symbol_root: Root symbol (e.g., 'ES', 'NQ', 'VX')
            interval_unit: Time unit for the interval ('minute', 'daily')
            interval_value: Value of the interval (e.g., 1, 15)
            force_full: Whether to force a full update
            fetch_mode: Mode for fetching data ('auto', 'full', 'incremental')
            lookback_days: Number of days to look back for updates
            roll_proximity_threshold_days: Number of days before roll to start using next contract
        """
        try:
            logger.info(f"Application.update_future_instrument_raw_data(symbol_root={symbol_root}, loader_symbol=@{symbol_root}, interval_unit={interval_unit}, interval_value={interval_value}, force_full={force_full}) called.")
            
            # Construct the loader symbol (e.g., @ES for ES)
            loader_symbol = f"@{symbol_root}"
            
            # Create MarketDataFetcher instance with both configs
            fetcher = MarketDataFetcher(
                db_path=self.db.db_path,
                config_path=self.config_path,
                db_connector=self.db
            )
            
            # Pass the futures config to the fetcher
            fetcher.futures_config = self.futures_config
            
            # Prepare arguments for the continuous contract loader
            args_for_loader = [
                '--symbol', loader_symbol,
                '--interval-unit', interval_unit,
                '--interval-value', str(interval_value),
                '--fetch-mode', fetch_mode,
                '--lookback-days', lookback_days,
                '--roll-proximity-threshold-days', roll_proximity_threshold_days,
                # Pass the db_path explicitly to the loader script in case it needs it, 
                # even if it uses the existing connection. This ensures its argparse default doesn't cause issues if not used.
                '--db-path', str(self.db.db_path) 
            ]
            
            if force_full:
                args_for_loader.append('--force')
            
            # Call the continuous contract loader, passing the existing connection
            logger.info(f"Calling continuous_contract_loader_main for {loader_symbol} futures with args: {args_for_loader}")
            continuous_contract_loader_main(args_list=args_for_loader, existing_conn=self.db.conn)
            
        except ImportError as e:
            logger.error(f"Failed to import continuous contract loader: {e}")
            raise
        except Exception as e:
            logger.error(f"Error updating future instrument data for {symbol_root}: {e}", exc_info=True)
            raise

    def update_es_futures(self, interval_unit: str, interval_value: int, force_full: bool = False) -> None:
        """Updates ES futures data by calling the generic future instrument update method."""
        logger.info(f"Application.update_es_futures (delegating to generic updater) for interval_unit={interval_unit}, interval_value={interval_value}, force_full={force_full}")
        self.update_future_instrument_raw_data(
            symbol_root="ES", 
            interval_unit=interval_unit, 
            interval_value=interval_value, 
            force_full=force_full
        )

    def update_nq_futures(self, interval_unit: str, interval_value: int, force_full: bool = False) -> None:
        """Updates NQ futures data by calling the generic future instrument update method."""
        logger.info(f"Application.update_nq_futures (delegating to generic updater) for interval_unit={interval_unit}, interval_value={interval_value}, force_full={force_full}")
        self.update_future_instrument_raw_data(
            symbol_root="NQ", 
            interval_unit=interval_unit, 
            interval_value=interval_value, 
            force_full=force_full
        )

    def update_individual_active_futures(self, symbol_root: str, item_config: Dict[str, Any], force_full: bool = False) -> None:
        """
        Updates data for active individual futures contracts for a given root symbol.
        Example: For ES, this would find ESM25, ESU25 (if active) and update them.
        """
        logger.info(f"Application.update_individual_active_futures for root: {symbol_root} called.")

        if self.db is None or self.db.conn is None:
            logger.error(f"Database not initialized. Cannot update individual futures for {symbol_root}.")
            return

        if MarketDataFetcher is None:
            logger.error(f"MarketDataFetcher not available. Cannot update individual futures for {symbol_root}.")
            return

        try:
            fetcher = MarketDataFetcher(
                config_path=str(self.config_path), # Main config (market_symbols.yaml wrapper)
                # futures_specific_config_path=str(self.project_root / 'config' / 'futures.yaml'), # Path to futures.yaml
                existing_conn=self.db.conn
            )

            logger.info(f"Determining active individual contracts for {symbol_root} using its config: {item_config.get('description', symbol_root)}")
            active_contracts = fetcher.get_active_futures_symbols(symbol_root, item_config)

            if not active_contracts:
                logger.info(f"No active individual contracts found or determined for {symbol_root}.")
                return

            logger.info(f"Found active contracts for {symbol_root}: {active_contracts}")

            global_freq_settings = self.config.get_section('settings')
            global_data_frequencies_map = {}
            if global_freq_settings and 'data_frequencies' in global_freq_settings:
                for freq_def in global_freq_settings['data_frequencies']:
                    if 'name' in freq_def and 'unit' in freq_def and 'interval' in freq_def:
                        global_data_frequencies_map[freq_def['name']] = freq_def
            
            configured_frequencies_for_root = item_config.get('frequencies', [])
            frequencies_to_process = []

            if not configured_frequencies_for_root:
                logger.warning(f"No frequencies defined in YAML for future_group {symbol_root}. Skipping individual contract updates.")
                return

            for freq_entry in configured_frequencies_for_root:
                parsed_freq = self._parse_frequency_entry(freq_entry, global_data_frequencies_map, symbol_root)
                if parsed_freq:
                    frequencies_to_process.append(parsed_freq)
            
            if not frequencies_to_process:
                logger.warning(f"No valid frequencies resolved for {symbol_root}. Skipping individual contract updates.")
                return

            for contract_symbol in active_contracts:
                logger.info(f"Processing individual contract: {contract_symbol} for root {symbol_root}")
                for freq_detail in frequencies_to_process:
                    interval_val = freq_detail['interval']
                    interval_u = freq_detail['unit']
                    freq_name = freq_detail.get('name', f"{interval_val}{interval_u}")
                    logger.info(f"Updating {contract_symbol} for frequency: {freq_name} ({interval_val} {interval_u})")
                    try:
                        fetcher.process_symbol(
                            symbol=contract_symbol,
                            update_history=force_full,
                            force=force_full,
                            interval_value=int(interval_val),
                            interval_unit=str(interval_u)
                        )
                        logger.info(f"Successfully processed {contract_symbol} for {freq_name}.")
                    except Exception as e_process:
                        logger.error(f"Error processing symbol {contract_symbol} for {freq_name}: {e_process}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in update_individual_active_futures for {symbol_root}: {e}", exc_info=True)

    def fetch_specific_symbol_data(self, symbol_to_fetch: str, interval_value: int, interval_unit: str, force_full: bool = False) -> None:
        """Fetches market data for a specific symbol, interval, and unit.

        Args:
            symbol_to_fetch: The exact symbol string to fetch (e.g., "@ES=102XC", "SPY").
            interval_value: The numerical value of the interval (e.g., 1, 15).
            interval_unit: The unit of the interval (e.g., "daily", "minute").
            force_full: If True, re-fetches all data, ignoring existing records.
        """
        logger.info(f"Application.fetch_specific_symbol_data call for: {symbol_to_fetch}, interval: {interval_value} {interval_unit}, force_full: {force_full}")
        try:
            # Initialize MarketDataFetcher if not already available or pass connection
            fetcher = MarketDataFetcher(
                config_path=self.config_path, # Main config (market_symbols.yaml wrapper)
                # futures_specific_config_path=str(self.project_root / 'config' / 'futures.yaml'), # Path to futures.yaml
                db_path=self.db.db_path,      # Use current DB path
                existing_conn=self.db.conn      # Pass existing connection
            )
            # Ensure the fetcher's connection is the same as the app's
            fetcher.set_connection(self.db.conn)

            # process_symbol expects update_history and force.
            # If force_full is True, we want to update history and force overwrite.
            
            # Use _parse_frequency_entry to ensure interval_value and interval_unit are correctly determined
            # This requires a bit of a refactor as fetch_specific_symbol_data takes value/unit directly
            # For now, assume interval_value and interval_unit are passed correctly based on prior parsing
            # If this method is called directly with "15min" style strings, it would need its own parsing step.
            # However, based on the log, it seems to be called from a loop that *should* do the parsing.

            fetcher.process_symbol(
                symbol=symbol_to_fetch,
                update_history=force_full, # If forcing full, then update history from start
                force=force_full,          # If forcing full, then overwrite
                interval_value=interval_value,
                interval_unit=interval_unit
            )
            logger.info(f"Successfully processed specific symbol data for {symbol_to_fetch} - {interval_value} {interval_unit}")

        except Exception as e:
            logger.error(f"Error in Application.fetch_specific_symbol_data for {symbol_to_fetch} ({interval_value} {interval_unit}): {e}", exc_info=True)
            # Depending on desired behavior, you might want to re-raise or handle

    def trigger_inhouse_build_script(
        self,
        target_built_symbol: str,      # e.g., @ES=101XN_d
        base_ts_symbol_for_build: str, # e.g., @ES=101XN (becomes --root-symbol for the script)
        interval_value: int,
        interval_unit: str,
        item_config: Dict[str, Any],   # Full YAML config for the target_built_symbol
        force_build: bool
    ) -> None:
        """Triggers the 'generate_back_adjusted_futures.py' script for a specific built contract and frequency."""
        logger.info(f"Application.trigger_inhouse_build_script for {target_built_symbol} at {interval_value}-{interval_unit}")

        # It's assumed interval_value and interval_unit are already correctly parsed integers and strings here
        # If this method were to receive "15min", it would need parsing.

        script_path = self.project_root / 'src' / 'scripts' / 'scripts' / 'generate_back_adjusted_futures.py'
        if not script_path.exists():
            logger.error(f"Build script not found at {script_path}. Cannot generate {target_built_symbol}.")
            return

        # Determine Python executable (venv or system)
        python_executable = sys.executable # Default to current interpreter
        if platform.system() == "Windows":
            # Check common venv paths, adjust if necessary
            venv_python = self.project_root / 'venv' / 'Scripts' / 'python.exe'
            if venv_python.exists():
                python_executable = str(venv_python)
            else:
                python_executable = 'python' # Fallback to system python if venv not found at expected loc
        else: # Linux/macOS
            venv_python = self.project_root / 'venv' / 'bin' / 'python'
            if venv_python.exists():
                python_executable = str(venv_python)
            else:
                python_executable = 'python3' # Common fallback for non-Windows

        # --- Constructing arguments for generate_back_adjusted_futures.py --- 
        cmd_args = [
            python_executable,
            str(script_path),
            '--root-symbol', base_ts_symbol_for_build, # The TradeStation symbol it's based on
            '--interval-value', str(interval_value),
            '--interval-unit', interval_unit,
            # Output symbol is implicitly derived by the script based on root + suffix, 
            # but we might need to ensure the script uses target_built_symbol or we pass it.
            # For now, let's assume script handles output naming based on root + default suffix, or add an arg if needed.
        ]

        # Extract script-specific build parameters from item_config
        # These keys should match what you intend to put in the YAML for built contracts
        roll_type = item_config.get('build_roll_type') # e.g., "02X"
        contract_position = item_config.get('build_contract_position') # e.g., 1
        adjustment_type = item_config.get('build_adjustment_type') # e.g., "N", "C"
        output_symbol_suffix = item_config.get('build_output_symbol_suffix') # e.g., "_d"
        # Removed build_source_identifier from being passed as a command-line argument
        # if source_identifier:
        #     cmd_args.extend(['--source-identifier', str(source_identifier)])
        
        if roll_type:
            cmd_args.extend(['--roll-type', str(roll_type)])
        if contract_position is not None: # Could be 0
            cmd_args.extend(['--contract-position', str(contract_position)])
        if adjustment_type:
            cmd_args.extend(['--adjustment-type', str(adjustment_type)])
        if output_symbol_suffix: # This suffix is used by the script to form the final symbol
            cmd_args.extend(['--output-symbol-suffix', str(output_symbol_suffix)])
        
        # The build_source_identifier is derived by the script, so we don't pass it.
        # if source_identifier:
        #     cmd_args.extend(['--source-identifier', str(source_identifier)])
        
        # Override output symbol if specified directly in config (more explicit)
        # The build script needs an argument for this if we want to control the exact output symbol name
        # Let's assume the script has --output-symbol argument.
        # If your YAML defines 'symbol: @ES=101XN_d', that's the target.
        # The script by default might create @ES=101XN_d (if root is @ES=101XN and suffix is _d)
        # For clarity, explicitly passing the target_built_symbol to the script might be better.
        # Adding a placeholder argument --explicit-output-symbol to the script if it needs it.
        # cmd_args.extend(['--output-symbol', target_built_symbol]) # This was identified as problematic and is removed.

        if force_build:
            cmd_args.append('--force-delete')
        
        cmd_args.extend(['--db-path', str(self.db.db_path)]) # Pass the database path
        cmd_args.extend(['--config-path', str(self.config_path)]) # Pass the main config path

        logger.info(f"Executing build command: {' '.join(cmd_args)}")

        try:
            # Close the database connection before running the script to avoid locking issues
            if self.db and self.db.conn:
                logger.info("Closing main app database connection before script execution.")
                self.db.close()
                self.db = None # Ensure it's re-initialized if needed later by the app

            process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=self.project_root)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully built {target_built_symbol} for {interval_value}-{interval_unit}. Script stdout:")
                for line in stdout.splitlines():
                    logger.info(f"[BUILD SCRIPT STDOUT]: {line}")
            else:
                logger.error(f"Failed to build {target_built_symbol} for {interval_value}-{interval_unit}. Return code: {process.returncode}")
                logger.error(f"[BUILD SCRIPT STDOUT]:\n{stdout}")
                logger.error(f"[BUILD SCRIPT STDERR]:\n{stderr}")

        except Exception as e:
            logger.error(f"Exception while triggering build for {target_built_symbol}: {e}", exc_info=True)
        finally:
            # Re-initialize database connection for the main app if it was closed
            if self.db is None:
                logger.info("Re-initializing main app database connection after script execution.")
                # Assuming read_only=False is the typical state for the app after such an operation
                self.init_database(db_path_override=cmd_args[cmd_args.index('--db-path')+1] , read_only=False)

    def get_continuous_contract_symbols(self, specific_root_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of symbol configurations for continuous contract generation,
        reflecting the new structure where futures.yaml defines continuous contracts
        and market_symbols.yaml provides root symbol defaults and raw data frequencies.
        """
        logger.info(f"Application.get_continuous_contract_symbols(specific_root_symbol={specific_root_symbol}) called.")
        
        results_list = []

        if not self.futures_config or 'futures' not in self.futures_config:
            logger.error("futures.yaml content not loaded or missing 'futures' key. Cannot get continuous contract symbols.")
            return results_list

        # Iterate through each base future defined in futures.yaml (e.g., ES, NQ)
        for root_key, future_product_config in self.futures_config.get('futures', {}).items():
            if specific_root_symbol and root_key != specific_root_symbol:
                continue

            logger.debug(f"Processing root_key '{root_key}' from futures.yaml")

            # 1. Get the corresponding base_symbol config from market_symbols.yaml for defaults and raw frequencies
            market_symbols_root_config = None
            if self.config: # self.config holds market_symbols.yaml content
                for item_cfg in self.config.get_section('futures') or []: 
                    if item_cfg.get('base_symbol') == root_key and item_cfg.get('asset_type') == 'future_group':
                        market_symbols_root_config = item_cfg.copy() # Use a copy
                        break
            
            if not market_symbols_root_config:
                logger.warning(f"No corresponding 'future_group' config found in market_symbols.yaml for base_symbol '{root_key}'. Some defaults/raw frequencies may be missing.")
                market_symbols_root_config = {} # Avoid NoneErrors, proceed with what futures.yaml has

            # 2. Prepare the main entry for this root_key
            # Defaults for root_entry come from market_symbols_root_config, then overridden by future_product_config from futures.yaml
            root_entry = {
                'root_symbol': root_key,
                'asset_type': future_product_config.get('asset_class', market_symbols_root_config.get('asset_type', 'future_group')),
                'description': future_product_config.get('description', market_symbols_root_config.get('description', f"{root_key} Futures")),
                'exchange': future_product_config.get('exchange', market_symbols_root_config.get('exchange')),
                'calendar': future_product_config.get('calendar', market_symbols_root_config.get('calendar')),
                'default_raw_table': market_symbols_root_config.get('default_raw_table', 'market_data'),
                'continuous_data_table': market_symbols_root_config.get('continuous_data_table', 'continuous_contracts'),
                'roll_calendar_table': market_symbols_root_config.get('roll_calendar_table', 'futures_roll_dates'),
                
                # Frequencies for the RAW generic continuous symbol (e.g., @ES), from market_symbols.yaml
                'raw_data_update_frequencies': market_symbols_root_config.get('frequencies', []),
                
                'continuous_symbols_to_process': [], # This will be populated next
                
                'config_from_market_symbols': market_symbols_root_config, # Storing the market_symbols part
                'config_from_futures_yaml': future_product_config # Storing the futures.yaml part (which includes its own inheritance)
            }

            # Apply overrides from futures.yaml general section to root_entry if they exist
            # (e.g., if futures.yaml has 'exchange' at the 'ES:' level)
            for key_to_override in ['asset_type', 'description', 'exchange', 'calendar']:
                if key_to_override in future_product_config:
                    root_entry[key_to_override] = future_product_config[key_to_override]

            # 3. Process defined continuous contracts from futures.yaml
            defined_continuous_contracts = []
            # A. Direct list of continuous contracts
            if 'continuous_contracts' in future_product_config:
                # --- ADDED DEBUG LOGGING ---
                if root_key == 'ES':
                    logger.debug(f"APP_DEBUG: Processing 'ES' root. future_product_config keys: {list(future_product_config.keys())}")
                    logger.debug(f"APP_DEBUG: ES continuous_contracts from YAML: {future_product_config.get('continuous_contracts')}")
                # --- END ADDED DEBUG LOGGING ---
                for cc_def_orig in future_product_config['continuous_contracts']:
                    cc_def = cc_def_orig.copy() # Work with a copy
                    # --- ADDED DEBUG LOGGING ---
                    if root_key == 'ES':
                        logger.debug(f"APP_DEBUG: ES cc_def_orig: {cc_def_orig}")
                        logger.debug(f"APP_DEBUG: ES cc_def (copy): {cc_def}")
                    # --- END ADDED DEBUG LOGGING ---
                    if not isinstance(cc_def, dict) or 'identifier' not in cc_def:
                        logger.warning(f"Skipping invalid continuous_contract definition for {root_key}: {cc_def}")
                        continue
                    # Ensure essential fields for processing are present, using root_entry for defaults if needed
                    cc_def['type'] = cc_def.get('type', 'continuous_future')
                    cc_def['default_source'] = cc_def.get('default_source', root_entry.get('default_source', 'tradestation'))
                    cc_def['exchange'] = cc_def.get('exchange', root_entry.get('exchange'))
                    cc_def['calendar'] = cc_def.get('calendar', root_entry.get('calendar'))
                    # Frequencies for this specific continuous symbol are from its own definition
                    cc_def['frequencies'] = cc_def.get('frequencies', []) 
                    defined_continuous_contracts.append(cc_def)
            
            # B. Continuous contract group
            if 'continuous_contract_group' in future_product_config:
                group_cfg = future_product_config['continuous_contract_group']
                base_identifier_in_group = group_cfg.get('identifier_base') # e.g., "@VX"
                
                # Ensure group belongs to the current root_key being processed
                # (e.g., if root_key is "VX", base_identifier_in_group should be "@VX")
                if base_identifier_in_group and base_identifier_in_group.lstrip('@') == root_key:
                    month_codes = group_cfg.get('month_codes', []) # e.g., ["1", "2", ..., "9"]
                    settings_code_template = group_cfg.get('settings_code', '01XN') # e.g., "01XN"
                    
                    for pos_str in month_codes: # mc_part is the position string like "1", "2"
                        # Construct the final identifier, e.g., @VX=101XN
                        # For settings_code "01XN" and pos_str "1", we want "101XN"
                        # This means: pos_str + settings_code_template = "1" + "01XN" = "101XN"
                        final_identifier = f"{base_identifier_in_group}={pos_str}{settings_code_template}"
                        
                        group_cc_def = {
                            'identifier': final_identifier,
                            'description': group_cfg.get('description_template', f"Group CC {pos_str} for {root_key}").format(nth_month=pos_str),
                            'type': group_cfg.get('type', 'continuous_future'),
                            'default_source': group_cfg.get('default_source', 'tradestation'),
                            'exchange': group_cfg.get('exchange', root_entry.get('exchange')),
                            'calendar': group_cfg.get('calendar', root_entry.get('calendar')),
                            'frequencies': group_cfg.get('frequencies', []), # Frequencies for this specific group-generated symbol
                            'start_date': group_cfg.get('start_date', market_symbols_root_config.get('start_date')),
                            'method': group_cfg.get('method', 'none'), 
                            'position': int(pos_str) if pos_str.isdigit() else None,
                            # Build params typically not in groups for TS-sourced, but good to have a placeholder
                            'base_contract_for_build': group_cfg.get('base_contract_for_build'),
                            'build_roll_type': group_cfg.get('build_roll_type'),
                            'build_contract_position': group_cfg.get('build_contract_position', int(pos_str) if pos_str.isdigit() else None),
                            'build_adjustment_type': group_cfg.get('build_adjustment_type'),
                            'build_output_symbol_suffix': group_cfg.get('build_output_symbol_suffix'),
                            # Removed build_source_identifier from being passed as a command-line argument
                            # 'build_source_identifier': group_cfg.get('build_source_identifier')
                        }
                        defined_continuous_contracts.append(group_cc_def)
                else:
                    logger.warning(f"identifier_base '{base_identifier_in_group}' in continuous_contract_group for {root_key} does not match. Group ignored.")

            if defined_continuous_contracts:
                root_entry['continuous_symbols_to_process'] = defined_continuous_contracts
                # --- ADDED DEBUG LOGGING ---
                if root_key == 'ES':
                    logger.debug(f"APP_DEBUG_POST_ASSIGN: ES root_entry['continuous_symbols_to_process']: {root_entry['continuous_symbols_to_process']}")
                # --- END ADDED DEBUG LOGGING ---
                results_list.append(root_entry)
                # --- ADDED ID LOGGING ---
                if root_key == 'ES':
                    logger.debug(f"APP_DEBUG_ID: ES root_entry appended to results_list with id: {id(root_entry)}")
                    logger.debug(f"APP_DEBUG_ID: ES continuous_symbols_to_process in appended root_entry (id: {id(root_entry['continuous_symbols_to_process'])}): {root_entry['continuous_symbols_to_process']}")
                # --- END ID LOGGING ---
                logger.info(f"Prepared entry for '{root_key}' with {len(defined_continuous_contracts)} continuous symbols to process.")
            elif not specific_root_symbol: 
                logger.info(f"No continuous contracts explicitly defined or generated via group for root_key '{root_key}' in futures.yaml. It will be skipped for continuous processing steps that rely on these definitions.")
        
        logger.info(f"get_continuous_contract_symbols returning {len(results_list)} root symbol configurations.")
        # Detailed debug logging can be added here if needed
        return results_list

    def get_vx_futures_symbols(self) -> List[str]:
        """
        Get a list of specific VX futures contract symbols from the configuration
        based on the new get_continuous_contract_symbols method.
        Example: ['@VX=101XN', '@VX=201XN'] (if defined in futures.yaml for VX)
        """
        logger.info("Application.get_vx_futures_symbols() called.")
        vx_symbols = []
        # Get only for VX by passing specific_root_symbol
        vx_root_configs = self.get_continuous_contract_symbols(specific_root_symbol='VX') 

        if not vx_root_configs: # Should be at most one entry if specific_root_symbol='VX' was processed
            logger.warning("No configuration entry returned by get_continuous_contract_symbols for root 'VX'.")
            return vx_symbols

        # Assuming the first (and likely only) entry is for VX
        vx_config_entry = vx_root_configs[0]
        
        specific_symbols_details = vx_config_entry.get('continuous_symbols_to_process', [])
        for detail in specific_symbols_details:
            identifier = detail.get('identifier')
            source = detail.get('default_source')
            # We are looking for directly sourced TradeStation continuous contracts like @VX=101XN
            if identifier and source == 'tradestation' and re.match(r"@VX=\d+XN$", identifier):
                vx_symbols.append(identifier)
        
        if not vx_symbols:
            logger.warning("No TradeStation-sourced VX futures symbols (e.g., @VX=101XN) found within the processed continuous symbols for VX. Check 'futures.yaml' for VX continuous_contracts or continuous_contract_group with default_source: tradestation.")
            
        logger.info(f"Found VX futures symbols for direct fetching: {vx_symbols}")
        return vx_symbols

    def verify_continuous_contracts(self) -> None:
        """Placeholder for verifying continuous contracts data."""
        logger.warning("Method verify_continuous_contracts is called but not fully implemented.")
        # TODO: Implement actual verification logic

    def verify_market_data(self) -> None:
        """Placeholder for verifying raw market data."""
        logger.warning("Method verify_market_data is called but not fully implemented.")
        # TODO: Implement actual verification logic

    def close(self) -> None:
        """Close all resources."""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("Database connection closed")

    def _parse_frequency_entry(self, freq_entry: Union[str, Dict], global_data_frequencies_map: Dict, symbol_identifier: str) -> Optional[Dict]:
        """
        Parses a frequency entry from YAML into a dict with interval, unit, and name.
        Handles strings (like '15min', 'daily', or a global key) or dicts.
        """
        freq_detail_to_process = None
        if isinstance(freq_entry, str):
            if freq_entry in global_data_frequencies_map:
                freq_detail_to_process = global_data_frequencies_map[freq_entry].copy()
            else: # Attempt to parse directly, e.g., "15min", "1minute", "daily", "1day"
                match_min = re.match(r"(\d+)\s*(min|minute|minutes)$", freq_entry, re.IGNORECASE)
                match_day = re.match(r"(\d*)\s*(d|day|days|daily)$", freq_entry, re.IGNORECASE)
                if match_min:
                    val = int(match_min.group(1))
                    freq_detail_to_process = {'name': freq_entry, 'interval': val, 'unit': 'minute'}
                elif match_day:
                    val_str = match_day.group(1)
                    val = int(val_str) if val_str else 1 # Default to 1 if no number (e.g., "daily")
                    freq_detail_to_process = {'name': freq_entry, 'interval': val, 'unit': 'daily'}
                else:
                    logger.warning(f"Frequency string '{freq_entry}' for {symbol_identifier} is not a recognized global key or parseable format (e.g., '15min', 'daily'). Skipping.")
        elif isinstance(freq_entry, dict):
            if 'name' in freq_entry and 'interval' in freq_entry and 'unit' in freq_entry:
                freq_detail_to_process = freq_entry.copy()
            elif 'interval' in freq_entry and 'unit' in freq_entry: # Handle case where name is missing
                logger.warning(f"Frequency for {symbol_identifier} is missing a 'name', using unit/interval directly: {freq_entry}")
                # Construct a name if missing
                name = f"{freq_entry['interval']}{freq_entry['unit']}"
                freq_detail_to_process = freq_entry.copy()
                freq_detail_to_process['name'] = freq_detail_to_process.get('name', name)
            else:
                logger.warning(f"Invalid frequency dict format for {symbol_identifier}: {freq_entry}. Skipping.")
        
        if freq_detail_to_process:
            # Validate and convert unit
            unit = str(freq_detail_to_process.get('unit','')).lower()
            if unit in ['min', 'minute', 'minutes']:
                freq_detail_to_process['unit'] = 'minute'
            elif unit in ['d', 'day', 'days', 'daily']:
                freq_detail_to_process['unit'] = 'daily'
            else:
                logger.warning(f"Invalid unit '{unit}' in frequency detail for {symbol_identifier}: {freq_detail_to_process}. Skipping.")
                return None
            
            try:
                freq_detail_to_process['interval'] = int(freq_detail_to_process['interval'])
            except ValueError:
                logger.warning(f"Invalid interval value in frequency detail for {symbol_identifier}: {freq_detail_to_process}. Skipping.")
                return None
                
        return freq_detail_to_process

# Singleton instance
_app = None

def get_app(config_path: Optional[Union[str, Path]] = None) -> Application:
    """
    Get the application singleton instance.
    
    Args:
        config_path: Path to configuration file (None for default)
        
    Returns:
        Application instance
    """
    global _app
    if _app is None:
        _app = Application(config_path)
    return _app