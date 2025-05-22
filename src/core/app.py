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

from .config import ConfigManager
from .database import Database
from .logging import configure_logging, create_timestamped_log_file

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
    from ..scripts.market_data.continuous_contract_loader import main as continuous_contract_loader_main
except ImportError as e:
    continuous_contract_loader_main = None
    logger.error(f"Failed to import continuous_contract_loader_main: {e}")

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
            logger.info(f"Database initialized: {db_path} (read_only={read_only})")
            
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
        """Updates raw data for a generic future instrument by calling continuous_contract_loader.py."""
        loader_symbol = f"@{symbol_root.upper()}" # e.g., @ES, @VX
        logger.info(f"Application.update_future_instrument_raw_data(symbol_root={symbol_root}, loader_symbol={loader_symbol}, interval_unit={interval_unit}, interval_value={interval_value}, force_full={force_full}) called.")

        if self.db is None or self.db.conn is None:
            logger.error(f"Database not initialized. Cannot update {loader_symbol} futures.")
            return

        if continuous_contract_loader_main is None:
            logger.error(f"continuous_contract_loader_main function not imported. Cannot update {loader_symbol} futures.")
            return
        
        try:
            logger.info(f"Calling continuous_contract_loader_main for {loader_symbol} futures.")
            args_for_script_list = [
                loader_symbol,
                "--db-path", str(self.db.db_path),
                "--config-path", str(self.config_path),
                "--interval-unit", interval_unit,
                "--interval-value", str(interval_value),
                "--fetch-mode", fetch_mode,
                "--lookback-days", lookback_days, # Defaulted to 90 to match old ES/NQ calls
                "--roll-proximity-threshold-days", roll_proximity_threshold_days # Defaulted to 7
            ]
            if force_full:
                args_for_script_list.append("--force")
            
            # Ensure continuous_contract_loader_main can accept existing_conn if needed, or handles its own
            # Based on its import and usage in update_es_futures, it takes args_list
            continuous_contract_loader_main(args_list=args_for_script_list)
            logger.info(f"{loader_symbol} futures update function (continuous_contract_loader) completed.")
        except Exception as e:
            logger.error(f"Error calling continuous_contract_loader_main for {loader_symbol}: {e}", exc_info=True)

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

    def get_continuous_contract_symbols(self) -> List[Dict[str, Any]]:
        """Get a list of symbol configurations for continuous contract generation."""
        logger.info("Application.get_continuous_contract_symbols() called.")

        # Load global frequency definitions
        global_freq_settings = self.config.get_section('settings')
        global_data_frequencies = {}
        if global_freq_settings and 'data_frequencies' in global_freq_settings:
            for freq_def in global_freq_settings['data_frequencies']:
                if 'name' in freq_def and 'unit' in freq_def and 'interval' in freq_def:
                    global_data_frequencies[freq_def['name']] = freq_def
                else:
                    logger.warning(f"Invalid frequency definition in settings.data_frequencies: {freq_def}")
        if not global_data_frequencies:
            logger.warning("No valid global data_frequencies found in settings. Could not map frequency names.")

        # Helper to get or create a root symbol entry in our results list
        def get_or_create_root_entry(results_list, root_sym, item_cfg_for_defaults):
            for entry in results_list:
                if entry['root_symbol'] == root_sym:
                    return entry
            # Not found, create it
            new_entry = {
                'root_symbol': root_sym,
                'asset_type': item_cfg_for_defaults.get('asset_type', 'future_group'),
                'roll_calendar_table': item_cfg_for_defaults.get('roll_calendar_table', 'futures_roll_dates'),
                'market_data_table': item_cfg_for_defaults.get('market_data_table', 'market_data'),
                'continuous_data_table': item_cfg_for_defaults.get('continuous_data_table', 'continuous_contracts'),
                'continuous_symbols_to_generate': [],
                'update_frequencies': [], # Initialize update_frequencies
                'original_config': item_cfg_for_defaults.copy()
            }
            results_list.append(new_entry)
            return new_entry

        symbols_for_generation = []
        futures_config_list = self.config.get_section('futures')

        if not futures_config_list:
            logger.warning("No 'futures' section found in the main configuration.")
            return symbols_for_generation

        for item_config in futures_config_list:
            root_symbol_candidate = None
            current_config_for_frequencies = item_config # Default to item_config

            # Case 1: Item is a root definition (e.g., ES, NQ with panama_positions)
            if 'base_symbol' in item_config:
                root_symbol_candidate = item_config['base_symbol']
                current_config_for_frequencies = item_config
                panama_positions = item_config.get('panama_positions')
                if panama_positions:
                    logger.debug(f"Processing root symbol '{root_symbol_candidate}' with panama_positions.")
                    root_entry = get_or_create_root_entry(symbols_for_generation, root_symbol_candidate, item_config)
                    
                    for i, pos_detail in enumerate(panama_positions):
                        position_number = i + 1
                        settings_code = "102XC" # Default for Panama
                        if isinstance(pos_detail, int):
                            position_number = pos_detail
                        elif isinstance(pos_detail, dict):
                            position_number = pos_detail.get('position', position_number)
                            settings_code = pos_detail.get('settings_code', settings_code)
                        
                        specific_symbol_name = f"@{root_symbol_candidate}={settings_code}"
                        if len(panama_positions) > 1 and isinstance(pos_detail, int):
                             specific_symbol_name = f"@{root_symbol_candidate}={position_number}{settings_code.lstrip('0123456789')}"

                        root_entry['continuous_symbols_to_generate'].append({
                            'symbol': specific_symbol_name,
                            'position': position_number,
                            'adjustment_method': 'panama',
                            'settings_code': settings_code
                        })
                else: # It's a base_symbol definition but might not have panama_positions, still need to process its frequencies
                    logger.debug(f"Processing root symbol '{root_symbol_candidate}' (no panama_positions, for frequency processing).")
                    root_entry = get_or_create_root_entry(symbols_for_generation, root_symbol_candidate, item_config)

            # Case 2: Item is a continuous_group definition
            elif 'continuous_group' in item_config:
                group_details = item_config['continuous_group']
                current_config_for_frequencies = group_details # Frequencies come from group_details here
                identifier_base = group_details.get('identifier_base')
                if identifier_base:
                    root_symbol_candidate = identifier_base.lstrip('@')
                    logger.debug(f"Processing continuous_group for identifier_base '{identifier_base}' (root: '{root_symbol_candidate}').")
                    root_entry = get_or_create_root_entry(symbols_for_generation, root_symbol_candidate, group_details)
                    
                    month_codes = group_details.get('month_codes', [])
                    settings_code_base = group_details.get('settings_code', "01XN")
                    adj_method = group_details.get('adjustment_method', 'panama')

                    for i, month_code_part in enumerate(month_codes):
                        final_settings_code = f"{i+1}{settings_code_base}"
                        specific_symbol_name = f"{identifier_base}={final_settings_code}"

                        root_entry['continuous_symbols_to_generate'].append({
                            'symbol': specific_symbol_name,
                            'position': i + 1,
                            'adjustment_method': adj_method,
                            'settings_code': final_settings_code
                        })
            
            # Case 3: Item is an explicitly defined individual continuous contract
            # These typically don't define raw data frequencies themselves but are derived.
            # Their frequencies will be handled by their root if defined.
            elif 'symbol' in item_config and item_config.get('type') == 'continuous_future':
                specific_symbol_name = item_config['symbol'] 
                parsed_root_match = Path(specific_symbol_name).stem.lstrip('@') 
                if parsed_root_match and '=' in specific_symbol_name:
                     root_symbol_candidate = specific_symbol_name.split('=')[0].lstrip('@')

                if root_symbol_candidate:
                    logger.debug(f"Processing explicit continuous_future '{specific_symbol_name}' for root '{root_symbol_candidate}'. Its frequencies will be determined by the root config.")
                    # Ensure root_entry exists so that continuous_symbols_to_generate can be populated.
                    # The frequencies themselves come from the main root_symbol config (e.g. 'ES' block)
                    root_entry = get_or_create_root_entry(symbols_for_generation, root_symbol_candidate, item_config) # item_config for defaults if root doesn't exist yet
                    
                    root_entry['continuous_symbols_to_generate'].append({
                        'symbol': specific_symbol_name,
                        'position': item_config.get('position', 1), 
                        'adjustment_method': item_config.get('adjustment_method', 'panama'), 
                        'settings_code': item_config.get('settings_code', specific_symbol_name.split('=')[-1] if '=' in specific_symbol_name else '')
                    })
                    # This path does not set current_config_for_frequencies as it's an individual symbol
                    # Frequencies are tied to the root (ES, NQ) not the specific continuous symbol (@ES=101XN)
                else:
                    logger.warning(f"Could not determine root symbol for explicit continuous_future: {item_config}. Skipping.")
                    # Set root_symbol_candidate to None so frequency processing logic below is skipped for this item
                    # as frequencies are handled by the main 'base_symbol' block.
                    root_symbol_candidate = None 
            
            else:
                logger.debug(f"Skipping item as it does not match known continuous contract definition patterns: {item_config.get('base_symbol') or item_config.get('symbol') or 'N/A'}")
                root_symbol_candidate = None # Ensure frequency processing is skipped

            # Process frequencies if a root_symbol_candidate was identified from base_symbol or continuous_group
            if root_symbol_candidate:
                # Find the root_entry again, as it might have been created in a different branch
                current_root_entry = None
                for entry in symbols_for_generation:
                    if entry['root_symbol'] == root_symbol_candidate:
                        current_root_entry = entry
                        break
                
                if not current_root_entry:
                    logger.warning(f"Could not find/create root_entry for {root_symbol_candidate} for frequency processing. This should not happen.")
                    continue

                raw_frequencies = current_config_for_frequencies.get('frequencies')
                processed_frequencies = []
                if isinstance(raw_frequencies, list):
                    for freq_item in raw_frequencies:
                        if isinstance(freq_item, str): # e.g., "1min"
                            if freq_item in global_data_frequencies:
                                processed_frequencies.append(global_data_frequencies[freq_item].copy()) # Add a copy
                            else:
                                logger.warning(f"Frequency name '{freq_item}' for {root_symbol_candidate} not found in global settings.data_frequencies. Skipping.")
                        elif isinstance(freq_item, dict): # e.g., {'name': 'daily', 'unit': 'daily', 'interval': 1}
                            if 'unit' in freq_item and 'interval' in freq_item:
                                processed_frequencies.append(freq_item.copy()) # Add a copy
                            else:
                                logger.warning(f"Invalid frequency dictionary for {root_symbol_candidate}: {freq_item}. Missing 'unit' or 'interval'. Skipping.")
                        else:
                            logger.warning(f"Invalid frequency item for {root_symbol_candidate}: {freq_item}. Skipping.")
                elif raw_frequencies: # Not a list, but not None/empty
                    logger.warning(f"Invalid 'frequencies' format for {root_symbol_candidate}. Expected list, got {type(raw_frequencies)}. Skipping frequency processing.")

                if processed_frequencies:
                    # Only update if not already populated (e.g. by another part of the config for the same root)
                    # and if the new list is not empty.
                    # A simple way is to extend if current is empty, or be more selective if merging.
                    # For now, let's assume one primary definition of frequencies per root symbol.
                    # If current_root_entry['update_frequencies'] is already populated, we might be overwriting.
                    # This can happen if 'ES' base_symbol appears multiple times or with continuous_future items.
                    # The get_or_create_root_entry uses the first item_config for defaults.
                    # Frequencies should ideally come from the 'base_symbol' or 'continuous_group' primary definition.
                    if not current_root_entry['update_frequencies']: # If empty, populate
                        current_root_entry['update_frequencies'] = processed_frequencies
                        logger.info(f"Populated update_frequencies for {root_symbol_candidate}: {processed_frequencies}")
                    elif current_root_entry['update_frequencies'] != processed_frequencies:
                        # If frequencies are already there AND different, it's a conflict/ambiguity.
                        logger.warning(f"Attempting to overwrite existing update_frequencies for {root_symbol_candidate}. "
                                       f"Old: {current_root_entry['update_frequencies']}, New: {processed_frequencies}. Keeping old.")
                        # Or decide on a merging strategy if necessary. For now, keep first.
                
                # Ensure original_config in root_entry is from the primary definition (base_symbol or continuous_group)
                # This is important if an explicit continuous_future item created the root_entry first.
                if current_root_entry.get('original_config', {}).get('symbol') and \
                   (item_config.get('base_symbol') or item_config.get('continuous_group')):
                    logger.debug(f"Updating original_config for {root_symbol_candidate} from primary definition.")
                    current_root_entry['original_config'] = current_config_for_frequencies.copy()

        # Clean up entries that didn't get any specific symbols generated OR don't have update frequencies
        # We still need symbols for continuous generation even if raw updates are not specified.
        # The filtering for continuous generation is separate.
        # Log symbols that have continuous_symbols_to_generate but no update_frequencies
        for entry in symbols_for_generation:
            if entry['continuous_symbols_to_generate'] and not entry['update_frequencies']:
                logger.warning(f"Root symbol {entry['root_symbol']} has continuous contracts to generate but no 'update_frequencies' defined. Raw data might not be updated.")
            # Also copy root-level properties like 'calendar', 'exchange', 'description' into the root_entry
            # from its 'original_config' for easier access by the caller.
            if 'original_config' in entry:
                for key_to_copy in ['calendar', 'exchange', 'description', 'data_source', 'default_raw_table', 'asset_type', 'start_date', 'contract_specs', 'expiry_rule', 'num_active_contracts']:
                    if key_to_copy in entry['original_config'] and key_to_copy not in entry:
                        entry[key_to_copy] = entry['original_config'][key_to_copy]

        final_symbols_for_generation = [
            entry for entry in symbols_for_generation if entry['continuous_symbols_to_generate']
        ]
        
        logger.info(f"get_continuous_contract_symbols returning: {len(final_symbols_for_generation)} root symbol configs.")
        if final_symbols_for_generation:
             logger.debug(f"First root symbol config details: {final_symbols_for_generation[0]}")
        return final_symbols_for_generation

    def get_vx_futures_symbols(self) -> List[str]:
        """
        Get a list of specific VX futures contract symbols from the configuration.
        Example: ['@VX=101XN', '@VX=201XN']
        """
        logger.info("Application.get_vx_futures_symbols() called.")
        vx_symbols = []
        all_continuous_configs = self.get_continuous_contract_symbols()

        for root_config_entry in all_continuous_configs:
            # Assuming VIX futures root symbol is 'VX' based on how get_continuous_contract_symbols
            # processes 'identifier_base' like '@VX'
            if root_config_entry.get('root_symbol', '').upper() == 'VX':
                specific_symbols_details = root_config_entry.get('continuous_symbols_to_generate', [])
                for detail in specific_symbols_details:
                    if 'symbol' in detail:
                        vx_symbols.append(detail['symbol'])
                # Assuming only one main configuration entry for 'VX' root.
                break 
        
        if not vx_symbols:
            logger.warning("No VX futures symbols found. Check 'futures' configuration for a group with identifier_base '@VX' or similar for VIX.")
            
        logger.info(f"Found VX futures symbols: {vx_symbols}")
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