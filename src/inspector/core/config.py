"""
Configuration module for DB Inspector.

This module handles loading and managing configuration settings for
the DB Inspector tool.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

# Default paths
def normalize_path(path):
    """
    Normalize path for cross-platform compatibility.

    In WSL, convert Windows paths to WSL paths if needed.
    """
    str_path = str(path)

    # Check if we're in WSL
    in_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            in_wsl = 'microsoft' in f.read().lower()
    except:
        pass

    # Handle Windows paths in WSL
    if in_wsl and str_path.startswith('/mnt/c/'):
        # Already in correct format
        return path
    elif in_wsl and str_path.startswith('C:'):
        # Convert Windows path to WSL path
        return Path('/mnt/c/' + str_path[3:].replace('\\', '/'))

    # Regular path handling
    return path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Default paths with normalization
DEFAULT_CONFIG_PATH = normalize_path(BASE_DIR / "config" / "inspector_config.yaml")
DEFAULT_SQL_PATH = normalize_path(BASE_DIR / "src" / "sql" / "sql")
DEFAULT_DB_PATH = normalize_path(BASE_DIR / "data" / "financial_data.duckdb")
DEFAULT_HISTORY_PATH = normalize_path(BASE_DIR / "data" / "inspector_history.json")
DEFAULT_EXPORT_PATH = normalize_path(BASE_DIR / "output" / "exports")
DEFAULT_REPORTS_PATH = normalize_path(BASE_DIR / "output" / "reports")
DEFAULT_BACKUP_PATH = normalize_path(BASE_DIR / "backups")
DEFAULT_LOG_PATH = normalize_path(BASE_DIR / "logs" / "inspector.log")
DEFAULT_THEME = "dark"

# Create directories if they don't exist
for path in [Path(BASE_DIR / "data"), DEFAULT_EXPORT_PATH, DEFAULT_REPORTS_PATH, DEFAULT_BACKUP_PATH, Path(BASE_DIR / "logs")]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")
        # Not fatal, will be handled during operations

class InspectorConfig:
    """Configuration manager for DB Inspector."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "paths": {
                "db": str(DEFAULT_DB_PATH),
                "sql": str(DEFAULT_SQL_PATH),
                "history": str(DEFAULT_HISTORY_PATH),
                "export": str(DEFAULT_EXPORT_PATH),
                "reports": str(DEFAULT_REPORTS_PATH),
                "backup": str(DEFAULT_BACKUP_PATH),
                "log": str(DEFAULT_LOG_PATH)
            },
            "ui": {
                "theme": DEFAULT_THEME,
                "max_rows_display": 1000,
                "syntax_highlighting": True,
                "show_query_time": True
            },
            "data_quality": {
                "outlier_threshold": 3.0,  # Z-score threshold
                "missing_data_threshold": 0.1,  # 10% missing data
                "price_spike_threshold": 10.0,  # 10% price change
                "volume_spike_threshold": 5.0  # 5x normal volume
            },
            "performance": {
                "cache_schema": True,
                "cache_results": True,
                "max_cache_size": 50,  # MB
                "max_query_time": 300  # seconds
            },
            "visualization": {
                "default_chart_type": "line",
                "default_color_scheme": "viridis",
                "interactive": True,
                "max_points": 10000
            },
            "backups": {
                "auto_backup_enabled": True,
                "backup_before_changes": True,
                "max_backups": 10,
                "compression": True
            },
            "market_structure": {
                "default_roll_method": "volume",
                "panama_ratio": 0.75,
                "correlation_period": 60  # days
            }
        }

        # If configuration file exists, load it
        config_loaded = False
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)

                # Merge file configuration with default
                merged_config = self._merge_configs(default_config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
                config_loaded = True
                return merged_config
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")

        # If we reach here, either the file doesn't exist or there was an error
        if not config_loaded:
            print(f"Creating default configuration at {self.config_path}")
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default configuration at {self.config_path}")
            except Exception as e:
                logger.error(f"Error creating default configuration at {self.config_path}: {e}")
                print(f"Warning: Could not create configuration file: {e}")

        # Validate and normalize paths
        self._validate_paths(default_config)

        return default_config

    def _validate_paths(self, config: Dict[str, Any]) -> None:
        """
        Validate and ensure paths in config exist.

        Args:
            config: Configuration dictionary
        """
        paths = config.get("paths", {})

        # Check if database path exists, use fallbacks if needed
        db_path = paths.get("db")
        if db_path and not os.path.exists(db_path):
            # Try alternate paths
            alternate_paths = [
                str(normalize_path(BASE_DIR / "data" / "financial_data.duckdb")),
                "data/financial_data.duckdb",
                "./financial_data.duckdb",
                "/mnt/c/Users/alexp/OneDrive/Gdrive/Trading/GitHub Projects/data-management/financial-data-system/data/financial_data.duckdb"
            ]

            for alt_path in alternate_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Using alternate database path: {alt_path}")
                    paths["db"] = alt_path
                    break
            else:
                logger.warning(f"Database path not found: {db_path} (will create new DB if needed)")

        # Ensure log directory exists
        log_path = paths.get("log")
        if log_path:
            log_dir = os.path.dirname(log_path)
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating log directory {log_dir}: {e}")

                # Fallback to a safe location
                fallback_log = "./inspector.log"
                logger.warning(f"Using fallback log path: {fallback_log}")
                paths["log"] = fallback_log
    
    def _merge_configs(self, default_config: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default configuration with file configuration.
        
        Args:
            default_config: Default configuration dictionary
            file_config: Configuration dictionary from file
            
        Returns:
            Merged configuration dictionary
        """
        result = default_config.copy()
        
        for section, section_config in file_config.items():
            if section not in result:
                result[section] = section_config
            elif isinstance(section_config, dict) and isinstance(result[section], dict):
                for key, value in section_config.items():
                    result[section][key] = value
        
        return result
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (if None, returns the entire section)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")

# Global instance
config = InspectorConfig()

def get_config() -> InspectorConfig:
    """
    Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return config