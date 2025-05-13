#!/usr/bin/env python
"""
Configuration Loader for Financial Data System

This module provides a sophisticated configuration loader that can:
1. Load and merge multiple YAML configuration files
2. Support references across files
3. Handle template inheritance
4. Provide environment-specific overrides
5. Validate configurations against schemas
"""

import os
import re
import yaml
import json
import logging
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Tuple
import jsonschema

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Advanced configuration loader that intelligently merges specialized YAML files.
    
    Features:
    - Load and merge multiple YAML files
    - Support for cross-file references using ${...} syntax
    - Support for environment variable references using ${env:...} syntax
    - Template inheritance with 'inherit' field
    - Schema validation for each configuration type
    - Environment-specific overrides
    """
    
    def __init__(self, config_dir: str = None, schema_dir: str = None, 
                 environment: str = None, legacy_support: bool = True):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to the directory containing YAML configuration files
            schema_dir: Path to the directory containing JSON schema files
            environment: Environment name for environment-specific overrides
            legacy_support: Whether to support the legacy single-file configuration
        """
        # Set default paths if not provided
        self.config_dir = config_dir or os.path.join(os.getcwd(), 'config')
        self.yaml_dir = os.path.join(self.config_dir, 'yaml')
        self.schema_dir = schema_dir or os.path.join(self.config_dir, 'schemas')
        
        # Environment settings
        self.environment = environment
        
        # Legacy support settings
        self.legacy_support = legacy_support
        self.legacy_file = os.path.join(self.config_dir, 'market_symbols.yaml')
        
        # Internal state
        self._config_cache: Dict[str, Any] = {}
        self._loaded_files: Set[str] = set()
        self._resolved_references: Dict[str, Any] = {}
        self._required_files = [
            'exchanges.yaml', 
            'futures.yaml',
            'indices.yaml',
            'etfs.yaml',
            'equities.yaml',
            'data_sources.yaml',
            'cleaning_rules.yaml'
        ]
        
        # Initialize schema validators
        self._schema_validators: Dict[str, jsonschema.Draft7Validator] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load schema validators for each configuration type."""
        if not os.path.exists(self.schema_dir):
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
            
        for schema_file in os.listdir(self.schema_dir):
            if not schema_file.endswith('.json'):
                continue
                
            schema_path = os.path.join(self.schema_dir, schema_file)
            schema_name = schema_file.replace('.json', '')
            
            try:
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                    
                self._schema_validators[schema_name] = jsonschema.Draft7Validator(schema)
                logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Error loading schema {schema_file}: {e}")
    
    def load_config(self, specific_files: List[str] = None) -> Dict[str, Any]:
        """
        Load and merge all configuration files.
        
        Args:
            specific_files: List of specific files to load (if None, loads all required files)
            
        Returns:
            Merged configuration dictionary
        """
        # Reset internal state
        self._config_cache = {}
        self._loaded_files = set()
        self._resolved_references = {}
        
        # Determine which files to load
        files_to_load = specific_files or self._required_files
        
        # Check if we need to fall back to legacy configuration
        if self.legacy_support and (not os.path.exists(self.yaml_dir) or not self._check_required_files()):
            logger.info("Using legacy configuration file")
            return self._load_legacy_config()
            
        # Load each configuration file
        for file_name in files_to_load:
            self._load_config_file(file_name)
            
        # Resolve cross-file references
        self._resolve_references()
        
        # Apply template inheritance
        self._apply_inheritance()
        
        # Apply environment-specific overrides
        if self.environment:
            self._apply_environment_overrides()
            
        # Validate configurations
        self._validate_configs()
        
        # Build integrated configuration
        integrated_config = self._build_integrated_config()
        
        return integrated_config
    
    def _check_required_files(self) -> bool:
        """Check if all required configuration files exist."""
        for file_name in self._required_files:
            file_path = os.path.join(self.yaml_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Required configuration file not found: {file_path}")
                return False
        return True
    
    def _load_legacy_config(self) -> Dict[str, Any]:
        """Load the legacy single-file configuration."""
        if not os.path.exists(self.legacy_file):
            logger.error(f"Legacy configuration file not found: {self.legacy_file}")
            return {}
            
        try:
            with open(self.legacy_file, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Loaded legacy configuration file: {self.legacy_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading legacy configuration file: {e}")
            return {}
    
    def _load_config_file(self, file_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            Configuration dictionary from the file
        """
        # Check if already loaded
        if file_name in self._loaded_files:
            return self._config_cache.get(file_name, {})
            
        # Build file path
        file_path = os.path.join(self.yaml_dir, file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            self._config_cache[file_name] = {}
            self._loaded_files.add(file_name)
            return {}
            
        # Load and parse YAML file
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.debug(f"Loaded configuration file: {file_path}")
            
            # Store in cache
            self._config_cache[file_name] = config
            self._loaded_files.add(file_name)
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            self._config_cache[file_name] = {}
            self._loaded_files.add(file_name)
            return {}
    
    def _resolve_references(self):
        """Resolve cross-file references in all loaded configurations."""
        for file_name, config in self._config_cache.items():
            self._config_cache[file_name] = self._resolve_references_in_obj(config)
    
    def _resolve_references_in_obj(self, obj: Any) -> Any:
        """
        Recursively resolve references in an object.
        
        Args:
            obj: Object to process
            
        Returns:
            Object with resolved references
        """
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[k] = self._resolve_references_in_obj(v)
            return result
        elif isinstance(obj, list):
            return [self._resolve_references_in_obj(item) for item in obj]
        elif isinstance(obj, str):
            # Check for reference pattern ${...}
            if obj.startswith("${") and obj.endswith("}"):
                ref_path = obj[2:-1]
                
                # Check if it's an environment variable reference
                if ref_path.startswith("env:"):
                    env_var = ref_path[4:]
                    env_value = os.environ.get(env_var)
                    if env_value is None:
                        logger.warning(f"Environment variable not found: {env_var}")
                        return obj
                    return env_value
                
                # Check if we've already resolved this reference
                if ref_path in self._resolved_references:
                    return self._resolved_references[ref_path]
                
                # Split the reference path
                parts = ref_path.split('.')
                
                # The first part is the file name without extension
                file_name = f"{parts[0]}.yaml"
                
                # Load the referenced file if not already loaded
                if file_name not in self._loaded_files:
                    self._load_config_file(file_name)
                
                # Navigate to the referenced object
                ref_obj = self._config_cache.get(file_name, {})
                for part in parts[1:]:
                    if isinstance(ref_obj, dict) and part in ref_obj:
                        ref_obj = ref_obj[part]
                    else:
                        logger.warning(f"Reference not found: {ref_path}")
                        return obj
                
                # Store the resolved reference
                self._resolved_references[ref_path] = ref_obj
                return ref_obj
            
            return obj
        else:
            return obj
    
    def _apply_inheritance(self):
        """Apply template inheritance in all loaded configurations."""
        for file_name, config in self._config_cache.items():
            self._apply_inheritance_in_file(file_name, config)
    
    def _apply_inheritance_in_file(self, file_name: str, config: Dict[str, Any]):
        """
        Apply template inheritance in a specific file.
        
        Args:
            file_name: Name of the file
            config: Configuration dictionary
        """
        # Check if the config has templates
        templates = config.get('templates', {})
        
        # Get the main section based on the file name
        section_name = file_name.split('.')[0]
        section = config.get(section_name, {})
        
        # Apply inheritance to items in the section
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, dict) and 'inherit' in value:
                    template_name = value['inherit']
                    if template_name in templates:
                        # Start with template and override with item properties
                        template = copy.deepcopy(templates[template_name])
                        self._merge_dict(template, value)
                        # Remove the inherit field
                        if 'inherit' in template:
                            del template['inherit']
                        config[section_name][key] = template
                    else:
                        logger.warning(f"Template '{template_name}' not found in {file_name}")
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge two dictionaries recursively.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key == 'inherit':
                continue
                
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific overrides."""
        env_dir = os.path.join(self.yaml_dir, 'environments', self.environment)
        
        if not os.path.exists(env_dir):
            logger.debug(f"Environment directory not found: {env_dir}")
            return
            
        # Load environment-specific overrides
        for file_name in self._loaded_files:
            env_file_path = os.path.join(env_dir, file_name)
            
            if not os.path.exists(env_file_path):
                continue
                
            try:
                with open(env_file_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    
                if env_config:
                    # Merge environment-specific config with main config
                    main_config = self._config_cache.get(file_name, {})
                    self._merge_dict(main_config, env_config)
                    logger.debug(f"Applied environment overrides for {file_name}")
            except Exception as e:
                logger.error(f"Error loading environment override {env_file_path}: {e}")
    
    def _validate_configs(self):
        """Validate each configuration against its schema."""
        for file_name, config in self._config_cache.items():
            schema_name = file_name.split('.')[0]
            
            if schema_name in self._schema_validators:
                validator = self._schema_validators[schema_name]
                
                try:
                    validator.validate(config)
                    logger.debug(f"Validated configuration: {file_name}")
                except jsonschema.exceptions.ValidationError as e:
                    logger.error(f"Validation error in {file_name}: {e}")
    
    def _build_integrated_config(self) -> Dict[str, Any]:
        """
        Build the integrated configuration from all loaded files.
        
        Returns:
            Integrated configuration dictionary
        """
        integrated_config = {}
        
        # Include each section from its respective file
        for file_name, config in self._config_cache.items():
            section_name = file_name.split('.')[0]
            
            if section_name in config:
                integrated_config[section_name] = config[section_name]
            
            # Also include version if available
            if 'version' in config:
                integrated_config.setdefault('versions', {})[section_name] = config['version']
        
        # Add metadata
        integrated_config['_metadata'] = {
            'loader': 'ConfigLoader',
            'environment': self.environment,
            'loaded_files': list(self._loaded_files)
        }
        
        return integrated_config
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section from the loaded configuration.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as a dictionary
        """
        file_name = f"{section}.yaml"
        
        if file_name not in self._loaded_files:
            self._load_config_file(file_name)
            
        config = self._config_cache.get(file_name, {})
        return config.get(section, {})
    
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Get a specific schema.
        
        Args:
            schema_name: Schema name
            
        Returns:
            Schema as a dictionary
        """
        schema_path = os.path.join(self.schema_dir, f"{schema_name}.json")
        
        if not os.path.exists(schema_path):
            logger.warning(f"Schema file not found: {schema_path}")
            return {}
            
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                
            return schema
        except Exception as e:
            logger.error(f"Error loading schema {schema_path}: {e}")
            return {}
    
    def save_integrated_config(self, output_path: str):
        """
        Save the integrated configuration to a file.
        
        Args:
            output_path: Path to save the configuration
        """
        config = self.load_config()
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Saved integrated configuration to {output_path}")
        except Exception as e:
            logger.error(f"Error saving integrated configuration: {e}")
    
    def convert_legacy_to_new(self, output_dir: str = None):
        """
        Convert legacy configuration to the new structure.
        
        Args:
            output_dir: Directory to save the new configuration files
        """
        if not os.path.exists(self.legacy_file):
            logger.error(f"Legacy configuration file not found: {self.legacy_file}")
            return
            
        # Set default output directory
        output_dir = output_dir or self.yaml_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load legacy configuration
            with open(self.legacy_file, 'r') as f:
                legacy_config = yaml.safe_load(f)
                
            # Extract sections
            exchanges_config = self._extract_exchanges(legacy_config)
            futures_config = self._extract_futures(legacy_config)
            indices_config = self._extract_indices(legacy_config)
            etfs_config = self._extract_etfs(legacy_config)
            equities_config = self._extract_equities(legacy_config)
            data_sources_config = self._extract_data_sources(legacy_config)
            cleaning_rules_config = self._extract_cleaning_rules(legacy_config)
            
            # Save to files
            self._save_yaml(exchanges_config, os.path.join(output_dir, 'exchanges.yaml'))
            self._save_yaml(futures_config, os.path.join(output_dir, 'futures.yaml'))
            self._save_yaml(indices_config, os.path.join(output_dir, 'indices.yaml'))
            self._save_yaml(etfs_config, os.path.join(output_dir, 'etfs.yaml'))
            self._save_yaml(equities_config, os.path.join(output_dir, 'equities.yaml'))
            self._save_yaml(data_sources_config, os.path.join(output_dir, 'data_sources.yaml'))
            self._save_yaml(cleaning_rules_config, os.path.join(output_dir, 'cleaning_rules.yaml'))
            
            logger.info(f"Successfully converted legacy configuration to new structure in {output_dir}")
        except Exception as e:
            logger.error(f"Error converting legacy configuration: {e}")
    
    def _extract_exchanges(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract exchanges configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        return {
            "version": "1.0",
            "exchanges": {}
        }
    
    def _extract_futures(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract futures configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        futures = legacy_config.get('futures', [])
        return {
            "version": "1.0",
            "templates": {},
            "futures": {}
        }
    
    def _extract_indices(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract indices configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        indices = legacy_config.get('indices', [])
        return {
            "version": "1.0",
            "templates": {},
            "indices": {}
        }
    
    def _extract_etfs(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ETFs configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        etfs = []
        for item in legacy_config.get('equities', []):
            if item.get('type') == 'ETF':
                etfs.append(item)
        
        return {
            "version": "1.0",
            "templates": {},
            "etfs": {}
        }
    
    def _extract_equities(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract equities configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        stocks = []
        for item in legacy_config.get('equities', []):
            if item.get('type') == 'Stock':
                stocks.append(item)
        
        return {
            "version": "1.0",
            "templates": {},
            "equities": {}
        }
    
    def _extract_data_sources(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data sources configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        return {
            "version": "1.0",
            "templates": {},
            "data_sources": {}
        }
    
    def _extract_cleaning_rules(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cleaning rules configuration from legacy config."""
        # This is a placeholder - you would implement extraction logic based on your legacy format
        return {
            "version": "1.0",
            "templates": {},
            "cleaning_rules": {}
        }
    
    def _save_yaml(self, data: Dict[str, Any], file_path: str):
        """Save data to a YAML file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        logger.debug(f"Saved configuration to {file_path}")


class ConfigManager:
    """
    Configuration manager that provides easy access to configuration sections.
    
    This class is a wrapper around ConfigLoader that provides a simpler interface
    for accessing configuration sections.
    """
    
    def __init__(self, config_dir: str = None, schema_dir: str = None, 
                 environment: str = None, legacy_support: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to the directory containing YAML configuration files
            schema_dir: Path to the directory containing JSON schema files
            environment: Environment name for environment-specific overrides
            legacy_support: Whether to support the legacy single-file configuration
        """
        self.loader = ConfigLoader(
            config_dir=config_dir,
            schema_dir=schema_dir,
            environment=environment,
            legacy_support=legacy_support
        )
        
        # Load the full configuration
        self.config = self.loader.load_config()
    
    def reload(self):
        """Reload the configuration."""
        self.config = self.loader.load_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section from the configuration.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as a dictionary
        """
        return self.config.get(section, {})
    
    def get_item(self, section: str, item_name: str) -> Dict[str, Any]:
        """
        Get a specific item from a section.
        
        Args:
            section: Section name
            item_name: Item name
            
        Returns:
            Item as a dictionary
        """
        section_data = self.get_section(section)
        return section_data.get(item_name, {})
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get a specific value using a dot-notation path.
        
        Args:
            path: Path to the value (e.g., "futures.ES.contract_info.patterns")
            default: Default value if not found
            
        Returns:
            The value at the specified path, or the default if not found
        """
        parts = path.split('.')
        
        # Navigate through the configuration
        current = self.config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def get_environment(self) -> str:
        """Get the current environment."""
        return self.loader.environment
    
    def set_environment(self, environment: str):
        """
        Set the environment.
        
        Args:
            environment: Environment name
        """
        self.loader.environment = environment
        self.reload()
    
    def save_integrated_config(self, output_path: str):
        """
        Save the integrated configuration to a file.
        
        Args:
            output_path: Path to save the configuration
        """
        self.loader.save_integrated_config(output_path)
    
    def convert_legacy_to_new(self, output_dir: str = None):
        """
        Convert legacy configuration to the new structure.
        
        Args:
            output_dir: Directory to save the new configuration files
        """
        self.loader.convert_legacy_to_new(output_dir)