"""
Registry for continuous futures contract generators.

This module provides a centralized registry for all continuous contract generation methods,
allowing for dynamic loading and configuration of different generators based on configuration
settings or user preferences.
"""

import logging
import importlib
from typing import Dict, List, Optional, Type, Any, Set

from .base import ContinuousContractBase
from .panama import PanamaContractGenerator

# Logger for this module
logger = logging.getLogger(__name__)

class ContinuousContractRegistry:
    """Registry for continuous contract generation methods."""
    
    def __init__(self):
        """Initialize the registry."""
        self._generators: Dict[str, Type[ContinuousContractBase]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register the default generators."""
        try:
            # Register the Panama method
            self.register('panama', PanamaContractGenerator)
            
            # Add more default generators here
            
            logger.debug(f"Registered {len(self._generators)} default generators")
        except Exception as e:
            logger.error(f"Error registering default generators: {e}")
    
    def register(self, name: str, generator_class: Type[ContinuousContractBase]) -> bool:
        """
        Register a new generator.
        
        Args:
            name: Name of the generator
            generator_class: Generator class
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate the generator class
            if not issubclass(generator_class, ContinuousContractBase):
                logger.error(f"Cannot register {name}: {generator_class.__name__} "
                           f"is not a subclass of ContinuousContractBase")
                return False
            
            # Register the generator
            self._generators[name.lower()] = generator_class
            logger.debug(f"Registered generator: {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering generator {name}: {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a generator.
        
        Args:
            name: Name of the generator
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        try:
            name = name.lower()
            if name in self._generators:
                del self._generators[name]
                logger.debug(f"Unregistered generator: {name}")
                return True
            else:
                logger.warning(f"Generator {name} not found in registry")
                return False
        except Exception as e:
            logger.error(f"Error unregistering generator {name}: {e}")
            return False
    
    def get(self, name: str) -> Optional[Type[ContinuousContractBase]]:
        """
        Get a generator by name.
        
        Args:
            name: Name of the generator
            
        Returns:
            Generator class or None if not found
        """
        return self._generators.get(name.lower())
    
    def create(self, name: str, **kwargs) -> Optional[ContinuousContractBase]:
        """
        Create a new instance of a generator.
        
        Args:
            name: Name of the generator
            **kwargs: Arguments to pass to the generator constructor
            
        Returns:
            Generator instance or None if not found
        """
        try:
            generator_class = self.get(name)
            
            if generator_class is None:
                logger.error(f"Generator {name} not found in registry")
                return None
            
            return generator_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating generator {name}: {e}")
            return None
    
    def list_generators(self) -> List[str]:
        """
        Get a list of all registered generators.
        
        Returns:
            List of generator names
        """
        return list(self._generators.keys())
    
    def load_from_module(self, module_path: str) -> Set[str]:
        """
        Load generators from a module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            Set of loaded generator names
        """
        loaded = set()
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find all classes that inherit from ContinuousContractBase
            for name in dir(module):
                try:
                    obj = getattr(module, name)
                    
                    if (isinstance(obj, type) and 
                        issubclass(obj, ContinuousContractBase) and 
                        obj is not ContinuousContractBase):
                        
                        # Register the generator
                        # Use the snake_case version of the class name as the key
                        key = ''.join(['_' + c.lower() if c.isupper() else c
                                      for c in name]).lstrip('_')
                        
                        # Remove 'contract_generator' or 'generator' suffix if present
                        for suffix in ['_contract_generator', '_generator']:
                            if key.endswith(suffix):
                                key = key[:-len(suffix)]
                        
                        if self.register(key, obj):
                            loaded.add(key)
                except Exception as e:
                    logger.warning(f"Error processing {name} from {module_path}: {e}")
            
            logger.info(f"Loaded {len(loaded)} generators from {module_path}")
            return loaded
            
        except ImportError as e:
            logger.error(f"Error importing module {module_path}: {e}")
            return set()
        except Exception as e:
            logger.error(f"Error loading generators from {module_path}: {e}")
            return set()
    
    def create_from_config(self, config: Dict[str, Any]) -> Optional[ContinuousContractBase]:
        """
        Create a generator instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Generator instance or None if creation failed
            
        Expected config format:
        {
            "method": "panama",
            "root_symbol": "ES",
            "position": 1,
            "roll_strategy": "volume",
            "ratio_limit": 0.75,
            ...
        }
        """
        try:
            # Extract method
            method = config.get('method', 'panama').lower()
            
            # Extract common parameters
            kwargs = config.copy()
            
            # Remove method from kwargs
            if 'method' in kwargs:
                del kwargs['method']
            
            # Create the generator
            return self.create(method, **kwargs)
            
        except Exception as e:
            logger.error(f"Error creating generator from config: {e}")
            return None


# Singleton instance
registry = ContinuousContractRegistry()

def get_registry() -> ContinuousContractRegistry:
    """Get the global registry instance."""
    return registry