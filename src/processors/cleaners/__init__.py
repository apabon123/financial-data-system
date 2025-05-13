"""
Data Cleaners Package

This package contains modules for cleaning and validating market data.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

def register_all_cleaners() -> List[str]:
    """
    Register all available data cleaners.
    
    Returns:
        List of registered cleaner names
    """
    # Import all cleaner modules to trigger registration
    try:
        from .vx_cleaner import VXDataCleaner
        logger.debug("Registered VX data cleaner")
    except ImportError:
        logger.warning("Failed to import VX data cleaner")
    
    # Import other cleaners as needed
    
    # Return the list of registered cleaners
    from .base import DataCleanerRegistry
    return DataCleanerRegistry.list_cleaners()


from .base import (
    DataCleaner,
    DataCleanerRegistry,
    DataCleanerError
)

__all__ = [
    'register_all_cleaners',
    'DataCleaner',
    'DataCleanerRegistry',
    'DataCleanerError'
]