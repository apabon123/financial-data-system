"""
Processors Package

This package contains data processing modules for the Financial Data System,
including continuous futures generation and data cleaning.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

def register_all_processors() -> List[str]:
    """
    Register all available data processors.
    
    Returns:
        List of registered processor types
    """
    # Register continuous contract builders
    try:
        from .continuous import register_all_builders
        builders = register_all_builders()
        logger.debug(f"Registered {len(builders)} continuous contract builders: {', '.join(builders)}")
    except ImportError:
        logger.warning("Failed to import continuous contract modules")
        builders = []
    
    # Register data cleaners
    try:
        from .cleaners import register_all_cleaners
        cleaners = register_all_cleaners()
        logger.debug(f"Registered {len(cleaners)} data cleaners: {', '.join(cleaners)}")
    except ImportError:
        logger.warning("Failed to import data cleaner modules")
        cleaners = []
    
    # Return all registered processor types
    return builders + cleaners


__all__ = ['register_all_processors']