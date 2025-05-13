"""
Data Sources Package

This package contains data source plugins for fetching data from various sources.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

def register_all_plugins() -> List[str]:
    """
    Register all available data source plugins.
    
    Returns:
        List of registered plugin names
    """
    # Import all plugin modules to trigger registration
    try:
        from .tradestation import TradeStationDataSource
        logger.debug("Registered TradeStation data source plugin")
    except ImportError:
        logger.warning("Failed to import TradeStation data source plugin")
    
    try:
        from .cboe import CBOEDataSource
        logger.debug("Registered CBOE data source plugin")
    except ImportError:
        logger.warning("Failed to import CBOE data source plugin")
    
    # Import other plugins as needed
    
    # Return the list of registered plugins
    from .base import DataSourceRegistry
    return DataSourceRegistry.list_sources()


__all__ = ['register_all_plugins']