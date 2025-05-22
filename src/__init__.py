"""
Financial Data System

A system for downloading, updating, managing, and inspecting financial market data,
particularly focusing on futures contracts (VIX, ES, NQ) using DuckDB.

This package provides modules for:
- Market data retrieval from multiple sources
- Continuous futures contract generation
- Data quality validation and cleaning
- Database management and visualization
"""

import logging
from pathlib import Path

# Package version
__version__ = '2.0.0'

# Set up a default null logger
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Import core components
from .core import get_app
from .core.config import init_config
from .core.database import get_database
from .core.logging import configure_logging

# Import data source components
from .data_sources.base import DataSourceRegistry

# Import processor components
from .processors.continuous import ContinuousContractRegistry
from .processors.cleaners import DataCleanerRegistry

__all__ = [
    '__version__',
    'get_app',
    'init_config',
    'get_database',
    'configure_logging',
    'DataSourceRegistry',
    'ContinuousContractRegistry',
    'DataCleanerRegistry',
    'PROJECT_ROOT'
]