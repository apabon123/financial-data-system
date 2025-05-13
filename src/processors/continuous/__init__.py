"""
Continuous Futures Package

This package contains modules for generating continuous futures contracts
using various methodologies (Panama, unadjusted).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

def register_all_builders() -> List[str]:
    """
    Register all available continuous contract builders.
    
    Returns:
        List of registered builder names
    """
    # Import all builder modules to trigger registration
    try:
        from .panama import PanamaContractBuilder
        logger.debug("Registered Panama continuous contract builder")
    except ImportError:
        logger.warning("Failed to import Panama contract builder")
    
    try:
        from .unadjusted import UnadjustedContractBuilder
        logger.debug("Registered Unadjusted continuous contract builder")
    except ImportError:
        logger.warning("Failed to import Unadjusted contract builder")
    
    # Import other builders as needed
    
    # Return the list of registered builders
    from .base import ContinuousContractRegistry
    return ContinuousContractRegistry.list_builders()


from .base import (
    ContinuousContractBuilder,
    ContinuousContractRegistry,
    ContinuousContractError,
    ContractRollover
)

__all__ = [
    'register_all_builders',
    'ContinuousContractBuilder',
    'ContinuousContractRegistry',
    'ContinuousContractError',
    'ContractRollover'
]