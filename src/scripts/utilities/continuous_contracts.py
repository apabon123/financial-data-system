"""
Utilities for working with continuous futures contracts.
"""

import logging
import duckdb
import pandas as pd
from datetime import datetime
from typing import Optional, Union, List, Dict, Tuple

from .database import get_db_engine

# Get the logger
logger = logging.getLogger(__name__)

def get_active_contract(continuous_symbol: str, date: Union[str, datetime], 
                      db_path: Optional[str] = None) -> Optional[str]:
    """
    Get the active underlying contract for a continuous contract on a specific date.
    
    Args:
        continuous_symbol: The continuous contract symbol (e.g., @VX=101XN)
        date: The date to check, either as a string 'YYYY-MM-DD' or datetime object
        db_path: Optional path to the database file
        
    Returns:
        The underlying contract symbol (e.g., VXH25) or None if not found
    """
    # Convert date to string if it's a datetime
    if isinstance(date, datetime):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = date
        
    try:
        # Use default path if not provided
        if db_path is None:
            db_path = 'data/financial_data.duckdb'
            
        # Connect to the database directly rather than using get_db_engine
        conn = duckdb.connect(database=db_path, read_only=True)
        
        # Query the mapping table
        query = """
            SELECT underlying_symbol
            FROM continuous_contract_mapping
            WHERE continuous_symbol = ?
            AND date = ?
        """
        result = conn.execute(query, [continuous_symbol, date_str]).fetchdf()
        
        if result.empty:
            # If no exact match, find the nearest date before the lookup date
            query = """
                SELECT underlying_symbol
                FROM continuous_contract_mapping
                WHERE continuous_symbol = ?
                AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            result = conn.execute(query, [continuous_symbol, date_str]).fetchdf()
            
            if result.empty:
                logger.warning(f"No mapping found for {continuous_symbol} on or before {date_str}")
                return None
        
        # Return the underlying symbol
        return result['underlying_symbol'].iloc[0]
        
    except Exception as e:
        logger.error(f"Error looking up active contract for {continuous_symbol} on {date_str}: {e}")
        return None
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()

def get_all_active_contracts(date: Union[str, datetime], 
                           root_symbol: Optional[str] = None,
                           db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get all active contracts for a specific date, optionally filtered by root symbol.
    
    Args:
        date: The date to check, either as a string 'YYYY-MM-DD' or datetime object
        root_symbol: Optional root symbol to filter by (e.g., 'VX')
        db_path: Optional path to the database file
        
    Returns:
        Dictionary mapping continuous symbols to their underlying contracts
    """
    # Convert date to string if it's a datetime
    if isinstance(date, datetime):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = date
        
    try:
        # Use default path if not provided
        if db_path is None:
            db_path = 'data/financial_data.duckdb'
            
        # Connect to the database directly rather than using get_db_engine
        conn = duckdb.connect(database=db_path, read_only=True)
        
        # Prepare the query
        if root_symbol:
            query = """
                SELECT continuous_symbol, underlying_symbol
                FROM continuous_contract_mapping
                WHERE date = ?
                AND continuous_symbol LIKE ?
                ORDER BY continuous_symbol
            """
            params = [date_str, f"@{root_symbol}=%"]
        else:
            query = """
                SELECT continuous_symbol, underlying_symbol
                FROM continuous_contract_mapping
                WHERE date = ?
                ORDER BY continuous_symbol
            """
            params = [date_str]
            
        # Execute the query
        result = conn.execute(query, params).fetchdf()
        
        if result.empty:
            logger.warning(f"No mappings found for date {date_str}")
            return {}
            
        # Convert to dictionary
        mapping = dict(zip(result['continuous_symbol'], result['underlying_symbol']))
        return mapping
        
    except Exception as e:
        logger.error(f"Error getting active contracts for {date_str}: {e}")
        return {}
    finally:
        # Close the connection
        if 'conn' in locals():
            conn.close()

def create_continuous_mapping(root_symbol: str, start_date: str, end_date: str, 
                            num_contracts: int = 9, db_path: Optional[str] = None) -> bool:
    """
    Create or update the continuous contract mapping table for a root symbol.
    This is a wrapper around the generate_continuous_mapping function in the 
    create_continuous_contract_mapping.py script.
    
    Args:
        root_symbol: The root symbol (e.g., 'VX')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        num_contracts: Number of continuous contracts to generate
        db_path: Optional path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the function from the script
        # Using local import to avoid circular imports
        from src.scripts.market_data.create_continuous_contract_mapping import generate_continuous_mapping, connect_db
        
        # Connect to the database
        conn = connect_db(db_path or 'data/financial_data.duckdb', read_only=False)
        
        # Generate the mapping
        generate_continuous_mapping(conn, root_symbol, num_contracts, start_date, end_date)
        
        # Close the connection
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating continuous mapping for {root_symbol}: {e}")
        return False 