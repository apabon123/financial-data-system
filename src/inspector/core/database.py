"""
Database module for DB Inspector.

This module handles database connections and operations for the DB Inspector tool.
"""

import time
import logging
import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .config import get_config

# Setup logging
logger = logging.getLogger(__name__)

class QueryResult:
    """Class to hold query results with metadata."""
    
    def __init__(self, 
                 dataframe: Optional[pd.DataFrame] = None, 
                 error: Optional[str] = None,
                 execution_time: float = 0.0,
                 query: str = "",
                 is_success: bool = True,
                 affected_rows: int = 0):
        """
        Initialize query result.
        
        Args:
            dataframe: Result DataFrame
            error: Error message if query failed
            execution_time: Query execution time in seconds
            query: SQL query that was executed
            is_success: Whether the query was successful
            affected_rows: Number of rows affected by the query
        """
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.error = error
        self.execution_time = execution_time
        self.query = query
        self.is_success = is_success
        self.affected_rows = affected_rows
        
    @property
    def is_empty(self) -> bool:
        """Check if the result is empty."""
        return self.dataframe.empty if self.dataframe is not None else True
    
    @property
    def row_count(self) -> int:
        """Get the number of rows in the result."""
        return len(self.dataframe) if self.dataframe is not None else 0
    
    @property
    def column_count(self) -> int:
        """Get the number of columns in the result."""
        return len(self.dataframe.columns) if self.dataframe is not None else 0

class DatabaseManager:
    """Manager for database connections and operations."""
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None, read_only: bool = True):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to database file
            read_only: Whether to open the database in read-only mode
        """
        config = get_config()
        self.db_path = Path(db_path) if db_path else Path(config.get("paths", "db"))
        self.read_only = read_only
        self.connection = None
        self.schema_cache = {}
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "max_execution_time": 0.0,
            "last_query_time": 0.0
        }
        self.query_history = []
        
        # Connect to database
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if connection was successful, False otherwise
        """
        if self.connection:
            self.close()
            
        try:
            self.connection = duckdb.connect(database=str(self.db_path), read_only=self.read_only)
            logger.info(f"Connected to database: {self.db_path} (read-only: {self.read_only})")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def execute_query(self, query: str) -> QueryResult:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            
        Returns:
            QueryResult object
        """
        self.performance_stats["total_queries"] += 1
        
        if not self.connection:
            self.connect()
            if not self.connection:
                return QueryResult(
                    error="No database connection",
                    query=query,
                    is_success=False
                )
        
        start_time = time.time()
        try:
            # Execute query
            result_relation = self.connection.execute(query)
            
            # Process result
            if result_relation is None:
                # Non-SELECT query with no result set
                end_time = time.time()
                execution_time = end_time - start_time
                
                self.performance_stats["successful_queries"] += 1
                self.performance_stats["total_execution_time"] += execution_time
                self.performance_stats["last_query_time"] = execution_time
                
                if execution_time > self.performance_stats["max_execution_time"]:
                    self.performance_stats["max_execution_time"] = execution_time
                
                # Add to query history
                self._add_to_history(query, execution_time, True)
                
                return QueryResult(
                    execution_time=execution_time,
                    query=query,
                    is_success=True,
                    affected_rows=0  # We don't know the exact count
                )
            else:
                # Query with result set
                df = result_relation.fetchdf()
                end_time = time.time()
                execution_time = end_time - start_time
                
                self.performance_stats["successful_queries"] += 1
                self.performance_stats["total_execution_time"] += execution_time
                self.performance_stats["last_query_time"] = execution_time
                
                if execution_time > self.performance_stats["max_execution_time"]:
                    self.performance_stats["max_execution_time"] = execution_time
                
                # Add to query history
                self._add_to_history(query, execution_time, True)
                
                return QueryResult(
                    dataframe=df,
                    execution_time=execution_time,
                    query=query,
                    is_success=True,
                    affected_rows=len(df)
                )
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.performance_stats["failed_queries"] += 1
            self.performance_stats["total_execution_time"] += execution_time
            self.performance_stats["last_query_time"] = execution_time
            
            # Add to query history
            self._add_to_history(query, execution_time, False, str(e))
            
            return QueryResult(
                error=str(e),
                execution_time=execution_time,
                query=query,
                is_success=False
            )
    
    def execute_multiple_queries(self, queries: List[str], 
                                stop_on_error: bool = False) -> List[QueryResult]:
        """
        Execute multiple SQL queries.
        
        Args:
            queries: List of SQL queries to execute
            stop_on_error: Whether to stop execution on error
            
        Returns:
            List of QueryResult objects
        """
        results = []
        
        for query in queries:
            result = self.execute_query(query)
            results.append(result)
            
            if stop_on_error and not result.is_success:
                break
                
        return results
    
    def execute_script(self, script: str, 
                      stop_on_error: bool = False) -> List[QueryResult]:
        """
        Execute a SQL script with multiple statements.
        
        Args:
            script: SQL script to execute
            stop_on_error: Whether to stop execution on error
            
        Returns:
            List of QueryResult objects
        """
        # Split script into individual queries
        queries = self._split_sql_script(script)
        return self.execute_multiple_queries(queries, stop_on_error)
    
    def _split_sql_script(self, script: str) -> List[str]:
        """
        Split a SQL script into individual queries.
        
        Args:
            script: SQL script to split
            
        Returns:
            List of individual SQL queries
        """
        # This is a basic implementation - could be improved to handle comments, quotes, etc.
        return [q.strip() for q in script.split(';') if q.strip()]
    
    def _add_to_history(self, query: str, execution_time: float, 
                      success: bool, error: str = "") -> None:
        """
        Add a query to the history.
        
        Args:
            query: SQL query
            execution_time: Query execution time in seconds
            success: Whether the query was successful
            error: Error message if query failed
        """
        max_history = get_config().get("performance", "max_query_history", 100)
        
        history_entry = {
            "query": query,
            "execution_time": execution_time,
            "success": success,
            "error": error,
            "timestamp": time.time()
        }
        
        self.query_history.append(history_entry)
        
        # Trim history if it gets too long
        if len(self.query_history) > max_history:
            self.query_history = self.query_history[-max_history:]
    
    def get_tables(self) -> List[str]:
        """
        Get a list of all tables in the database.
        
        Returns:
            List of table names
        """
        result = self.execute_query("""
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'table' 
            AND name NOT LIKE 'sqlite_%'
        """)
        if result.is_success and not result.is_empty:
            return result.dataframe['name'].tolist()
        return []
    
    def get_views(self) -> List[str]:
        """
        Get a list of all views in the database.
        
        Returns:
            List of view names
        """
        result = self.execute_query("""
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'view'
        """)
        if result.is_success and not result.is_empty:
            return result.dataframe['name'].tolist()
        return []
    
    def get_table_schema(self, table_name: str) -> QueryResult:
        """
        Get the schema of a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            QueryResult object with table schema
        """
        return self.execute_query(f"DESCRIBE {table_name}")
    
    def get_table_preview(self, table_name: str, limit: int = 10) -> QueryResult:
        """
        Get a preview of a table's data.
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            QueryResult object with table preview
        """
        return self.execute_query(f"SELECT * FROM {table_name} LIMIT {limit}")
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        result = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
        if result.is_success and not result.is_empty:
            return result.dataframe['count'].iloc[0]
        return 0
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get information about columns in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        result = self.execute_query(f"DESCRIBE {table_name}")
        if result.is_success and not result.is_empty:
            return result.dataframe.to_dict(orient='records')
        return []
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """
        Get primary key columns for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of primary key column names
        """
        columns = self.get_table_columns(table_name)
        return [col['column_name'] for col in columns if col.get('is_primary_key', False)]
    
    def backup_database(self, backup_path: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save backup (if None, use default)
            
        Returns:
            Tuple of (success, message)
        """
        config = get_config()
        
        if not backup_path:
            backup_dir = Path(config.get("paths", "backup"))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"financial_data_backup_{timestamp}.duckdb"
        else:
            backup_path = Path(backup_path)
        
        try:
            # Make sure backup directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Create backup
            result = self.execute_query(f"EXPORT DATABASE '{backup_path}'")
            
            if result.is_success:
                return True, f"Database backup created at {backup_path}"
            else:
                return False, f"Error creating backup: {result.error}"
        except Exception as e:
            return False, f"Error creating backup: {e}"
    
    def restore_database(self, backup_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Restore database from a backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Tuple of (success, message)
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            return False, f"Backup file not found: {backup_path}"
        
        try:
            # Close current connection
            self.close()
            
            # Connect to backup database
            backup_conn = duckdb.connect(database=str(backup_path), read_only=True)
            
            # Connect to target database
            target_conn = duckdb.connect(database=str(self.db_path), read_only=False)
            
            # Export from backup and import to target
            backup_conn.execute(f"EXPORT DATABASE '{self.db_path}'")
            
            # Close connections
            backup_conn.close()
            target_conn.close()
            
            # Reconnect to the database
            self.connect()
            
            return True, f"Database restored from {backup_path}"
        except Exception as e:
            # Try to reconnect
            self.connect()
            return False, f"Error restoring database: {e}"

# Global instance
db_manager = None

def get_db_manager(db_path: Optional[Union[str, Path]] = None,
                 read_only: bool = True) -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Args:
        db_path: Path to database file
        read_only: Whether to open the database in read-only mode
        
    Returns:
        Global database manager instance
    """
    global db_manager
    
    if db_manager is None or db_path is not None:
        db_manager = DatabaseManager(db_path, read_only)
        
    return db_manager