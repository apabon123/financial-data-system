#!/usr/bin/env python
"""
Database Management Module

This module provides a standardized interface for database operations
in the Financial Data System, specifically for DuckDB management.

Features:
- Connection management
- Schema initialization
- Backup and restore operations
- Utility functions for common database operations
"""

import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Any, Tuple

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Exception raised for database-related errors."""
    pass


class Database:
    """DuckDB database manager for the Financial Data System."""
    
    def __init__(self, db_path: Union[str, Path], read_only: bool = False):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the DuckDB database file
            read_only: Whether to open the database in read-only mode
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.conn = None
        
        # Initialize schema paths
        self.schema_dir = Path(__file__).parent.parent.parent / 'sql' / 'schema'
        self.schema_files = {
            'init': self.schema_dir / 'init_schema.sql',
            'views': self.schema_dir / 'create_views.sql',
            'indices': self.schema_dir / 'create_indices.sql'
        }
        
        # Connect to the database
        self._connect()
    
    def _connect(self) -> None:
        """Connect to the DuckDB database."""
        try:
            # Create the directory if it doesn't exist
            if not self.read_only:
                os.makedirs(self.db_path.parent, exist_ok=True)
            
            self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
            logger.info(f"Connected to database: {self.db_path} (read_only={self.read_only})")
            
        except duckdb.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info(f"Closed database connection: {self.db_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic connection closing."""
        self.close()
    
    def execute(self, query: str, params: List = None) -> duckdb.DuckDBPyResult:
        """
        Execute a SQL query with optional parameters.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            DuckDB query result
            
        Raises:
            DatabaseError: If the query fails
        """
        if not self.conn:
            self._connect()
            
        try:
            if params:
                return self.conn.execute(query, params)
            else:
                return self.conn.execute(query)
        except duckdb.Error as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}")
            raise DatabaseError(f"Query execution failed: {e}")
    
    def query_to_df(self, query: str, params: List = None) -> pd.DataFrame:
        """
        Execute a query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            Query results as a pandas DataFrame
            
        Raises:
            DatabaseError: If the query fails
        """
        try:
            result = self.execute(query, params)
            return result.fetchdf()
        except duckdb.Error as e:
            logger.error(f"Error executing query to DataFrame: {e}\nQuery: {query}")
            raise DatabaseError(f"Query to DataFrame failed: {e}")
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        query = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        result = self.execute(query).fetchone()
        return result[0] > 0
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get the column names for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names
            
        Raises:
            DatabaseError: If the table doesn't exist
        """
        if not self.table_exists(table_name):
            raise DatabaseError(f"Table does not exist: {table_name}")
            
        query = f"PRAGMA table_info('{table_name}')"
        result = self.execute(query).fetchdf()
        return result['name'].tolist()
    
    def initialize_schema(self) -> None:
        """
        Initialize the database schema.
        
        Creates all required tables, views, and indices if they don't exist.
        
        Raises:
            DatabaseError: If schema initialization fails
        """
        if self.read_only:
            raise DatabaseError("Cannot initialize schema in read-only mode")
            
        try:
            # Run the main schema initialization script
            if self.schema_files['init'].exists():
                with open(self.schema_files['init'], 'r') as f:
                    schema_sql = f.read()
                self.execute(schema_sql)
                logger.info("Schema initialization completed successfully")
            else:
                logger.warning(f"Schema file not found: {self.schema_files['init']}")
                
            # Create views
            if self.schema_files['views'].exists():
                with open(self.schema_files['views'], 'r') as f:
                    views_sql = f.read()
                self.execute(views_sql)
                logger.info("Views creation completed successfully")
                
            # Create indices
            if self.schema_files['indices'].exists():
                with open(self.schema_files['indices'], 'r') as f:
                    indices_sql = f.read()
                self.execute(indices_sql)
                logger.info("Indices creation completed successfully")
                
        except duckdb.Error as e:
            logger.error(f"Schema initialization failed: {e}")
            raise DatabaseError(f"Schema initialization failed: {e}")
    
    def create_backup(self, backup_dir: Union[str, Path], retention_days: int = 30) -> Path:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store the backup
            retention_days: Number of days to keep backups (0 to keep all)
            
        Returns:
            Path to the created backup file
            
        Raises:
            DatabaseError: If backup creation fails
        """
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            os.makedirs(backup_path, exist_ok=True)
            
        # Close the connection to ensure all data is flushed
        if self.conn:
            self.conn.close()
            self.conn = None
            
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = self.db_path.stem
            backup_file = backup_path / f"{db_name}_{timestamp}.duckdb"
            
            # Copy the database file
            shutil.copy2(self.db_path, backup_file)
            logger.info(f"Created backup: {backup_file}")
            
            # Cleanup old backups if retention is specified
            if retention_days > 0:
                self._cleanup_old_backups(backup_path, retention_days)
                
            # Reconnect to the database
            self._connect()
            
            return backup_file
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            # Reconnect anyway
            self._connect()
            raise DatabaseError(f"Backup creation failed: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, days: int) -> None:
        """
        Remove backups older than the specified number of days.
        
        Args:
            backup_dir: Directory containing backups
            days: Number of days to keep backups
        """
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
        
        count = 0
        try:
            for backup_file in backup_dir.glob("*.duckdb"):
                # Check if the file is a backup (contains timestamp)
                if not backup_file.stem.split("_")[-1].isdigit():
                    continue
                    
                # Remove if older than retention period
                file_time = backup_file.stat().st_mtime
                if file_time < cutoff_time:
                    backup_file.unlink()
                    count += 1
                    
            logger.info(f"Cleaned up {count} old backup files")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def restore_from_backup(self, backup_path: Union[str, Path]) -> None:
        """
        Restore the database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Raises:
            DatabaseError: If restoration fails
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise DatabaseError(f"Backup file does not exist: {backup_file}")
            
        # Close the connection
        if self.conn:
            self.conn.close()
            self.conn = None
            
        try:
            # Create a backup of the current database first
            current_backup_dir = self.db_path.parent / "emergency_backups"
            os.makedirs(current_backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_backup = current_backup_dir / f"{self.db_path.stem}_before_restore_{timestamp}.duckdb"
            
            if self.db_path.exists():
                shutil.copy2(self.db_path, current_backup)
                logger.info(f"Created emergency backup before restore: {current_backup}")
            
            # Restore from the backup
            shutil.copy2(backup_file, self.db_path)
            logger.info(f"Restored database from backup: {backup_file}")
            
            # Reconnect to the database
            self._connect()
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            # Reconnect anyway
            self._connect()
            raise DatabaseError(f"Restore failed: {e}")
    
    def execute_file(self, sql_file: Union[str, Path]) -> None:
        """
        Execute SQL queries from a file.
        
        Args:
            sql_file: Path to the SQL file
            
        Raises:
            DatabaseError: If execution fails
        """
        file_path = Path(sql_file)
        if not file_path.exists():
            raise DatabaseError(f"SQL file does not exist: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                sql = f.read()
                
            statements = self._split_sql_statements(sql)
            for stmt in statements:
                if stmt.strip():
                    self.execute(stmt)
                    
            logger.info(f"Executed SQL file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error executing SQL file {file_path}: {e}")
            raise DatabaseError(f"SQL file execution failed: {e}")
    
    @staticmethod
    def _split_sql_statements(sql: str) -> List[str]:
        """
        Split SQL content into individual statements.
        
        Args:
            sql: SQL content with multiple statements
            
        Returns:
            List of individual SQL statements
        """
        # This is a simple implementation and may not handle all edge cases
        # For complex SQL parsing, consider using a dedicated SQL parser
        statements = []
        current = []
        
        for line in sql.splitlines():
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('--'):
                current.append(line)  # Keep comments for context
                continue
                
            current.append(line)
            
            if stripped.endswith(';'):
                statements.append('\n'.join(current))
                current = []
        
        # Add any remaining content as the last statement
        if current:
            statements.append('\n'.join(current))
            
        return statements
    
    def maintain_database(self) -> None:
        """
        Perform database maintenance operations (VACUUM, ANALYZE).
        
        Raises:
            DatabaseError: If maintenance fails
        """
        if self.read_only:
            raise DatabaseError("Cannot perform maintenance in read-only mode")
            
        try:
            # Execute VACUUM to reclaim space
            self.execute("VACUUM")
            logger.info("VACUUM completed successfully")
            
            # Execute ANALYZE to update statistics
            self.execute("ANALYZE")
            logger.info("ANALYZE completed successfully")
            
        except duckdb.Error as e:
            logger.error(f"Database maintenance failed: {e}")
            raise DatabaseError(f"Database maintenance failed: {e}")
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                         if_exists: str = 'append', 
                         on_conflict: Optional[str] = None) -> int:
        """
        Insert a pandas DataFrame into a table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: Action if table exists ('fail', 'replace', 'append')
            on_conflict: ON CONFLICT clause (e.g., 'DO NOTHING' or 'DO UPDATE SET ...')
            
        Returns:
            Number of rows inserted
            
        Raises:
            DatabaseError: If insertion fails
        """
        if self.read_only:
            raise DatabaseError("Cannot insert data in read-only mode")
            
        if df.empty:
            logger.warning(f"Attempted to insert empty DataFrame into {table_name}")
            return 0
            
        try:
            # Register DataFrame as a temporary view
            temp_view_name = f"temp_insert_{table_name}_{id(df)}"
            self.conn.register(temp_view_name, df)
            
            # Get column names from DataFrame
            columns = df.columns.tolist()
            columns_str = ", ".join([f'"{col}"' for col in columns])
            
            if if_exists == 'replace' and self.table_exists(table_name):
                self.execute(f'DELETE FROM {table_name}')
            
            # Construct the SQL query
            if on_conflict:
                sql = f"""
                INSERT INTO {table_name} ({columns_str})
                SELECT {columns_str} FROM {temp_view_name}
                ON CONFLICT {on_conflict}
                """
            else:
                sql = f"""
                INSERT INTO {table_name} ({columns_str})
                SELECT {columns_str} FROM {temp_view_name}
                """
                
            # Execute the query
            self.execute(sql)
            
            # Unregister the temporary view
            self.conn.unregister(temp_view_name)
            
            row_count = len(df)
            logger.info(f"Inserted {row_count} rows into {table_name}")
            return row_count
            
        except Exception as e:
            logger.error(f"Error inserting DataFrame into {table_name}: {e}")
            # Try to unregister the view if it exists
            try:
                self.conn.unregister(temp_view_name)
            except:
                pass
            raise DatabaseError(f"DataFrame insertion failed: {e}")
            
    def upsert_dataframe(self, df: pd.DataFrame, table_name: str, 
                         key_columns: List[str], update_columns: Optional[List[str]] = None) -> int:
        """
        Insert or update rows in a table based on key columns.
        
        Args:
            df: DataFrame to upsert
            table_name: Target table name
            key_columns: Columns that form the unique key
            update_columns: Columns to update (None for all except key columns)
            
        Returns:
            Number of rows upserted
            
        Raises:
            DatabaseError: If upsert fails
        """
        if update_columns is None:
            update_columns = [col for col in df.columns if col not in key_columns]
            
        # Construct ON CONFLICT clause
        key_str = ", ".join(key_columns)
        updates = [f'"{col}" = excluded."{col}"' for col in update_columns]
        update_str = ", ".join(updates)
        
        on_conflict = f"""
        ({key_str}) DO UPDATE SET {update_str}
        """
        
        return self.insert_dataframe(df, table_name, if_exists='append', on_conflict=on_conflict)
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get information about a table's structure.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with table information
            
        Raises:
            DatabaseError: If the table doesn't exist
        """
        if not self.table_exists(table_name):
            raise DatabaseError(f"Table does not exist: {table_name}")
            
        query = f"PRAGMA table_info('{table_name}')"
        return self.query_to_df(query)


def get_database(db_path: Union[str, Path], read_only: bool = False) -> Database:
    """
    Helper function to get a Database instance.
    
    Args:
        db_path: Path to the database file
        read_only: Whether to open in read-only mode
        
    Returns:
        Database instance
    """
    return Database(db_path, read_only)