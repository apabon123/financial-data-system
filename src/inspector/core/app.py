"""
Main application module for DB Inspector.

This module provides the main application class that integrates all components
of the DB Inspector tool.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from .config import get_config
from .database import get_db_manager
from .schema import get_schema_manager

# Setup logging
logger = logging.getLogger(__name__)

class DBInspectorApp:
    """Main application class for DB Inspector."""
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None, read_only: bool = True):
        """
        Initialize the DB Inspector application.
        
        Args:
            db_path: Path to database file
            read_only: Whether to open the database in read-only mode
        """
        self.config = get_config()
        self.db_manager = get_db_manager(db_path, read_only)
        self.schema_manager = get_schema_manager()
        
        # Application state
        self.current_query = ""
        self.current_table = ""
        self.current_view = ""
        self.current_symbol = ""
        self.query_history = []
        self.visualization_cache = {}
        
        # Module registry - will be populated by individual modules
        self.modules = {}
        
        # UI registry - will be populated by UI components
        self.ui_components = {}
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize the application."""
        logger.info("Initializing DB Inspector application")
        
        # Load saved state if available
        self._load_state()
        
        # Register built-in modules
        self._register_modules()
    
    def _load_state(self):
        """Load saved application state."""
        state_path = Path(self.config.get("paths", "history"))
        
        if state_path.exists():
            try:
                import json
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    
                self.query_history = state.get('query_history', [])
                self.current_table = state.get('current_table', '')
                self.current_view = state.get('current_view', '')
                self.current_symbol = state.get('current_symbol', '')
                
                logger.info(f"Loaded saved state from {state_path}")
            except Exception as e:
                logger.error(f"Error loading saved state: {e}")
    
    def _register_modules(self):
        """Register built-in application modules."""
        # This will be populated by individual modules
        pass
    
    def save_state(self):
        """Save application state."""
        state_path = Path(self.config.get("paths", "history"))
        
        try:
            import json
            
            state = {
                'query_history': self.query_history[-100:],  # Keep last 100 queries
                'current_table': self.current_table,
                'current_view': self.current_view,
                'current_symbol': self.current_symbol,
                'last_saved': time.time()
            }
            
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved application state to {state_path}")
        except Exception as e:
            logger.error(f"Error saving application state: {e}")
    
    def execute_query(self, query: str) -> Any:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query result
        """
        result = self.db_manager.execute_query(query)
        
        # Add to history if successful
        if result.is_success:
            self.query_history.append({
                'query': query,
                'timestamp': time.time(),
                'execution_time': result.execution_time,
                'row_count': result.row_count
            })
            
            # Keep history at a reasonable size
            if len(self.query_history) > 100:
                self.query_history = self.query_history[-100:]
        
        return result
    
    def execute_sql_file(self, file_path: Union[str, Path]) -> List[Any]:
        """
        Execute a SQL file.
        
        Args:
            file_path: Path to SQL file
            
        Returns:
            List of query results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"SQL file not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
                
            return self.db_manager.execute_script(sql_script)
        except Exception as e:
            logger.error(f"Error executing SQL file {file_path}: {e}")
            return []
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables.
        
        Returns:
            List of table names
        """
        return self.schema_manager.tables
    
    def get_views(self) -> List[str]:
        """
        Get list of all views.
        
        Returns:
            List of view names
        """
        return self.schema_manager.views
    
    def get_table_preview(self, table_name: str, limit: int = 10) -> Any:
        """
        Get a preview of table data.
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows
            
        Returns:
            Query result with table preview
        """
        self.current_table = table_name
        return self.db_manager.get_table_preview(table_name, limit)
    
    def get_table_schema(self, table_name: str) -> Any:
        """
        Get table schema.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Query result with table schema
        """
        return self.db_manager.get_table_schema(table_name)
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get table statistics.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary of table statistics
        """
        return self.schema_manager.get_table_stats().get(table_name, {})
    
    def get_related_tables(self, table_name: str) -> List[str]:
        """
        Get related tables.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of related table names
        """
        return self.schema_manager.get_related_tables(table_name)
    
    def get_sample_queries(self, table_name: str = None) -> List[Dict[str, str]]:
        """
        Get sample queries.
        
        Args:
            table_name: Optional table name to get table-specific queries
            
        Returns:
            List of sample query dictionaries
        """
        samples = []
        
        if table_name:
            # Table specific queries
            samples.append({
                'name': f"Select from {table_name}",
                'query': self.schema_manager.generate_table_query(table_name)
            })
            
            # Add join queries with related tables
            related_tables = self.schema_manager.get_related_tables(table_name, max_distance=1)
            for related in related_tables:
                join_query = self.schema_manager.generate_join_query(table_name, related)
                if join_query:
                    samples.append({
                        'name': f"Join {table_name} with {related}",
                        'query': join_query
                    })
        else:
            # General queries
            samples.append({
                'name': "List tables",
                'query': "SHOW TABLES;"
            })
            
            samples.append({
                'name': "List views",
                'query': "SHOW VIEWS;"
            })
            
            samples.append({
                'name': "List symbols in market_data",
                'query': "SELECT DISTINCT symbol FROM market_data ORDER BY symbol;"
            })
            
            samples.append({
                'name': "Show latest prices",
                'query': "SELECT * FROM latest_prices ORDER BY timestamp DESC LIMIT 20;"
            })
        
        return samples
    
    def backup_database(self, backup_path: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """
        Create a database backup.
        
        Args:
            backup_path: Optional custom path for backup
            
        Returns:
            Tuple of (success, message)
        """
        return self.db_manager.backup_database(backup_path)
    
    def restore_database(self, backup_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Tuple of (success, message)
        """
        return self.db_manager.restore_database(backup_path)
    
    def set_readonly_mode(self, readonly: bool) -> None:
        """
        Set database readonly mode.
        
        Args:
            readonly: Whether to open the database in read-only mode
        """
        # Need to reconnect with new mode
        db_path = self.db_manager.db_path
        self.db_manager.close()
        self.db_manager = get_db_manager(db_path, readonly)
    
    def shutdown(self) -> None:
        """Clean shutdown of the application."""
        # Save current state
        self.save_state()
        
        # Close database connection
        self.db_manager.close()
        
        logger.info("DB Inspector application shutdown complete")

# Global instance
app: Optional[DBInspectorApp] = None # Added type hint

def get_app(db_path: Optional[Union[str, Path]] = None, read_only: bool = True) -> DBInspectorApp:
    """
    Get the global application instance.
    If db_path is provided, or if the existing app's read_only status differs,
    a new instance is created.
    
    Args:
        db_path: Path to database file
        read_only: Whether to open the database in read-only mode
        
    Returns:
        Global application instance
    """
    global app
    
    should_reinitialize = False
    if app is None:
        should_reinitialize = True
    elif db_path is not None and Path(app.db_manager.db_path) != Path(db_path):
        logger.info(f"DB path changed from {app.db_manager.db_path} to {db_path}. Re-initializing app.")
        app.shutdown() # Shutdown old instance
        should_reinitialize = True
    elif app.db_manager.read_only != read_only:
        logger.info(f"Read-only mode changed from {app.db_manager.read_only} to {read_only}. Re-initializing app.")
        app.shutdown() # Shutdown old instance
        should_reinitialize = True
    
    if should_reinitialize:
        logger.info(f"Creating new DBInspectorApp instance. DB: {db_path or 'default'}, Read-only: {read_only}")
        app = DBInspectorApp(db_path, read_only)
    
    return app

def close_and_clear_global_app_instance() -> None:
    """
    Shuts down the current global app instance (if it exists) and sets it to None.
    This is useful for ensuring resources like database connections are released,
    especially before an external script might need to access the same resources.
    """
    global app
    if app is not None:
        logger.info("Shutting down and clearing global DBInspectorApp instance.")
        try:
            app.shutdown()
        except Exception as e:
            logger.error(f"Error during explicit shutdown of global app instance: {e}", exc_info=True)
        finally:
            app = None
    else:
        logger.info("No global DBInspectorApp instance to clear.")