"""
Interactive SQL Execution Module.

This module provides functionality for interactive SQL query execution,
syntax highlighting, and query history management.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Import optional dependencies with fallbacks
try:
    from pygments import highlight
    from pygments.lexers import SqlLexer
    from pygments.formatters import TerminalFormatter
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False
    logging.warning("Pygments is not installed. Syntax highlighting will be disabled.")

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.filters import Condition, has_completions
    from prompt_toolkit.key_binding import KeyBindings
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    logging.error("Prompt-toolkit is not installed. Interactive SQL mode will be limited.")

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    logging.error("Rich is not installed. CLI interface will be degraded.")

from ..core.app import get_app
from ..core.config import get_config
from ..core.database import QueryResult

# Setup logging
logger = logging.getLogger(__name__)

class SQLExecutor:
    """Interactive SQL query executor with syntax highlighting."""

    def __init__(self):
        """Initialize SQL executor."""
        self.app = get_app()
        self.config = get_config()

        # Check for required dependencies
        self._check_dependencies()

        # Set up history file
        history_path = Path(self.config.get("paths", "history")).parent / "sql_history.txt"
        os.makedirs(history_path.parent, exist_ok=True)

        # Initialize components based on available dependencies
        if HAS_PROMPT_TOOLKIT:
            self.history = FileHistory(str(history_path))

            # Initialize prompt session
            style = Style.from_dict({
                'completion-menu.completion': 'bg:#008888 #ffffff',
                'completion-menu.completion.current': 'bg:#00aaaa #000000',
                'scrollbar.background': 'bg:#88aaaa',
                'scrollbar.button': 'bg:#222222',
            })

            # Create SQL keyword completer
            self.keywords = self._get_sql_keywords()
            self.table_names = self.app.get_tables() + self.app.get_views()
            self.completer = self._create_completer()

            # Create key bindings
            kb = KeyBindings()

            @kb.add('f2')
            def _(event):
                """Run query, display results, and clear buffer."""
                buffer = event.app.current_buffer
                query_text = buffer.text
                
                if query_text.strip():
                    # print(f"SQL> {query_text.rstrip()}") # Optional: echo the query
                    start_time = time.time()
                    result = self.execute_query(query_text)
                    elapsed = time.time() - start_time
                    
                    if result.is_success:
                        if not result.is_empty:
                            self._display_result_table(result, elapsed)
                        else:
                            print(f"Query executed successfully in {elapsed:.2f}s, but returned no results.")
                    else:
                        print(f"Query error: {result.error}")
                    buffer.text = "" # Clear buffer after execution
                event.app.invalidate()

            @kb.add('f3')
            def _(event):
                """Format query."""
                buffer = event.app.current_buffer
                buffer.text = self.format_query(buffer.text)

            @kb.add('f4')
            def _(event):
                """Show tables."""
                self.show_tables()

            @kb.add('escape')
            def _(event):
                """Dismiss completion menu."""
                buffer = event.app.current_buffer
                if buffer.complete_state:
                    buffer.complete_state = None
                event.app.invalidate()

            # Conditional 'Enter' for accepting completions
            @kb.add('enter', filter=has_completions, eager=True)
            def _(event):
                """Accept current completion if completion menu is visible."""
                buffer = event.app.current_buffer
                if buffer.complete_state and buffer.complete_state.current_completion:
                    buffer.apply_completion(buffer.complete_state.current_completion)
                # Explicitly clear completion state after applying
                # to prevent any other Enter handlers from acting on it
                # and to ensure the UI updates correctly.
                if buffer.complete_state is not None: # Check before accessing
                    buffer.complete_state = None 
                event.app.invalidate() # Request a redraw

            # Set up lexer if pygments is available
            lexer = None
            if HAS_PYGMENTS:
                lexer = PygmentsLexer(SqlLexer)

            # Create prompt session
            self.session = PromptSession(
                history=self.history,
                auto_suggest=AutoSuggestFromHistory(),
                lexer=lexer,
                completer=self.completer,
                style=style,
                key_bindings=kb,
                complete_while_typing=False,
                enable_history_search=True
            )
        else:
            self.session = None
            self.history = None
            self.keywords = self._get_sql_keywords()
            self.table_names = self.app.get_tables() + self.app.get_views()
            self.completer = None

        # Query storage
        self.saved_queries = self._load_saved_queries()
        self.last_result = None

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        missing_deps = []

        if not HAS_RICH:
            missing_deps.append("rich")

        # These are optional but affect functionality
        optional_missing = []
        if not HAS_PYGMENTS:
            optional_missing.append("pygments")

        if not HAS_PROMPT_TOOLKIT:
            optional_missing.append("prompt_toolkit")

        # Log warnings about missing dependencies
        if missing_deps:
            logger.error(f"Required dependencies missing: {', '.join(missing_deps)}")
            print(f"\n❌ Error: Required dependencies missing: {', '.join(missing_deps)}")
            print("SQL execution requires these packages.")
            print(f"Install with: pip install {' '.join(missing_deps)}")

        if optional_missing:
            logger.warning(f"Optional dependencies missing: {', '.join(optional_missing)}")
            print(f"\n⚠️ Warning: Optional dependencies missing: {', '.join(optional_missing)}")
            print("Interactive SQL mode will have limited functionality.")
            print(f"Install with: pip install {' '.join(optional_missing)}")

        # Only continue with full functionality if critical dependencies are available
        self.can_highlight = HAS_PYGMENTS
        self.can_interactive = HAS_PROMPT_TOOLKIT
    
    def _get_sql_keywords(self) -> List[str]:
        """
        Get list of SQL keywords for autocompletion.
        
        Returns:
            List of SQL keywords
        """
        return [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 
            'OUTER JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'UNION',
            'INSERT INTO', 'UPDATE', 'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE', 
            'DROP TABLE', 'CREATE VIEW', 'DROP VIEW', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN',
            'LIKE', 'IS NULL', 'IS NOT NULL', 'AS', 'ON', 'DESC', 'ASC', 'DISTINCT',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'WITH', 'VALUES', 'UNION ALL', 'INTERSECT', 'EXCEPT', 'EXPLAIN'
        ]
    
    def _create_completer(self) -> WordCompleter:
        """
        Create word completer for SQL keywords and table names.
        
        Returns:
            WordCompleter instance
        """
        words = self.keywords + self.table_names
        column_names = self._get_column_names()
        words.extend(column_names)
        
        return WordCompleter(words, ignore_case=True)
    
    def _get_column_names(self) -> List[str]:
        """
        Get list of column names from tables for autocompletion.
        
        Returns:
            List of column names
        """
        columns = []
        
        for table in self.table_names:
            try:
                table_columns = self.app.schema_manager.table_columns.get(table, [])
                if not table_columns:
                    continue
                    
                for col in table_columns:
                    if 'column_name' in col:
                        columns.append(col['column_name'])
            except Exception:
                pass
        
        return list(set(columns))  # Remove duplicates
    
    def _load_saved_queries(self) -> Dict[str, str]:
        """
        Load saved queries from file.
        
        Returns:
            Dictionary of saved queries
        """
        saved_queries_path = Path(self.config.get("paths", "history")).parent / "saved_queries.json"
        
        if saved_queries_path.exists():
            try:
                with open(saved_queries_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading saved queries: {e}")
        
        return {}
    
    def _save_queries(self) -> None:
        """Save queries to file."""
        saved_queries_path = Path(self.config.get("paths", "history")).parent / "saved_queries.json"
        
        try:
            os.makedirs(saved_queries_path.parent, exist_ok=True)
            with open(saved_queries_path, 'w') as f:
                json.dump(self.saved_queries, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queries: {e}")
    
    def highlight_query(self, query: str) -> str:
        """
        Apply syntax highlighting to SQL query.

        Args:
            query: SQL query string

        Returns:
            Highlighted query string
        """
        if not getattr(self, 'can_highlight', False):
            return query

        if self.config.get("ui", "syntax_highlighting", True):
            try:
                return highlight(query, SqlLexer(), TerminalFormatter())
            except Exception as e:
                logger.error(f"Error highlighting query: {e}")
                return query
        return query
    
    def format_query(self, query: str) -> str:
        """
        Format SQL query for readability.
        
        Args:
            query: SQL query string
            
        Returns:
            Formatted query string
        """
        # This is a basic implementation - could be improved with a proper SQL formatter
        import re
        
        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)
        
        # Format keywords
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 
            'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'UNION', 'INSERT INTO',
            'UPDATE', 'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE', 'DROP TABLE'
        ]
        
        # Format main clauses
        for keyword in keywords:
            pattern = re.compile(rf'\b{keyword}\b', re.IGNORECASE)
            query = pattern.sub(f"\n{keyword}", query)
        
        # Format JOIN ON
        query = re.sub(r'\bON\b', '\n  ON', query, flags=re.IGNORECASE)
        
        # Format AND/OR in WHERE clauses
        query = re.sub(r'\bAND\b', '\n  AND', query, flags=re.IGNORECASE)
        query = re.sub(r'\bOR\b', '\n  OR', query, flags=re.IGNORECASE)
        
        # Format commas in SELECT
        parts = query.split('\n')
        for i, part in enumerate(parts):
            if part.strip().upper().startswith('SELECT'):
                parts[i] = re.sub(r',', ',\n  ', part)
        
        query = '\n'.join(parts)
        
        return query
    
    def execute_query(self, query: str) -> QueryResult:
        """
        Execute SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            QueryResult object
        """
        # Save current query
        self.app.current_query = query
        
        # Execute query
        result = self.app.execute_query(query)
        self.last_result = result
        
        return result
    
    def execute_sql_file(self, file_path: Union[str, Path]) -> List[QueryResult]:
        """
        Execute a SQL file.
        
        Args:
            file_path: Path to SQL file
            
        Returns:
            List of QueryResult objects
        """
        return self.app.execute_sql_file(file_path)
    
    def save_query(self, name: str, query: str) -> None:
        """
        Save a named query.
        
        Args:
            name: Query name
            query: SQL query string
        """
        self.saved_queries[name] = query
        self._save_queries()
    
    def delete_saved_query(self, name: str) -> bool:
        """
        Delete a saved query.
        
        Args:
            name: Query name
            
        Returns:
            True if query was deleted, False otherwise
        """
        if name in self.saved_queries:
            del self.saved_queries[name]
            self._save_queries()
            return True
        return False
    
    def get_saved_query(self, name: str) -> Optional[str]:
        """
        Get a saved query.
        
        Args:
            name: Query name
            
        Returns:
            Query string or None if not found
        """
        return self.saved_queries.get(name)
    
    def get_saved_queries(self) -> Dict[str, str]:
        """
        Get all saved queries.
        
        Returns:
            Dictionary of saved queries
        """
        return self.saved_queries
    
    def get_history(self, limit: int = 20) -> List[str]:
        """
        Get query history.
        
        Args:
            limit: Maximum number of history items
            
        Returns:
            List of query strings
        """
        # Get from app history
        return [item['query'] for item in self.app.query_history[-limit:]]
    
    def show_tables(self) -> QueryResult:
        """
        Execute SHOW TABLES query.
        
        Returns:
            QueryResult object
        """
        return self.execute_query("SHOW TABLES;")
    
    def show_views(self) -> QueryResult:
        """
        Execute SHOW VIEWS query.
        
        Returns:
            QueryResult object
        """
        return self.execute_query("SHOW VIEWS;")
    
    def describe_table(self, table_name: str) -> QueryResult:
        """
        Execute DESCRIBE query for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            QueryResult object
        """
        return self.execute_query(f"DESCRIBE {table_name};")
    
    def get_sample_queries(self, table_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get sample queries for a table.
        
        Args:
            table_name: Name of the table (optional)
            
        Returns:
            List of sample query dictionaries
        """
        return self.app.get_sample_queries(table_name)
    
    def interactive_mode(self) -> None:
        """Run interactive SQL query mode."""
        print("\nEntering interactive SQL mode. Press Ctrl+D to exit.")

        # Check if we can run in interactive mode
        if not getattr(self, 'can_interactive', False):
            print("\n❌ Error: Interactive SQL mode requires prompt_toolkit.")
            print("Install with: pip install prompt_toolkit>=3.0.33 pygments>=2.15.0")
            print("Using fallback mode: Type your query and press Enter. Type 'exit' to quit.")
            self._fallback_interactive_mode()
            return

        print("F2: Execute query | F3: Format query | F4: Show tables\n")

        try:
            while True:
                # Update completer with latest tables and columns
                self.table_names = self.app.get_tables() + self.app.get_views()
                self.completer = self._create_completer()
                self.session.completer = self.completer

                # Get query from user
                try:
                    # Revised loop for Alt+Enter submission
                    query_from_multiline_submit = self.session.prompt('SQL> ', multiline=True)

                    if query_from_multiline_submit: # prompt_toolkit returns None on Ctrl-D (EOF)
                        if query_from_multiline_submit.strip(): # Check if it's not just empty due to F2 clearing
                            # This block handles Alt+Enter submission
                            start_time = time.time()
                            result = self.execute_query(query_from_multiline_submit)
                            elapsed = time.time() - start_time

                            if result.is_success:
                                if not result.is_empty:
                                    self._display_result_table(result, elapsed)
                                else:
                                    print(f"Query executed successfully in {elapsed:.2f}s, but returned no results.")
                            else:
                                print(f"Query error: {result.error}")
                        # If F2 was pressed, its binding handled everything, and buffer was cleared.
                        # query_from_multiline_submit might be empty if F2 clears before prompt returns,
                        # or it might be the text F2 processed if clearing happens after.
                        # The F2 binding now clears its own buffer, so this path should be safe.
                    else: # None typically means EOF (Ctrl-D)
                        print("\nExiting SQL mode.")
                        break

                except EOFError: # Should be caught if prompt returns None or raises it directly
                    print("\nExiting SQL mode.")
                    break # Exit the while True loop
                except Exception as e:
                    logger.error(f"Error in prompt session: {e}")
                    print(f"Error in prompt session: {e}")
                    self._fallback_interactive_mode()
                    return

                # Execute query
                # This block is now largely handled by the F2 binding or the explicit Alt+Enter submission block above.
                # We can remove or comment out this section if F2 and Alt+Enter cover all execution paths.
                # start_time = time.time()
                # result = self.execute_query(query) # 'query' is undefined here now
                # elapsed = time.time() - start_time

                # # Display result
                # if result.is_success:
                #     if not result.is_empty:
                #         # Print result as table
                #         self._display_result_table(result, elapsed)
                #     else:
                #         print(f"Query executed successfully in {elapsed:.2f}s, but returned no results.")
                # else:
                #     print(f"Query error: {result.error}")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Unexpected error in interactive mode: {e}")
            print(f"Unexpected error in interactive mode: {e}")
            self._fallback_interactive_mode()

    def _fallback_interactive_mode(self) -> None:
        """Simple fallback interactive mode without fancy features."""
        print("\nFallback SQL Mode - Type 'exit' to quit")

        try:
            while True:
                # Get query from user
                query = input('SQL> ')

                if query.lower() in ('exit', 'quit'):
                    break

                # Execute query
                start_time = time.time()
                result = self.execute_query(query)
                elapsed = time.time() - start_time

                # Display result
                if result.is_success:
                    if not result.is_empty:
                        # Print result in simple format
                        self._display_result_simple(result, elapsed)
                    else:
                        print(f"Query executed successfully in {elapsed:.2f}s, but returned no results.")
                else:
                    print(f"Query error: {result.error}")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error in fallback mode: {e}")
            print(f"Error: {e}")

        print("\nExiting SQL mode.")

    def _display_result_table(self, result, elapsed):
        """Display query result using rich if available, otherwise use fallback."""
        if HAS_RICH:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title=f"Query Result ({result.row_count} rows, {elapsed:.2f}s)")

                # Add columns
                for col in result.dataframe.columns:
                    table.add_column(str(col))

                # Add rows (limit to max_rows by default)
                max_rows = self.config.get("ui", "max_rows_display", 100)
                for _, row in result.dataframe.head(max_rows).iterrows():
                    table.add_row(*[str(v) for v in row])

                console.print(table)

                if result.row_count > max_rows:
                    print(f"Note: Only showing first {max_rows} of {result.row_count} rows")
            except Exception as e:
                logger.error(f"Error displaying results with rich: {e}")
                self._display_result_simple(result, elapsed)
        else:
            self._display_result_simple(result, elapsed)

    def _display_result_simple(self, result, elapsed):
        """Simple fallback for displaying results without rich."""
        print(f"\nQuery Result ({result.row_count} rows, {elapsed:.2f}s)\n")

        # Print header
        headers = [str(col) for col in result.dataframe.columns]
        header_line = " | ".join(headers)
        print(header_line)
        print("-" * len(header_line))

        # Print rows (limited)
        max_rows = min(self.config.get("ui", "max_rows_display", 100), 20)  # Lower limit for simple display
        for i, (_, row) in enumerate(result.dataframe.head(max_rows).iterrows()):
            row_str = " | ".join(str(v) for v in row)
            print(row_str)

        if result.row_count > max_rows:
            print(f"\nNote: Only showing first {max_rows} of {result.row_count} rows")
    
    def run_query_and_print(self, query: str) -> None:
        """
        Run a query and print the results.

        Args:
            query: SQL query string
        """
        print(f"Executing query: {query}")
        result = self.execute_query(query)

        if result.is_success:
            if not result.is_empty:
                # Print result as table
                self._display_result_table(result, result.execution_time)
            else:
                print(f"Query executed successfully in {result.execution_time:.2f}s, but returned no results.")
        else:
            print(f"Query error: {result.error}")
    
    def run_file_and_print(self, file_path: Union[str, Path]) -> None:
        """
        Run a SQL file and print the results.

        Args:
            file_path: Path to SQL file
        """
        try:
            file_path = Path(file_path)
            print(f"Executing SQL file: {file_path}")

            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return

            try:
                results = self.execute_sql_file(file_path)
            except Exception as e:
                logger.error(f"Error executing SQL file: {e}")
                print(f"Error executing SQL file: {e}")
                return

            for i, result in enumerate(results):
                print(f"\n--- Query {i+1} ---")

                if result.is_success:
                    if not result.is_empty:
                        # Print result as table
                        self._display_result_table(result, result.execution_time)
                    else:
                        print(f"Query executed successfully in {result.execution_time:.2f}s, but returned no results.")
                else:
                    print(f"Query error: {result.error}")
        except Exception as e:
            logger.error(f"Unexpected error processing SQL file: {e}")
            print(f"Unexpected error: {e}")

# Global instance
sql_executor = None

def get_sql_executor() -> SQLExecutor:
    """
    Get the global SQL executor instance.
    
    Returns:
        Global SQL executor instance
    """
    global sql_executor
    
    if sql_executor is None:
        sql_executor = SQLExecutor()
        
    return sql_executor