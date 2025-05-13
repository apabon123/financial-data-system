"""
Schema Browser Module for DB Inspector.

This module provides interactive schema browsing and visualization capabilities
for exploring database structure and relationships.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set

# Import optional dependencies with fallbacks
try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("NetworkX is not installed. Schema visualization will be disabled.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib is not installed. Schema visualization will be disabled.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import SIMPLE
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    logging.error("Rich is not installed. CLI interface will be degraded.")
    # Define minimal fallbacks
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    class Table:
        def __init__(self, *args, **kwargs):
            self.rows = []
        def add_column(self, *args, **kwargs):
            pass
        def add_row(self, *args):
            self.rows.append(args)
    class Panel:
        def __init__(self, *args, **kwargs):
            self.content = args[0] if args else ""
    SIMPLE = None
    class Text:
        def __init__(self, text=""):
            self.text = text

from ..core.app import get_app
from ..core.config import get_config
from ..core.schema import get_schema_manager

# Setup logging
logger = logging.getLogger(__name__)

class SchemaBrowser:
    """Interactive database schema browser with visualization capabilities."""
    
    def __init__(self):
        """Initialize schema browser."""
        self.app = get_app()
        self.config = get_config()
        self.schema_manager = get_schema_manager()
        self.console = Console()
        
        # Current state
        self.current_table = None
        self.current_view = None
        self.filter_pattern = None
        
    def list_tables(self, pattern: Optional[str] = None) -> None:
        """
        List database tables, optionally filtered by pattern.
        
        Args:
            pattern: Optional filter pattern (case-insensitive substring)
        """
        tables = self.schema_manager.tables
        
        if pattern:
            tables = [t for t in tables if pattern.lower() in t.lower()]
            self.filter_pattern = pattern
        else:
            self.filter_pattern = None
        
        table = Table(title=f"Database Tables ({len(tables)})", box=SIMPLE)
        table.add_column("Table Name")
        table.add_column("Rows")
        table.add_column("Columns")
        table.add_column("Size")
        table.add_column("Related Tables")
        
        # Get stats for all tables
        stats = self.schema_manager.get_table_stats(refresh=True)
        
        for table_name in sorted(tables):
            table_stats = stats.get(table_name, {})
            row_count = table_stats.get('row_count', 0)
            column_count = table_stats.get('column_count', 0)
            size_estimation = table_stats.get('size_estimation', 'Unknown')
            
            # Get related tables (just count for the list view)
            related = self.schema_manager.get_related_tables(table_name)
            related_count = len(related)
            
            table.add_row(
                table_name, 
                f"{row_count:,}", 
                str(column_count), 
                size_estimation,
                f"{related_count} tables" if related_count > 0 else "None"
            )
        
        self.console.print(table)
        
    def list_views(self, pattern: Optional[str] = None) -> None:
        """
        List database views, optionally filtered by pattern.
        
        Args:
            pattern: Optional filter pattern (case-insensitive substring)
        """
        views = self.schema_manager.views
        
        if pattern:
            views = [v for v in views if pattern.lower() in v.lower()]
            self.filter_pattern = pattern
        else:
            self.filter_pattern = None
        
        table = Table(title=f"Database Views ({len(views)})", box=SIMPLE)
        table.add_column("View Name")
        table.add_column("Columns")
        
        for view_name in sorted(views):
            # Get view columns
            columns = self.schema_manager.view_columns.get(view_name, [])
            column_count = len(columns)
            
            table.add_row(view_name, str(column_count))
        
        self.console.print(table)
    
    def show_table_details(self, table_name: str) -> None:
        """
        Show detailed information about a specific table.
        
        Args:
            table_name: Table name
        """
        if table_name not in self.schema_manager.tables:
            self.console.print(f"[bold red]Table {table_name} not found.[/bold red]")
            return
        
        self.current_table = table_name
        stats = self.schema_manager.get_table_stats().get(table_name, {})
        columns = self.schema_manager.table_columns.get(table_name, [])
        relationships = self.schema_manager.get_table_relationships(table_name)
        
        # Print table info
        self.console.print(f"[bold cyan]Table:[/bold cyan] {table_name}")
        self.console.print(f"[bold cyan]Row Count:[/bold cyan] {stats.get('row_count', 0):,}")
        self.console.print(f"[bold cyan]Column Count:[/bold cyan] {stats.get('column_count', 0)}")
        self.console.print(f"[bold cyan]Size Estimation:[/bold cyan] {stats.get('size_estimation', 'Unknown')}")
        
        # Print primary keys
        pk_columns = stats.get('primary_keys', [])
        if pk_columns:
            self.console.print(f"[bold cyan]Primary Keys:[/bold cyan] {', '.join(pk_columns)}")
        else:
            self.console.print("[bold cyan]Primary Keys:[/bold cyan] None")
        
        # Print columns
        col_table = Table(title=f"Columns in {table_name}", box=SIMPLE)
        col_table.add_column("Column Name")
        col_table.add_column("Data Type")
        col_table.add_column("Nullable")
        col_table.add_column("Default")
        
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('column_type', '')
            nullable = "Yes" if col.get('null', 'YES').upper() == 'YES' else "No"
            default = col.get('default', '')
            
            # Highlight primary key columns
            if col_name in pk_columns:
                col_name = f"[bold yellow]{col_name} (PK)[/bold yellow]"
            
            col_table.add_row(col_name, col_type, nullable, str(default))
        
        self.console.print(col_table)
        
        # Print relationships
        if relationships['incoming'] or relationships['outgoing']:
            rel_table = Table(title=f"Relationships for {table_name}", box=SIMPLE)
            rel_table.add_column("Relationship Type")
            rel_table.add_column("Related Table")
            rel_table.add_column("Local Column")
            rel_table.add_column("Foreign Column")
            rel_table.add_column("Confidence")
            
            for rel in relationships['incoming']:
                rel_table.add_row(
                    "Referenced By",
                    rel['from_table'],
                    rel['to_column'],
                    rel['from_column'],
                    rel['confidence']
                )
            
            for rel in relationships['outgoing']:
                rel_table.add_row(
                    "References",
                    rel['to_table'],
                    rel['from_column'],
                    rel['to_column'],
                    rel['confidence']
                )
            
            self.console.print(rel_table)
        else:
            self.console.print("[bold cyan]Relationships:[/bold cyan] None detected")
        
        # Print sample query
        sample_query = self.schema_manager.generate_table_query(table_name)
        self.console.print(Panel(
            sample_query,
            title="Sample Query",
            expand=False
        ))
    
    def show_view_details(self, view_name: str) -> None:
        """
        Show detailed information about a specific view.
        
        Args:
            view_name: View name
        """
        if view_name not in self.schema_manager.views:
            self.console.print(f"[bold red]View {view_name} not found.[/bold red]")
            return
        
        self.current_view = view_name
        columns = self.schema_manager.view_columns.get(view_name, [])
        
        # Print view info
        self.console.print(f"[bold cyan]View:[/bold cyan] {view_name}")
        self.console.print(f"[bold cyan]Column Count:[/bold cyan] {len(columns)}")
        
        # Print columns
        col_table = Table(title=f"Columns in {view_name}", box=SIMPLE)
        col_table.add_column("Column Name")
        col_table.add_column("Data Type")
        col_table.add_column("Nullable")
        col_table.add_column("Default")
        
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('column_type', '')
            nullable = "Yes" if col.get('null', 'YES').upper() == 'YES' else "No"
            default = col.get('default', '')
            
            col_table.add_row(col_name, col_type, nullable, str(default))
        
        self.console.print(col_table)
        
        # Try to get view definition if available
        try:
            result = self.app.execute_query(f"SELECT sql FROM sqlite_master WHERE type='view' AND name='{view_name}'")
            if result.is_success and not result.is_empty:
                view_sql = result.dataframe['sql'].iloc[0]
                self.console.print(Panel(
                    view_sql,
                    title="View Definition",
                    expand=False
                ))
        except Exception:
            # DuckDB might not expose view definitions in the same way as SQLite
            pass
    
    def visualize_schema(self, focus_table: Optional[str] = None,
                       max_distance: int = 2) -> None:
        """
        Generate and display a visualization of database schema relationships.

        Args:
            focus_table: Table to focus visualization around (optional)
            max_distance: Maximum relationship distance from focus table
        """
        if nx is None or plt is None:
            print("\n❌ Error: Schema visualization requires NetworkX and Matplotlib.")
            print("Please install these packages with:")
            print("pip install networkx>=3.0 matplotlib>=3.7.0")
            return

        try:
            # Create a subgraph if focus table is specified
            if focus_table:
                if focus_table not in self.schema_manager.schema_graph:
                    print(f"Table {focus_table} not found in schema")
                    return

                # Get subgraph of tables within max_distance of focus_table
                tables_to_include = set([focus_table])

                try:
                    # Direct connections (outgoing)
                    for node, distance in nx.single_source_shortest_path_length(
                            self.schema_manager.schema_graph, focus_table, cutoff=max_distance).items():
                        if distance <= max_distance:
                            tables_to_include.add(node)

                    # Incoming connections
                    for node, distance in nx.single_source_shortest_path_length(
                            self.schema_manager.schema_graph.reverse(), focus_table, cutoff=max_distance).items():
                        if distance <= max_distance:
                            tables_to_include.add(node)
                except nx.NetworkXError as e:
                    logger.error(f"NetworkX error: {e}")
                    print(f"Error creating schema subgraph: {e}")
                    return

                graph = self.schema_manager.schema_graph.subgraph(tables_to_include)
                title = f"Schema Relationships for {focus_table} (max distance: {max_distance})"
            else:
                graph = self.schema_manager.schema_graph
                title = "Full Database Schema Relationships"

            # If the graph is empty, inform the user
            if len(graph.nodes) == 0:
                print("No tables or relationships found to visualize.")
                return

            # Create figure
            plt.figure(figsize=(12, 10))
            plt.title(title)

            try:
                # Set up node positions using spring layout
                pos = nx.spring_layout(graph)

                # Draw nodes
                table_nodes = [n for n in graph.nodes if graph.nodes[n].get('type') == 'table']
                view_nodes = [n for n in graph.nodes if graph.nodes[n].get('type') == 'view']

                nx.draw_networkx_nodes(graph, pos,
                                    nodelist=table_nodes,
                                    node_color='lightblue',
                                    node_size=500,
                                    alpha=0.8)

                nx.draw_networkx_nodes(graph, pos,
                                    nodelist=view_nodes,
                                    node_color='lightgreen',
                                    node_size=500,
                                    alpha=0.8)

                # Draw highlight node if focus table is specified
                if focus_table and focus_table in graph.nodes:
                    nx.draw_networkx_nodes(graph, pos,
                                        nodelist=[focus_table],
                                        node_color='red',
                                        node_size=700,
                                        alpha=0.8)

                # Draw edges with different styles based on relationship type
                edge_colors = []
                edge_widths = []

                for u, v, data in graph.edges(data=True):
                    if data.get('relationship_type') == 'potential_fk':
                        if data.get('confidence') == 'high':
                            edge_colors.append('blue')
                            edge_widths.append(2.0)
                        else:
                            edge_colors.append('gray')
                            edge_widths.append(1.0)
                    else:
                        edge_colors.append('black')
                        edge_widths.append(1.0)

                nx.draw_networkx_edges(graph, pos,
                                    edge_color=edge_colors,
                                    width=edge_widths,
                                    arrowsize=15,
                                    arrowstyle='->')

                # Draw labels
                nx.draw_networkx_labels(graph, pos, font_size=10)
            except Exception as e:
                logger.error(f"Error drawing graph: {e}")
                print(f"Error drawing graph: {e}")
                plt.close()
                return

            # Save figure to temp file and display path
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    print(f"\nSchema visualization saved to: {tmp.name}")
                    print("You can open this file to view the schema diagram.")
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")
                print(f"Error saving visualization: {e}")
                plt.close()

        except Exception as e:
            logger.error(f"Error visualizing schema: {e}")
            print(f"Error visualizing schema: {e}")
            print("This is likely due to missing or incompatible dependencies.")
            print("Please ensure NetworkX (>=3.0) and Matplotlib (>=3.7.0) are installed.")
    
    def list_related_tables(self, table_name: str, max_distance: int = 2) -> None:
        """
        List tables related to a specific table.
        
        Args:
            table_name: Table name
            max_distance: Maximum relationship distance
        """
        if table_name not in self.schema_manager.tables:
            self.console.print(f"[bold red]Table {table_name} not found.[/bold red]")
            return
        
        related_tables = self.schema_manager.get_related_tables(table_name, max_distance)
        
        if not related_tables:
            self.console.print(f"No tables related to {table_name} found within {max_distance} steps.")
            return
        
        # Get direct relationships
        direct_relationships = set()
        for rel in self.schema_manager.relationships:
            if rel['from_table'] == table_name:
                direct_relationships.add(rel['to_table'])
            if rel['to_table'] == table_name:
                direct_relationships.add(rel['from_table'])
        
        table = Table(title=f"Tables Related to {table_name} (max distance: {max_distance})", box=SIMPLE)
        table.add_column("Table Name")
        table.add_column("Relationship")
        table.add_column("Join Query Available")
        
        for related in sorted(related_tables):
            if related in direct_relationships:
                relationship = "Direct"
            else:
                relationship = "Indirect"
            
            # Check if join query can be generated
            join_query = self.schema_manager.generate_join_query(table_name, related)
            join_available = "Yes" if join_query else "No"
            
            table.add_row(related, relationship, join_available)
        
        self.console.print(table)
    
    def show_join_query(self, from_table: str, to_table: str) -> None:
        """
        Show a sample join query between two tables.
        
        Args:
            from_table: First table name
            to_table: Second table name
        """
        if from_table not in self.schema_manager.tables:
            self.console.print(f"[bold red]Table {from_table} not found.[/bold red]")
            return
        
        if to_table not in self.schema_manager.tables:
            self.console.print(f"[bold red]Table {to_table} not found.[/bold red]")
            return
        
        join_query = self.schema_manager.generate_join_query(from_table, to_table)
        
        if join_query:
            self.console.print(Panel(
                join_query,
                title=f"Join Query: {from_table} ↔ {to_table}",
                expand=False
            ))
        else:
            self.console.print(f"[bold yellow]Could not generate join query between {from_table} and {to_table}.[/bold yellow]")
            self.console.print("No relationship path found between these tables.")
    
    def interactive_browser(self) -> None:
        """Run interactive schema browser."""
        print("\nEntering interactive schema browser. Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                # Get command from user
                command = input("\nschema> ").strip()
                
                if command.lower() in ('exit', 'quit'):
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.lower() == 'tables':
                    self.list_tables()
                elif command.lower() == 'views':
                    self.list_views()
                elif command.lower().startswith('table '):
                    table_name = command[6:].strip()
                    self.show_table_details(table_name)
                elif command.lower().startswith('view '):
                    view_name = command[5:].strip()
                    self.show_view_details(view_name)
                elif command.lower() == 'visualize':
                    self.visualize_schema()
                elif command.lower().startswith('visualize '):
                    table_name = command[10:].strip()
                    self.visualize_schema(focus_table=table_name)
                elif command.lower().startswith('related '):
                    table_name = command[8:].strip()
                    self.list_related_tables(table_name)
                elif command.lower().startswith('join '):
                    parts = command[5:].strip().split()
                    if len(parts) == 2:
                        self.show_join_query(parts[0], parts[1])
                    else:
                        print("Usage: join <from_table> <to_table>")
                elif command.lower().startswith('filter '):
                    pattern = command[7:].strip()
                    self.list_tables(pattern)
                else:
                    print(f"Unknown command: {command}")
                    self._show_help()
            
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting schema browser.")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        Available commands:
        
        tables                  - List all tables
        views                   - List all views
        table <name>            - Show details for a specific table
        view <name>             - Show details for a specific view
        visualize               - Visualize full database schema
        visualize <table>       - Visualize schema around a specific table
        related <table>         - List tables related to a specific table
        join <table1> <table2>  - Show sample join query between two tables
        filter <pattern>        - Filter tables by name pattern
        help                    - Show this help
        exit                    - Exit schema browser
        """
        
        print(help_text)

# Global instance
schema_browser = None

def get_schema_browser() -> SchemaBrowser:
    """
    Get the global schema browser instance.
    
    Returns:
        Global schema browser instance
    """
    global schema_browser
    
    if schema_browser is None:
        schema_browser = SchemaBrowser()
        
    return schema_browser