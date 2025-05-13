"""
Schema management module for DB Inspector.

This module handles database schema analysis and relationship management.
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Set, Optional, Tuple

from .database import get_db_manager

# Setup logging
logger = logging.getLogger(__name__)

class SchemaManager:
    """Manager for database schema analysis and relationship management."""
    
    def __init__(self):
        """Initialize schema manager."""
        self.db_manager = get_db_manager()
        self.tables = []
        self.views = []
        self.table_columns = {}
        self.view_columns = {}
        self.table_stats = {}
        self.relationships = []
        self.schema_graph = nx.DiGraph()
        
        # Initialize
        self.refresh_schema()
    
    def refresh_schema(self) -> None:
        """Refresh schema information from the database."""
        self._load_tables_and_views()
        self._load_columns()
        self._load_relationships()
        self._build_schema_graph()
    
    def _load_tables_and_views(self) -> None:
        """Load tables and views from the database."""
        self.tables = self.db_manager.get_tables()
        self.views = self.db_manager.get_views()
        
        logger.info(f"Loaded {len(self.tables)} tables and {len(self.views)} views")
    
    def _load_columns(self) -> None:
        """Load column information for tables and views."""
        self.table_columns = {}
        self.view_columns = {}
        
        # Load table columns
        for table in self.tables:
            columns = self.db_manager.get_table_columns(table)
            self.table_columns[table] = columns
        
        # Load view columns
        for view in self.views:
            result = self.db_manager.execute_query(f"DESCRIBE {view}")
            if result.is_success and not result.is_empty:
                self.view_columns[view] = result.dataframe.to_dict(orient='records')
    
    def _load_relationships(self) -> None:
        """
        Analyze and load table relationships.
        
        This method attempts to identify relationships between tables based on:
        1. Primary key / foreign key patterns
        2. Column name patterns (e.g., table_id in another table)
        3. Data analysis (matching values between columns)
        """
        self.relationships = []
        
        # Check for primary key / foreign key patterns
        for table in self.tables:
            pk_columns = self._get_primary_key_columns(table)
            
            for pk_column in pk_columns:
                # Look for this column name in other tables
                for other_table in self.tables:
                    if other_table == table:
                        continue
                        
                    other_columns = [col['column_name'] for col in self.table_columns.get(other_table, [])]
                    
                    # Check for exact column name match
                    if pk_column in other_columns:
                        self.relationships.append({
                            'from_table': table,
                            'from_column': pk_column,
                            'to_table': other_table,
                            'to_column': pk_column,
                            'relationship_type': 'potential_fk',
                            'confidence': 'high'
                        })
                    
                    # Check for pattern like 'table_id'
                    pattern = f"{table}_id"
                    if pattern in other_columns:
                        self.relationships.append({
                            'from_table': table,
                            'from_column': pk_column,
                            'to_table': other_table,
                            'to_column': pattern,
                            'relationship_type': 'potential_fk',
                            'confidence': 'medium'
                        })
    
    def _get_primary_key_columns(self, table: str) -> List[str]:
        """
        Get primary key columns for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of primary key column names
        """
        columns = self.table_columns.get(table, [])
        pk_columns = []
        
        for col in columns:
            # Look for primary key field in column info
            if col.get('primary_key') or col.get('is_primary_key'):
                pk_columns.append(col['column_name'])
        
        # If no explicit PK found, look for common patterns
        if not pk_columns:
            column_names = [col['column_name'] for col in columns]
            for name in ['id', f"{table}_id", 'primary_key']:
                if name in column_names:
                    pk_columns.append(name)
                    break
        
        return pk_columns
    
    def _build_schema_graph(self) -> None:
        """Build a graph representation of the database schema."""
        self.schema_graph = nx.DiGraph()
        
        # Add tables as nodes
        for table in self.tables:
            self.schema_graph.add_node(table, type='table', columns=self.table_columns.get(table, []))
        
        # Add views as nodes
        for view in self.views:
            self.schema_graph.add_node(view, type='view', columns=self.view_columns.get(view, []))
        
        # Add relationships as edges
        for rel in self.relationships:
            self.schema_graph.add_edge(
                rel['from_table'],
                rel['to_table'],
                from_column=rel['from_column'],
                to_column=rel['to_column'],
                relationship_type=rel['relationship_type'],
                confidence=rel['confidence']
            )
    
    def get_table_stats(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tables.
        
        Args:
            refresh: Force refresh of statistics
            
        Returns:
            Dictionary mapping table names to statistics
        """
        if self.table_stats and not refresh:
            return self.table_stats
        
        self.table_stats = {}
        
        for table in self.tables:
            self.table_stats[table] = self._get_single_table_stats(table)
        
        return self.table_stats
    
    def _get_single_table_stats(self, table: str) -> Dict[str, Any]:
        """
        Get statistics for a single table.
        
        Args:
            table: Table name
            
        Returns:
            Dictionary of table statistics
        """
        stats = {
            'name': table,
            'row_count': self.db_manager.get_table_count(table),
            'column_count': len(self.table_columns.get(table, [])),
            'primary_keys': self._get_primary_key_columns(table),
            'foreign_keys': [],
            'referenced_by': [],
            'has_indexes': False,
            'size_estimation': 'Unknown'
        }
        
        # Add foreign keys and referenced_by information
        for rel in self.relationships:
            if rel['to_table'] == table:
                stats['foreign_keys'].append({
                    'column': rel['to_column'],
                    'references': f"{rel['from_table']}({rel['from_column']})",
                    'confidence': rel['confidence']
                })
            
            if rel['from_table'] == table:
                stats['referenced_by'].append({
                    'table': rel['to_table'],
                    'column': rel['to_column'],
                    'confidence': rel['confidence']
                })
        
        # Check for indexes (approximate by looking for index-related info in column descriptions)
        for col in self.table_columns.get(table, []):
            if col.get('is_indexed') or col.get('index') or col.get('has_index'):
                stats['has_indexes'] = True
                break
        
        # Estimate size if row count is available
        if stats['row_count'] > 0:
            avg_row_size = self._estimate_row_size(table)
            estimated_size_bytes = avg_row_size * stats['row_count']
            
            if estimated_size_bytes < 1024:
                stats['size_estimation'] = f"{estimated_size_bytes} bytes"
            elif estimated_size_bytes < 1024 * 1024:
                stats['size_estimation'] = f"{estimated_size_bytes / 1024:.2f} KB"
            elif estimated_size_bytes < 1024 * 1024 * 1024:
                stats['size_estimation'] = f"{estimated_size_bytes / (1024 * 1024):.2f} MB"
            else:
                stats['size_estimation'] = f"{estimated_size_bytes / (1024 * 1024 * 1024):.2f} GB"
        
        return stats
    
    def _estimate_row_size(self, table: str) -> int:
        """
        Estimate average row size for a table.
        
        Args:
            table: Table name
            
        Returns:
            Estimated average row size in bytes
        """
        columns = self.table_columns.get(table, [])
        total_size = 0
        
        for col in columns:
            data_type = col.get('column_type', '').lower()
            
            # Estimate based on data type
            if 'int' in data_type:
                if 'big' in data_type:
                    total_size += 8
                else:
                    total_size += 4
            elif 'double' in data_type or 'float' in data_type:
                total_size += 8
            elif 'timestamp' in data_type or 'date' in data_type:
                total_size += 8
            elif 'boolean' in data_type:
                total_size += 1
            elif 'varchar' in data_type or 'string' in data_type:
                # Extract size from varchar(N) if available
                size_parts = data_type.split('(')
                if len(size_parts) > 1 and ')' in size_parts[1]:
                    size_str = size_parts[1].split(')')[0]
                    try:
                        size = int(size_str)
                        total_size += min(size, 50)  # Estimate average usage
                    except ValueError:
                        total_size += 50  # Default estimate
                else:
                    total_size += 50  # Default estimate for strings
            else:
                total_size += 8  # Default for unknown types
        
        # Add overhead
        total_size += 16  # Row header overhead
        
        return total_size
    
    def get_table_relationships(self, table: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get relationships for a specific table.
        
        Args:
            table: Table name
            
        Returns:
            Dictionary with incoming and outgoing relationships
        """
        incoming = []
        outgoing = []
        
        for rel in self.relationships:
            if rel['from_table'] == table:
                outgoing.append({
                    'to_table': rel['to_table'],
                    'from_column': rel['from_column'],
                    'to_column': rel['to_column'],
                    'relationship_type': rel['relationship_type'],
                    'confidence': rel['confidence']
                })
            
            if rel['to_table'] == table:
                incoming.append({
                    'from_table': rel['from_table'],
                    'from_column': rel['from_column'],
                    'to_column': rel['to_column'],
                    'relationship_type': rel['relationship_type'],
                    'confidence': rel['confidence']
                })
        
        return {
            'incoming': incoming,
            'outgoing': outgoing
        }
    
    def get_related_tables(self, table: str, max_distance: int = 2) -> List[str]:
        """
        Get tables related to a specific table up to a maximum distance.
        
        Args:
            table: Table name
            max_distance: Maximum distance in relationship graph
            
        Returns:
            List of related table names
        """
        if table not in self.schema_graph:
            return []
        
        related = set()
        
        # Get tables within specified distance
        for node, distance in nx.single_source_shortest_path_length(self.schema_graph, table, cutoff=max_distance).items():
            if node != table and distance <= max_distance:
                related.add(node)
        
        # Also get tables that reference this table
        for node, distance in nx.single_source_shortest_path_length(self.schema_graph.reverse(), table, cutoff=max_distance).items():
            if node != table and distance <= max_distance:
                related.add(node)
        
        return list(related)
    
    def generate_table_query(self, table: str) -> str:
        """
        Generate a sample query for a table with all columns.
        
        Args:
            table: Table name
            
        Returns:
            SQL query string
        """
        columns = self.table_columns.get(table, [])
        
        if not columns:
            return f"SELECT * FROM {table} LIMIT 100;"
        
        column_names = [col['column_name'] for col in columns]
        column_list = ",\n    ".join(column_names)
        
        return f"SELECT\n    {column_list}\nFROM {table}\nLIMIT 100;"
    
    def generate_join_query(self, from_table: str, to_table: str) -> Optional[str]:
        """
        Generate a sample join query between two tables.
        
        Args:
            from_table: First table name
            to_table: Second table name
            
        Returns:
            SQL query string or None if no relationship found
        """
        # Find direct relationship
        for rel in self.relationships:
            if (rel['from_table'] == from_table and rel['to_table'] == to_table):
                return self._build_join_query(
                    from_table, to_table,
                    rel['from_column'], rel['to_column']
                )
            
            if (rel['from_table'] == to_table and rel['to_table'] == from_table):
                return self._build_join_query(
                    from_table, to_table,
                    rel['to_column'], rel['from_column']
                )
        
        # No direct relationship found - try to find a path
        try:
            path = nx.shortest_path(self.schema_graph, from_table, to_table)
            
            if len(path) == 2:  # Direct path
                # Find the relationship
                for rel in self.relationships:
                    if rel['from_table'] == path[0] and rel['to_table'] == path[1]:
                        return self._build_join_query(
                            path[0], path[1],
                            rel['from_column'], rel['to_column']
                        )
            
            elif len(path) > 2:  # Indirect path
                # Build multi-table join
                from_alias = "a"
                query_parts = [f"SELECT\n    a.*,\n    {chr(ord(from_alias) + len(path) - 1)}.*"]
                query_parts.append(f"FROM {path[0]} AS {from_alias}")
                
                for i in range(1, len(path)):
                    to_alias = chr(ord(from_alias) + i)
                    
                    # Find relationship between path[i-1] and path[i]
                    join_found = False
                    for rel in self.relationships:
                        if rel['from_table'] == path[i-1] and rel['to_table'] == path[i]:
                            query_parts.append(
                                f"JOIN {path[i]} AS {to_alias} ON {from_alias}.{rel['from_column']} = {to_alias}.{rel['to_column']}"
                            )
                            join_found = True
                            break
                        
                        if rel['from_table'] == path[i] and rel['to_table'] == path[i-1]:
                            query_parts.append(
                                f"JOIN {path[i]} AS {to_alias} ON {from_alias}.{rel['to_column']} = {to_alias}.{rel['from_column']}"
                            )
                            join_found = True
                            break
                    
                    if not join_found:
                        # No explicit relationship, use common column names
                        prev_columns = [col['column_name'] for col in self.table_columns.get(path[i-1], [])]
                        curr_columns = [col['column_name'] for col in self.table_columns.get(path[i], [])]
                        
                        # Find common columns
                        common_columns = set(prev_columns) & set(curr_columns)
                        
                        if common_columns:
                            common_col = next(iter(common_columns))
                            query_parts.append(
                                f"JOIN {path[i]} AS {to_alias} ON {from_alias}.{common_col} = {to_alias}.{common_col}"
                            )
                        else:
                            # Last resort - join on IDs
                            prev_id = f"{path[i-1]}_id"
                            curr_id = f"{path[i]}_id"
                            
                            if prev_id in curr_columns:
                                query_parts.append(
                                    f"JOIN {path[i]} AS {to_alias} ON {from_alias}.id = {to_alias}.{prev_id}"
                                )
                            elif curr_id in prev_columns:
                                query_parts.append(
                                    f"JOIN {path[i]} AS {to_alias} ON {from_alias}.{curr_id} = {to_alias}.id"
                                )
                            else:
                                # Can't determine join condition
                                return None
                
                query_parts.append("LIMIT 100;")
                return "\n".join(query_parts)
            
        except nx.NetworkXNoPath:
            return None
        
        return None
    
    def _build_join_query(self, from_table: str, to_table: str, 
                       from_column: str, to_column: str) -> str:
        """
        Build a join query between two tables.
        
        Args:
            from_table: First table name
            to_table: Second table name
            from_column: Join column in first table
            to_column: Join column in second table
            
        Returns:
            SQL query string
        """
        from_columns = [col['column_name'] for col in self.table_columns.get(from_table, [])]
        to_columns = [col['column_name'] for col in self.table_columns.get(to_table, [])]
        
        # Select limited number of columns to avoid overwhelming results
        max_columns = 5
        from_selected = from_columns[:max_columns]
        to_selected = to_columns[:max_columns]
        
        from_list = ", ".join([f"a.{col}" for col in from_selected])
        to_list = ", ".join([f"b.{col}" for col in to_selected])
        
        query = f"""SELECT
    {from_list},
    {to_list}
FROM {from_table} AS a
JOIN {to_table} AS b ON a.{from_column} = b.{to_column}
LIMIT 100;"""
        
        return query

# Global instance
schema_manager = None

def get_schema_manager() -> SchemaManager:
    """
    Get the global schema manager instance.
    
    Returns:
        Global schema manager instance
    """
    global schema_manager
    
    if schema_manager is None:
        schema_manager = SchemaManager()
        
    return schema_manager