"""
Main entry point for DB Inspector application.

This module serves as the main entry point when the package is run as a module.
"""

import os
import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Configure logging first
import logging
import os
from pathlib import Path

# Safely create logs directory and configure logging
try:
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "logs"
    
    # Try both absolute and relative paths to handle WSL/Windows issues
    log_paths = [
        log_dir / "inspector.log",
        Path("logs/inspector.log"),
        Path("./logs/inspector.log"),
        Path("inspector.log")
    ]
    
    # Try to create log directory
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory {log_dir}: {e}")
        # Try creating a local logs directory
        try:
            os.makedirs("logs", exist_ok=True)
        except Exception:
            # Last resort - use current directory
            log_paths = [Path("inspector.log")]
    
    # Find first writable log path
    log_file = None
    for path in log_paths:
        try:
            # Check if we can write to this path
            if path.parent.exists() or os.access(os.path.dirname(path) if os.path.dirname(path) else ".", os.W_OK):
                log_file = path
                break
        except Exception:
            continue
    
    # If no writable path found, use null handler
    if not log_file:
        print("Warning: Could not find writable log location, logging to console only")
        handlers = [logging.StreamHandler()]
    else:
        print(f"Logging to: {log_file}")
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
except Exception as e:
    # Last resort - console logging only
    print(f"Warning: Error setting up logging: {e}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

logger = logging.getLogger(__name__)

def run_fallback_mode():
    """Run DB Inspector in fallback mode using only standard library."""
    print("\n" + "="*60)
    print("DB INSPECTOR - FALLBACK MODE")
    print("="*60)
    print("\nRunning with limited functionality (standard library only)")
    print("This mode provides basic database information without advanced features.")

    # Import only standard library modules
    import os
    import sqlite3
    import json
    import time
    import re
    from datetime import datetime

    try:
        # Try to locate the database
        db_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "financial_data.duckdb"),
            "data/financial_data.duckdb",
            "./financial_data.duckdb"
        ]

        db_file = None
        for path in db_paths:
            if os.path.exists(path):
                db_file = path
                break

        if not db_file:
            print(f"\n❌ Error: Could not find database file.")
            print("Expected at: data/financial_data.duckdb")
            return

        print(f"\nDatabase: {db_file}")
        
        # Main menu loop
        while True:
            print("\nDB Inspector Fallback Mode - Main Menu")
            print("=" * 40)
            print("1. Database Information")
            print("2. List Tables")
            print("3. View Table Schema")
            print("4. Sample Data")
            print("5. Simple Query")
            print("6. View System Tables")
            print("7. View Database Stats")
            print("8. Exit")

            try:
                choice = input("\nEnter choice (1-8): ").strip()

                if choice == '8' or choice.lower() == 'exit':
                    print("Exiting fallback mode.")
                    break

                # Basic functionality with sqlite3 (limited DuckDB compatibility)
                try:
                    # Try to use SQLite to read DuckDB file (might work for basic metadata)
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()

                    if choice == '1':
                        # Show database information
                        print("\nDatabase Information:")
                        print("-" * 40)
                        
                        db_size_bytes = os.path.getsize(db_file)
                        size_kb = db_size_bytes / 1024
                        size_mb = size_kb / 1024
                        
                        if size_mb >= 1:
                            size_str = f"{size_mb:.2f} MB"
                        else:
                            size_str = f"{size_kb:.2f} KB"
                            
                        mod_time = os.path.getmtime(db_file)
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                        
                        print(f"Path: {db_file}")
                        print(f"Size: {size_str} ({db_size_bytes:,} bytes)")
                        print(f"Last Modified: {mod_date}")
                        
                        # Count tables and views
                        try:
                            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                            table_count = cursor.fetchone()[0]
                            
                            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='view'")
                            view_count = cursor.fetchone()[0]
                            
                            print(f"Tables: {table_count}")
                            print(f"Views: {view_count}")
                        except sqlite3.OperationalError:
                            print("Could not query metadata tables")

                    elif choice == '2':
                        # List tables - might work for basic DuckDB files
                        try:
                            cursor.execute("SELECT name, type FROM sqlite_master WHERE type='table' OR type='view' ORDER BY type, name")
                            objects = cursor.fetchall()
                            
                            if not objects:
                                print("\nNo tables or views found in the database.")
                            else:
                                print("\nDatabase objects:")
                                print("-" * 40)
                                
                                # Group by type
                                tables = [obj[0] for obj in objects if obj[1] == 'table']
                                views = [obj[0] for obj in objects if obj[1] == 'view']
                                
                                if tables:
                                    print("\nTables:")
                                    for i, table in enumerate(sorted(tables), 1):
                                        print(f"  {i}. {table}")
                                
                                if views:
                                    print("\nViews:")
                                    for i, view in enumerate(sorted(views), 1):
                                        print(f"  {i}. {view}")
                        except sqlite3.OperationalError as e:
                            # Fallback to simple table list
                            try:
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                                tables = cursor.fetchall()
                                print("\nTables in database:")
                                for i, (table,) in enumerate(tables, 1):
                                    print(f"  {i}. {table}")
                            except sqlite3.OperationalError:
                                print(f"\n❌ Error: Could not list tables: {e}")
                                print("DuckDB files may not be fully compatible with SQLite.")

                    elif choice == '3':
                        # Show table schema
                        table_name = input("Enter table name: ")
                        try:
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()
                            
                            if not columns:
                                print(f"\nNo schema information found for table '{table_name}'")
                            else:
                                print(f"\nSchema for {table_name}:")
                                print("-" * 40)
                                print(f"{'Column':<20} {'Type':<15} {'Nullable':<10} {'Default':<15}")
                                print("-" * 60)
                                
                                for col in columns:
                                    col_id, name, type_, notnull, default_val, pk = col
                                    nullable = "NOT NULL" if notnull else "NULL"
                                    default = str(default_val) if default_val is not None else ""
                                    print(f"{name:<20} {type_:<15} {nullable:<10} {default:<15}")
                                    
                                # Try to get primary key info
                                try:
                                    pk_cols = [col[1] for col in columns if col[5]]
                                    if pk_cols:
                                        print(f"\nPrimary Key: {', '.join(pk_cols)}")
                                except Exception:
                                    pass
                        except sqlite3.OperationalError as e:
                            print(f"\n❌ Error: Could not get schema for table '{table_name}': {e}")

                    elif choice == '4':
                        # Show sample data
                        table_name = input("Enter table name: ")
                        try:
                            # Get column names first
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = [col[1] for col in cursor.fetchall()]
                            
                            if not columns:
                                print(f"\n❌ Error: Could not get columns for table '{table_name}'")
                                continue
                                
                            # Get sample data
                            limit_str = input("Enter number of rows to display (default: 5): ").strip()
                            limit = int(limit_str) if limit_str and limit_str.isdigit() else 5
                            
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                            rows = cursor.fetchall()

                            print(f"\nSample data from {table_name} ({len(rows)} rows):")
                            print("-" * 60)
                            
                            # Format output as a table
                            # Determine column widths (min 10, max 30)
                            col_widths = [min(30, max(10, len(col))) for col in columns]
                            
                            # Print header
                            header = " | ".join(f"{col:{width}}" for col, width in zip(columns, col_widths))
                            print(header)
                            print("-" * len(header))
                            
                            # Print rows
                            for row in rows:
                                formatted_row = []
                                for i, val in enumerate(row):
                                    # Truncate long values
                                    str_val = str(val)
                                    if len(str_val) > col_widths[i]:
                                        str_val = str_val[:col_widths[i]-3] + "..."
                                    formatted_row.append(f"{str_val:{col_widths[i]}}")
                                print(" | ".join(formatted_row))
                                
                        except sqlite3.OperationalError as e:
                            print(f"\n❌ Error: Could not query table '{table_name}': {e}")

                    elif choice == '5':
                        # Simple query execution
                        print("\nSimple Query Mode")
                        print("Enter your SQL query below (or 'exit' to return to menu)")
                        print("Note: Only simple SELECT queries are supported in fallback mode")
                        
                        while True:
                            query = input("\nSQL> ").strip()
                            
                            if query.lower() == 'exit':
                                break
                                
                            if not query:
                                continue
                                
                            # Basic security check - only allow SELECT statements
                            if not query.lower().startswith('select'):
                                print("❌ Error: Only SELECT queries are allowed in fallback mode")
                                continue
                                
                            try:
                                start_time = time.time()
                                cursor.execute(query)
                                
                                # Get column names from cursor description
                                if cursor.description:
                                    columns = [desc[0] for desc in cursor.description]
                                    rows = cursor.fetchall()
                                    
                                    query_time = time.time() - start_time
                                    print(f"\nResults ({len(rows)} rows, {query_time:.3f} seconds):")
                                    print("-" * 60)
                                    
                                    if rows:
                                        # Print header
                                        print(" | ".join(columns))
                                        print("-" * len(" | ".join(columns)))
                                        
                                        # Print rows (max 20)
                                        for row in rows[:20]:
                                            print(" | ".join(str(val) for val in row))
                                            
                                        if len(rows) > 20:
                                            print(f"... {len(rows) - 20} more rows (showing 20/{len(rows)})")
                                    else:
                                        print("No results returned")
                                else:
                                    print("Query executed successfully, but returned no results")
                            except sqlite3.OperationalError as e:
                                print(f"❌ Error executing query: {e}")

                    elif choice == '6':
                        # View system tables
                        print("\nSystem Tables:")
                        print("-" * 40)
                        
                        try:
                            # Find system tables (usually prefixed with sqlite_ or duckdb_)
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE 'sqlite_%' OR name LIKE 'duckdb_%')")
                            system_tables = cursor.fetchall()
                            
                            if system_tables:
                                for i, (table,) in enumerate(system_tables, 1):
                                    print(f"  {i}. {table}")
                                    
                                # Allow viewing a system table
                                choice = input("\nEnter table number to view (or press Enter to skip): ").strip()
                                if choice and choice.isdigit() and 1 <= int(choice) <= len(system_tables):
                                    table_name = system_tables[int(choice)-1][0]
                                    
                                    try:
                                        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
                                        rows = cursor.fetchall()
                                        
                                        # Get column names
                                        columns = [desc[0] for desc in cursor.description]
                                        
                                        print(f"\nContents of {table_name} (max 10 rows):")
                                        print("-" * 60)
                                        
                                        # Print header
                                        print(" | ".join(columns))
                                        print("-" * len(" | ".join(columns)))
                                        
                                        # Print rows
                                        for row in rows:
                                            print(" | ".join(str(val) for val in row))
                                    except sqlite3.OperationalError as e:
                                        print(f"❌ Error viewing system table: {e}")
                            else:
                                print("No system tables found")
                        except sqlite3.OperationalError as e:
                            print(f"❌ Error: Could not query system tables: {e}")

                    elif choice == '7':
                        # Database statistics
                        print("\nDatabase Statistics:")
                        print("-" * 40)
                        
                        try:
                            # Get list of all tables
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                            tables = cursor.fetchall()
                            
                            if not tables:
                                print("No tables found")
                                continue
                                
                            print(f"Total tables: {len(tables)}")
                            
                            # Get row counts and size estimates
                            table_stats = []
                            
                            for table_name, in tables:
                                try:
                                    # Skip system tables
                                    if table_name.startswith('sqlite_') or table_name.startswith('duckdb_'):
                                        continue
                                        
                                    # Get row count
                                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                                    row_count = cursor.fetchone()[0]
                                    
                                    # Get column count
                                    cursor.execute(f"PRAGMA table_info({table_name})")
                                    col_count = len(cursor.fetchall())
                                    
                                    table_stats.append((table_name, row_count, col_count))
                                except sqlite3.OperationalError:
                                    # Skip tables that can't be queried
                                    continue
                            
                            # Sort by row count (descending)
                            table_stats.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display table statistics
                            print("\nTop tables by row count:")
                            print(f"{'Table Name':<30} {'Rows':<10} {'Columns':<10}")
                            print("-" * 50)
                            
                            for name, rows, cols in table_stats[:10]:  # Show top 10
                                print(f"{name:<30} {rows:<10,} {cols:<10}")
                                
                            # Calculate totals
                            total_rows = sum(stats[1] for stats in table_stats)
                            
                            print(f"\nTotal rows across all tables: {total_rows:,}")
                            
                        except sqlite3.OperationalError as e:
                            print(f"❌ Error calculating database statistics: {e}")

                    conn.close()

                except sqlite3.OperationalError as e:
                    print(f"\n❌ Error accessing database: {e}")
                    print("DuckDB files may not be fully compatible with SQLite.")
                    print("Please install the required dependencies for full functionality.")
            except KeyboardInterrupt:
                print("\nOperation cancelled. Exiting fallback mode.")
                break
            except Exception as e:
                print(f"\n❌ Error in fallback mode: {e}")

    except Exception as e:
        print(f"\n❌ Critical error in fallback mode: {e}")

    print("\nTo use full functionality, install dependencies with:")
    print("pip install -r requirements.txt")

def main_with_error_handling():
    """Run main function with robust error handling."""
    try:
        # Check for fallback mode
        if "--fallback" in sys.argv:
            run_fallback_mode()
            return

        # Check dependencies first
        try:
            from inspector.utils.dependency_checker import check_dependencies, print_dependency_report
            all_deps_available, missing_deps = check_dependencies()

            if not all_deps_available:
                print_dependency_report()
                print("\n⚠️ DB Inspector cannot run with missing dependencies.")
                print("Please install the required dependencies and try again.")
                print("\nAlternatives:")
                print("1. Install dependencies: pip install -r requirements.txt")
                print("2. Run in fallback mode: DB_inspect_enhanced.bat --fallback")
                print("3. Use the dependency checker: check_dependencies.bat")
                sys.exit(1)
        except ImportError:
            # If we can't import the dependency checker, we can still try to run
            logger.warning("Could not import dependency checker, proceeding anyway")

        # Import CLI main function only after dependency check
        try:
            from inspector.cli import main
            main()
        except ImportError as e:
            logger.error(f"Failed to import CLI module: {e}")
            print(f"\n❌ Error: Could not load CLI module: {e}")
            print("This is likely due to missing dependencies.")
            print("\nYou can use fallback mode with: --fallback")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        try:
            # Ensure traceback is available
            import traceback
            tb_str = traceback.format_exc()
            
            logger.error(f"Unhandled exception: {e}")
            logger.error(tb_str)
            
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            print("\nAn unexpected error occurred. Check the logs for details.")
            print("Log file: Check the logs directory")
            
            if "--debug" in sys.argv:
                print("\nDetailed error information (debug mode):")
                print(tb_str)
        except Exception as inner_e:
            # Last resort error handling
            print(f"\n❌ Critical error: {e}")
            print(f"Error during error handling: {inner_e}")
        
        sys.exit(1)

if __name__ == "__main__":
    main_with_error_handling()