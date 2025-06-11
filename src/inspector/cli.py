"""
Command Line Interface for DB Inspector.

This module provides a CLI menu system for interacting with the DB Inspector tool.
"""

import os
import sys
import logging
import argparse
import traceback
import subprocess
import time # Added for delay
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import SIMPLE
from rich.prompt import Prompt, Confirm

from .core.app import get_app, close_and_clear_global_app_instance
from .core.config import get_config
from .modules.sql_executor import get_sql_executor
from .modules.schema_browser import get_schema_browser
from .modules.data_quality import get_quality_analyzer
from .utils.terminal import run_terminal_cmd

# Setup logging
logger = logging.getLogger(__name__)

class DBInspectorCLI:
    """Command line interface for DB Inspector."""
    
    def __init__(self):
        """Initialize CLI."""
        self.console = Console()
        self.app = get_app()
        self.config = get_config()
        self.sql_executor = get_sql_executor()
        self.schema_browser = get_schema_browser()
        self.quality_analyzer = get_quality_analyzer()
        
        # Parse command line arguments
        self.args = self._parse_args()
        
        # Apply command line arguments
        if self.args.database:
            self.app = get_app(self.args.database, not self.args.write)
        
        # Set up logging
        self._setup_logging()
    
    def _parse_args(self) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(description="DB Inspector - Financial Market Data Inspection Tool")
        
        parser.add_argument('--database', '-d', type=str, help='Path to database file')
        parser.add_argument('--write', '-w', action='store_true', help='Open database in write mode')
        parser.add_argument('--query', '-q', type=str, help='Execute SQL query and exit')
        parser.add_argument('--file', '-f', type=str, help='Execute SQL file and exit')
        parser.add_argument('--schema', '-s', action='store_true', help='Show database schema and exit')
        parser.add_argument('--analyze', '-a', type=str, help='Analyze data quality for symbol and exit')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        
        return parser.parse_args()
    
    def _setup_logging(self) -> None:
        """Set up logging."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.get("paths", "log"))
            ]
        )
        
        logger.info("DB Inspector CLI started")
    
    def run(self) -> None:
        """Run DB Inspector CLI."""
        # Check for single command execution
        if self.args.query:
            self.sql_executor.run_query_and_print(self.args.query)
            return
        
        if self.args.file:
            self.sql_executor.run_file_and_print(self.args.file)
            return
        
        if self.args.schema:
            self.show_database_info()
            self.schema_browser.list_tables()
            return
        
        if self.args.analyze:
            results = self.quality_analyzer.analyze_market_data_table(symbol=self.args.analyze)
            self.quality_analyzer.display_analysis_results(results)
            return
        
        # Run interactive menu
        self.show_welcome()
        self.main_menu()
    
    def show_welcome(self) -> None:
        """Show welcome banner."""
        banner = r"""
    ____  ____      ____                               __
   / __ \/ __ )    /  _/___  _________  ___  _________/ /_____  _____
  / / / / __  |    / // __ \/ ___/ __ \/ _ \/ ___/ __  / ___/ / / / _ \
 / /_/ / /_/ /   _/ // / / (__  ) /_/ /  __/ /__/ /_/ / /  / /_/ /  __/
/_____/_____/   /___/_/ /_/____/ .___/\___/\___/\__,_/_/   \__, /\___/
                              /_/                          /____/
        """

        try:
            self.console.print(Panel(banner, title="Financial Market Data Inspection Tool", subtitle="v1.0.0"))
        except Exception as e:
            # Fallback in case of rich error
            print(banner)
            print("Financial Market Data Inspection Tool v1.0.0")
            logger.warning(f"Error printing welcome banner: {e}")
        
        # Show database info
        db_path = self.app.db_manager.db_path
        read_only = self.app.db_manager.read_only
        
        self.console.print(f"[bold cyan]Database:[/bold cyan] {db_path}")
        self.console.print(f"[bold cyan]Mode:[/bold cyan] {'Read-Only' if read_only else 'Read-Write'}")
        
        # Show quick stats
        table_count = len(self.app.get_tables())
        view_count = len(self.app.get_views())
        
        self.console.print(f"[bold cyan]Tables:[/bold cyan] {table_count}")
        self.console.print(f"[bold cyan]Views:[/bold cyan] {view_count}")
    
    def show_database_info(self) -> None:
        """Show database information."""
        db_path = self.app.db_manager.db_path
        read_only = self.app.db_manager.read_only
        
        self.console.print(f"[bold cyan]Database:[/bold cyan] {db_path}")
        self.console.print(f"[bold cyan]Mode:[/bold cyan] {'Read-Only' if read_only else 'Read-Write'}")
        
        # Get database stats
        tables = self.app.get_tables()
        views = self.app.get_views()
        
        self.console.print(f"[bold cyan]Tables:[/bold cyan] {len(tables)}")
        self.console.print(f"[bold cyan]Views:[/bold cyan] {len(views)}")
        
        # Get table stats
        table_stats = self.app.schema_manager.get_table_stats()
        
        total_rows = sum(stat.get('row_count', 0) for stat in table_stats.values())
        
        self.console.print(f"[bold cyan]Total Rows:[/bold cyan] {total_rows:,}")
        
        # Show largest tables
        largest_tables = sorted(
            [(name, stat) for name, stat in table_stats.items()],
            key=lambda x: x[1].get('row_count', 0),
            reverse=True
        )[:5]
        
        if largest_tables:
            table = Table(title="Largest Tables", box=SIMPLE)
            table.add_column("Table Name")
            table.add_column("Rows")
            table.add_column("Columns")
            table.add_column("Size")
            
            for name, stat in largest_tables:
                table.add_row(
                    name,
                    f"{stat.get('row_count', 0):,}",
                    str(stat.get('column_count', 0)),
                    stat.get('size_estimation', 'Unknown')
                )
            
            self.console.print(table)
    
    def main_menu(self) -> None:
        """Display main menu and handle user input."""
        while True:
            self.console.print("\n[bold cyan]Main Menu[/bold cyan]")
            
            options = [
                "Execute SQL Query",
                "Browse Schema",
                "Analyze Data Quality",
                "Market Structure Tools",
                "Data Management",
                "Database Information",
                "Update/Backup/Restore",
                "Exit"
            ]
            
            for i, option in enumerate(options, 1):
                self.console.print(f"{i}. {option}")
            
            choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])
            
            if choice == '1':
                self.sql_executor.interactive_mode()
            elif choice == '2':
                self.schema_browser.interactive_browser()
            elif choice == '3':
                self.quality_analyzer.interactive_analyzer()
            elif choice == '4':
                self.market_structure_menu()
            elif choice == '5':
                self.data_management_menu()
            elif choice == '6':
                self.database_info_menu()
            elif choice == '7':
                self.backup_restore_menu()
            elif choice == '8':
                self.exit_application()
                break
    
    def market_structure_menu(self) -> None:
        """Display market structure tools menu."""
        self.console.print("\n[bold cyan]Market Structure Tools[/bold cyan]")
        
        options = [
            "Analyze Futures Contracts",
            "Explore Continuous Contracts",
            "Visualize Roll Calendar",
            "Correlation Matrix",
            "Volatility/Volume Profile",
            "Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"{i}. {option}")
        
        choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])
        
        if choice == '1':
            self.console.print("Futures contract analysis not yet implemented")
        elif choice == '2':
            self.console.print("Continuous contract exploration not yet implemented")
        elif choice == '3':
            self.console.print("Roll calendar visualization not yet implemented")
        elif choice == '4':
            self.console.print("Correlation matrix not yet implemented")
        elif choice == '5':
            self.console.print("Volatility/Volume profile not yet implemented")
        elif choice == '6':
            return
    
    def data_management_menu(self) -> None:
        """Display data management menu."""
        self.console.print("\n[bold cyan]Data Management[/bold cyan]")
        
        options = [
            "Correct Data",
            "Import/Export Data",
            "Clean/Filter Data",
            "Delete Old Contracts",
            "Manage Symbol Metadata",
            "Generate Continuous Futures Series",
            "Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"{i}. {option}")
        
        choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])
        
        if choice == '1':
            self.console.print("Data correction not yet implemented")
        elif choice == '2':
            self.console.print("Import/export not yet implemented")
        elif choice == '3':
            self.console.print("Data cleaning not yet implemented")
        elif choice == '4':
            self.console.print("Contract deletion not yet implemented")
        elif choice == '5':
            self.console.print("Symbol metadata manager not yet implemented")
        elif choice == '6':
            self._generate_continuous_futures_series()
        elif choice == '7':
            return
    
    def _generate_continuous_futures_series(self) -> None:
        """Prompt for parameters and run the generate_back_adjusted_futures.py script."""
        self.console.print("\n[bold cyan]Generate Continuous Futures Series[/bold cyan]")
        self.console.print("This will run the `src/scripts/scripts/generate_back_adjusted_futures.py` script.")

        try:
            # Get Python executable
            python_executable = sys.executable
            if not python_executable:
                self.console.print("[bold red]Error: Could not determine Python executable path.[/bold red]")
                return

            # Get script path (assuming execution from project root)
            # Resolve script path relative to the project root.
            # Assuming this cli.py file is at src/inspector/cli.py
            project_root = Path(__file__).resolve().parents[2]
            script_path = project_root / "src" / "scripts" / "scripts" / "generate_back_adjusted_futures.py"
            
            if not script_path.exists():
                self.console.print(f"[bold red]Error: Script not found at {script_path}[/bold red]")
                self.console.print("Ensure you are running from the project's root directory and the script exists.")
                return

            # --- Prompt for parameters ---
            root_symbol = Prompt.ask("Enter Root Symbol (e.g., ES, NQ)")
            if not root_symbol.strip():
                self.console.print("[bold red]Root Symbol cannot be empty.[/bold red]")
                return

            roll_type = Prompt.ask("Enter Roll Type (e.g., 01X, volume)")
            if not roll_type.strip():
                self.console.print("[bold red]Roll Type cannot be empty.[/bold red]")
                return

            contract_position_str = Prompt.ask("Enter Contract Position (e.g., 1, 2)", default="1")
            try:
                contract_position = int(contract_position_str)
                if contract_position <= 0:
                    raise ValueError("Contract position must be positive.")
            except ValueError as e:
                self.console.print(f"[bold red]Invalid Contract Position: {e}. Must be a positive integer.[/bold red]")
                return

            interval_value_str = Prompt.ask("Enter Interval Value (e.g., 1, 15)")
            try:
                interval_value = int(interval_value_str)
                if interval_value <= 0:
                    raise ValueError("Interval value must be positive.")
            except ValueError as e:
                self.console.print(f"[bold red]Invalid Interval Value: {e}. Must be a positive integer.[/bold red]")
                return
            
            interval_unit = Prompt.ask("Enter Interval Unit", choices=["minute", "daily", "hour"], default="daily")
            
            adjustment_type = Prompt.ask("Enter Adjustment Type (C for constant, N for none)", choices=["C", "N"], default="C")

            # Hardcoded values from batch script
            output_symbol_suffix = "_d"
            source_identifier_base = "inhouse_backadj"

            # Optional flags
            cmd_flags = []
            if Confirm.ask("Force delete existing data for this symbol?", default=False):
                cmd_flags.append("--force-delete")
            
            if Confirm.ask("Recreate table on PK mismatch (DEV ONLY)?", default=False):
                self.console.print("[bold red]WARNING: --recreate-table-on-pk-mismatch is a developer option and can lead to data loss. Use with extreme caution.[/bold red]")
                if Confirm.ask("Are you sure you want to enable --recreate-table-on-pk-mismatch?", default=False):
                    cmd_flags.append("--recreate-table-on-pk-mismatch")
                else:
                    self.console.print("[italic]--recreate-table-on-pk-mismatch was NOT enabled.[/italic]")


            # Construct command
            command_to_run = [
                str(python_executable), str(script_path),
                "--root-symbol", root_symbol,
                "--roll-type", roll_type,
                "--contract-position", str(contract_position),
                "--interval-value", str(interval_value),
                "--interval-unit", interval_unit,
                "--adjustment-type", adjustment_type,
                "--output-symbol-suffix", output_symbol_suffix,
                "--source-identifier", source_identifier_base
            ]
            command_to_run.extend(cmd_flags)

            self.console.print("\n[bold yellow]The following command will be executed:[/bold yellow]")
            self.console.print(f"`{' '.join(command_to_run)}`")
            
            if not Confirm.ask("\nRun this command?", default=True):
                self.console.print("[italic]Operation cancelled by user.[/italic]")
                return

            # Execute script
            self.console.print("\n[italic blue]Attempting to close database connection before script execution...[/italic blue]")
            close_and_clear_global_app_instance()
            
            # Add a small delay to allow OS to release file locks
            time.sleep(0.5) 
            
            self.console.print(f"[italic blue]Executing script: {script_path}...[/italic blue]")
            process = subprocess.run(command_to_run, capture_output=True, text=True, check=False)
            
            self.console.print("\n[bold green]Script Standard Output:[/bold green]")
            if process.stdout:
                self.console.print(process.stdout)
            else:
                self.console.print("[italic]No standard output.[/italic]")

            if process.stderr:
                self.console.print("\n[bold red]Script Standard Error:[/bold red]")
                self.console.print(process.stderr)
            
            if process.returncode == 0:
                self.console.print("\n[bold green]Script executed successfully.[/bold green]")
            else:
                self.console.print(f"\n[bold red]Script failed with exit code {process.returncode}.[/bold red]")

        except Exception as e:
            self.console.print("\n[bold red]An error occurred: {e}[/bold red]")
            logger.error(f"Error in _generate_continuous_futures_series: {e}", exc_info=True)
        finally:
            self.console.print("\n[italic blue]Re-initializing application and database connection...[/italic blue]")
            
            db_path_for_reinit = self.args.database
            read_only_for_reinit = not self.args.write
            
            self.app = get_app(db_path_for_reinit, read_only_for_reinit)
            
            # Re-fetch module handlers for the CLI instance to use the new app state
            self.sql_executor = get_sql_executor(self.app)
            self.schema_browser = get_schema_browser(self.app)
            self.quality_analyzer = get_quality_analyzer(self.app)
            
            self.console.print("[italic blue]Application re-initialized.[/italic blue]")

    def backup_restore_menu(self) -> None:
        """Display backup/restore menu."""
        self.console.print("\n[bold cyan]Backup/Restore[/bold cyan]")
        
        options = [
            "Create Backup",
            "Restore from Backup",
            "List Backups",
            "Update Market Data",
            "Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"{i}. {option}")
        
        choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])
        
        if choice == '1':
            self.create_backup()
        elif choice == '2':
            self.restore_backup()
        elif choice == '3':
            self.list_backups()
        elif choice == '4':
            self._run_market_data_update_script()
        elif choice == '5':
            return
    
    def _run_market_data_update_script(self) -> None:
        """Runs the market data update Python script directly, with --skip-panama."""
        # Construct the path to the Python script
        project_root = Path(__file__).resolve().parent.parent.parent # financial-data-system
        script_path = project_root / "src" / "scripts" / "market_data" / "update_all_market_data_v2.py"
        python_executable = project_root / "venv" / "Scripts" / "python.exe"

        command_to_run = f'"{python_executable}" "{script_path}" --skip-panama'
        
        self.console.print(f"\n[bold yellow]Attempting to run market data update script:[/bold yellow]\n{command_to_run}")
        
        confirm_run = Confirm.ask("Do you want to proceed?", default=True)
        if not confirm_run:
            self.console.print("[italic]Market data update cancelled.[/italic]")
            return

        self.console.print("[italic blue]Executing script...[/italic blue]")
        logger.info(f"User confirmed to run market data update script: {command_to_run}")
        
        # Close database connection before running script
        self.console.print("[italic blue]Closing database connection...[/italic blue]")
        self.app.shutdown()
        
        try:
            # Execute the command using run_terminal_cmd
            # Note: The run_terminal_cmd in the tool definition doesn't have require_user_approval
            # We will rely on the user confirming above.
            run_terminal_cmd_result = self.tool_registry.get_tool("run_terminal_cmd").call({
                "command": command_to_run,
                "is_background": False,
                # "require_user_approval": True, # This arg is not in the tool definition
                "explanation": "Execute the market data update Python script with --skip-panama."
            })
            
            # This part assumes run_terminal_cmd_result might give some feedback or raise on error
            # For now, we just log and proceed.
            logger.info(f"Terminal command proposed. Result: {run_terminal_cmd_result}")

            # Reinitialize app and database connection
            self.console.print("[italic blue]Reopening database connection...[/italic blue]")
            
            db_path_for_reinit = self.args.database
            read_only_for_reinit = not self.args.write
            self.app = get_app(db_path_for_reinit, read_only_for_reinit)
            
            # Re-fetch module handlers
            self.sql_executor = get_sql_executor(self.app)
            self.schema_browser = get_schema_browser(self.app)
            self.quality_analyzer = get_quality_analyzer(self.app)
            
        except Exception as e:
            # Reinitialize app and database connection even if script fails
            self.console.print("[italic blue]Reopening database connection after error...[/italic blue]")
            db_path_for_reinit = self.args.database
            read_only_for_reinit = not self.args.write
            self.app = get_app(db_path_for_reinit, read_only_for_reinit)

            # Re-fetch module handlers
            self.sql_executor = get_sql_executor(self.app)
            self.schema_browser = get_schema_browser(self.app)
            self.quality_analyzer = get_quality_analyzer(self.app)
            raise e

    def create_backup(self) -> None:
        """Create a database backup."""
        if self.app.db_manager.read_only:
            self.console.print("[bold yellow]Note:[/bold yellow] Database is in read-only mode. Backup will use default method.")
        
        # Ask for custom path or use default
        use_custom = Confirm.ask("Would you like to specify a backup location?")
        
        backup_path = None
        if use_custom:
            default_dir = Path(self.config.get("paths", "backup"))
            backup_path = Prompt.ask("Enter backup path", default=str(default_dir))
        
        self.console.print("Creating backup...")
        success, message = self.app.backup_database(backup_path)
        
        if success:
            self.console.print(f"[bold green]Backup created successfully:[/bold green] {message}")
        else:
            self.console.print(f"[bold red]Backup failed:[/bold red] {message}")
    
    def restore_backup(self) -> None:
        """Restore database from backup."""
        if self.app.db_manager.read_only:
            self.console.print("[bold red]Error:[/bold red] Database is in read-only mode. Cannot restore.")
            self.console.print("Restart with --write/-w flag to enable write mode.")
            return
        
        # Show available backups
        backup_dir = Path(self.config.get("paths", "backup"))
        
        if not backup_dir.exists():
            self.console.print(f"[bold yellow]Backup directory not found:[/bold yellow] {backup_dir}")
            return
        
        backups = list(backup_dir.glob("*.duckdb"))
        
        if not backups:
            self.console.print("[bold yellow]No backups found.[/bold yellow]")
            return
        
        # Sort backups by modification time (newest first)
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        table = Table(title="Available Backups", box=SIMPLE)
        table.add_column("#")
        table.add_column("Backup File")
        table.add_column("Date")
        table.add_column("Size")
        
        for i, backup in enumerate(backups, 1):
            # Get file stats
            stats = backup.stat()
            mod_time = stats.st_mtime
            size = stats.st_size
            
            # Format date and size
            from datetime import datetime
            date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            elif size < 1024 * 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
            
            table.add_row(str(i), backup.name, date_str, size_str)
        
        self.console.print(table)
        
        # Ask user to select backup
        choices = [str(i) for i in range(1, len(backups) + 1)]
        choice = Prompt.ask("Enter backup number to restore", choices=choices)
        
        selected_backup = backups[int(choice) - 1]
        
        # Confirm restore
        confirm = Confirm.ask(f"Are you sure you want to restore from {selected_backup.name}? This will overwrite the current database.")
        
        if not confirm:
            self.console.print("Restore cancelled.")
            return
        
        # Perform restore
        self.console.print(f"Restoring from {selected_backup}...")
        success, message = self.app.restore_database(selected_backup)
        
        if success:
            self.console.print(f"[bold green]Restore completed successfully:[/bold green] {message}")
        else:
            self.console.print(f"[bold red]Restore failed:[/bold red] {message}")
    
    def list_backups(self) -> None:
        """List available database backups."""
        backup_dir = Path(self.config.get("paths", "backup"))
        
        if not backup_dir.exists():
            self.console.print(f"[bold yellow]Backup directory not found:[/bold yellow] {backup_dir}")
            return
        
        backups = list(backup_dir.glob("*.duckdb"))
        
        if not backups:
            self.console.print("[bold yellow]No backups found.[/bold yellow]")
            return
        
        # Sort backups by modification time (newest first)
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        table = Table(title="Available Backups", box=SIMPLE)
        table.add_column("Backup File")
        table.add_column("Date")
        table.add_column("Size")
        
        for backup in backups:
            # Get file stats
            stats = backup.stat()
            mod_time = stats.st_mtime
            size = stats.st_size
            
            # Format date and size
            from datetime import datetime
            date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            elif size < 1024 * 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
            
            table.add_row(backup.name, date_str, size_str)
        
        self.console.print(table)
    
    def exit_application(self) -> None:
        """Exit the application."""
        self.console.print("\nShutting down DB Inspector...")
        self.app.shutdown()
        self.console.print("[bold green]Goodbye![/bold green]")

    def database_info_menu(self) -> None:
        """Display database information menu."""
        while True:
            self.console.print("\n[bold cyan]Database Information[/bold cyan]")
            options = [
                "Show General Info",
                "Generate Data Summary Report",
                "Back to Main Menu"
            ]
            for i, option in enumerate(options, 1):
                self.console.print(f"{i}. {option}")
            
            choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])

            if choice == '1':
                self.show_database_info()
            elif choice == '2':
                self.generate_data_summary_report()
            elif choice == '3':
                break

    def generate_data_summary_report(self) -> None:
        """Generates and displays a summary report for key market data tables,
        optionally filtered by a root symbol."""
        self.console.print("\n[bold cyan]Generating Data Summary Report...[/bold cyan]")

        root_symbol_input = Prompt.ask(
            "Enter root symbol (e.g., ES, VX) or leave blank for all", default=""
        ).strip().upper()

        base_queries = {
            "market_data": """
                SELECT
                    symbol,
                    interval_value,
                    interval_unit,
                    source,
                    MIN(timestamp) AS first_date,
                    MAX(timestamp) AS last_date,
                    COUNT(*) AS observation_count
                FROM market_data
                {where_clause}
                GROUP BY symbol, interval_value, interval_unit, source
                ORDER BY
                    interval_unit,
                    interval_value,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$'
                        THEN CAST(SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) AS INTEGER)
                        ELSE NULL
                    END,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$' AND LENGTH(symbol) > 2
                        THEN
                            CASE SUBSTRING(symbol FROM LENGTH(symbol)-2 FOR 1)
                                WHEN 'F' THEN 1 WHEN 'G' THEN 2 WHEN 'H' THEN 3 WHEN 'J' THEN 4
                                WHEN 'K' THEN 5 WHEN 'M' THEN 6 WHEN 'N' THEN 7 WHEN 'Q' THEN 8
                                WHEN 'U' THEN 9 WHEN 'V' THEN 10 WHEN 'X' THEN 11 WHEN 'Z' THEN 12
                                ELSE 99
                            END
                        ELSE 99
                    END,
                    symbol,
                    source;
            """,
            "market_data_cboe": """
                SELECT
                    symbol,
                    interval_value,
                    interval_unit,
                    source,
                    MIN(timestamp) AS first_date,
                    MAX(timestamp) AS last_date,
                    COUNT(*) AS observation_count
                FROM market_data_cboe
                {where_clause}
                GROUP BY symbol, interval_value, interval_unit, source
                ORDER BY
                    interval_unit,
                    interval_value,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$'
                        THEN CAST(SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) AS INTEGER)
                        ELSE NULL
                    END,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$' AND LENGTH(symbol) > 2
                        THEN
                            CASE SUBSTRING(symbol FROM LENGTH(symbol)-2 FOR 1)
                                WHEN 'F' THEN 1 WHEN 'G' THEN 2 WHEN 'H' THEN 3 WHEN 'J' THEN 4
                                WHEN 'K' THEN 5 WHEN 'M' THEN 6 WHEN 'N' THEN 7 WHEN 'Q' THEN 8
                                WHEN 'U' THEN 9 WHEN 'V' THEN 10 WHEN 'X' THEN 11 WHEN 'Z' THEN 12
                                ELSE 99
                            END
                        ELSE 99
                    END,
                    symbol,
                    source;
            """,
            "continuous_contracts": """
                SELECT
                    symbol,
                    interval_value,
                    interval_unit,
                    source,
                    MIN(timestamp) AS first_date,
                    MAX(timestamp) AS last_date,
                    COUNT(*) AS observation_count
                FROM continuous_contracts
                {where_clause}
                GROUP BY symbol, interval_value, interval_unit, source
                ORDER BY
                    interval_unit,
                    interval_value,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$'
                        THEN CAST(SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) AS INTEGER)
                        ELSE NULL
                    END,
                    CASE
                        WHEN SUBSTRING(symbol FROM LENGTH(symbol)-1 FOR 2) ~ '^[0-9]{{2}}$' AND LENGTH(symbol) > 2
                        THEN
                            CASE SUBSTRING(symbol FROM LENGTH(symbol)-2 FOR 1)
                                WHEN 'F' THEN 1 WHEN 'G' THEN 2 WHEN 'H' THEN 3 WHEN 'J' THEN 4
                                WHEN 'K' THEN 5 WHEN 'M' THEN 6 WHEN 'N' THEN 7 WHEN 'Q' THEN 8
                                WHEN 'U' THEN 9 WHEN 'V' THEN 10 WHEN 'X' THEN 11 WHEN 'Z' THEN 12
                                ELSE 99
                            END
                        ELSE 99
                    END,
                    symbol,
                    source;
            """
        }

        where_clause_sql = ""
        if root_symbol_input:
            # Escape single quotes in root_symbol_input to prevent SQL injection
            # हालांकि, LIKE पैटर्न में, हमें % वाइल्डकार्ड की अनुमति देनी चाहिए
            # इसलिए, हम केवल सिंगल कोट्स को एस्केप करेंगे।
            safe_root_symbol = root_symbol_input.replace("'", "''")
            where_clause_sql = f"WHERE (symbol LIKE '{safe_root_symbol}%' OR symbol LIKE '@{safe_root_symbol}%')"
            self.console.print(f"[italic blue]Filtering by root symbol: {root_symbol_input}[/italic blue]")
        else:
            # Ensures a valid query even if no specific where clause is needed.
            # Some DBs might require a WHERE clause if others are conditional.
            # For DuckDB, an empty string for {where_clause} in f-string formatting is fine.
             where_clause_sql = " " # Add a space to maintain formatting if no WHERE clause

        for table_name, base_query_template in base_queries.items():
            query = base_query_template.format(where_clause=where_clause_sql)
            
            self.console.print(f"\n[bold green]Summary for {table_name}:[/bold green]")
            try:
                result = self.app.db_manager.execute_query(query)
                
                if result.is_success and not result.is_empty:
                    summary_table = Table(box=SIMPLE)
                    # Add columns based on the first row's keys, if dataframe exists
                    if result.dataframe is not None and not result.dataframe.empty:
                        for col_name in result.dataframe.columns:
                            summary_table.add_column(col_name.replace('_', ' ').title())
                        
                        for index, row in result.dataframe.iterrows():
                            summary_table.add_row(*(str(row[col]) for col in result.dataframe.columns))
                        self.console.print(summary_table)
                    else:
                        self.console.print("No data found or dataframe is empty.")
                        
                elif not result.is_success:
                    self.console.print(f"[red]Error executing query for {table_name}: {result.error}[/red]")
                else: # result.is_empty
                    self.console.print("No data found in this table matching the criteria.")
            except Exception as e:
                self.console.print(f"[red]Failed to generate summary for {table_name}: {e}[/red]")
                logger.error(f"Error generating summary for {table_name}: {e}", exc_info=True)

def main():
    """Main entry point for DB Inspector CLI."""
    try:
        cli = DBInspectorCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

        print(f"\n❌ Error: {e}")
        print("An unexpected error occurred. Check the logs for details.")

        if "--debug" in sys.argv:
            print("\nDetailed error information (debug mode):")
            traceback.print_exc()

        sys.exit(1)

if __name__ == "__main__":
    main()