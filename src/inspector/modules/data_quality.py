"""
Data Quality Analysis Module.

This module provides tools for analyzing data quality, including missing data detection,
outlier identification, and consistency checks across data sources.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime, timedelta

# Import required dependencies with fallbacks
try:
    import numpy as np
except ImportError:
    np = None
    logging.error("NumPy is not installed. Data quality analysis will be limited.")

try:
    import pandas as pd
except ImportError:
    pd = None
    logging.error("Pandas is not installed. Data quality analysis will not function.")

# Import optional visualization dependencies
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib is not installed. Visualization features will be disabled.")

try:
    import seaborn as sns
except ImportError:
    sns = None
    logging.warning("Seaborn is not installed. Enhanced visualizations will be disabled.")

# Import UI dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import SIMPLE
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.prompt import Prompt
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
    class Progress:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def add_task(self, *args, **kwargs):
            return 0
        def update(self, *args, **kwargs):
            pass

from ..core.app import get_app
from ..core.config import get_config
from ..core.schema import get_schema_manager

# Setup logging
logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Data quality analysis tools for financial market data."""

    def __init__(self):
        """Initialize data quality analyzer."""
        # Check if required dependencies are available
        self._check_dependencies()

        self.app = get_app(db_path="data/financial_data.duckdb")
        self.config = get_config()
        self.schema_manager = get_schema_manager()
        self.console = Console()

        # Default configuration
        self.default_config = {
            'outlier_threshold_std': 3.0,  # Standard deviations for outlier detection
            'zero_price_detection': True,  # Detect zero prices
            'min_expected_volume': 0,  # Minimum expected volume
            'gap_detection_days': 1,  # Maximum allowed gap in days
            'market_hours_check': False,  # Check if data is within market hours
        }

        # Current configuration (can be overridden)
        self.config_values = self.default_config.copy()

    def _parse_interval_string(self, interval_str: Optional[str]) -> Optional[Tuple[int, str]]:
        """
        Parse an interval string (e.g., "1D", "30Min", "H") into value and unit.

        Args:
            interval_str: The interval string.

        Returns:
            A tuple (value, unit_name) or None if parsing fails.
            Units: 'minute', 'hourly', 'daily', 'weekly', 'monthly'.
        """
        if not interval_str:
            return None

        interval_str = interval_str.strip().lower()
        if not interval_str:
            return None

        import re
        match = re.match(r"(\d*)?\s*([a-z]+)", interval_str)
        if not match:
            return None

        value_str, unit_abbr = match.groups()
        value = int(value_str) if value_str else 1

        if unit_abbr in ("m", "min", "minute", "minutes"):
            unit = "minute"
        elif unit_abbr in ("h", "hr", "hour", "hours"):
            unit = "hourly"
        elif unit_abbr in ("d", "day", "days", "daily"):
            unit = "daily"
        elif unit_abbr in ("w", "wk", "week", "weeks", "weekly"):
            unit = "weekly"
        elif unit_abbr in ("mon", "month", "months", "monthly"):
            unit = "monthly"
        else:
            logger.warning(f"Could not parse interval unit: {unit_abbr} from string {interval_str}")
            return None
        
        return value, unit

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        missing_deps = []

        if pd is None:
            missing_deps.append("pandas")

        if np is None:
            missing_deps.append("numpy")

        # These are optional but affect functionality
        optional_missing = []
        if plt is None:
            optional_missing.append("matplotlib")

        if sns is None:
            optional_missing.append("seaborn")

        # Log warnings about missing dependencies
        if missing_deps:
            logger.error(f"Required dependencies missing: {', '.join(missing_deps)}")
            print(f"\n❌ Error: Required dependencies missing: {', '.join(missing_deps)}")
            print("Data quality analysis requires these packages.")
            print(f"Install with: pip install {' '.join(missing_deps)}")

        if optional_missing:
            logger.warning(f"Optional dependencies missing: {', '.join(optional_missing)}")
            print(f"\n⚠️ Warning: Optional dependencies missing: {', '.join(optional_missing)}")
            print("Some visualization features will be disabled.")
            print(f"Install with: pip install {' '.join(optional_missing)}")

        # Only continue if critical dependencies are available
        self.can_analyze = (pd is not None and np is not None)
        self.can_visualize = (plt is not None)
    
    def set_config(self, **kwargs) -> None:
        """
        Set configuration values.
        
        Args:
            **kwargs: Configuration key-value pairs
        """
        for key, value in kwargs.items():
            if key in self.config_values:
                self.config_values[key] = value
                logger.info(f"Set {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def analyze_market_data_table(self, symbol: Optional[str] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               interval: Optional[str] = None, # User input string e.g. "1D", "30Min"
                               is_continuous: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis on market data.

        Args:
            symbol: Symbol to analyze (optional)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval string (e.g., "1D", "30Min")
            is_continuous: Whether the data is for a continuous contract

        Returns:
            Dictionary with analysis results
        """
        if not getattr(self, 'can_analyze', False):
            return {
                'success': False,
                'message': 'Required dependencies (pandas, numpy) are missing',
                'error': 'Install required dependencies with: pip install pandas numpy'
            }

        params: List[Any] = []
        
        # Parse the user-provided interval string
        parsed_interval: Optional[Tuple[int, str]] = self._parse_interval_string(interval)
        query_symbol = symbol # Symbol to be used in the SQL query

        if is_continuous:
            target_table = "continuous_contracts"
            date_column = "timestamp"
            source_name = "Continuous Contracts Data"
            order_by_columns = ["symbol", date_column, "interval_value", "interval_unit"]

            # Determine actual interval to query for continuous contracts
            if parsed_interval:
                query_interval_value = parsed_interval[0]
                query_interval_unit = parsed_interval[1]
                logger.info(f"User specified interval for continuous contract: {query_interval_value} {query_interval_unit}")

                # If a specific continuous contract (e.g., @ES=102XC) is given 
                # AND a sub-daily interval is requested, switch to the root symbol (e.g., @ES)
                if symbol and "=" in symbol and query_interval_unit != "daily":
                    root_symbol_candidate = symbol.split("=")[0]
                    # Basic validation: check if it looks like a root (e.g., starts with @, doesn't have too many chars after @)
                    if root_symbol_candidate.startswith("@") and len(root_symbol_candidate) < 6: # Heuristic
                        logger.info(f"Sub-daily interval ('{query_interval_unit}') requested for specific continuous symbol '{symbol}'. Switching to root symbol '{root_symbol_candidate}' for analysis.")
                        query_symbol = root_symbol_candidate
                    else:
                        logger.warning(f"Could not reliably determine root symbol from '{symbol}' for sub-daily analysis. Proceeding with original symbol.")
            else:
                # Default to daily if no interval is specified by the user for continuous contracts
                query_interval_value = 1
                query_interval_unit = "daily"
                logger.info(f"No interval specified for continuous contract. Defaulting to {query_interval_value} {query_interval_unit}.")
        else:
            target_table = "market_data"
            date_column = "timestamp"
            order_by_columns = ["symbol", date_column, "interval_value", "interval_unit"]

            query_interval_value = parsed_interval[0] if parsed_interval else None
            query_interval_unit = parsed_interval[1] if parsed_interval else None

        query = f"SELECT * FROM {target_table} WHERE 1=1"

        if query_symbol: # Use query_symbol here
            query += f" AND symbol = ?"
            params.append(query_symbol)
        if start_date:
            # Use CAST(date_column AS DATE) for DuckDB compatibility to compare dates
            query += f" AND CAST({date_column} AS DATE) >= ?"
            params.append(start_date)
        if end_date:
            query += f" AND CAST({date_column} AS DATE) <= ?"
            params.append(end_date)
        
        if query_interval_value is not None and query_interval_unit is not None:
            query += f" AND interval_value = ? AND interval_unit = ?"
            params.append(query_interval_value)
            params.append(query_interval_unit)
        
        query += f" ORDER BY {', '.join(order_by_columns)}"

        df: Optional[pd.DataFrame] = None
        error_message: Optional[str] = None
        actual_source_table = target_table # Keep track of where data actually came from
        
        logger.info(f"Executing query for {source_name}: {query} with params: {params}")
        try:
            result = self.app.db_manager.execute_query(query, params=params)
            logger.info(f"Query result for {source_name} - success: {result.is_success}, empty: {result.is_empty}")
            if result.is_success and not result.is_empty:
                df = result.dataframe
                logger.info(f"Found {len(df)} rows in {source_name}")
            elif result.is_success and result.is_empty:
                error_message = f"No data found matching criteria in {source_name}."
            else: # Query failed
                error_message = f"Error executing database query for {source_name}: {result.error}"
                
        except Exception as e:
            logger.error(f"Database query error for {source_name}: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Error executing database query for {source_name}',
                'error': str(e)
            }

        # Fallback to market_data_cboe ONLY if not continuous and primary query failed or yielded no data
        if not is_continuous and df is None:
            logger.info(f"No data from {target_table} or query failed, trying market_data_cboe.")
            cboe_table = "market_data_cboe"
            # CBOE uses 'timestamp', 'interval_value', 'interval_unit' (always 1, 'daily' as per DATABASE.MD)
            cboe_date_column = "timestamp"
            cboe_order_by_columns = ["symbol", cboe_date_column] 
            cboe_source_name = "Market Data CBOE"
            
            cboe_query = f"SELECT * FROM {cboe_table} WHERE 1=1"
            cboe_params: List[Any] = []
            actual_source_table = cboe_table # Update if we fetch from here

            if query_symbol:
                cboe_query += f" AND symbol = ?"
                cboe_params.append(query_symbol)
            if start_date:
                # Use CAST(cboe_date_column AS DATE) for DuckDB compatibility
                cboe_query += f" AND CAST({cboe_date_column} AS DATE) >= ?"
                cboe_params.append(start_date)
            if end_date:
                # Use CAST(cboe_date_column AS DATE) for DuckDB compatibility
                cboe_query += f" AND CAST({cboe_date_column} AS DATE) <= ?"
                cboe_params.append(end_date)
            
            # CBOE data is daily, so filter for interval_value=1 and interval_unit='daily'
            # This aligns with the DATABASE.MD note: "interval_unit: (text) - 'daily' (always for CBOE)"
            cboe_query += f" AND interval_value = ? AND interval_unit = ?"
            cboe_params.append(1)
            cboe_params.append("daily")
            
            cboe_query += f" ORDER BY {', '.join(cboe_order_by_columns)}"
            
            logger.info(f"Executing query for {cboe_source_name}: {cboe_query} with params: {cboe_params}")
            try:
                result_cboe = self.app.db_manager.execute_query(cboe_query, params=cboe_params)
                logger.info(f"Query result for {cboe_source_name} - success: {result_cboe.is_success}, empty: {result_cboe.is_empty}")
                if result_cboe.is_success and not result_cboe.is_empty:
                    df = result_cboe.dataframe
                    # Overwrite date_column for downstream processing if it was different
                    date_column = cboe_date_column 
                    logger.info(f"Found {len(df)} rows in {cboe_source_name}")
                    error_message = None # Clear previous error if CBOE data found
                elif result_cboe.is_success and result_cboe.is_empty:
                    if not error_message or "No data found" in error_message:
                         error_message = f"No data found matching criteria in {source_name} or {cboe_source_name}."
                else: # CBOE Query failed
                    error_message = f"Error executing database query for {cboe_source_name}: {result_cboe.error}"

            except Exception as e:
                logger.error(f"Database query error for {cboe_source_name}: {e}", exc_info=True)
                if error_message is None: 
                    error_message = f"Error executing database query for {cboe_source_name}: {str(e)}"
        
        if df is None: 
            return {
                'success': False,
                'message': error_message or "No data found or query failed for specified criteria.",
                'error': error_message 
            }

        # Ensure the date column used for analysis exists and is consistently named 'date'.
        final_date_col_for_analysis = 'date' 
        if date_column == 'timestamp' and 'timestamp' in df.columns:
            if 'date' in df.columns and not df['date'].equals(df['timestamp']):
                logger.warning("DataFrame has both 'date' and 'timestamp' columns with different data. Using 'timestamp' and renaming to 'date'.")
                df = df.drop(columns=['date']) # Drop the conflicting 'date' column
            if 'date' not in df.columns: # Rename if 'date' doesn't exist or was just dropped
                 logger.info(f"Renaming '{date_column}' to '{final_date_col_for_analysis}' for analysis consistency.")
                 df = df.rename(columns={date_column: final_date_col_for_analysis})
        elif final_date_col_for_analysis not in df.columns:
            logger.error(f"Critical: DataFrame does not have the expected date column '{final_date_col_for_analysis}' (original: '{date_column}') for analysis.")
            return {
                'success': False,
                'message': f"Data retrieved but essential date column '{final_date_col_for_analysis}' (from '{date_column}') is missing or could not be prepared.",
                'error': "Missing/unprepared date column"
            }
        # Begin analysis
        try:
            with Progress(console=self.console, transient=True) as progress:
                analysis_task = progress.add_task("[cyan]Analyzing data quality...", total=5)

                progress.update(analysis_task, advance=1, description="Calculating basic statistics...")
                stats = self._calculate_statistics(df, date_col_name=final_date_col_for_analysis)

                progress.update(analysis_task, advance=1, description="Analyzing missing data...")
                # Pass the determined query_interval_value and query_interval_unit as a tuple
                current_query_interval_tuple = None
                if 'query_interval_value' in locals() and 'query_interval_unit' in locals() and query_interval_value is not None and query_interval_unit is not None:
                    current_query_interval_tuple = (query_interval_value, query_interval_unit)
                
                missing_data = self._analyze_missing_data(df, date_col_name=final_date_col_for_analysis, query_interval_tuple=current_query_interval_tuple)

                progress.update(analysis_task, advance=1, description="Detecting outliers...")
                outliers = self._detect_outliers(df, date_col_name=final_date_col_for_analysis)

                progress.update(analysis_task, advance=1, description="Checking time series consistency...")
                consistency = self._check_time_series_consistency(df, date_col_name=final_date_col_for_analysis)
                
                progress.update(analysis_task, advance=1, description="Generating visualizations...")
                visualizations = {}
                if getattr(self, 'can_visualize', False):
                    visualizations = self._generate_visualizations(df, date_col_name=final_date_col_for_analysis)
                else:
                    logger.warning("Visualizations skipped due to missing dependencies")
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Error during analysis: {str(e)}',
                'error': str(e),
                'partial_results': {
                    'symbol': query_symbol,
                    'date_range': [df[final_date_col_for_analysis].min(), df[final_date_col_for_analysis].max()] if final_date_col_for_analysis in df.columns and not df.empty else None,
                    'row_count': len(df),
                    'statistics': stats if 'stats' in locals() else {},
                }
            }

        # Return comprehensive results
        return {
            'success': True,
            'symbol': query_symbol, # Return the symbol actually used for querying
            'source_table': actual_source_table, 
            'date_range': [df[final_date_col_for_analysis].min(), df[final_date_col_for_analysis].max()] if final_date_col_for_analysis in df.columns and not df.empty else None,
            'row_count': len(df),
            'statistics': stats,
            'missing_data': missing_data,
            'outliers': outliers,
            'consistency': consistency,
            'visualizations': visualizations
        }
    
    def _calculate_statistics(self, df: pd.DataFrame, date_col_name: str) -> Dict[str, Any]:
        """
        Calculate basic statistics for market data.
        
        Args:
            df: Market data DataFrame
            date_col_name: Name of the date column
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Number of unique symbols
        if 'symbol' in df.columns:
            stats['unique_symbols'] = df['symbol'].nunique()
            
            # Group by symbol
            symbol_stats = {}
            for symbol, group in df.groupby('symbol'):
                symbol_stats[symbol] = {
                    'row_count': len(group),
                    'date_range': [group[date_col_name].min(), group[date_col_name].max()],
                    'intervals': group['interval'].unique().tolist() if 'interval' in group.columns else None
                }
            stats['by_symbol'] = symbol_stats
        
        # Calculate OHLCV statistics if columns exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                stats[f'{col}_stats'] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'zeros': (df[col] == 0).sum()
                }
        
        # Calculate returns if close exists
        if 'close' in df.columns and len(df) > 1:
            # Try to sort by date if available
            if date_col_name in df.columns:
                df_sorted = df.sort_values(['symbol', date_col_name])
            else:
                df_sorted = df
                
            # Calculate returns by symbol
            returns_stats = {}
            for symbol, group in df_sorted.groupby('symbol'):
                if len(group) <= 1:
                    continue
                
                # Calculate returns
                returns = group['close'].pct_change().dropna()
                
                if len(returns) > 0:
                    returns_stats[symbol] = {
                        'min': returns.min(),
                        'max': returns.max(),
                        'mean': returns.mean(),
                        'std': returns.std(),
                        'positive_days': (returns > 0).sum(),
                        'negative_days': (returns < 0).sum()
                    }
            
            stats['returns'] = returns_stats
        
        return stats
    
    def _analyze_missing_data(self, df: pd.DataFrame, date_col_name: str, query_interval_tuple: Optional[Tuple[int, str]] = None) -> Dict[str, Any]:
        """
        Analyze missing data in market data.
        
        Args:
            df: Market data DataFrame
            date_col_name: Name of the date column
            query_interval_tuple: Optional tuple (value, unit) of the queried interval.
            
        Returns:
            Dictionary with missing data analysis
        """
        missing = {}
        
        # Overall missing values
        missing['overall'] = {
            'total_cells': df.size,
            'missing_cells': df.isna().sum().sum(),
            'missing_percentage': (df.isna().sum().sum() / df.size) * 100
        }
        
        # Missing by column
        missing['by_column'] = {}
        for col in df.columns:
            missing['by_column'][col] = {
                'missing_count': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
        
        # Analyze time series gaps if date column exists
        if date_col_name in df.columns and 'symbol' in df.columns:
            missing['time_series_gaps'] = {}
            
            # Convert query_interval_tuple back to string for _find_date_gaps if needed, or modify _find_date_gaps to accept tuple
            # For now, let's reconstruct a simple string if the tuple is provided.
            interval_str_for_gaps = None
            if query_interval_tuple:
                value, unit = query_interval_tuple
                if unit == "minute": unit_abbr = "Min"
                elif unit == "hourly": unit_abbr = "H"
                elif unit == "daily": unit_abbr = "D"
                elif unit == "weekly": unit_abbr = "W"
                elif unit == "monthly": unit_abbr = "M"
                else: unit_abbr = unit
                interval_str_for_gaps = f"{value}{unit_abbr}"

            for symbol, group in df.groupby('symbol'):
                if len(group) <= 1:
                    continue
                
                # Log the actual date range of the group being processed for gaps
                if not group.empty and date_col_name in group.columns:
                    min_date_in_group = group[date_col_name].min()
                    max_date_in_group = group[date_col_name].max()
                    logger.info(f"_analyze_missing_data: Processing symbol '{symbol}'. Group date range: {min_date_in_group} to {max_date_in_group}. Points: {len(group)}.")

                # If query_interval_tuple is provided, use it directly for gap analysis.
                # This overrides trying to guess from a df['interval'] column which might be inconsistent.
                if interval_str_for_gaps:
                    gaps = self._find_date_gaps(group, interval=interval_str_for_gaps, date_col_name=date_col_name)
                    if gaps:
                        # Store gaps directly under symbol, not nested by DataFrame's interval column values
                        missing['time_series_gaps'][symbol] = gaps 
                else: # Fallback to old behavior if no specific query interval was given (should be rare for main analysis)
                    logger.warning(f"No specific query interval provided to _analyze_missing_data for symbol {symbol}. Analyzing gaps by df['interval'] column if present.")
                    if 'interval' in df.columns:
                        interval_gaps_dict = {}
                        for interval_val_from_df, interval_group in group.groupby('interval'):
                            gaps = self._find_date_gaps(interval_group, interval=str(interval_val_from_df), date_col_name=date_col_name)
                            if gaps:
                                interval_gaps_dict[str(interval_val_from_df)] = gaps
                        if interval_gaps_dict:
                            missing['time_series_gaps'][symbol] = interval_gaps_dict
                    else:
                        # Analyze without interval if column not present
                        gaps = self._find_date_gaps(group, date_col_name=date_col_name) # Interval will be None, _find_date_gaps defaults
                        if gaps:
                            missing['time_series_gaps'][symbol] = gaps
        
        return missing
    
    def _find_date_gaps(self, df: pd.DataFrame, interval: Optional[str] = None, date_col_name: str = 'date') -> List[Dict[str, Any]]:
        """
        Find gaps in date sequence. More performant for high-frequency data.
        
        Args:
            df: DataFrame for a single symbol and a single interval.
            interval: Data interval string (e.g., "1min", "1D").
            date_col_name: Name of the date column.
            
        Returns:
            List of gap dictionaries.
        """
        if df.empty or len(df) < 2:
            return []

        df_sorted = df.sort_values(date_col_name).copy() # Ensure it's sorted and a copy
        dates = pd.to_datetime(df_sorted[date_col_name])
        gaps = []

        # Determine expected frequency based on interval string
        parsed_interval_tuple = self._parse_interval_string(interval)
        if not parsed_interval_tuple:
            logger.warning(f"Could not parse interval '{interval}' in _find_date_gaps. Defaulting to business day frequency for gap detection.")
            expected_delta = pd.tseries.offsets.BDay()
        else:
            val, unit = parsed_interval_tuple
            if unit == "minute":
                expected_delta = pd.Timedelta(minutes=val)
            elif unit == "hourly":
                expected_delta = pd.Timedelta(hours=val)
            elif unit == "daily":
                expected_delta = pd.Timedelta(days=val) # or BDay() if only business days expected
                 # For daily, consider if it should be calendar days or business days.
                 # If strictly business days, pd.tseries.offsets.BDay() is better.
                 # For now, using Timedelta for simplicity, adjust if stricter business day logic needed.
            elif unit == "weekly":
                expected_delta = pd.Timedelta(weeks=val)
            elif unit == "monthly": # This is trickier due to varying month lengths
                # For monthly, a simple timedelta is often not accurate.
                # This part might need a more sophisticated approach if precise monthly gaps are critical.
                logger.warning("Monthly gap detection with simple timedelta may be imprecise.")
                expected_delta = pd.Timedelta(days=val * 30) # Approximation
            else:
                logger.warning(f"Unknown interval unit '{unit}' in _find_date_gaps. Defaulting to business day frequency.")
                expected_delta = pd.tseries.offsets.BDay()
        
        # Iterate through sorted dates to find gaps
        current_gap_start = None
        max_deltas_logged = 0 # Helper to limit logging

        for i in range(len(dates) - 1):
            # Calculate the difference between consecutive timestamps
            actual_delta = dates.iloc[i+1] - dates.iloc[i]
            
            # Log large deltas for diagnosis, especially for the full run
            if actual_delta > pd.Timedelta(days=1) and max_deltas_logged < 20: # Log up to 20 large deltas
                logger.info(f"_find_date_gaps: Large actual_delta at index {i}: {actual_delta}. Date[i]: {dates.iloc[i]}, Date[i+1]: {dates.iloc[i+1]}")
                max_deltas_logged +=1

            # A gap exists if actual_delta is significantly larger than expected_delta.
            # Using 1.5 times expected_delta as a threshold to allow for minor timing variations.
            is_gap = False
            if isinstance(expected_delta, pd.Timedelta):
                if actual_delta > expected_delta * 1.5: # If actual is > 1.5x expected, it's a gap
                    is_gap = True
            elif isinstance(expected_delta, pd.tseries.offsets.DateOffset):
                 # For offsets like BDay, check if next expected date is missing
                 next_expected_date = dates.iloc[i] + expected_delta
                 if dates.iloc[i+1] > next_expected_date: # if next actual date is after next expected date
                     is_gap = True
            
            if is_gap:
                gap_start_time = dates.iloc[i] + expected_delta
                gap_end_time = dates.iloc[i+1] - expected_delta 
                
                # Ensure gap_end_time is not before gap_start_time for tiny intervals
                if gap_end_time < gap_start_time:
                    gap_end_time = gap_start_time

                if isinstance(expected_delta, pd.Timedelta) and expected_delta.total_seconds() > 0:
                    # Ensure we are calculating points *within* the gap, so actual_delta / expected_delta, then subtract 1 for the known point after gap.
                    # If actual_delta is 2*expected_delta, there's 1 missing point.
                    num_missing_points = int(round( (actual_delta.total_seconds() / expected_delta.total_seconds()) ) -1 )
                    if num_missing_points < 0: num_missing_points = 0 # Should not be negative
                else: # Fallback for offsets or zero delta
                    # For BDay or other offsets, direct point counting is harder without generating range.
                    # Let's keep it 0 if not a simple Timedelta, or consider a more complex estimation if needed.
                    num_missing_points = 0 # Default to 0 if cannot estimate reliably from simple timedelta arithmetic

                gaps.append({
                    'start_date': gap_start_time,
                    'end_date': gap_end_time, # This is the timestamp before the next actual data point
                    'gap_duration_seconds': (gap_end_time - gap_start_time).total_seconds(),
                    'missing_points_estimated': num_missing_points
                })
        
        return gaps
    
    def _detect_outliers(self, df: pd.DataFrame, date_col_name: str) -> Dict[str, Any]:
        """
        Detect outliers in market data.
        
        Args:
            df: Market data DataFrame
            date_col_name: Name of the date column
            
        Returns:
            Dictionary with outlier analysis
        """
        outliers = {}
        
        # Standard deviation based outlier detection for price columns
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in df.columns]
        
        # Get threshold
        threshold = self.config_values['outlier_threshold_std']
        
        # Detect outliers by symbol
        if 'symbol' in df.columns:
            outliers['by_symbol'] = {}
            
            for symbol, group in df.groupby('symbol'):
                symbol_outliers = {}
                
                for col in available_columns:
                    # Calculate mean and std
                    mean = group[col].mean()
                    std = group[col].std()
                    
                    if pd.isna(mean) or pd.isna(std) or std == 0:
                        continue
                    
                    # Find outliers
                    upper_limit = mean + (std * threshold)
                    lower_limit = mean - (std * threshold)
                    
                    upper_outliers = group[group[col] > upper_limit]
                    lower_outliers = group[group[col] < lower_limit]
                    
                    if len(upper_outliers) > 0 or len(lower_outliers) > 0:
                        symbol_outliers[col] = {
                            'mean': mean,
                            'std': std,
                            'upper_limit': upper_limit,
                            'lower_limit': lower_limit,
                            'upper_outliers_count': len(upper_outliers),
                            'lower_outliers_count': len(lower_outliers),
                            'upper_outliers': upper_outliers[['date', col]].to_dict('records') if len(upper_outliers) > 0 else [],
                            'lower_outliers': lower_outliers[['date', col]].to_dict('records') if len(lower_outliers) > 0 else []
                        }
                
                if symbol_outliers:
                    outliers['by_symbol'][symbol] = symbol_outliers
        
        # Check for zero prices (often indicates data errors)
        if self.config_values['zero_price_detection']:
            zero_prices = {}
            
            for col in available_columns:
                zero_count = (df[col] == 0).sum()
                
                if zero_count > 0:
                    zero_prices[col] = {
                        'count': zero_count,
                        'percentage': (zero_count / len(df)) * 100
                    }
            
            if zero_prices:
                outliers['zero_prices'] = zero_prices
        
        # Check for extreme returns if close is available
        if 'close' in df.columns and 'symbol' in df.columns:
            extreme_returns = {}
            
            for symbol, group in df.groupby('symbol'):
                if len(group) <= 1:
                    continue
                
                # Calculate returns
                sorted_group = group.sort_values(date_col_name)
                returns = sorted_group['close'].pct_change().dropna()
                
                if len(returns) == 0:
                    continue
                
                # Find extreme returns
                mean_return = returns.mean()
                std_return = returns.std()
                
                if pd.isna(mean_return) or pd.isna(std_return) or std_return == 0:
                    continue
                
                upper_limit = mean_return + (std_return * threshold)
                lower_limit = mean_return - (std_return * threshold)
                
                extreme_up = sorted_group.iloc[1:][returns > upper_limit]
                extreme_down = sorted_group.iloc[1:][returns < lower_limit]
                
                if len(extreme_up) > 0 or len(extreme_down) > 0:
                    extreme_returns[symbol] = {
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'upper_limit': upper_limit,
                        'lower_limit': lower_limit,
                        'extreme_up_count': len(extreme_up),
                        'extreme_down_count': len(extreme_down),
                        'extreme_up': extreme_up[['date', 'close']].to_dict('records') if len(extreme_up) > 0 else [],
                        'extreme_down': extreme_down[['date', 'close']].to_dict('records') if len(extreme_down) > 0 else []
                    }
            
            if extreme_returns:
                outliers['extreme_returns'] = extreme_returns
        
        return outliers
    
    def _check_time_series_consistency(self, df: pd.DataFrame, date_col_name: str) -> Dict[str, Any]:
        """
        Check time series consistency in market data.
        
        Args:
            df: Market data DataFrame
            date_col_name: Name of the date column
            
        Returns:
            Dictionary with consistency analysis
        """
        consistency = {}
        
        # Check high-low consistency (high should always be >= low)
        if 'high' in df.columns and 'low' in df.columns:
            high_low_inconsistent = df[df['high'] < df['low']]
            
            if len(high_low_inconsistent) > 0:
                consistency['high_low_issues'] = {
                    'count': len(high_low_inconsistent),
                    'percentage': (len(high_low_inconsistent) / len(df)) * 100,
                    'examples': high_low_inconsistent.head(10).to_dict('records')
                }
        
        # Check OHLC consistency (open and close should be between high and low)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            open_issues = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
            close_issues = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
            
            if len(open_issues) > 0:
                consistency['open_issues'] = {
                    'count': len(open_issues),
                    'percentage': (len(open_issues) / len(df)) * 100,
                    'examples': open_issues.head(10).to_dict('records')
                }
            
            if len(close_issues) > 0:
                consistency['close_issues'] = {
                    'count': len(close_issues),
                    'percentage': (len(close_issues) / len(df)) * 100,
                    'examples': close_issues.head(10).to_dict('records')
                }
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in df.columns]
        
        for col in available_columns:
            negative_prices = df[df[col] < 0]
            
            if len(negative_prices) > 0:
                consistency[f'negative_{col}'] = {
                    'count': len(negative_prices),
                    'percentage': (len(negative_prices) / len(df)) * 100,
                    'examples': negative_prices.head(10).to_dict('records')
                }
        
        # Check for negative volume
        if 'volume' in df.columns:
            negative_volume = df[df['volume'] < 0]
            
            if len(negative_volume) > 0:
                consistency['negative_volume'] = {
                    'count': len(negative_volume),
                    'percentage': (len(negative_volume) / len(df)) * 100,
                    'examples': negative_volume.head(10).to_dict('records')
                }
            
            # Check for minimum expected volume
            min_volume = self.config_values['min_expected_volume']
            if min_volume > 0:
                low_volume = df[df['volume'] < min_volume]
                
                if len(low_volume) > 0:
                    consistency['low_volume'] = {
                        'count': len(low_volume),
                        'percentage': (len(low_volume) / len(df)) * 100,
                        'min_expected': min_volume,
                        'examples': low_volume.head(10).to_dict('records')
                    }
        
        return consistency
    
    def _generate_visualizations(self, df: pd.DataFrame, date_col_name: str) -> Dict[str, str]:
        """
        Generate visualizations for data quality analysis.

        Args:
            df: Market data DataFrame
            date_col_name: Name of the date column

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        # Check if visualization dependencies are available
        if plt is None:
            logger.warning("Matplotlib is not installed. Visualizations will be skipped.")
            return visualizations
        
        try:
            import matplotlib
            matplotlib.use('Agg') 
            logger.info("Matplotlib backend set to Agg for visualization.")
        except ImportError:
            logger.warning("Could not import matplotlib to set backend. Visualizations might still cause issues.")
        except Exception as e:
            logger.warning(f"Error setting matplotlib backend: {e}. Visualizations might still cause issues.")

        # Determine output directory for visualizations
        # Assuming this script is in src/inspector/modules/data_quality.py
        # Project root would be Path(__file__).parent.parent.parent.parent
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        output_dir = project_root / "output" / "data_quality"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured visualization output directory exists: {output_dir}")
        except Exception as e:
            logger.error(f"Could not create visualization output directory {output_dir}: {e}. Saving to temp.")
            # Fallback to tempfile if directory creation fails
            output_dir = Path(tempfile.gettempdir())

        # Extract symbol and interval from DataFrame or analysis context if possible
        # This relies on the structure of 'df' or having access to the original query parameters
        # For simplicity, we'll try to get it from df if a single symbol/interval is present
        # If multiple symbols, chart names will be more generic or symbol-specific in loops.
        
        # Generate a base filename prefix from symbol and interval
        # df might contain data for multiple symbols if no specific symbol was requested for analysis
        # The `symbol` argument to this function is not directly available here. 
        # We'd need to pass it or derive it. Let's assume `df['symbol']` can be used if unique.
        # Similarly for interval.
        
        # For now, let's try a generic approach. If specific symbol/interval is available in df:
        base_filename_parts = []
        try:
            if 'symbol' in df.columns and df['symbol'].nunique() == 1:
                current_symbol = df['symbol'].iloc[0]
                base_filename_parts.append(str(current_symbol).replace("=", "_").replace("@","S_")) # Sanitize symbol
            
            #Interval info is not directly in df.columns as 'interval_value'/'interval_unit' generally
            # We need to reconstruct it or have it passed. For now, we will omit it from base filename if not easily derived.
            # Or, if it was part of the query that produced df, it might be available from analysis_results if passed.

        except Exception as e:
            logger.warning(f"Could not determine unique symbol/interval for filename: {e}")

        base_file_prefix = "_".join(filter(None, base_filename_parts))
        if not base_file_prefix: # Default if no parts were added
            base_file_prefix = "analysis_plot"

        # Check if we have price data
        price_columns = ['open', 'high', 'low', 'close']
        has_price_data = any(col in df.columns for col in price_columns)

        if not has_price_data:
            return visualizations

        # Set default style if seaborn is available
        if sns is not None:
            sns.set(style="whitegrid")

        # 1. Missing data heatmap
        if df.isna().sum().sum() > 0:
            try:
                plt.figure(figsize=(10, 6))

                # Use seaborn if available, otherwise matplotlib fallback
                if sns is not None:
                    sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
                else:
                    # Basic matplotlib fallback for missing data heatmap
                    plt.imshow(df.isna(), aspect='auto', cmap='viridis')
                    plt.colorbar(label='Missing')

                plt.title('Missing Data Heatmap')
                plt.tight_layout()

                # Construct filename: <output_dir>/<base_file_prefix>_missing_heatmap.png
                filename = output_dir / f"{base_file_prefix}_missing_heatmap.png"
                try:
                    plt.savefig(filename)
                    plt.close()
                    visualizations['missing_data_heatmap'] = str(filename)
                    logger.info(f"Saved missing data heatmap to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save missing_data_heatmap: {e}")
            except Exception as e:
                logger.error(f"Error generating missing data heatmap: {e}")
                logger.error(f"Error details: {str(e)}")

        # 2. Price series with outliers (by symbol if available)
        if 'close' in df.columns:
            try:
                if 'symbol' in df.columns:
                    # Limit to max 5 symbols for readability
                    symbols = df['symbol'].unique()[:5]

                    for symbol in symbols:
                        try:
                            symbol_df = df[df['symbol'] == symbol].sort_values(date_col_name)

                            if len(symbol_df) <= 1:
                                continue

                            # Downsample for plotting if too many points (e.g., > 50,000)
                            plot_df = symbol_df
                            if len(symbol_df) > 50000: # Arbitrary threshold for downsampling
                                logger.info(f"Downsampling data for symbol {symbol} for price plot ({len(symbol_df)} points to daily).")
                                try:
                                    # Ensure date_col_name is DatetimeIndex for resampling
                                    temp_plot_df = symbol_df.set_index(pd.to_datetime(symbol_df[date_col_name]))
                                    # Resample to daily, taking an aggregate (e.g., last close, or ohlc if we want to plot candles)
                                    # For a simple line plot of close, taking the last known close of the day is fine.
                                    resampled_close = temp_plot_df['close'].resample('D').last()
                                    # We need a dataframe with the date_col_name for plotting
                                    plot_df = pd.DataFrame({date_col_name: resampled_close.index, 'close': resampled_close.values}).dropna()
                                except Exception as resample_e:
                                    logger.warning(f"Could not downsample data for {symbol} due to: {resample_e}. Plotting raw data.")
                                    plot_df = symbol_df # Fallback to original if resampling fails

                            plt.figure(figsize=(12, 6))

                            # Plot close price
                            plt.plot(plot_df[date_col_name], plot_df['close'], label='Close Price')

                            # Highlight outliers (use original symbol_df for outlier calculation and coordinates)
                            mean = symbol_df['close'].mean()
                            std = symbol_df['close'].std()

                            if not pd.isna(mean) and not pd.isna(std) and std > 0:
                                threshold = self.config_values['outlier_threshold_std']
                                upper_limit = mean + (std * threshold)
                                lower_limit = mean - (std * threshold)

                                upper_outliers = symbol_df[symbol_df['close'] > upper_limit]
                                lower_outliers = symbol_df[symbol_df['close'] < lower_limit]

                                if len(upper_outliers) > 0:
                                    plt.scatter(upper_outliers[date_col_name], upper_outliers['close'],
                                            color='red', label='Upper Outliers')

                                if len(lower_outliers) > 0:
                                    plt.scatter(lower_outliers[date_col_name], lower_outliers['close'],
                                            color='green', label='Lower Outliers')

                            plt.title(f'Close Price with Outliers - {symbol}')
                            plt.xlabel('Date')
                            plt.ylabel('Price')
                            plt.legend(loc='upper left')
                            plt.tight_layout()

                            # Construct filename: <output_dir>/<symbol>_price_outliers.png
                            symbol_filename_part = str(symbol).replace("=","_").replace("@","S_")
                            filename = output_dir / f"{symbol_filename_part}_price_outliers.png"
                            try:
                                plt.savefig(filename)
                                plt.close()
                                visualizations[f'price_outliers_{symbol}'] = str(filename)
                                logger.info(f"Saved price outliers for {symbol} to {filename}")
                            except Exception as e:
                                logger.error(f"Failed to save price_outliers_{symbol}: {e}")
                        except Exception as symbol_e:
                            logger.error(f"Error processing symbol {symbol}: {symbol_e}")
                else:
                    # Single chart for all data (if no symbols column or only one symbol originally)
                    plot_df = df # Start with the original df
                    if len(df) > 50000: # Arbitrary threshold
                        logger.info(f"Downsampling data for general price plot ({len(df)} points to daily).")
                        try:
                            temp_plot_df = df.set_index(pd.to_datetime(df[date_col_name]))
                            resampled_close = temp_plot_df['close'].resample('D').last()
                            plot_df = pd.DataFrame({date_col_name: resampled_close.index, 'close': resampled_close.values}).dropna()
                        except Exception as resample_e:
                            logger.warning(f"Could not downsample general data due to: {resample_e}. Plotting raw data.")
                            plot_df = df # Fallback
                    
                    plt.figure(figsize=(12, 6))

                    # Plot close price
                    plt.plot(plot_df[date_col_name], plot_df['close'], label='Close Price')

                    # Highlight outliers (use original df for outlier calculation and coordinates)
                    mean = df['close'].mean()
                    std = df['close'].std()

                    if not pd.isna(mean) and not pd.isna(std) and std > 0:
                        threshold = self.config_values['outlier_threshold_std']
                        upper_limit = mean + (std * threshold)
                        lower_limit = mean - (std * threshold)

                        upper_outliers = df[df['close'] > upper_limit]
                        lower_outliers = df[df['close'] < lower_limit]

                        if len(upper_outliers) > 0:
                            plt.scatter(upper_outliers[date_col_name], upper_outliers['close'],
                                      color='red', label='Upper Outliers')

                        if len(lower_outliers) > 0:
                            plt.scatter(lower_outliers[date_col_name], lower_outliers['close'],
                                      color='green', label='Lower Outliers')

                    plt.title('Close Price with Outliers')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend(loc='upper left')
                    plt.tight_layout()

                    # Construct filename: <output_dir>/<base_file_prefix>_price_outliers.png
                    filename = output_dir / f"{base_file_prefix}_price_outliers.png"
                    try:
                        plt.savefig(filename)
                        plt.close()
                        visualizations['price_outliers'] = str(filename)
                        logger.info(f"Saved general price outliers to {filename}")
                    except Exception as e:
                        logger.error(f"Failed to save general price_outliers: {e}")
            except Exception as e:
                logger.error(f"Error generating price series visualization: {e}")

        # 3. Volume histogram (if available)
        if 'volume' in df.columns:
            try:
                plt.figure(figsize=(10, 6))

                # Use seaborn if available, otherwise matplotlib fallback
                if sns is not None:
                    sns.histplot(df['volume'], kde=True)
                else:
                    plt.hist(df['volume'].dropna(), bins=30)

                plt.title('Volume Distribution')
                plt.xlabel('Volume')
                plt.ylabel('Frequency')
                plt.yscale('log')
                
                # Set x-axis limits to 99.5th percentile for better visibility of main distribution
                if 'volume' in df.columns and not df['volume'].empty:
                    max_vol_display = df['volume'].quantile(0.995) 
                    if pd.notna(max_vol_display) and max_vol_display > 0:
                        plt.xlim(-0.01 * max_vol_display, max_vol_display) # Start slightly before 0 for visual
                        plt.title('Volume Distribution (up to 99.5th percentile)') # Update title
                    else:
                        plt.title('Volume Distribution') # Default title if percentile is problematic
                else:
                    plt.title('Volume Distribution')

                plt.tight_layout()

                # Construct filename: <output_dir>/<base_file_prefix>_volume_histogram.png
                filename = output_dir / f"{base_file_prefix}_volume_histogram.png"
                try:
                    plt.savefig(filename)
                    plt.close()
                    visualizations['volume_histogram'] = str(filename)
                    logger.info(f"Saved volume histogram to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save volume_histogram: {e}")
            except Exception as e:
                logger.error(f"Error generating volume histogram: {e}")

        # 4. Returns distribution (if close price available)
        if 'close' in df.columns and 'symbol' in df.columns and sns is not None:
            # This visualization requires seaborn
            try:
                plt.figure(figsize=(10, 6))

                # Calculate returns by symbol
                all_returns = []

                for symbol, group in df.groupby('symbol'):
                    if len(group) <= 1:
                        continue

                    # Sort and calculate returns
                    sorted_group = group.sort_values(date_col_name)
                    returns = sorted_group['close'].pct_change().dropna()

                    if len(returns) > 0:
                        returns_df = pd.DataFrame({
                            'symbol': symbol,
                            'returns': returns.values
                        })
                        all_returns.append(returns_df)

                if all_returns:
                    returns_df = pd.concat(all_returns)

                    # Plot returns distribution
                    if returns_df['symbol'].nunique() > 1:
                        sns.histplot(data=returns_df, x='returns', hue='symbol', kde=True, common_norm=False)
                    else:
                        sns.histplot(data=returns_df, x='returns', kde=True)
                        
                    plt.title('Returns Distribution by Symbol' if returns_df['symbol'].nunique() > 1 else 'Returns Distribution')
                    plt.xlabel('Returns')
                    plt.ylabel('Frequency')
                    plt.yscale('log') # Add log scale to Y-axis for returns

                    # Set x-axis limits based on percentiles to see tails better
                    if not returns_df['returns'].empty:
                        lower_percentile = returns_df['returns'].quantile(0.001) # 0.1th percentile
                        upper_percentile = returns_df['returns'].quantile(0.999) # 99.9th percentile
                        # Add a small buffer or ensure range is not zero
                        if lower_percentile == upper_percentile:
                            padding = 0.01 * abs(lower_percentile) if lower_percentile != 0 else 0.01
                            lower_percentile -= padding
                            upper_percentile += padding
                        plt.xlim(lower_percentile, upper_percentile)

                    plt.legend(loc='upper right')
                    plt.tight_layout()

                    # Construct filename: <output_dir>/<base_file_prefix>_returns_dist.png
                    filename = output_dir / f"{base_file_prefix}_returns_dist.png"
                    try:
                        plt.savefig(filename)
                        plt.close()
                        visualizations['returns_distribution'] = str(filename)
                        logger.info(f"Saved returns distribution to {filename}")
                    except Exception as e:
                        logger.error(f"Failed to save returns_distribution: {e}")
            except Exception as e:
                logger.error(f"Error generating returns distribution: {e}")

        return visualizations
    
    def display_analysis_results(self, results: Dict[str, Any]) -> None:
        """
        Display data quality analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        if not results.get('success', False):
            self.console.print(f"[bold red]Analysis failed:[/bold red] {results.get('message', 'Unknown error')}")
            return
        
        # Print summary
        self.console.print(f"[bold cyan]Data Quality Analysis Summary[/bold cyan]")
        self.console.print(f"Symbol: {results.get('symbol', 'All symbols')}")
        
        if results.get('date_range'):
            self.console.print(f"Date Range: {results['date_range'][0]} to {results['date_range'][1]}")
        
        self.console.print(f"Total Rows: {results['row_count']:,}")
        
        # Print statistics
        stats = results.get('statistics', {})
        
        if 'unique_symbols' in stats:
            self.console.print(f"\n[bold cyan]Symbols:[/bold cyan] {stats['unique_symbols']} unique symbols")
        
        # Print OHLCV statistics
        ohlcv_table = Table(title="OHLCV Statistics", box=SIMPLE)
        ohlcv_table.add_column("Measure")
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if f'{col}_stats' in stats:
                ohlcv_table.add_column(col.capitalize())
        
        for measure in ['min', 'max', 'mean', 'median', 'std', 'zeros']:
            row = [measure.capitalize()]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if f'{col}_stats' in stats:
                    value = stats[f'{col}_stats'].get(measure, 'N/A')
                    
                    if isinstance(value, (int, float)):
                        if measure == 'zeros' and value > 0:
                            row.append(f"[bold red]{value:,}[/bold red]")
                        elif col == 'volume' and measure in ['min', 'max', 'mean', 'median']:
                            row.append(f"{value:,.0f}")
                        else:
                            row.append(f"{value:,.6g}")
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            
            ohlcv_table.add_row(*row)
        
        self.console.print(ohlcv_table)
        
        # Print missing data summary
        missing = results.get('missing_data', {})
        
        if missing:
            overall = missing.get('overall', {})
            
            if overall:
                self.console.print(f"\n[bold cyan]Missing Data:[/bold cyan] {overall.get('missing_cells', 0):,} cells ({overall.get('missing_percentage', 0):.2f}% of all data)")
            
            # Print missing by column if significant
            by_column = missing.get('by_column', {})
            
            if by_column:
                columns_with_missing = {col: info for col, info in by_column.items() 
                                     if info.get('missing_count', 0) > 0}
                
                if columns_with_missing:
                    missing_table = Table(title="Missing Data by Column", box=SIMPLE)
                    missing_table.add_column("Column")
                    missing_table.add_column("Missing Count")
                    missing_table.add_column("Missing %")
                    
                    for col, info in columns_with_missing.items():
                        missing_table.add_row(
                            col,
                            f"{info.get('missing_count', 0):,}",
                            f"{info.get('missing_percentage', 0):.2f}%"
                        )
                    
                    self.console.print(missing_table)
            
            # Print time series gaps
            gaps = missing.get('time_series_gaps', {})
            
            if gaps:
                self.console.print(f"\n[bold cyan]Time Series Gaps:[/bold cyan] Found in {len(gaps)} symbols")
                
                for symbol, symbol_gaps_data in list(gaps.items())[:3]:  # Limit to 3 symbols for readability
                    actual_gaps_list = []
                    if isinstance(symbol_gaps_data, dict): # Old: Gaps by interval from DataFrame column
                        for interval_key, interval_specific_gaps in symbol_gaps_data.items():
                            if interval_specific_gaps:
                                actual_gaps_list.extend(interval_specific_gaps)
                    elif isinstance(symbol_gaps_data, list): # New: Direct list of gaps for the query interval
                        actual_gaps_list = symbol_gaps_data
                    
                    if actual_gaps_list:
                        # Sort gaps by duration (descending) to show largest first
                        # Ensure 'gap_duration_seconds' exists and is numeric for sorting
                        actual_gaps_list.sort(key=lambda x: x.get('gap_duration_seconds', 0) if isinstance(x.get('gap_duration_seconds'), (int, float)) else 0, reverse=True)
                        
                        gap_table_title = f"Largest Gaps for {symbol}"
                        if isinstance(symbol_gaps_data, dict) and len(symbol_gaps_data) == 1:
                             # If it was from old structure with one interval key
                            interval_key = list(symbol_gaps_data.keys())[0]
                            gap_table_title = f"Largest Gaps for {symbol} (Interval: {interval_key})"
                        
                        gap_table = Table(title=gap_table_title, box=SIMPLE)
                        gap_table.add_column("Start Date")
                        gap_table.add_column("End Date")
                        gap_table.add_column("Duration")
                        gap_table.add_column("Missing Pts (est)")
                        
                        for gap in actual_gaps_list[:10]:  # Show top 10 largest gaps
                            duration_val = gap.get('gap_duration_seconds', 'N/A')
                            formatted_duration = _format_duration(duration_val) if duration_val != 'N/A' else 'N/A'
                            gap_table.add_row(
                                str(gap.get('start_date', '')),
                                str(gap.get('end_date', '')),
                                formatted_duration,
                                str(gap.get('missing_points_estimated', ''))
                            )
                        self.console.print(gap_table)
                    else:
                        self.console.print(f"No gaps found or reported for {symbol} after processing.")
        
        # Print outliers
        outliers = results.get('outliers', {})
        
        if outliers:
            by_symbol = outliers.get('by_symbol', {})
            
            if by_symbol:
                total_outliers = sum(
                    info.get('upper_outliers_count', 0) + info.get('lower_outliers_count', 0)
                    for symbol_data in by_symbol.values()
                    for info in symbol_data.values()
                )
                
                self.console.print(f"\n[bold cyan]Outliers:[/bold cyan] Found {total_outliers} price outliers across {len(by_symbol)} symbols")
                
                # Show examples of outliers
                for symbol, symbol_outliers in list(by_symbol.items())[:2]:  # Limit to 2 symbols
                    for col, info in list(symbol_outliers.items())[:2]:  # Limit to 2 columns
                        if info.get('upper_outliers_count', 0) > 0 or info.get('lower_outliers_count', 0) > 0:
                            self.console.print(f"\n[bold yellow]{symbol} {col}:[/bold yellow] {info.get('upper_outliers_count', 0)} high outliers, {info.get('lower_outliers_count', 0)} low outliers")
                            self.console.print(f"Mean: {info.get('mean', 0):.6g}, Std: {info.get('std', 0):.6g}")
                            self.console.print(f"Limits: {info.get('lower_limit', 0):.6g} to {info.get('upper_limit', 0):.6g}")
            
            # Zero prices
            zero_prices = outliers.get('zero_prices', {})
            
            if zero_prices:
                zero_table = Table(title="Zero Prices", box=SIMPLE)
                zero_table.add_column("Column")
                zero_table.add_column("Count")
                zero_table.add_column("Percentage")
                
                for col, info in zero_prices.items():
                    zero_table.add_row(
                        col,
                        f"{info.get('count', 0):,}",
                        f"{info.get('percentage', 0):.2f}%"
                    )
                
                self.console.print(zero_table)
            
            # Extreme returns
            extreme_returns = outliers.get('extreme_returns', {})
            
            if extreme_returns:
                total_extreme = sum(
                    info.get('extreme_up_count', 0) + info.get('extreme_down_count', 0)
                    for info in extreme_returns.values()
                )
                
                self.console.print(f"\n[bold cyan]Extreme Returns:[/bold cyan] Found {total_extreme} instances across {len(extreme_returns)} symbols")
        
        # Print consistency issues
        consistency = results.get('consistency', {})
        
        if consistency:
            total_issues = sum(info.get('count', 0) for info in consistency.values())
            
            if total_issues > 0:
                self.console.print(f"\n[bold red]Consistency Issues:[/bold red] Found {total_issues} issues")
                
                # Create issue table
                issue_table = Table(title="Data Consistency Issues", box=SIMPLE)
                issue_table.add_column("Issue Type")
                issue_table.add_column("Count")
                issue_table.add_column("Percentage")
                
                for issue_type, info in consistency.items():
                    issue_table.add_row(
                        issue_type.replace('_', ' ').title(),
                        f"{info.get('count', 0):,}",
                        f"{info.get('percentage', 0):.2f}%"
                    )
                
                self.console.print(issue_table)
        
        # Print visualization paths
        visualizations = results.get('visualizations', {})
        
        if visualizations:
            self.console.print(f"\n[bold cyan]Visualizations:[/bold cyan] Generated {len(visualizations)} charts")
            
            for name, path in visualizations.items():
                self.console.print(f"- {name}: {path}")
    
    def interactive_analyzer(self) -> None:
        """Run interactive data quality analyzer."""
        self.console.print("\nEntering interactive data quality analyzer. Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                # Get command from user
                command_str = self.console.input("\n[bold]quality>[/bold] ").strip()
                command_parts = command_str.split()
                command = command_parts[0].lower() if command_parts else ""

                if command in ('exit', 'quit'):
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'analyze':
                    # --- Contract Type Selection ---
                    contract_type_prompt = "Analyze [1] Individual Contract or [2] Continuous Contract?"
                    contract_choice = Prompt.ask(contract_type_prompt, choices=["1", "2"], default="1").strip()
                    is_continuous = contract_choice == "2"
                    
                    if is_continuous:
                        symbol_prompt = "Enter continuous contract symbol (e.g., @ES=102XC, @VX=101XN):"
                    else:
                        symbol_prompt = "Enter individual contract symbol (e.g., ESU23, NQZ23):"
                    symbol = Prompt.ask(symbol_prompt, default=None)
                    
                    # --- Date Range Selection ---
                    start_date_str = Prompt.ask("Enter start date (YYYY-MM-DD, optional)", default=None)
                    end_date_str = Prompt.ask("Enter end date (YYYY-MM-DD, optional)", default=None)
                    
                    # --- Interval Selection ---
                    interval_prompt = "Enter data interval (e.g., 1D, 1H, 30Min, optional, default: 1D for individual, best available for continuous)"
                    default_interval = None if is_continuous else "1D" # Continuous might infer best, individual defaults to Daily
                    interval = Prompt.ask(interval_prompt, default=default_interval)

                    # Validate and parse dates
                    start_date, end_date = None, None
                    if start_date_str:
                        try:
                            datetime.strptime(start_date_str, "%Y-%m-%d")
                            start_date = start_date_str
                        except ValueError:
                            self.console.print(f"[bold red]Invalid start date format: {start_date_str}. Please use YYYY-MM-DD.[/bold red]")
                            continue
                    if end_date_str:
                        try:
                            datetime.strptime(end_date_str, "%Y-%m-%d")
                            end_date = end_date_str
                        except ValueError:
                            self.console.print(f"[bold red]Invalid end date format: {end_date_str}. Please use YYYY-MM-DD.[/bold red]")
                            continue
                    
                    if start_date and end_date and start_date > end_date:
                        self.console.print("[bold red]Start date cannot be after end date.[/bold red]")
                        continue

                    # Run analysis (Pass is_continuous to the analysis function)
                    # self.console.print(f"Analyzing: Symbol={symbol}, Start={start_date}, End={end_date}, Interval={interval}, Continuous={is_continuous}")
                    results = self.analyze_market_data_table(
                        symbol=symbol, 
                        start_date=start_date, 
                        end_date=end_date, 
                        interval=interval,
                        is_continuous=is_continuous # Pass the new flag
                    )
                    self.display_analysis_results(results)

                elif command == 'config':
                    if len(command_parts) == 1:
                        self._show_config()
                    elif len(command_parts) >= 3:
                        key = command_parts[1]
                        value = " ".join(command_parts[2:]) # Allow spaces in value
                        
                        if key in self.config_values:
                            try:
                                current_type = type(self.config_values[key])
                                if current_type == bool:
                                    new_value = value.lower() in ('true', 'yes', '1')
                                elif current_type == int:
                                    new_value = int(value)
                                elif current_type == float:
                                    new_value = float(value)
                                else: # str or other
                                    new_value = value
                                self.config_values[key] = new_value
                                self.console.print(f"[green]Set {key} = {self.config_values[key]}[/green]")
                            except ValueError:
                                self.console.print(f"[bold red]Invalid value '{value}' for {key} (expected {current_type.__name__}).[/bold red]")
                        else:
                            self.console.print(f"[bold red]Unknown config key: {key}[/bold red]")
                    else:
                        self.console.print("Usage: config <key> <value> (or 'config' to show current)")
                elif command == 'reset':
                    self.config_values = self.default_config.copy()
                    self.console.print("[green]Configuration reset to defaults.[/green]")
                else:
                    self.console.print(f"[bold red]Unknown command: {command_str}[/bold red]")
                    self._show_help()
            
            except KeyboardInterrupt:
                self.console.print("\nOperation cancelled by user.")
                break
            except EOFError: # Handle Ctrl+D
                self.console.print("\nExiting data quality analyzer.")
                break
            except Exception as e:
                logger.error(f"Error in interactive analyzer: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        
        self.console.print("\nExiting data quality analyzer.")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        Available commands:
        
        analyze                - Interactively analyze market data quality.
                                 You will be prompted for contract type (individual/continuous),
                                 symbol, date range, and interval.
        config <key> <value>   - Set a specific configuration value.
        config                 - Show the current data quality configuration.
        reset                  - Reset all configuration values to their defaults.
        help                   - Show this help message.
        exit                   - Exit the data quality analyzer.
        
        Examples for config:
        config outlier_threshold_std 4.0
                               - Set outlier threshold to 4.0 standard deviations.
        config zero_price_detection false
                               - Disable zero price detection.
        """
        
        self.console.print(Panel(help_text, title="Data Quality Analyzer Help", expand=False))
    
    def _show_config(self) -> None:
        """Show current configuration."""
        config_table = Table(title="Current Configuration", box=SIMPLE)
        config_table.add_column("Setting")
        config_table.add_column("Value")
        config_table.add_column("Description")
        
        descriptions = {
            'outlier_threshold_std': "Standard deviations for outlier detection",
            'zero_price_detection': "Detect zero prices as data errors",
            'min_expected_volume': "Minimum expected volume (0 = no check)",
            'gap_detection_days': "Maximum allowed gap in days",
            'market_hours_check': "Check if data is within market hours"
        }
        
        for key, value in self.config_values.items():
            config_table.add_row(
                key,
                str(value),
                descriptions.get(key, "")
            )
        
        self.console.print(config_table)

def _format_duration(seconds: Union[float, int]) -> str:
    if not isinstance(seconds, (float, int)) or seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0s"

    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, sec = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{int(days)}d")
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if sec > 0 or not parts: # show seconds if it's the only unit or if there's a remainder
        parts.append(f"{int(sec)}s")
    
    return " ".join(parts)

# Global instance
quality_analyzer = None

def get_quality_analyzer() -> DataQualityAnalyzer:
    """
    Get the global data quality analyzer instance.
    
    Returns:
        Global data quality analyzer instance
    """
    global quality_analyzer
    
    if quality_analyzer is None:
        quality_analyzer = DataQualityAnalyzer()
        
    return quality_analyzer