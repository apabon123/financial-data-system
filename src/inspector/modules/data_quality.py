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

        self.app = get_app()
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
                               interval: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis on market data.

        Args:
            symbol: Symbol to analyze (optional)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            Dictionary with analysis results
        """
        # Check if analysis is possible
        if not getattr(self, 'can_analyze', False):
            return {
                'success': False,
                'message': 'Required dependencies (pandas, numpy) are missing',
                'error': 'Install required dependencies with: pip install pandas numpy'
            }

        try:
            # Build query based on filters
            query = "SELECT * FROM market_data WHERE 1=1"

            if symbol:
                query += f" AND symbol = '{symbol}'"

            if start_date:
                query += f" AND date >= '{start_date}'"

            if end_date:
                query += f" AND date <= '{end_date}'"

            if interval:
                query += f" AND interval = '{interval}'"

            query += " ORDER BY symbol, date, interval"

            # Execute query
            try:
                result = self.app.db_manager.execute_query(query)
            except Exception as e:
                logger.error(f"Database query error: {e}")
                return {
                    'success': False,
                    'message': 'Error executing database query',
                    'error': str(e)
                }

            if not result.is_success or result.is_empty:
                return {
                    'success': False,
                    'message': 'No data found matching criteria',
                    'error': result.error if not result.is_success else None
                }

            # Get data as DataFrame
            df = result.dataframe

            # Begin analysis
            try:
                with Progress() as progress:
                    analysis_task = progress.add_task("Analyzing data quality...", total=5)

                    # 1. Basic statistics
                    progress.update(analysis_task, advance=1, description="Calculating basic statistics...")
                    stats = self._calculate_statistics(df)

                    # 2. Missing data analysis
                    progress.update(analysis_task, advance=1, description="Analyzing missing data...")
                    missing_data = self._analyze_missing_data(df)

                    # 3. Outlier detection
                    progress.update(analysis_task, advance=1, description="Detecting outliers...")
                    outliers = self._detect_outliers(df)

                    # 4. Time series consistency
                    progress.update(analysis_task, advance=1, description="Checking time series consistency...")
                    consistency = self._check_time_series_consistency(df)

                    # 5. Generate visualizations (if possible)
                    progress.update(analysis_task, advance=1, description="Generating visualizations...")
                    visualizations = {}
                    if getattr(self, 'can_visualize', False):
                        visualizations = self._generate_visualizations(df)
                    else:
                        logger.warning("Visualizations skipped due to missing dependencies")
            except Exception as e:
                # If analysis fails, return error with partial results if available
                logger.error(f"Analysis error: {e}")
                return {
                    'success': False,
                    'message': f'Error during analysis: {str(e)}',
                    'error': str(e),
                    'partial_results': {
                        'symbol': symbol,
                        'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None,
                        'row_count': len(df),
                        'statistics': stats if 'stats' in locals() else {},
                    }
                }

            # Return comprehensive results
            return {
                'success': True,
                'symbol': symbol,
                'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None,
                'row_count': len(df),
                'statistics': stats,
                'missing_data': missing_data,
                'outliers': outliers,
                'consistency': consistency,
                'visualizations': visualizations
            }
        except Exception as e:
            logger.error(f"Unexpected error in data analysis: {e}")
            return {
                'success': False,
                'message': f'Unexpected error in data analysis: {str(e)}',
                'error': str(e)
            }
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics for market data.
        
        Args:
            df: Market data DataFrame
            
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
                    'date_range': [group['date'].min(), group['date'].max()] if 'date' in group.columns else None,
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
            if 'date' in df.columns:
                df_sorted = df.sort_values(['symbol', 'date'])
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
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data in market data.
        
        Args:
            df: Market data DataFrame
            
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
        if 'date' in df.columns and 'symbol' in df.columns:
            missing['time_series_gaps'] = {}
            
            for symbol, group in df.groupby('symbol'):
                if len(group) <= 1:
                    continue
                
                # Try to analyze gaps by interval
                if 'interval' in df.columns:
                    interval_gaps = {}
                    
                    for interval, interval_group in group.groupby('interval'):
                        gaps = self._find_date_gaps(interval_group, interval)
                        if gaps:
                            interval_gaps[interval] = gaps
                    
                    if interval_gaps:
                        missing['time_series_gaps'][symbol] = interval_gaps
                else:
                    # Analyze without interval
                    gaps = self._find_date_gaps(group)
                    if gaps:
                        missing['time_series_gaps'][symbol] = gaps
        
        return missing
    
    def _find_date_gaps(self, df: pd.DataFrame, interval: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find gaps in date sequence.
        
        Args:
            df: DataFrame for a single symbol
            interval: Data interval
            
        Returns:
            List of gap dictionaries
        """
        # Sort by date
        df_sorted = df.sort_values('date')
        dates = pd.to_datetime(df_sorted['date'])
        
        gaps = []
        
        # Determine expected frequency based on interval
        if interval and 'day' in interval.lower():
            freq = 'B'  # Business days
        elif interval and 'hour' in interval.lower():
            freq = 'H'  # Hourly
        elif interval and 'minute' in interval.lower() or interval and 'min' in interval.lower():
            # Try to extract number of minutes
            try:
                minutes = int(''.join(filter(str.isdigit, interval)))
                freq = f'{minutes}min'
            except:
                freq = 'min'
        else:
            # Default to daily
            freq = 'B'
        
        # Generate expected date range
        if len(dates) >= 2:
            expected_dates = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
            
            # Find missing dates
            missing_dates = set(expected_dates) - set(dates)
            
            if missing_dates:
                # Group consecutive missing dates
                missing_dates_sorted = sorted(missing_dates)
                gap_start = missing_dates_sorted[0]
                current_gap = [gap_start]
                
                for i in range(1, len(missing_dates_sorted)):
                    if missing_dates_sorted[i] - missing_dates_sorted[i-1] <= pd.Timedelta(days=1):
                        current_gap.append(missing_dates_sorted[i])
                    else:
                        # End of gap, record it
                        gaps.append({
                            'start_date': current_gap[0],
                            'end_date': current_gap[-1],
                            'gap_days': (current_gap[-1] - current_gap[0]).days + 1,
                            'missing_points': len(current_gap)
                        })
                        
                        # Start new gap
                        gap_start = missing_dates_sorted[i]
                        current_gap = [gap_start]
                
                # Don't forget the last gap
                if current_gap:
                    gaps.append({
                        'start_date': current_gap[0],
                        'end_date': current_gap[-1],
                        'gap_days': (current_gap[-1] - current_gap[0]).days + 1,
                        'missing_points': len(current_gap)
                    })
        
        return gaps
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in market data.
        
        Args:
            df: Market data DataFrame
            
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
                sorted_group = group.sort_values('date')
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
    
    def _check_time_series_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check time series consistency in market data.
        
        Args:
            df: Market data DataFrame
            
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
    
    def _generate_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate visualizations for data quality analysis.

        Args:
            df: Market data DataFrame

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        # Check if visualization dependencies are available
        if plt is None:
            logger.warning("Matplotlib is not installed. Visualizations will be skipped.")
            return visualizations

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

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    visualizations['missing_data_heatmap'] = tmp.name
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
                            symbol_df = df[df['symbol'] == symbol].sort_values('date')

                            if len(symbol_df) <= 1:
                                continue

                            plt.figure(figsize=(12, 6))

                            # Plot close price
                            plt.plot(symbol_df['date'], symbol_df['close'], label='Close Price')

                            # Highlight outliers
                            mean = symbol_df['close'].mean()
                            std = symbol_df['close'].std()

                            if not pd.isna(mean) and not pd.isna(std) and std > 0:
                                threshold = self.config_values['outlier_threshold_std']
                                upper_limit = mean + (std * threshold)
                                lower_limit = mean - (std * threshold)

                                upper_outliers = symbol_df[symbol_df['close'] > upper_limit]
                                lower_outliers = symbol_df[symbol_df['close'] < lower_limit]

                                if len(upper_outliers) > 0:
                                    plt.scatter(upper_outliers['date'], upper_outliers['close'],
                                            color='red', label='Upper Outliers')

                                if len(lower_outliers) > 0:
                                    plt.scatter(lower_outliers['date'], lower_outliers['close'],
                                            color='green', label='Lower Outliers')

                            plt.title(f'Close Price with Outliers - {symbol}')
                            plt.xlabel('Date')
                            plt.ylabel('Price')
                            plt.legend()
                            plt.tight_layout()

                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                plt.savefig(tmp.name)
                                plt.close()
                                visualizations[f'price_outliers_{symbol}'] = tmp.name
                        except Exception as symbol_e:
                            logger.error(f"Error processing symbol {symbol}: {symbol_e}")
                else:
                    # Single chart for all data
                    plt.figure(figsize=(12, 6))

                    # Plot close price
                    plt.plot(df['date'], df['close'], label='Close Price')

                    # Highlight outliers
                    mean = df['close'].mean()
                    std = df['close'].std()

                    if not pd.isna(mean) and not pd.isna(std) and std > 0:
                        threshold = self.config_values['outlier_threshold_std']
                        upper_limit = mean + (std * threshold)
                        lower_limit = mean - (std * threshold)

                        upper_outliers = df[df['close'] > upper_limit]
                        lower_outliers = df[df['close'] < lower_limit]

                        if len(upper_outliers) > 0:
                            plt.scatter(upper_outliers['date'], upper_outliers['close'],
                                      color='red', label='Upper Outliers')

                        if len(lower_outliers) > 0:
                            plt.scatter(lower_outliers['date'], lower_outliers['close'],
                                      color='green', label='Lower Outliers')

                    plt.title('Close Price with Outliers')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.tight_layout()

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        plt.savefig(tmp.name)
                        plt.close()
                        visualizations['price_outliers'] = tmp.name
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
                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    visualizations['volume_histogram'] = tmp.name
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
                    sorted_group = group.sort_values('date')
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
                    sns.histplot(data=returns_df, x='returns', hue='symbol', kde=True, common_norm=False)
                    plt.title('Returns Distribution by Symbol')
                    plt.xlabel('Returns')
                    plt.ylabel('Frequency')
                    plt.tight_layout()

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        plt.savefig(tmp.name)
                        plt.close()
                        visualizations['returns_distribution'] = tmp.name
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
                
                for symbol, symbol_gaps in list(gaps.items())[:3]:  # Limit to 3 symbols for readability
                    if isinstance(symbol_gaps, dict):
                        # Gaps by interval
                        for interval, interval_gaps in symbol_gaps.items():
                            if interval_gaps:
                                gap_table = Table(title=f"Gaps for {symbol} ({interval})", box=SIMPLE)
                                gap_table.add_column("Start Date")
                                gap_table.add_column("End Date")
                                gap_table.add_column("Gap Days")
                                gap_table.add_column("Missing Points")
                                
                                for gap in interval_gaps[:5]:  # Limit to 5 gaps
                                    gap_table.add_row(
                                        str(gap.get('start_date', '')),
                                        str(gap.get('end_date', '')),
                                        str(gap.get('gap_days', '')),
                                        str(gap.get('missing_points', ''))
                                    )
                                
                                self.console.print(gap_table)
                    else:
                        # Direct list of gaps
                        if symbol_gaps:
                            gap_table = Table(title=f"Gaps for {symbol}", box=SIMPLE)
                            gap_table.add_column("Start Date")
                            gap_table.add_column("End Date")
                            gap_table.add_column("Gap Days")
                            gap_table.add_column("Missing Points")
                            
                            for gap in symbol_gaps[:5]:  # Limit to 5 gaps
                                gap_table.add_row(
                                    str(gap.get('start_date', '')),
                                    str(gap.get('end_date', '')),
                                    str(gap.get('gap_days', '')),
                                    str(gap.get('missing_points', ''))
                                )
                            
                            self.console.print(gap_table)
        
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
        print("\nEntering interactive data quality analyzer. Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                # Get command from user
                command = input("\nquality> ").strip()
                
                if command.lower() in ('exit', 'quit'):
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.lower().startswith('analyze'):
                    # Parse parameters
                    parts = command.split()
                    symbol = None
                    start_date = None
                    end_date = None
                    interval = None
                    
                    # Look for parameters
                    for i, part in enumerate(parts):
                        if part == '-s' and i + 1 < len(parts):
                            symbol = parts[i + 1]
                        elif part == '-d' and i + 2 < len(parts):
                            start_date = parts[i + 1]
                            end_date = parts[i + 2]
                        elif part == '-i' and i + 1 < len(parts):
                            interval = parts[i + 1]
                    
                    # Run analysis
                    results = self.analyze_market_data_table(symbol, start_date, end_date, interval)
                    self.display_analysis_results(results)
                elif command.lower().startswith('config'):
                    # Parse config settings
                    parts = command.split()
                    
                    if len(parts) == 1:
                        # Show current config
                        self._show_config()
                    elif len(parts) >= 3:
                        # Update config
                        key = parts[1]
                        value = parts[2]
                        
                        if key in self.config_values:
                            # Convert to appropriate type
                            try:
                                if isinstance(self.config_values[key], bool):
                                    self.config_values[key] = value.lower() in ('true', 'yes', '1')
                                elif isinstance(self.config_values[key], int):
                                    self.config_values[key] = int(value)
                                elif isinstance(self.config_values[key], float):
                                    self.config_values[key] = float(value)
                                else:
                                    self.config_values[key] = value
                                
                                print(f"Set {key} = {self.config_values[key]}")
                            except ValueError:
                                print(f"Invalid value for {key}: {value}")
                        else:
                            print(f"Unknown config key: {key}")
                    else:
                        print("Usage: config <key> <value>")
                elif command.lower() == 'reset':
                    # Reset config to defaults
                    self.config_values = self.default_config.copy()
                    print("Reset configuration to defaults")
                else:
                    print(f"Unknown command: {command}")
                    self._show_help()
            
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting data quality analyzer.")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        Available commands:
        
        analyze [-s SYMBOL] [-d START_DATE END_DATE] [-i INTERVAL]
                               - Analyze market data quality
        config <key> <value>   - Set configuration value
        config                 - Show current configuration
        reset                  - Reset configuration to defaults
        help                   - Show this help
        exit                   - Exit data quality analyzer
        
        Examples:
        
        analyze                - Analyze all market data
        analyze -s ES          - Analyze data for ES symbol
        analyze -d 2023-01-01 2023-03-31
                               - Analyze data within date range
        analyze -s ES -i DAY   - Analyze daily data for ES
        config outlier_threshold_std 4.0
                               - Set outlier threshold to 4 standard deviations
        """
        
        print(help_text)
    
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