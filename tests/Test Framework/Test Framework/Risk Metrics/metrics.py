import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List

from ..config import Config, LoggingConfig # Relative import
from ..models import MetricType, MetricResult, BaseReturns, LeveredReturns # Relative import

class RiskMetrics:
    \"\"\"Calculates and manages risk metrics for trading strategies.\"\"\"

    def __init__(self, config: Config):
        self.logger = LoggingConfig.get_logger(__name__)
        self.logger.info("Initializing RiskMetrics") # Added init log

        # Initialize with config attribute access
        # Get risk_metrics settings from config
        rm_config = config.risk_metrics if hasattr(config, 'risk_metrics') else {}
        self.windows = rm_config.get('windows', {
            'short_term': 21,
            'medium_term': 63,
            'long_term': 252
        }) # Provide default windows
        self.min_periods = rm_config.get('min_periods', 5)
        self.required_windows = rm_config.get('required_windows', ['short', 'medium'])
        self.risk_free_rate = rm_config.get('risk_free_rate', 0.02)

        self._validate_config()
        self._reset_caches()

    def _validate_required_metrics(self, metrics: pd.DataFrame) -> None:
        \"\"\"
        Validate that all required metrics are present and valid.

        Args:
            metrics: DataFrame of calculated metrics

        Raises:
            ValueError: If required metrics are missing or invalid
        \"\"\"
        if metrics is None or metrics.empty:
            raise ValueError("Metrics DataFrame is empty or None, cannot validate.")

        for window in self.required_windows:
            required_metric_names = [
                f'vol_{window}',
                f'return_{window}',
                f'sharpe_{window}'
            ]

            missing = [m for m in required_metric_names if m not in metrics.columns]
            if missing:
                raise ValueError(f"Missing required metrics in DataFrame: {missing}")

            # Check for invalid values (all NaN or infinite)
            for metric in required_metric_names:
                if metrics[metric].isnull().all():
                    raise ValueError(f"Required metric '{metric}' contains only NaN values.")
                if np.isinf(metrics[metric]).any():
                     self.logger.warning(f"Metric '{metric}' contains infinite values.") # Warn, don't raise

        self.logger.debug("Required metrics validation successful.")

    def _validate_config(self) -> None:
        \"\"\"Validate configuration parameters.\"\"\"
        if not isinstance(self.windows, dict):
             raise ValueError("'windows' configuration must be a dictionary.")

        # Validate window sizes
        for window_name, size in self.windows.items():
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Window size must be a positive integer: {window_name}={size}")
            if size < self.min_periods:
                raise ValueError(
                    f"Window size ({size} for {window_name}) cannot be smaller than min_periods ({self.min_periods})"
                )

        # Validate required windows exist
        window_keys = set(name.split('_')[0] for name in self.windows.keys())
        missing_windows = set(self.required_windows) - window_keys
        if missing_windows:
            raise ValueError(f"Required window types not found in configuration: {missing_windows}")

        # Validate risk-free rate
        if not isinstance(self.risk_free_rate, (int, float)) or not (0 <= self.risk_free_rate <= 1):
            raise ValueError(f"Risk-free rate must be a number between 0 and 1: {self.risk_free_rate}")

        self.logger.debug("RiskMetrics configuration validation successful.")


    def _reset_caches(self) -> None:
        \"\"\"Reset calculation caches.\"\"\"
        self._cache = {
            'rolling': {},
            'summary': {},
            'intermediate': {}  # For storing intermediate calculations
        }
        self.logger.debug("RiskMetrics caches reset.")

    def calculate_rolling_metrics(self, returns_obj: Union[pd.Series, BaseReturns, LeveredReturns]) -> MetricResult:
        \"\"\"
        Calculate rolling metrics from either a Series or returns object with enhanced validation.

        Args:
            returns_obj: Either a returns Series or a BaseReturns/LeveredReturns object

        Returns:
            MetricResult containing rolling metrics DataFrame and calculation metadata
        \"\"\"
        try:
            self.logger.info("Starting rolling metrics calculation")
            warning_messages = []

            # Get returns series based on input type with validation
            if isinstance(returns_obj, (BaseReturns, LeveredReturns)):
                returns = returns_obj.returns
                obj_hash = hash(returns_obj.returns.to_string()) # Cache key based on returns data
            elif isinstance(returns_obj, pd.Series):
                returns = returns_obj
                obj_hash = hash(returns.to_string()) # Cache key based on returns data
            else:
                raise TypeError(f"Unsupported returns type: {type(returns_obj)}")

            # Check cache first
            if obj_hash in self._cache['rolling']:
                 self.logger.info("Returning cached rolling metrics.")
                 return self._cache['rolling'][obj_hash]

            # Validate data
            if not isinstance(returns, pd.Series):
                 raise TypeError("Input returns must be a pandas Series.")
            if returns.empty:
                self.logger.warning("Empty returns series provided for rolling metrics. Returning empty result.")
                return MetricResult(
                    value=pd.DataFrame(),
                    calculation_time=pd.Timestamp.now(),
                    metric_type=MetricType.ROLLING,
                    input_rows=0,
                    warnings=["Input returns series was empty."]
                )

            # Log initial state
            initial_rows = len(returns)
            self.logger.info(f"Initial returns data: {initial_rows} rows")
            self.logger.info(f"Returns index is unique: {returns.index.is_unique}")

            # Handle duplicate dates with logging
            if not returns.index.is_unique:
                duplicate_count = returns.index.duplicated().sum()
                self.logger.info(f"Found {duplicate_count} duplicate dates in returns index. Keeping first occurrence.")
                returns = returns[~returns.index.duplicated(keep='first')]
                self.logger.info(f"After removing duplicates: {len(returns)} rows")

            # Handle NaN/inf values
            nan_mask = returns.isna()
            inf_mask = np.isinf(returns)
            invalid_mask = nan_mask | inf_mask
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                self.logger.info(f"Found {invalid_count} NaN/inf values in returns. Filtering them out.")
                returns = returns[~invalid_mask]
                warning_messages.append(f"Filtered out {invalid_count} NaN/inf values.")

            available_rows = len(returns)
            self.logger.info(f"Processing {available_rows} valid rows of returns data")
            if initial_rows != available_rows:
                self.logger.info(f"Total rows filtered out: {initial_rows - available_rows}")

            if available_rows == 0:
                 self.logger.warning("No valid returns data left after filtering. Returning empty result.")
                 return MetricResult(
                     value=pd.DataFrame(),
                     calculation_time=pd.Timestamp.now(),
                     metric_type=MetricType.ROLLING,
                     input_rows=0,
                     warnings=warning_messages + ["No valid returns data after filtering."]
                 )

            # Initialize metrics DataFrame
            metrics = pd.DataFrame(index=returns.index)

            # Calculate windows and log
            windows = self._get_windows(available_rows)
            self.logger.info(f"Using windows: {windows} with {available_rows} available rows")

            # Calculate metrics for each window
            for window_name, window_size in windows.items():
                if window_size > available_rows:
                    msg = f"Window size {window_size} ('{window_name}') larger than available data {available_rows}. Skipping window."
                    self.logger.warning(msg)
                    warning_messages.append(msg)
                    continue

                # Dynamic min_periods: at least 5, but no more than half the window, capped by window size
                min_periods = min(max(self.min_periods, window_size // 2), window_size)

                # --- Calculate Individual Metrics --- #
                metrics = self._calculate_window_metrics(metrics, returns, window_name, window_size, min_periods)

            # Calculate equity curve and drawdown (independent of windows)
            equity = (1 + returns).cumprod()
            metrics['equity'] = equity # Add equity curve
            metrics['high_water_mark'] = equity.cummax()
            metrics['drawdown'] = self._calculate_drawdown(equity)

            # Validate required metrics are present after calculation
            try:
                self._validate_required_metrics(metrics)
            except ValueError as e:
                self.logger.error(f"Validation failed after calculating rolling metrics: {e}")
                warning_messages.append(f"Validation Error: {e}")
                # Decide if we should return partial results or raise
                # Returning partial results with warnings for now.

            result = MetricResult(
                value=metrics,
                calculation_time=pd.Timestamp.now(),
                metric_type=MetricType.ROLLING,
                input_rows=available_rows,
                warnings=warning_messages
            )

            # Cache the result
            self._cache['rolling'][obj_hash] = result
            self.logger.info("Finished rolling metrics calculation.")
            return result

        except Exception as e:
            self.logger.error(f"Rolling metrics calculation failed unexpectedly: {str(e)}", exc_info=True)
            # Return an empty MetricResult on failure?
            return MetricResult(
                value=pd.DataFrame(),
                calculation_time=pd.Timestamp.now(),
                metric_type=MetricType.ROLLING,
                input_rows=0,
                warnings=[f"Calculation failed: {e}"]
            )

    def calculate_summary_metrics(self, returns_obj: Union[pd.Series, BaseReturns, LeveredReturns]) -> Dict[str, float]:
        \"\"\"
        Calculate summary metrics with proper compound returns handling.

        Args:
            returns_obj: Either a returns Series or a BaseReturns/LeveredReturns object

        Returns:
            Dictionary of summary metrics including total return, Sharpe ratio, etc.
        \"\"\"
        try:
            self.logger.info("Starting summary metrics calculation.")
            # Extract returns series based on input type
            if isinstance(returns_obj, (BaseReturns, LeveredReturns)):
                returns = returns_obj.returns
                obj_hash = hash(returns_obj.returns.to_string()) # Cache key
            elif isinstance(returns_obj, pd.Series):
                returns = returns_obj
                obj_hash = hash(returns.to_string()) # Cache key
            else:
                raise TypeError(f"Unsupported returns data type: {type(returns_obj)}")

            # Check cache
            if obj_hash in self._cache['summary']:
                 self.logger.info("Returning cached summary metrics.")
                 return self._cache['summary'][obj_hash]

            # Validate and clean data
            if not isinstance(returns, pd.Series):
                 raise TypeError("Input returns must be a pandas Series.")
            if returns.empty:
                self.logger.warning("Empty returns series provided for summary metrics. Returning empty dict.")
                return {}

            # Handle NaN/inf values
            nan_mask = returns.isna()
            inf_mask = np.isinf(returns)
            invalid_mask = nan_mask | inf_mask
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                self.logger.info(f"Found {invalid_count} NaN/inf values in returns. Filtering them out for summary calculation.")
                returns = returns[~invalid_mask]

            if returns.empty:
                self.logger.warning("No valid returns data left after filtering for summary metrics. Returning empty dict.")
                return {}

            # Calculate proper compound equity curve
            equity = (1 + returns).cumprod()
            trading_days = len(returns)
            years = trading_days / 252.0 # Use float for division

            # Calculate metrics using compound returns
            total_return = equity.iloc[-1] / equity.iloc[0] - 1 if not equity.empty else 0.0
            annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and not equity.empty else 0.0

            volatility = returns.std() * np.sqrt(252)

            excess_return = annualized_return - self.risk_free_rate
            sharpe = excess_return / volatility if volatility is not None and volatility > 1e-9 else 0.0

            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0

            # Handle potential division by zero in profit factor
            sum_pos_returns = returns[returns > 0].sum()
            sum_neg_returns = returns[returns < 0].sum()
            profit_factor = abs(sum_pos_returns / sum_neg_returns) if sum_neg_returns != 0 else float('inf')

            summary = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility if volatility is not None else 0.0,
                'sharpe_ratio': sharpe,
                'sortino_ratio': excess_return / downside_vol if downside_vol is not None and downside_vol > 1e-9 else 0.0,
                'max_drawdown': (equity / equity.cummax() - 1).min() if not equity.empty else 0.0,
                'calmar_ratio': annualized_return / abs((equity / equity.cummax() - 1).min()) if (equity / equity.cummax() - 1).min() != 0 else 0.0, # Added Calmar
                'skewness': returns.skew(),
                'kurtosis': returns.kurt(),
                'var_95': returns.quantile(0.05),
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean() if not returns[returns <= returns.quantile(0.05)].empty else 0.0,
                'best_return': returns.max(),
                'worst_return': returns.min(),
                'avg_return': returns.mean(),
                'avg_pos_return': returns[returns > 0].mean() if not returns[returns > 0].empty else 0.0,
                'avg_neg_return': returns[returns < 0].mean() if not returns[returns < 0].empty else 0.0,
                'pos_return_ratio': (returns > 0).mean(),
                'profit_factor': profit_factor,
                'trading_days': float(trading_days),
                'years': years
            }

            # Replace NaN results with 0.0 for consistency
            summary = {k: (v if pd.notna(v) else 0.0) for k, v in summary.items()}

            self._cache['summary'][obj_hash] = summary
            self.logger.info("Finished summary metrics calculation.")
            return summary

        except Exception as e:
            self.logger.error(f"Summary metrics calculation failed unexpectedly: {str(e)}", exc_info=True)
            return {} # Return empty dict on error

    def calculate_metrics(self, returns_data: Union[pd.Series, BaseReturns, LeveredReturns],
                          metric_type: MetricType = MetricType.ROLLING,
                          caller: str = "") -> Union[MetricResult, Dict[str, float]]:
        \"\"\"Calculate either rolling or summary metrics.\"\"\"
        try:
            self.logger.info(f"\nMetrics calculation requested by: {caller} | Type: {metric_type}")

            # Extract returns series based on input type
            if isinstance(returns_data, (BaseReturns, LeveredReturns)):
                returns = returns_data.returns
            elif isinstance(returns_data, pd.Series):
                returns = returns_data
            else:
                raise TypeError(f"Unsupported returns data type: {type(returns_data)}")

            if not isinstance(returns, pd.Series):
                 raise TypeError("Extracted returns data is not a pandas Series.")

            self.logger.info(f"Returns length: {len(returns)}")
            if not returns.empty:
                self.logger.info(f"Returns date range: {returns.index.min()} to {returns.index.max()}")
            else:
                self.logger.info("Returns series is empty.")

            if metric_type == MetricType.ROLLING:
                return self.calculate_rolling_metrics(returns)
            elif metric_type == MetricType.SUMMARY:
                return self.calculate_summary_metrics(returns)
            else:
                 raise ValueError(f"Invalid metric_type specified: {metric_type}")

        except Exception as e:
            self.logger.error(f"Metrics calculation failed in main dispatcher: {str(e)}", exc_info=True)
            # Return default based on expected type
            return MetricResult(value=pd.DataFrame(), calculation_time=pd.Timestamp.now(), metric_type=metric_type, input_rows=0, warnings=[f"Failed: {e}"]) if metric_type == MetricType.ROLLING else {}


    def _calculate_window_metrics(self, metrics: pd.DataFrame, returns: pd.Series,
                                  window_name: str, window_size: int,
                                  min_periods: int) -> pd.DataFrame:
        \"\"\"
        Calculate all metrics for a specific window.

        Args:
            metrics: DataFrame to store results
            returns: Return series to analyze
            window_name: Name of the window period (e.g., 'short')
            window_size: Size of the rolling window
            min_periods: Minimum periods required

        Returns:
            Updated metrics DataFrame
        \"\"\"
        # Calculate volatility
        vol = self._calculate_volatility(returns, window_size, min_periods)
        metrics[f'vol_{window_name}'] = vol

        # Calculate compound returns (annualized)
        ret = self._calculate_rolling_returns(returns, window_size, min_periods)
        metrics[f'return_{window_name}'] = ret

        # Calculate excess returns (annualized)
        annualized_rf = self.risk_free_rate
        excess_ret = ret - annualized_rf # Already annualized
        metrics[f'excess_return_{window_name}'] = excess_ret

        # Calculate Sharpe ratio (using annualized returns and vol)
        metrics[f'sharpe_{window_name}'] = self._calculate_rolling_sharpe(
            excess_ret, vol
        )

        # Calculate Sortino ratio (using annualized returns and vol)
        metrics[f'sortino_{window_name}'] = self._calculate_rolling_sortino(
            returns, excess_ret, window_size, min_periods
        )

        return metrics

    def _calculate_volatility(self, returns: pd.Series, window: int,
                              min_periods: int) -> pd.Series:
        \"\"\"
        Calculate rolling annualized volatility.

        Args:
            returns: Return series
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of annualized volatility values
        \"\"\"
        cache_key = f'vol_{window}_{min_periods}' # Include min_periods in cache key
        if cache_key in self._cache['intermediate']:
            return self._cache['intermediate'][cache_key]

        vol = returns.rolling(
            window=window,
            min_periods=min_periods
        ).std() * np.sqrt(252)

        self._cache['intermediate'][cache_key] = vol
        return vol

    def _calculate_rolling_returns(self, returns: pd.Series, window: int,
                                   min_periods: int) -> pd.Series:
        \"\"\"
        Calculate rolling annualized compound returns.

        Args:
            returns: Return series
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of rolling annualized returns
        \"\"\"
        cache_key = f'returns_{window}_{min_periods}'
        if cache_key in self._cache['intermediate']:
            return self._cache['intermediate'][cache_key]

        # Calculate rolling geometric mean return per period
        # (1 + R_avg) = product(1 + R_i)^(1/N)
        # Then annualize: (1 + R_avg)^252 - 1
        # Using apply for custom rolling product calculation for annualization
        # Ensure raw=True for potential performance gain if function supports it
        roll_returns = returns.rolling(
            window=window,
            min_periods=min_periods
        ).apply(lambda x: (np.prod(1 + x) ** (252.0 / len(x))) - 1 if len(x) > 0 else np.nan, raw=True)


        self._cache['intermediate'][cache_key] = roll_returns
        return roll_returns

    def _calculate_rolling_sharpe(self, annualized_excess_returns: pd.Series,
                                  annualized_volatility: pd.Series) -> pd.Series:
        \"\"\"
        Calculate rolling Sharpe ratio using annualized inputs.

        Args:
            annualized_excess_returns: Annualized excess return series
            annualized_volatility: Annualized volatility series

        Returns:
            Series of Sharpe ratios
        \"\"\"
        # Avoid division by zero or near-zero volatility
        sharpe = np.where(
            annualized_volatility.notna() & (annualized_volatility > 1e-9),
            annualized_excess_returns / annualized_volatility,
            0.0 # Return 0 if vol is zero/NaN
        )
        return pd.Series(sharpe, index=annualized_excess_returns.index).fillna(0.0)

    def _calculate_rolling_sortino(self, returns: pd.Series, annualized_excess_returns: pd.Series,
                                   window: int, min_periods: int) -> pd.Series:
        \"\"\"
        Calculate rolling Sortino ratio using annualized excess returns.

        Args:
            returns: Daily return series
            annualized_excess_returns: Annualized excess return series
            window: Rolling window size
            min_periods: Minimum periods required for downside deviation

        Returns:
            Series of Sortino ratios
        \"\"\"
        # Calculate downside returns (set positive returns to 0)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0.0

        # Calculate rolling downside deviation (annualized)
        downside_vol = downside_returns.rolling(
            window=window,
            min_periods=min_periods
        ).std() * np.sqrt(252)

        # Calculate Sortino Ratio
        sortino = np.where(
            downside_vol.notna() & (downside_vol > 1e-9),
            annualized_excess_returns / downside_vol,
            0.0 # Return 0 if downside vol is zero/NaN
        )
        return pd.Series(sortino, index=annualized_excess_returns.index).fillna(0.0)

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        \"\"\"
        Calculate drawdown series from equity curve.

        Args:
            equity: Equity curve series

        Returns:
            Series of drawdown values (negative or zero)
        \"\"\"
        if equity.empty:
            return pd.Series(dtype=float)

        high_water_mark = equity.cummax()
        # Avoid division by zero if HWM is zero (can happen if equity starts at 0)
        drawdown = np.where(high_water_mark > 1e-9, equity / high_water_mark - 1, 0.0)
        return pd.Series(drawdown, index=equity.index).fillna(0.0)

    def _validate_input_data(self, daily_performance: pd.DataFrame) -> None:
        \"\"\"Validate input data for risk metrics calculation.\"\"\"
        # This seems unused, internal methods handle validation directly.
        # Consider removing if not called externally.
        pass

    def _get_windows(self, available_days: Optional[int] = None) -> Dict[str, int]:
        \"\"\"Get dictionary of usable window names and sizes based on available data.\"\"\"
        usable_windows = {}
        for name, size in self.windows.items():
             # Extract base name like 'short', 'medium', 'long'
             base_name = name.split('_')[0]
             if available_days is None or available_days >= size:
                 usable_windows[base_name] = size
             elif base_name in self.required_windows:
                 # Keep required windows even if data is insufficient, but log warning
                 self.logger.warning(f"Required window '{base_name}' (size {size}) has insufficient data ({available_days} days). Calculations may be less reliable.")
                 usable_windows[base_name] = size
             else:
                  self.logger.info(f"Skipping window '{base_name}' (size {size}) due to insufficient data ({available_days} days)." )

        # Ensure all required windows are present, even if logged as insufficient
        for req_win_name in self.required_windows:
             if req_win_name not in usable_windows:
                 # Find the full name (e.g., 'short_term') and size
                 found = False
                 for win_key, win_size in self.windows.items():
                     if win_key.startswith(req_win_name):
                         usable_windows[req_win_name] = win_size
                         self.logger.warning(f"Including required window '{req_win_name}' (size {win_size}) despite insufficient data ({available_days} days).")
                         found = True
                         break
                 if not found:
                      # This case should ideally be caught by _validate_config
                      self.logger.error(f"Required window '{req_win_name}' not found in self.windows config.")

        self.logger.debug(f"Determined usable windows: {usable_windows}")
        return usable_windows

    def clear_cache(self) -> None:
        \"\"\"Clear all calculation caches.\"\"\"
        self._reset_caches()
        self.logger.info("RiskMetrics cache cleared.") 