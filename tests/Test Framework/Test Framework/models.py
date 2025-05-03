from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Optional, List, Union
import pandas as pd
from datetime import time, datetime

# --- Enums ---

class ExitReason(Enum):
    \"\"\"Enumeration of possible trade exit reasons\"\"\"
    MARKET_CLOSE = "MARKET_CLOSE"
    VWAP_STOP = "VWAP_STOP"
    BOUNDARY_STOP = "BOUNDARY_STOP"

class MetricType(Enum):
    \"\"\"Types of metrics that can be calculated.\"\"\"
    ROLLING = "rolling"
    SUMMARY = "summary"


# --- Configuration Related Dataclasses ---

@dataclass
class StrategyParameters:
    \"\"\"Data class for strategy parameters\"\"\"
    lookback_days: int
    volatility_multiplier: float
    min_holding_period: pd.Timedelta = pd.Timedelta(minutes=1)
    entry_times: List[int] = None  # List of valid entry minute marks (e.g., [0, 30])

    def __post_init__(self):
        \"\"\"Set default entry times if none provided\"\"\"
        if self.entry_times is None:
            self.entry_times = [0, 30]  # Default to trading on hour and half hour

@dataclass
class ContractSpecification:
    \"\"\"Data class for contract specifications\"\"\"
    symbol: str  # Add this line
    tick_size: float
    multiplier: float
    margin: float
    market_open: time
    market_close: time
    last_entry: time

@dataclass
class TransactionCosts:
    \"\"\"Configuration for transaction costs\"\"\"
    commission_rate: float
    slippage_rate: float
    min_commission: float = 0.0
    fixed_costs: float = 0.0

@dataclass
class RiskParameters:
    \"\"\"Base configuration for risk management.\"\"\"
    min_size: float
    max_size: float
    min_scalar: float
    max_scalar: float

@dataclass
class RiskLimits:
    \"\"\"Risk limits configuration\"\"\"
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    position_limit_pct: float  # Max position size as % of equity
    concentration_limit: float  # Max exposure to single instrument

@dataclass
class VolatilityParams:
    \"\"\"Volatility-based sizing parameters\"\"\"
    target_volatility: float
    estimation_window: int
    min_scaling: float
    max_scaling: float
    adaptation_rate: float  # Added this field
    vol_target_range: Tuple[float, float] = (0.10, 0.20)  # Added with default

@dataclass
class SharpeParams:
    \"\"\"Parameters for Sharpe-based risk management\"\"\"
    target_sharpe: float
    target_volatility: float
    min_scaling: float
    max_scaling: float
    adaptation_rate: float = 0.1
    min_trades: int = 5
    risk_free_rate: float = 0.02
    target_range: Tuple[float, float] = (0.5, 2.0)
    window_type: str = "medium"  # Use 'short', 'medium', or 'long' to match RiskMetrics windows

@dataclass
class AdaptiveParams:
    \"\"\"Parameters for adaptive risk management\"\"\"
    base_volatility: float
    regime_window: int
    adaptation_rate: float
    min_scaling: float
    max_scaling: float
    vol_target_range: Tuple[float, float] = (0.10, 0.20)
    regime_thresholds: Tuple[float, float] = (0.8, 1.2)  # Added for regime detection

@dataclass
class RegimeConfig:
    \"\"\"Configuration for regime detection\"\"\"
    enabled: bool = True
    method: str = "volatility"  # "volatility", "trend", or "combined"
    lookback: int = 252
    vol_threshold: float = 1.5  # Multiplier for regime change detection
    trend_threshold: float = 0.5  # Z-score for trend detection


# --- Trading Related Dataclasses ---

@dataclass
class Trade:
    \"\"\"Represents a completed trade.\"\"\"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    direction: int
    pnl: float
    costs: float
    exit_reason: str
    base_size: float
    final_size: float
    base_return: float  # Added field
    final_return: float  # Added field
    strategy_name: str
    symbol: str
    contract_spec: ContractSpecification


@dataclass
class MetricResult:
    value: Union[pd.DataFrame, Dict[str, float]]
    calculation_time: pd.Timestamp
    metric_type: MetricType
    input_rows: int
    warnings: List[str] = field(default_factory=list)

@dataclass
class BaseReturns:
    \"\"\"Class representing base strategy returns before position sizing.\"\"\"
    returns: pd.Series
    metrics: Optional[Union[pd.DataFrame, MetricResult]] = None
    summary_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        \"\"\"Validate returns data and log statistics.\"\"\"
        # Note: Logging and validation logic referencing _validate_returns and RiskMetrics
        # will need to be adjusted or moved after refactoring.
        # For now, keeping the structure.
        # logger = logging.getLogger(__name__)
        # _validate_returns(self.returns, "BaseReturns", logger) # Requires _validate_returns
        pass # Placeholder for validation

    @property
    def start_date(self) -> pd.Timestamp:
        \"\"\"Get the start date of returns data.\"\"\"
        return self.returns.index.min()

    @property
    def end_date(self) -> pd.Timestamp:
        \"\"\"Get the end date of returns data.\"\"\"
        return self.returns.index.max()

    @property
    def trading_days(self) -> int:
        \"\"\"Get the number of trading days.\"\"\"
        return len(self.returns)

    def calculate_metrics(self, risk_metrics: 'RiskMetrics') -> None:
        \"\"\"Calculate metrics using validated returns data.\"\"\"
        # Requires RiskMetrics to be imported and available
        if len(self.returns) == 0:
            return

        # self.metrics = risk_metrics.calculate_metrics(
        #     self.returns,
        #     metric_type=MetricType.ROLLING,
        #     caller="BaseReturns"
        # )

        # self.summary_metrics = risk_metrics.calculate_metrics(
        #     self.returns,
        #     metric_type=MetricType.SUMMARY,
        #     caller="BaseReturns"
        # )
        pass # Placeholder for calculation logic

    def subset(self, start_date: Optional[pd.Timestamp] = None,
               end_date: Optional[pd.Timestamp] = None) -> 'BaseReturns':
        \"\"\"Create a new BaseReturns object with a subset of the data.\"\"\"
        mask = pd.Series(True, index=self.returns.index)
        if start_date:
            mask &= self.returns.index >= start_date
        if end_date:
            mask &= self.returns.index <= end_date

        return BaseReturns(returns=self.returns[mask])


@dataclass
class LeveredReturns:
    \"\"\"Class representing returns after position sizing.\"\"\"
    position_sizes: pd.Series
    base_returns: BaseReturns
    metrics: Optional[Union[pd.DataFrame, 'MetricResult']] = None
    summary_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        \"\"\"Validate data and calculate levered returns.\"\"\"
        if not isinstance(self.position_sizes, pd.Series):
            raise TypeError("position_sizes must be a pandas Series")

        if not isinstance(self.base_returns, BaseReturns):
            raise TypeError("base_returns must be a BaseReturns instance")

        # Align position sizes with returns index
        self.position_sizes = self.position_sizes.reindex(
            self.base_returns.returns.index
        ).fillna(1.0)

    @property
    def returns(self) -> pd.Series:
        # import logging # Requires logging setup
        # logger = logging.getLogger(__name__)
        # logger.debug("LeveredReturns - Base returns (head):\\n%s", self.base_returns.returns.head())
        # logger.debug("LeveredReturns - Position sizes (head):\\n%s", self.position_sizes.head())

        levered_equity = (1 + self.base_returns.returns * self.position_sizes).cumprod()
        # logger.debug("LeveredReturns - Levered equity (head):\\n%s", levered_equity.head())

        levered_returns = levered_equity.pct_change().fillna(0)
        # logger.debug("LeveredReturns - Levered returns (head):\\n%s", levered_returns.head())

        return levered_returns


    def calculate_metrics(self, risk_metrics: 'RiskMetrics') -> None:
        \"\"\"Calculate metrics using the levered returns.\"\"\"
        # Requires RiskMetrics to be imported and available
        levered_returns = self.returns

        if len(levered_returns) == 0:
            return

        # self.metrics = risk_metrics.calculate_metrics(
        #     levered_returns,
        #     metric_type=MetricType.ROLLING,
        #     caller="LeveredReturns"
        # )

        # self.summary_metrics = risk_metrics.calculate_metrics(
        #     levered_returns,
        #     metric_type=MetricType.SUMMARY,
        #     caller="LeveredReturns"
        # )
        pass # Placeholder for calculation logic


@dataclass
class TradingResults:
    \"\"\"Contains all results from a strategy execution.\"\"\"

    symbol: str
    strategy_name: str
    base_trades: List[Trade]
    final_trades: List[Trade]
    trade_metrics: List[Dict]
    daily_performance: pd.DataFrame
    execution_data: pd.DataFrame
    # config: Config # Requires Config class import
    contract_spec: ContractSpecification
    timestamp: Optional[pd.Timestamp] = None
    base_returns: Optional[BaseReturns] = None
    levered_returns: Optional[LeveredReturns] = None
    risk_metrics: Optional['RiskMetrics'] = None # Store the RiskMetrics instance

    def __post_init__(self):
        \"\"\"Initialize TradingResults, potentially calculate metrics.\"\"\"
        self.timestamp = self.timestamp or pd.Timestamp.now()
        # self.risk_metrics = risk_metrics or RiskMetrics(config) # Requires Config and RiskMetrics

        # Initialize returns objects if not provided and daily_performance is
        if self.daily_performance is not None:
            if self.base_returns is None and 'base_returns' in self.daily_performance.columns:
                self.base_returns = BaseReturns(
                    returns=self.daily_performance['base_returns']
                )

            if (self.levered_returns is None and self.base_returns is not None and
                    'position_size' in self.daily_performance.columns):
                 # Simplified levered returns calculation for structure
                levered_returns_calc = (self.daily_performance['base_returns'] *
                                        self.daily_performance['position_size'])
                self.levered_returns = LeveredReturns(
                    # returns=levered_returns_calc, # Avoid calculating here, use property
                    position_sizes=self.daily_performance['position_size'],
                    base_returns=self.base_returns
                )

        # Calculate metrics if RiskMetrics instance is available
        if self.risk_metrics:
             self.calculate_all_metrics()


    def calculate_all_metrics(self) -> None:
        \"\"\"Calculate all metrics for both base and levered returns.\"\"\"
        if not self.risk_metrics:
             # logger.warning("RiskMetrics instance not available, cannot calculate metrics.") # Requires logging
             return

        if self.base_returns:
            self.base_returns.calculate_metrics(self.risk_metrics)
        if self.levered_returns:
            self.levered_returns.calculate_metrics(self.risk_metrics)

    @property
    def metrics(self) -> Dict:
        \"\"\"Return calculated performance metrics combined from all sources.\"\"\"
        metrics = {}

        # Trade-based metrics
        if self.base_trades:
            if len(self.base_trades) > 0:
                metrics.update({
                    'base_total_trades': len(self.base_trades),
                    'base_win_rate': sum(1 for t in self.base_trades if t.pnl > t.costs) / len(self.base_trades),
                    'base_total_pnl': sum(t.pnl for t in self.base_trades),
                    'base_total_costs': sum(t.costs for t in self.base_trades)
                })
            else:
                 metrics.update({
                    'base_total_trades': 0, 'base_win_rate': 0,
                    'base_total_pnl': 0, 'base_total_costs': 0
                 })


        if self.final_trades:
            if len(self.final_trades) > 0:
                metrics.update({
                    'final_total_trades': len(self.final_trades),
                    'final_win_rate': sum(1 for t in self.final_trades if t.pnl > t.costs) / len(self.final_trades),
                    'final_total_pnl': sum(t.pnl for t in self.final_trades),
                    'final_total_costs': sum(t.costs for t in self.final_trades)
                })
            else:
                metrics.update({
                    'final_total_trades': 0, 'final_win_rate': 0,
                    'final_total_pnl': 0, 'final_total_costs': 0
                 })

        # Get metrics from returns objects
        if self.base_returns and self.base_returns.summary_metrics:
            metrics.update({f'base_{k}': v for k, v in self.base_returns.summary_metrics.items()})

        if self.levered_returns and self.levered_returns.summary_metrics:
            metrics.update({f'levered_{k}': v for k, v in self.levered_returns.summary_metrics.items()})

        return metrics

    def get_metric(self, metric_name: str, returns_type: str = 'levered') -> pd.Series:
        \"\"\"
        Get a specific metric time series.

        Args:
            metric_name (str): Name of the metric to retrieve
            returns_type (str): Either 'base' or 'levered'

        Returns:
            pd.Series: Time series of the requested metric
        \"\"\"
        returns_obj = self.base_returns if returns_type == 'base' else self.levered_returns

        if returns_obj is None or returns_obj.metrics is None:
            raise ValueError(f"No metrics available for returns_type '{returns_type}'")

        metrics_df = returns_obj.metrics.value if isinstance(returns_obj.metrics, MetricResult) else returns_obj.metrics

        if not isinstance(metrics_df, pd.DataFrame) or metric_name not in metrics_df.columns:
             raise KeyError(f"Metric '{metric_name}' not found or metrics are not a DataFrame for {returns_type} returns")

        return metrics_df[metric_name]

    def get_returns(self, returns_type: str = 'levered') -> pd.Series:
        \"\"\"Get returns series of specified type.\"\"\"
        returns_obj = self.base_returns if returns_type == 'base' else self.levered_returns
        return returns_obj.returns if returns_obj else pd.Series(dtype=float) # Return empty series if none

    def get_position_sizes(self) -> pd.Series:
        \"\"\"Get position sizes series.\"\"\"
        return (self.levered_returns.position_sizes
                if self.levered_returns
                else pd.Series(dtype=float)) # Return empty series if none 