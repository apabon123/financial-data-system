from __future__ import annotations # Required for type hinting RiskManager in factory
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, time

from ..config import Config, LoggingConfig
from ..models import (
    ContractSpecification, RiskLimits, VolatilityParams, SharpeParams, AdaptiveParams,
    BaseReturns, LeveredReturns, MetricResult
)
from ..Risk_Metrics.metrics import RiskMetrics # Use folder name


class RiskManager(ABC):
    \"\"\"Abstract base class for risk management.\"\"\"

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        self.logger = LoggingConfig.get_logger(__name__)
        self.config = config
        self.risk_metrics = risk_metrics

        # Ensure risk_limits is a dictionary before unpacking
        risk_limits_dict = config.risk_limits if isinstance(config.risk_limits, dict) else {}
        self.risk_limits = RiskLimits(**risk_limits_dict)

        # Get contract specification for the configured symbol
        # Use helper method to avoid code duplication
        self.contract_spec = self._get_contract_spec()
        if not self.contract_spec:
             # Log error if spec not found, initialization might fail later
             self.logger.error(f"ContractSpecification could not be initialized for symbol '{config.symbol}'")

        # Initialize risk tracking
        self.current_drawdown = 0.0
        self.peak_equity = config.initial_equity
        self.daily_pnl = 0.0

    def _get_contract_spec(self) -> Optional[ContractSpecification]:
        \"\"\"Get contract specification from config safely.\"\"\"
        if self.config.symbol is None:
            self.logger.error("Cannot get contract spec: config.symbol is None.")
            return None
        if not hasattr(self.config, 'contract_specs') or self.config.symbol not in self.config.contract_specs:
            self.logger.error(f"No contract specifications found for symbol '{self.config.symbol}' in config.")
            return None
        if not hasattr(self.config, 'trading_hours') or self.config.symbol not in self.config.trading_hours:
            self.logger.error(f"No trading hours found for symbol '{self.config.symbol}' in config.")
            return None

        spec_data = self.config.contract_specs[self.config.symbol]
        hours_data = self.config.trading_hours[self.config.symbol]

        # Check required keys
        required_spec_keys = {'tick_size', 'multiplier', 'margin'}
        required_hour_keys = {'market_open', 'market_close', 'last_entry'}
        if not required_spec_keys.issubset(spec_data.keys()):
             self.logger.error(f"Missing required keys in contract_specs for {self.config.symbol}: {required_spec_keys - set(spec_data.keys())}")
             return None
        if not required_hour_keys.issubset(hours_data.keys()):
             self.logger.error(f"Missing required keys in trading_hours for {self.config.symbol}: {required_hour_keys - set(hours_data.keys())}")
             return None

        try:
            return ContractSpecification(
                symbol=self.config.symbol,
                tick_size=float(spec_data['tick_size']),
                multiplier=float(spec_data['multiplier']),
                margin=float(spec_data['margin']),
                market_open=self._parse_time(hours_data['market_open']),
                market_close=self._parse_time(hours_data['market_close']),
                last_entry=self._parse_time(hours_data['last_entry'])
            )
        except (ValueError, TypeError) as e:
             self.logger.error(f"Error creating ContractSpecification for {self.config.symbol}: {e}")
             return None


    def prepare_base_strategy_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Prepare DataFrame with base position sizes for evaluating strategy.\"\"\"
        try:
            df = df.copy()

            if self.contract_spec is None:
                 raise ValueError("Cannot prepare base positions without valid ContractSpecification.")

            # Calculate fixed base position size - no risk limits applied yet
            base_size = self.calculate_base_position_size(self.config.initial_equity)

            if 'signal' not in df.columns:
                 raise ValueError("'signal' column not found in DataFrame. Cannot prepare base positions.")

            # Set fixed position sizes from signals
            df['base_position_size'] = df['signal'] * base_size

            # Add signal verification
            signal_times = df[df['signal'] != 0].index
            self.logger.info(f"\nVerifying signals for base position preparation:")
            if not signal_times.empty:
                self.logger.info(f"First signal time: {signal_times[0]}")
                self.logger.info(f"Last signal time: {signal_times[-1]}")
                self.logger.info(f"Sample of signals:")
                for time in signal_times[:min(5, len(signal_times))]: # Limit sample size
                    self.logger.info(f"  {time}: {df.loc[time, 'signal']}")
            else:
                self.logger.info("No signals found in the provided DataFrame.")


            # Initialize tracking columns (used later by TradeManager)
            df['position'] = 0 # Current live position (-1, 0, 1)
            # Initialize 'position_size' based on whether sizing will be applied later
            df['position_size'] = 0.0 # This will be overwritten by apply_position_sizing
            df['current_equity'] = self.config.initial_equity
            # df['base_equity'] = self.config.initial_equity # Might not be needed here

            # Log statistics about signals and positions
            self.logger.info("\nBase Position Preparation Statistics:")
            self.logger.info("-" * 30)
            self.logger.info(f"Initial equity: ${self.config.initial_equity:,.2f}")
            self.logger.info(f"Base position size calculated: {base_size:.2f} contracts")

            # Signal statistics
            total_signals = (df['signal'] != 0).sum()
            long_signals = (df['signal'] > 0).sum()
            short_signals = (df['signal'] < 0).sum()
            self.logger.info(f"\nSignal Distribution in Input Data:")
            self.logger.info(f"Total signals: {total_signals}")
            self.logger.info(f"Long signals: {long_signals}")
            self.logger.info(f"Short signals: {short_signals}")

            # Position statistics
            self.logger.info(f"\nBase Position Size Assigned:")
            self.logger.info(f"Fixed long base position: {base_size:.2f} contracts")
            self.logger.info(f"Fixed short base position: {-base_size:.2f} contracts")

            return df

        except Exception as e:
            self.logger.error(f"Base position preparation failed: {str(e)}", exc_info=True)
            raise

    def calculate_base_position_size(self, initial_equity: float) -> float:
        \"\"\"
        Calculate base position size using initial equity and contract margin.
        For base strategy, this is a fixed size based solely on initial equity.
        \"\"\"
        try:
            if self.contract_spec is None or self.contract_spec.margin <= 0:
                 self.logger.warning("Cannot calculate base position size: Invalid ContractSpecification or margin <= 0. Returning 1.0")
                 return 1.0

            # Simple calculation based on initial equity and margin only
            # Avoid division by zero
            base_size = initial_equity / self.contract_spec.margin

            self.logger.info(f"\nBase Position Size Calculation:")
            self.logger.info(f"Initial equity: ${initial_equity:,.2f}")
            self.logger.info(f"Contract margin: ${self.contract_spec.margin:,.2f}")
            self.logger.info(f"Base size: {base_size:.2f} contracts")

            # Add a basic sanity check
            if base_size <= 0:
                 self.logger.warning(f"Calculated base size ({base_size:.2f}) is non-positive. Returning 1.0 as fallback.")
                 return 1.0

            return base_size

        except Exception as e:
            self.logger.error(f"Base position size calculation failed: {str(e)}", exc_info=True)
            return 1.0  # Conservative default

    def _parse_time(self, time_str: Union[str, time]) -> Optional[time]:
        \"\"\"Parse time string to time object if needed, handling errors.\"\"\"
        if isinstance(time_str, time):
            return time_str
        if isinstance(time_str, str):
             try:
                 return datetime.strptime(time_str, '%H:%M').time()
             except ValueError:
                 self.logger.error(f"Invalid time format '{time_str}'. Expected HH:MM.")
                 return None
        self.logger.error(f"Invalid type for time parsing: {type(time_str)}. Expected str or time.")
        return None

    def initialize_risk_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Initialize risk tracking columns in DataFrame. (Seems less used now)\"\"\"
        # This method seems less relevant now as columns are initialized elsewhere.
        # Keeping structure but adding logging if called.
        self.logger.debug("Initializing risk tracking columns (if not present). Note: May be redundant.")
        df = df.copy()

        risk_columns = {
            'volatility': 0.0,  # Rolling volatility
            'risk_scaling': 1.0,  # Risk-based position scaling
            'risk_adjusted_size': 0.0,  # Risk-adjusted position size
            'exposure_pct': 0.0,  # Position exposure as % of equity
            'margin_used_pct': 0.0  # Margin utilization percentage
        }

        for col, default in risk_columns.items():
            if col not in df.columns:
                df[col] = default

        return df

    @abstractmethod
    def calculate_position_sizes(self, df: pd.DataFrame,
                                 base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> pd.Series:
        \"\"\"Calculate position size multipliers based on risk metrics or base returns.\n\n        Args:
            df: The main DataFrame containing signals and potentially other data.
            base_returns_or_metrics: Either a BaseReturns object or a DataFrame of pre-calculated metrics.
                                      Implementations should handle both possibilities.
\n        Returns:
            pd.Series: Position size multipliers, indexed like df.
        \"\"\"
        pass

    def apply_position_sizing(self, df: pd.DataFrame, position_size_multipliers: pd.Series) -> pd.DataFrame:
        \"\"\"
        Apply position sizing multipliers to the base positions with validation.
        This method calculates the final 'position_size' based on the multiplier,
        base size, current equity, and risk limits.

        Args:
            df: DataFrame with 'base_position_size' and 'current_equity' columns.
            position_size_multipliers: Series of multipliers (e.g., from calculate_position_sizes),
                                      indexed like df.

        Returns:
            DataFrame with added/updated 'position_size' and 'leverage' columns.
        \"\"\"
        try:
            self.logger.info("Applying position sizing multipliers.")
            df = df.copy()

            if 'base_position_size' not in df.columns:
                 raise ValueError("'base_position_size' column missing for applying sizing.")
            if 'current_equity' not in df.columns:
                 # Initialize if missing, although TradeManager should handle this
                 self.logger.warning("'current_equity' column missing, initializing with initial equity.")
                 df['current_equity'] = self.config.initial_equity

            if not isinstance(position_size_multipliers, pd.Series):
                 raise TypeError("position_size_multipliers must be a pandas Series.")

            # Validate indexes and align multipliers
            if not df.index.equals(position_size_multipliers.index):
                self.logger.warning("Position size multipliers index mismatch - attempting realignment using forward fill.")
                # Ensure multiplier index matches df index frequency if possible
                # Reindex with forward fill to propagate daily/windowed multipliers intraday
                position_size_multipliers = position_size_multipliers.reindex(df.index, method='ffill')

                # Check for NaNs after realignment (e.g., at the start)
                if position_size_multipliers.isna().any():
                    initial_nan_count = position_size_multipliers.isna().sum()
                    self.logger.warning(f"{initial_nan_count} NaN values found in multipliers after realignment. Filling with 1.0 (neutral scaling).")
                    position_size_multipliers = position_size_multipliers.fillna(1.0)

            # Store the raw multipliers as leverage
            df['leverage'] = position_size_multipliers

            # Calculate scaled position size before limits
            # Scale base size by multiplier and equity ratio
            scaled_size = (
                df['base_position_size'] *
                df['leverage'] *
                (df['current_equity'] / self.config.initial_equity)
            )

            # Apply risk limits (clip based on max contracts, margin, etc.)
            final_position_size = scaled_size.copy()
            if self.contract_spec and self.risk_limits:
                 # Vectorized limit application might be complex, loop for clarity if needed
                 # Simple example: clip to absolute max size from config
                 if hasattr(self.risk_limits, 'max_position_size') and self.risk_limits.max_position_size > 0:
                     final_position_size = final_position_size.clip(
                         lower=-self.risk_limits.max_position_size,
                         upper=self.risk_limits.max_position_size
                     )
                 # Add margin limit check (simplified example)
                 if self.contract_spec.margin > 0 and hasattr(self.risk_limits, 'position_limit_pct') and self.risk_limits.position_limit_pct > 0:
                     max_contracts_by_margin = (df['current_equity'] * self.risk_limits.position_limit_pct) / self.contract_spec.margin
                     final_position_size = final_position_size.clip(
                         lower=-max_contracts_by_margin,
                         upper=max_contracts_by_margin
                     )
            else:
                 self.logger.warning("Cannot apply detailed risk limits: ContractSpec or RiskLimits not fully defined.")

            df['position_size'] = final_position_size

            # Calculate exposure percentage after sizing and limits
            if self.contract_spec and self.contract_spec.margin > 0:
                 df['exposure_pct'] = (
                     abs(df['position_size']) *
                     self.contract_spec.margin /
                     df['current_equity'] # Use current equity for pct calculation
                 ) * 100
                 df['exposure_pct'].fillna(0.0, inplace=True) # Fill NaN exposure (e.g., if equity is zero)
            else:
                 df['exposure_pct'] = 0.0

            # Final validation
            if df['position_size'].isna().any():
                self.logger.error("NaN values found in final 'position_size' column. Filling with 0.")
                df['position_size'] = df['position_size'].fillna(0.0)

            # Log statistics
            self.logger.info("\nPosition Sizing Application Statistics:")
            self.logger.info(f"Average leverage applied: {df['leverage'].mean():.2f} (Range: [{df['leverage'].min():.2f}, {df['leverage'].max():.2f}])")
            self.logger.info(f"Average final position size: {abs(df['position_size']).mean():.2f}")
            self.logger.info(f"Max final position size: {abs(df['position_size']).max():.2f}")
            self.logger.info(f"Average exposure pct: {df['exposure_pct'].mean():.2f}%")
            self.logger.info(f"Max exposure pct: {df['exposure_pct'].max():.2f}%")

            return df

        except Exception as e:
            self.logger.error(f"Position sizing application failed: {str(e)}", exc_info=True)
            # Return original df or df with default sizes in case of error?
            # Returning df with potentially partial changes for debugging.
            df['position_size'] = df.get('position_size', df.get('base_position_size', 0.0)) # Fallback
            return df

    def check_risk_limits(self, df: pd.DataFrame) -> List[str]:
        \"\"\"Check risk limits based on *final* positions after sizing.\n\n        Args:
            df: DataFrame containing final 'position_size' and 'current_equity'.
                It should also contain 'pnl' if daily loss is checked.
\n        Returns:
            List of strings describing any violated limits.
        \"\"\"
        violations = []
        if df.empty or self.contract_spec is None or self.risk_limits is None:
            self.logger.warning("Cannot check risk limits: DataFrame empty or config missing.")
            return violations

        # Calculate exposure based on final position size and current equity
        if 'position_size' not in df.columns or 'current_equity' not in df.columns:
             self.logger.warning("Missing 'position_size' or 'current_equity' for risk limit check.")
             return violations

        if self.contract_spec.margin <= 0:
             self.logger.warning("Contract margin is zero or negative, cannot check margin-based limits.")
             return violations

        # Ensure current_equity is not zero for percentage calculations
        equity_for_calc = df['current_equity'].replace(0, np.nan)

        position_exposure_pct = (
            abs(df['position_size']) *
            self.contract_spec.margin /
            equity_for_calc
        )

        # Position size limit check (% of equity)
        if hasattr(self.risk_limits, 'position_limit_pct') and self.risk_limits.position_limit_pct > 0:
            if (position_exposure_pct > self.risk_limits.position_limit_pct).any():
                max_exposure = position_exposure_pct.max()
                violations.append(
                    f"Position size exceeds {self.risk_limits.position_limit_pct:.1%} equity limit "
                    f"(Max exposure reached: {max_exposure:.1%})"
                )

        # Concentration limit check (similar logic, depends on definition)
        # Assuming concentration limit is also % of equity for single instrument
        if hasattr(self.risk_limits, 'concentration_limit') and self.risk_limits.concentration_limit > 0:
            if (position_exposure_pct > self.risk_limits.concentration_limit).any():
                max_exposure = position_exposure_pct.max()
                violations.append(
                    f"Position concentration exceeds {self.risk_limits.concentration_limit:.1%} limit "
                    f"(Max exposure reached: {max_exposure:.1%})"
                )

        # Absolute position size check
        if hasattr(self.risk_limits, 'max_position_size') and self.risk_limits.max_position_size > 0:
             if (abs(df['position_size']) > self.risk_limits.max_position_size).any():
                 max_abs_pos = abs(df['position_size']).max()
                 violations.append(
                     f"Absolute position size exceeds {self.risk_limits.max_position_size:.2f} limit "
                     f"(Max size reached: {max_abs_pos:.2f})"
                 )

        # Daily loss limit check (requires daily P&L calculation first)
        if 'pnl' in df.columns and hasattr(self.risk_limits, 'max_daily_loss') and self.risk_limits.max_daily_loss > 0:
             # Ensure index is DatetimeIndex for grouping
             if isinstance(df.index, pd.DatetimeIndex):
                 daily_pnl = df['pnl'].groupby(df.index.date).sum()
                 # Need daily starting equity for percentage calculation - complex here
                 # Simple check: absolute daily loss value (if PNL represents daily delta)
                 if not daily_pnl.empty:
                      min_daily_pnl = daily_pnl.min()
                      # Need to define how max_daily_loss is interpreted (absolute value or percentage)
                      # Assuming it's a percentage of initial equity for this example
                      max_loss_value = self.config.initial_equity * self.risk_limits.max_daily_loss
                      if min_daily_pnl < -max_loss_value:
                           violations.append(
                               f"Daily loss of ${abs(min_daily_pnl):,.2f} exceeds limit of ${max_loss_value:,.2f} "
                               f"({self.risk_limits.max_daily_loss:.1%})"
                           )
             else:
                 self.logger.warning("Cannot perform daily loss check: Index is not DatetimeIndex.")


        # Max Drawdown check - typically done on daily equity curve, harder to check intraday here
        # This check is usually performed after aggregation or on a rolling basis.

        if violations:
            self.logger.warning(f"Risk limit violations detected: {violations}")
        else:
            self.logger.info("Risk limit checks passed.")

        return violations

    def update_risk_metrics(self, equity: float, timestamp: pd.Timestamp) -> None:
        \"\"\"Update risk tracking metrics (e.g., drawdown). Called periodically or after trades.\"\"\"
        # Update peak equity and drawdown
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0: # Avoid division by zero
             self.current_drawdown = (equity - self.peak_equity) / self.peak_equity
        else:
             self.current_drawdown = 0.0

        # Reset daily P&L at market open
        market_open_time = self.contract_spec.market_open if self.contract_spec else None
        if market_open_time and timestamp.time() == market_open_time:
            self.logger.debug(f"Resetting daily PNL tracker at market open {timestamp}")
            self.daily_pnl = 0.0

        # Update daily P&L (this assumes P&L is passed externally or calculated)
        # This method's responsibility might need refinement depending on where PnL is calculated.


    @abstractmethod
    def get_risk_summary(self) -> dict:
        \"\"\"Get current risk metrics summary.\"\"\"
        pass

    def _validate_position_limits(self, position_size: float, current_equity: float) -> float:
        \"\"\"Validate and adjust a *single* position size based on limits.\n           Note: Prefer vectorized `apply_position_sizing` where possible.\n        \"\"\"
        try:
            if self.contract_spec is None or self.risk_limits is None:
                self.logger.warning("_validate_position_limits: Config missing. Returning original size.")
                return position_size
            if current_equity <= 0 or self.contract_spec.margin <= 0:
                 self.logger.warning("_validate_position_limits: Equity or margin non-positive. Returning 0.")
                 return 0.0

            # Calculate maximum position based on margin requirements (% equity)
            margin_max = float('inf')
            if hasattr(self.risk_limits, 'position_limit_pct') and self.risk_limits.position_limit_pct > 0:
                 margin_max = (current_equity * self.risk_limits.position_limit_pct) / self.contract_spec.margin

            # Calculate maximum position based on concentration limit (% equity)
            # Assuming concentration limit is per instrument and similar calc
            concentration_max = float('inf')
            if hasattr(self.risk_limits, 'concentration_limit') and self.risk_limits.concentration_limit > 0:
                 # This calculation might be wrong depending on how concentration is defined
                 # Assuming multiplier is relevant for notional value, not just margin check
                 # Let's use the same as margin_max for simplicity unless clarified
                 concentration_max = (current_equity * self.risk_limits.concentration_limit) / self.contract_spec.margin

            # Absolute max position size
            abs_max = float('inf')
            if hasattr(self.risk_limits, 'max_position_size') and self.risk_limits.max_position_size > 0:
                 abs_max = self.risk_limits.max_position_size

            # Take the minimum of all limits
            effective_max_position = min(margin_max, concentration_max, abs_max)

            # Apply limits
            clipped_size = np.clip(position_size, -effective_max_position, effective_max_position)

            if clipped_size != position_size:
                 self.logger.debug(f"Position size {position_size:.2f} clipped to {clipped_size:.2f} based on limits (Max: {effective_max_position:.2f})")

            return clipped_size

        except Exception as e:
            self.logger.error(f"Position limit validation failed: {str(e)}", exc_info=True)
            return 0.0 # Return 0 as safe default

    def _calculate_exposure_metrics(self, position_size: float, price: float,
                                    current_equity: float) -> Dict[str, float]:
        \"\"\"Calculate exposure and risk metrics for a position.\n           Note: Use vectorized calculations in `apply_position_sizing` preferably.
        \"\"\"
        try:
            if self.contract_spec is None:
                 return {'error': "Missing contract spec"}
            if current_equity <= 0:
                 return {'error': "Non-positive equity"}

            notional_exposure = abs(position_size * price * self.contract_spec.multiplier)
            margin_used = abs(position_size * self.contract_spec.margin)

            return {
                'notional_exposure': notional_exposure,
                'exposure_pct': (notional_exposure / current_equity) * 100,
                'margin_used': margin_used,
                'margin_used_pct': (margin_used / current_equity) * 100
            }

        except Exception as e:
            self.logger.error(f"Exposure calculation failed: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _apply_position_limits(self, position_sizes: pd.Series, equity: pd.Series) -> pd.Series:
        \"\"\"Apply position limits based on risk parameters. (Seems duplicate/simplified)\"\"\"
        # This seems like a simplified version of logic within apply_position_sizing.
        # Consider consolidating or removing.
        self.logger.warning("_apply_position_limits called - may be redundant with apply_position_sizing logic.")
        if self.risk_limits is None:
             return position_sizes # No limits to apply

        # Calculate maximum position size based on equity percentage limit
        equity_based_limit = pd.Series(float('inf'), index=position_sizes.index) # Default to no limit
        if hasattr(self.risk_limits, 'position_limit_pct') and self.risk_limits.position_limit_pct > 0:
            if self.contract_spec and self.contract_spec.margin > 0:
                 equity_based_limit = (equity * self.risk_limits.position_limit_pct) / self.contract_spec.margin
            else:
                 self.logger.warning("Cannot apply equity pct limit without margin info.")

        # Absolute position limit
        abs_limit = float('inf')
        if hasattr(self.risk_limits, 'max_position_size') and self.risk_limits.max_position_size > 0:
            abs_limit = self.risk_limits.max_position_size

        # Apply combined limits (vectorized)
        effective_upper_limit = np.minimum(abs_limit, equity_based_limit)
        return position_sizes.clip(
            lower=-effective_upper_limit,
            upper=effective_upper_limit
        )

# --- Factory --- #

class RiskManagerFactory:
    \"\"\"Factory for creating risk manager instances.\"\"\"

    @staticmethod
    def create(config: Config, risk_metrics: RiskMetrics) -> RiskManager:
        \"\"\"
        Create a risk manager instance based on configuration.

        Args:
            config: Complete configuration object
            risk_metrics: Initialized RiskMetrics instance

        Returns:
            Configured risk manager instance
        \"\"\"
        risk_config = config.get_risk_manager_config()
        manager_type = risk_config.get('type', 'volatility').lower() # Default to volatility

        logger = LoggingConfig.get_logger(__name__)
        logger.info(f"Creating risk manager of type: {manager_type}")

        if manager_type == 'volatility':
            return VolatilityTargetRiskManager(config, risk_metrics)
        elif manager_type == 'sharpe':
            return SharpeRatioRiskManager(config, risk_metrics)
        elif manager_type == 'adaptive':
            return AdaptiveRiskManager(config, risk_metrics)
        elif manager_type == 'combined':
            logger.info("Creating combined risk manager components...")
            # Create all managers for combination
            vol_manager = VolatilityTargetRiskManager(config, risk_metrics)
            sharpe_manager = SharpeRatioRiskManager(config, risk_metrics)
            adaptive_manager = AdaptiveRiskManager(config, risk_metrics)

            # Get weights from config or use default equal weights
            weights = risk_config.get('combined_weights')
            num_managers = 3
            if weights is None or len(weights) != num_managers:
                logger.warning(f"Invalid or missing 'combined_weights' in config. Using equal weights.")
                weights = [1.0 / num_managers] * num_managers
            elif not np.isclose(sum(weights), 1.0):
                logger.warning(f"Combined weights {weights} do not sum to 1.0. Normalizing.")
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

            return CombinedRiskManager(
                config=config,
                risk_metrics=risk_metrics,
                managers=[vol_manager, sharpe_manager, adaptive_manager],
                weights=weights
            )
        else:
            logger.warning(f"Unknown risk manager type: '{manager_type}'. Using VolatilityTargetRiskManager as default.")
            return VolatilityTargetRiskManager(config, risk_metrics)

# --- Concrete Implementations --- #

class VolatilityTargetRiskManager(RiskManager):
    \"\"\"Concrete implementation of RiskManager using volatility targeting.\"\"\"

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)

        # Add debug logging for config values
        vol_params_dict = config.volatility_params if isinstance(config.volatility_params, dict) else {}
        self.logger.debug(f"Raw volatility params from config: {vol_params_dict}")

        # Extract and validate volatility parameters from config
        # Provide defaults directly in .get()
        vol_params_processed = {
            'target_volatility': float(vol_params_dict.get('target_volatility', 0.15)),
            'estimation_window': int(vol_params_dict.get('estimation_window', 63)),
            'min_scaling': float(vol_params_dict.get('min_scaling', 0.5)),
            'max_scaling': float(vol_params_dict.get('max_scaling', 2.0)),
            'adaptation_rate': float(vol_params_dict.get('adaptation_rate', 0.1)),
            'vol_target_range': tuple(vol_params_dict.get('vol_target_range', (0.10, 0.20)))
        }

        self.logger.info(f"Processed volatility parameters: {vol_params_processed}")

        try:
             # Initialize with validated parameters
             self.vol_params = VolatilityParams(**vol_params_processed)
        except TypeError as e:
             self.logger.error(f"Error initializing VolatilityParams dataclass: {e}. Check config structure.")
             # Handle error appropriately, maybe raise or use default params
             raise ValueError(f"Invalid volatility_params structure in config: {e}") from e

        self.current_vol = None # Store last known vol?
        self.current_scaling = 1.0 # Store last scaling factor?

        self.logger.info(f"Initialized VolatilityTargetRiskManager")

    def _get_metrics_df(self, base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> Optional[pd.DataFrame]:
        \"\"\"Helper to extract or calculate metrics DataFrame.\"\"\"
        metrics_df = None
        if isinstance(base_returns_or_metrics, pd.DataFrame):
            metrics_df = base_returns_or_metrics
            self.logger.debug("Using provided metrics DataFrame for vol targeting.")
        elif isinstance(base_returns_or_metrics, BaseReturns):
            if base_returns_or_metrics.metrics is None:
                self.logger.info("BaseReturns provided, but metrics not calculated. Calculating now...")
                base_returns_or_metrics.calculate_metrics(self.risk_metrics)

            if base_returns_or_metrics.metrics is not None:
                 metrics_data = base_returns_or_metrics.metrics
                 metrics_df = metrics_data.value if isinstance(metrics_data, MetricResult) else metrics_data
                 self.logger.debug("Using metrics from BaseReturns object for vol targeting.")
            else:
                 self.logger.warning("Could not get metrics from BaseReturns object.")
        else:
             self.logger.error(f"Unsupported type for metrics data: {type(base_returns_or_metrics)}")

        if metrics_df is not None and not isinstance(metrics_df, pd.DataFrame):
             self.logger.error(f"Expected metrics_df to be a DataFrame, but got {type(metrics_df)}")
             return None
        if metrics_df is None or metrics_df.empty:
             self.logger.warning("Metrics DataFrame is empty or None.")
             return None

        return metrics_df

    def calculate_position_sizes(self, df: pd.DataFrame,
                                 base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> pd.Series:
        \"\"\"Calculate position size multipliers using volatility targeting.\n\n        Args:
            df: The main intraday DataFrame.
            base_returns_or_metrics: BaseReturns object or DataFrame of metrics.
\n        Returns:
            pd.Series: Position size multipliers indexed like df.
        \"\"\"
        try:
            self.logger.info("Calculating position sizes using Volatility Targeting.")

            metrics_df = self._get_metrics_df(base_returns_or_metrics)
            if metrics_df is None:
                 self.logger.warning("Could not obtain valid metrics. Returning neutral scaling (1.0).")
                 return pd.Series(1.0, index=df.index)

            # Determine which volatility column to use (e.g., medium term)
            # Choose based on estimation_window or a fixed preference
            vol_window_map = {v: k for k, v in self.risk_metrics.windows.items()} # Map size to name base
            target_window_size = self.vol_params.estimation_window
            vol_col_base = None
            if target_window_size in vol_window_map:
                 vol_col_base = vol_window_map[target_window_size].split('_')[0] # e.g., 'medium'
            else:
                 # Fallback if exact window size not found
                 self.logger.warning(f"Estimation window {target_window_size} not in standard windows. Falling back to 'medium'.")
                 vol_col_base = 'medium'

            vol_col = f'vol_{vol_col_base}'
            if vol_col not in metrics_df.columns:
                self.logger.error(f"Required volatility column '{vol_col}' not found in metrics DataFrame. Available: {metrics_df.columns}. Returning neutral scaling.")
                return pd.Series(1.0, index=df.index)

            # Get daily volatility from metrics (indexed by date)
            daily_vol = metrics_df[vol_col]
            target_vol = self.vol_params.target_volatility

            self.logger.info(f"\nVolatility Analysis for Sizing:")
            self.logger.info(f"Using column: {vol_col}")
            self.logger.info(f"Target volatility: {target_vol:.2%}")
            if daily_vol.empty:
                 self.logger.warning(f"Volatility series '{vol_col}' is empty. Returning neutral scaling.")
                 return pd.Series(1.0, index=df.index)

            self.logger.info(f"Raw daily vol stats - mean: {daily_vol.mean():.2%}, min: {daily_vol.min():.2%}, max: {daily_vol.max():.2%}")

            # Replace zero/negative/NaN values with target vol for safe division
            daily_vol_safe = daily_vol.replace(0, target_vol).mask(daily_vol < 0, target_vol).fillna(target_vol)

            # --- Calculate Scaling --- #
            # Inverse relationship: higher vol -> lower size, lower vol -> higher size
            daily_scaling = target_vol / daily_vol_safe

            # Apply adaptation rate (smoothing)
            if self.vol_params.adaptation_rate > 0 and self.vol_params.adaptation_rate < 1:
                 self.logger.debug(f"Applying EWM smoothing with alpha={self.vol_params.adaptation_rate}")
                 daily_scaling = daily_scaling.ewm(alpha=self.vol_params.adaptation_rate, adjust=False).mean()

            # --- Apply Scaling Limits --- #
            daily_scaling_clipped = daily_scaling.clip(
                lower=self.vol_params.min_scaling,
                upper=self.vol_params.max_scaling
            )

            # --- Forward Fill Daily Scaling to Intraday --- #
            # Ensure metrics_df index is DatetimeIndex for date matching
            if not isinstance(metrics_df.index, pd.DatetimeIndex):
                 self.logger.error("Metrics index is not DatetimeIndex, cannot map to intraday.")
                 return pd.Series(1.0, index=df.index)

            # Create a temporary mapping Series
            scaling_map = pd.Series(daily_scaling_clipped.values, index=metrics_df.index.date)

            # Map daily scaling to the intraday DataFrame's date index
            intraday_scaling = df.index.date.map(scaling_map)

            # Handle potential NaNs at the beginning (before first scaling value)
            intraday_scaling = intraday_scaling.fillna(method='bfill') # Backfill first
            intraday_scaling = intraday_scaling.fillna(1.0) # Fill remaining (if any) with 1.0

            # Add debugging info
            self.logger.info(f"\nVolatility-Based Position Sizing Multipliers:")
            self.logger.info(f"Average daily vol used: {daily_vol_safe.mean():.2%}")
            self.logger.info(f"Average daily scaling (before clip): {daily_scaling.mean():.2f}")
            self.logger.info(f"Average clipped daily scaling: {daily_scaling_clipped.mean():.2f}")
            self.logger.info(f"Final intraday scaling stats - mean: {intraday_scaling.mean():.2f}, min: {intraday_scaling.min():.2f}, max: {intraday_scaling.max():.2f}")

            # Ensure the result has the same index as the input df
            intraday_scaling.index = df.index
            return intraday_scaling

        except Exception as e:
            self.logger.error(f"Volatility target position size calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=df.index) # Fallback to neutral scaling

    def _calculate_volatility_scaling(self, risk_metrics: pd.DataFrame) -> pd.Series:
        \"\"\"Calculate volatility-based position scaling with safety checks. (Internal helper - potentially redundant)\"\"\"
        # This logic seems better placed within calculate_position_sizes directly.
        # Marking as potentially redundant.
        self.logger.warning("_calculate_volatility_scaling called - may be redundant.")
        try:
            # Determine the vol column based on estimation window
            vol_col = f'vol_{self.vol_params.estimation_window}' # Simplistic mapping
            if vol_col not in risk_metrics.columns:
                 # Fallback or smarter selection needed
                 vol_col = 'vol_medium' # Example fallback
                 if vol_col not in risk_metrics.columns:
                     self.logger.warning(f"Rolling volatility column '{vol_col}' not found. Using default scaling.")
                     return pd.Series(1.0, index=risk_metrics.index)

            current_vol = risk_metrics[vol_col]

            # Handle zero or NaN volatility - replace with target for safe division
            current_vol_safe = current_vol.replace(0, self.vol_params.target_volatility)
            current_vol_safe = current_vol_safe.fillna(self.vol_params.target_volatility)

            target_vol = self.vol_params.target_volatility

            # Ensure volatility target stays within configured range
            if hasattr(self.vol_params, 'vol_target_range'):
                 min_vol_target, max_vol_target = self.vol_params.vol_target_range
                 target_vol = np.clip(target_vol, min_vol_target, max_vol_target)

            # Calculate scaling with safety checks for division by zero
            scaling = np.where(
                current_vol_safe > 1e-9, # Check against small number
                target_vol / current_vol_safe,
                1.0 # Default scaling if safe vol is near zero
            )

            # Clip scaling to prevent extreme values
            scaling = np.clip(
                scaling,
                self.vol_params.min_scaling,
                self.vol_params.max_scaling
            )

            return pd.Series(scaling, index=risk_metrics.index)

        except Exception as e:
            self.logger.error(f"Volatility scaling calculation failed: {str(e)}")
            return pd.Series(1.0, index=risk_metrics.index)

    def _calculate_drawdown_adjustment(self, risk_metrics: pd.DataFrame) -> pd.Series:
        \"\"\"Calculate drawdown-based position adjustment. (Could be part of combined manager)\"\"\"
        # This might be better suited for a combined manager or as an optional overlay
        self.logger.debug("Calculating drawdown adjustment (Note: currently simple linear scaling).")
        try:
            # Use drawdown from risk metrics if available
            if 'drawdown' not in risk_metrics.columns:
                self.logger.warning("Drawdown column not found in risk metrics. Using default adjustment (1.0).")
                return pd.Series(1.0, index=risk_metrics.index)

            drawdown = risk_metrics['drawdown'] # Drawdown is usually negative

            # Calculate adjustment factor (example: linear scaling based on drawdown)
            # Reduce size more as drawdown increases
            max_drawdown_limit = abs(self.risk_limits.max_drawdown) if hasattr(self.risk_limits, 'max_drawdown') else 0.5 # Default max DD 50%
            if max_drawdown_limit <= 0:
                 return pd.Series(1.0, index=risk_metrics.index) # No adjustment if limit is zero/negative

            # Scale from 1.0 (no drawdown) down to a minimum (e.g., 0.5) at max_drawdown_limit
            # Adjustment = 1.0 - (abs(current_drawdown) / max_drawdown_limit) * (1.0 - min_adjustment)
            min_adjustment = 0.5 # Example: never reduce below 50%
            adjustment = 1.0 - (abs(drawdown) / max_drawdown_limit) * (1.0 - min_adjustment)

            # Ensure adjustment stays within reasonable bounds [min_adjustment, 1.0]
            adjustment = adjustment.clip(min_adjustment, 1.0)

            return adjustment

        except Exception as e:
            self.logger.error(f"Drawdown adjustment calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=risk_metrics.index)

    def get_risk_summary(self) -> dict:
        \"\"\"Get current risk metrics summary for Volatility Targeting.\"\"\"
        summary = {
            'manager_type': 'VolatilityTarget',
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'current_volatility': self.current_vol if self.current_vol is not None else 'N/A',
            'current_scaling': self.current_scaling if self.current_scaling is not None else 'N/A',
            'config_params': self.vol_params.__dict__, # Include relevant params
            'risk_limits': self.risk_limits.__dict__
        }
        return summary

class SharpeRatioRiskManager(RiskManager):
    \"\"\"Risk management based on Sharpe ratio and volatility targeting.\"\"\"

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)

        sharpe_params_dict = config.sharpe_params if isinstance(config.sharpe_params, dict) else {}
        self.logger.debug(f"Raw Sharpe params from config: {sharpe_params_dict}")

        # Initialize Sharpe parameters with defaults
        params_processed = {
            'target_sharpe': float(sharpe_params_dict.get('target_sharpe', 1.0)),
            'min_scaling': float(sharpe_params_dict.get('min_scaling', 0.5)),
            'max_scaling': float(sharpe_params_dict.get('max_scaling', 2.0)),
            'target_volatility': float(sharpe_params_dict.get('target_volatility', 0.15)),
            'min_trades': int(sharpe_params_dict.get('min_trades', 5)), # Currently unused in calc
            'risk_free_rate': float(sharpe_params_dict.get('risk_free_rate', 0.02)), # Used by RiskMetrics
            'adaptation_rate': float(sharpe_params_dict.get('adaptation_rate', 0.1)),
            'target_range': tuple(sharpe_params_dict.get('target_range', (0.5, 2.0))), # Currently unused in calc
            'window_type': str(sharpe_params_dict.get('window_type', 'medium')).lower() # 'short', 'medium', 'long'
        }
        self.logger.info(f"Processed Sharpe parameters: {params_processed}")

        try:
            self.params = SharpeParams(**params_processed)
        except TypeError as e:
             self.logger.error(f"Error initializing SharpeParams dataclass: {e}. Check config structure.")
             raise ValueError(f"Invalid sharpe_params structure in config: {e}") from e

        self.current_scaling = 1.0 # Track last scaling factor
        self.current_sharpe = None # Track last sharpe used

        self.logger.info(f"Initialized SharpeRatioRiskManager")

    def _get_metrics_df(self, base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> Optional[pd.DataFrame]:
        \"\"\"Helper to extract or calculate metrics DataFrame.\"\"\"
        # Same helper as in VolatilityTargetRiskManager
        metrics_df = None
        if isinstance(base_returns_or_metrics, pd.DataFrame):
            metrics_df = base_returns_or_metrics
            self.logger.debug("Using provided metrics DataFrame for Sharpe targeting.")
        elif isinstance(base_returns_or_metrics, BaseReturns):
            if base_returns_or_metrics.metrics is None:
                self.logger.info("BaseReturns provided, metrics not calculated. Calculating now...")
                base_returns_or_metrics.calculate_metrics(self.risk_metrics)
            if base_returns_or_metrics.metrics is not None:
                 metrics_data = base_returns_or_metrics.metrics
                 metrics_df = metrics_data.value if isinstance(metrics_data, MetricResult) else metrics_data
                 self.logger.debug("Using metrics from BaseReturns object for Sharpe targeting.")
            else:
                 self.logger.warning("Could not get metrics from BaseReturns object.")
        else:
             self.logger.error(f"Unsupported type for metrics data: {type(base_returns_or_metrics)}")

        if metrics_df is not None and not isinstance(metrics_df, pd.DataFrame):
             self.logger.error(f"Expected metrics_df to be a DataFrame, but got {type(metrics_df)}")
             return None
        if metrics_df is None or metrics_df.empty:
             self.logger.warning("Metrics DataFrame is empty or None.")
             return None
        return metrics_df

    def calculate_position_sizes(self, df: pd.DataFrame,
                                 base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> pd.Series:
        \"\"\"Calculate position size multipliers based on Sharpe ratio and target volatility.\n\n        Args:
            df: The main intraday DataFrame.
            base_returns_or_metrics: BaseReturns object or DataFrame of metrics.
\n        Returns:
            pd.Series: Position size multipliers indexed like df.
        \"\"\"
        try:
            self.logger.info("Calculating position sizes using Sharpe Ratio Targeting.")
            metrics_df = self._get_metrics_df(base_returns_or_metrics)
            if metrics_df is None:
                self.logger.warning("Could not obtain valid metrics. Returning neutral scaling (1.0).")
                return pd.Series(1.0, index=df.index)

            # Get required metrics based on window_type
            window = self.params.window_type
            sharpe_col = f'sharpe_{window}'
            vol_col = f'vol_{window}'

            if not all(col in metrics_df.columns for col in [sharpe_col, vol_col]):
                missing = {col for col in [sharpe_col, vol_col] if col not in metrics_df.columns}
                self.logger.warning(f"Missing required metrics columns ({missing}) for window '{window}'. Returning neutral scaling.")
                return pd.Series(1.0, index=df.index)

            # --- Calculate Daily Scaling --- #
            daily_scaling = pd.Series(1.0, index=metrics_df.index) # Default to 1.0
            valid_mask = metrics_df[sharpe_col].notna() & metrics_df[vol_col].notna() & (metrics_df[vol_col] > 1e-9)

            if valid_mask.any():
                # Get valid data
                sharpe = metrics_df.loc[valid_mask, sharpe_col].clip(-10, 10) # Clip extreme Sharpe
                vol = metrics_df.loc[valid_mask, vol_col] # Already positive due to mask

                # Store current Sharpe for summary
                if not sharpe.empty:
                    self.current_sharpe = sharpe.iloc[-1]

                # 1. Sharpe Scaling Component
                sharpe_scaling = (sharpe / self.params.target_sharpe)
                # Clip immediately to prevent extremes before combining
                sharpe_scaling = sharpe_scaling.clip(self.params.min_scaling, self.params.max_scaling)

                # 2. Volatility Scaling Component
                vol_scaling = (self.params.target_volatility / vol)
                vol_scaling = vol_scaling.clip(self.params.min_scaling, self.params.max_scaling)

                # 3. Combine Scaling (Multiplicative)
                combined_scaling = sharpe_scaling * vol_scaling

                # 4. Apply Adaptation Rate (Smoothing)
                if self.params.adaptation_rate > 0 and self.params.adaptation_rate < 1:
                    self.logger.debug(f"Applying EWM smoothing with alpha={self.params.adaptation_rate}")
                    combined_scaling = combined_scaling.ewm(alpha=self.params.adaptation_rate, adjust=False).mean()

                # Store combined scaling for the valid days
                daily_scaling.loc[valid_mask] = combined_scaling
            else:
                 self.logger.warning("No valid data points found for Sharpe/Vol scaling calculation.")

            # --- Apply Final Limits & Forward Fill --- # 
            daily_scaling_clipped = daily_scaling.clip(
                self.params.min_scaling,
                self.params.max_scaling
            ).fillna(1.0) # Fill any initial NaNs with 1.0

            # Store last scaling factor
            if not daily_scaling_clipped.empty:
                 self.current_scaling = daily_scaling_clipped.iloc[-1]

            # Forward fill daily scaling to intraday points
            if not isinstance(metrics_df.index, pd.DatetimeIndex):
                 self.logger.error("Metrics index is not DatetimeIndex, cannot map to intraday.")
                 return pd.Series(1.0, index=df.index)

            scaling_map = pd.Series(daily_scaling_clipped.values, index=metrics_df.index.date)
            intraday_scaling = df.index.date.map(scaling_map)
            intraday_scaling = intraday_scaling.fillna(method='bfill').fillna(1.0)

            # Log sizing statistics
            self.logger.info(f"\nSharpe-Based Position Sizing Multipliers (Window: {window}):")
            self.logger.info(f"Target Sharpe: {self.params.target_sharpe:.2f}, Target Vol: {self.params.target_volatility:.2%}")
            self.logger.info(f"Average daily scaling (clipped): {daily_scaling_clipped.mean():.2f}")
            self.logger.info(f"Final intraday scaling stats - mean: {intraday_scaling.mean():.2f}, min: {intraday_scaling.min():.2f}, max: {intraday_scaling.max():.2f}")

            # Ensure result has the same index as df
            intraday_scaling.index = df.index
            return intraday_scaling

        except Exception as e:
            self.logger.error(f"Sharpe ratio position size calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=df.index) # Fallback to neutral scaling

    def _calculate_sharpe_scaling(self, sharpe_ratio: pd.Series) -> pd.Series:
        \"\"\"Calculate scaling based on Sharpe ratio with safety checks. (Internal helper - potentially redundant)\"\"\"
        self.logger.warning("_calculate_sharpe_scaling called - logic is likely within calculate_position_sizes.")
        try:
            # Replace invalid values (inf, NaN) with target Sharpe for scaling
            sharpe_ratio_safe = sharpe_ratio.replace([np.inf, -np.inf], self.params.target_sharpe)
            sharpe_ratio_safe = sharpe_ratio_safe.fillna(self.params.target_sharpe)

            # Calculate basic scaling factor
            scaling = sharpe_ratio_safe / self.params.target_sharpe

            # Apply adaptation rate for smoother transitions using EWM
            if self.params.adaptation_rate > 0 and self.params.adaptation_rate < 1:
                 adapted_scaling = scaling.ewm(alpha=self.params.adaptation_rate, adjust=False).mean()
                 # Update current scaling based on the last value of the smoothed series
                 if not adapted_scaling.empty:
                     self.current_scaling = adapted_scaling.iloc[-1]
                 return adapted_scaling
            else:
                 # No adaptation, update current scaling directly
                 if not scaling.empty:
                     self.current_scaling = scaling.iloc[-1]
                 return scaling

        except Exception as e:
            self.logger.error(f"Sharpe scaling calculation failed: {str(e)}")
            return pd.Series(1.0, index=sharpe_ratio.index)

    def _calculate_adaptation(self, risk_metrics: pd.DataFrame) -> float:
        \"\"\"Calculate Sharpe ratio adaptation factor. (Currently unused)\"\"\"
        # This method seems unused in the current calculate_position_sizes logic.
        # It might be useful for dynamically adjusting adaptation_rate itself.
        self.logger.debug("_calculate_adaptation (Sharpe) called but currently unused in sizing.")
        sharpe_col = f'sharpe_{self.params.window_type}'
        if sharpe_col in risk_metrics.columns and not risk_metrics[sharpe_col].empty:
            current_sharpe = risk_metrics[sharpe_col].iloc[-1]
            target_sharpe = self.params.target_sharpe
            if pd.notna(current_sharpe) and target_sharpe != 0:
                # Example adaptation logic (adjust scaling based on deviation from target)
                adaptation = np.exp(-self.params.adaptation_rate * abs(current_sharpe / target_sharpe - 1))
                return adaptation
        return 1.0

    def get_risk_summary(self) -> dict:
        \"\"\"Get current risk metrics summary for Sharpe Manager.\"\"\"
        return {
            'manager_type': 'SharpeRatio',
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'current_sharpe': self.current_sharpe if self.current_sharpe is not None else 'N/A',
            'current_scaling': self.current_scaling if self.current_scaling is not None else 'N/A',
            'config_params': self.params.__dict__,
            'risk_limits': self.risk_limits.__dict__
        }

class AdaptiveRiskManager(RiskManager):
    \"\"\"Risk management that adapts to market regimes (based on volatility)."\"\"

    def __init__(self, config: Config, risk_metrics: RiskMetrics):
        super().__init__(config, risk_metrics)

        adaptive_params_dict = config.adaptive_params if isinstance(config.adaptive_params, dict) else {}
        self.logger.debug(f"Raw Adaptive params from config: {adaptive_params_dict}")

        # Extract and validate adaptive parameters from config
        params_processed = {
            'base_volatility': float(adaptive_params_dict.get('base_volatility', 0.15)),
            'regime_window': int(adaptive_params_dict.get('regime_window', 252)), # Used for regime detection (external)
            'adaptation_rate': float(adaptive_params_dict.get('adaptation_rate', 0.1)),
            'min_scaling': float(adaptive_params_dict.get('min_scaling', 0.5)),
            'max_scaling': float(adaptive_params_dict.get('max_scaling', 2.0)),
            'vol_target_range': tuple(adaptive_params_dict.get('vol_target_range', (0.10, 0.20))), # Unused here?
            'regime_thresholds': tuple(adaptive_params_dict.get('regime_thresholds', (0.8, 1.2))) # Used for regime detection (external)
        }
        self.logger.info(f"Processed Adaptive parameters: {params_processed}")

        try:
             self.params = AdaptiveParams(**params_processed)
        except TypeError as e:
             self.logger.error(f"Error initializing AdaptiveParams dataclass: {e}. Check config structure.")
             raise ValueError(f"Invalid adaptive_params structure in config: {e}") from e

        self._current_regime = 'normal' # Default regime
        self.current_scaling = 1.0

        self.logger.info(f"Initialized AdaptiveRiskManager")

    def _get_metrics_df(self, base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> Optional[pd.DataFrame]:
        \"\"\"Helper to extract or calculate metrics DataFrame.\"\"\"
        # Same helper logic
        metrics_df = None
        if isinstance(base_returns_or_metrics, pd.DataFrame):
            metrics_df = base_returns_or_metrics
            self.logger.debug("Using provided metrics DataFrame for Adaptive targeting.")
        elif isinstance(base_returns_or_metrics, BaseReturns):
            if base_returns_or_metrics.metrics is None:
                self.logger.info("BaseReturns provided, metrics not calculated. Calculating now...")
                base_returns_or_metrics.calculate_metrics(self.risk_metrics)
            if base_returns_or_metrics.metrics is not None:
                 metrics_data = base_returns_or_metrics.metrics
                 metrics_df = metrics_data.value if isinstance(metrics_data, MetricResult) else metrics_data
                 self.logger.debug("Using metrics from BaseReturns object for Adaptive targeting.")
            else:
                 self.logger.warning("Could not get metrics from BaseReturns object.")
        else:
             self.logger.error(f"Unsupported type for metrics data: {type(base_returns_or_metrics)}")

        if metrics_df is not None and not isinstance(metrics_df, pd.DataFrame):
             self.logger.error(f"Expected metrics_df to be a DataFrame, but got {type(metrics_df)}")
             return None
        if metrics_df is None or metrics_df.empty:
             self.logger.warning("Metrics DataFrame is empty or None.")
             return None
        return metrics_df

    def _detect_regime(self, metrics_df: pd.DataFrame) -> pd.Series:
        \"\"\"Detect market regime based on rolling volatility.\n           Assumes volatility is calculated over `regime_window`.\n        \"\"\"
        # This logic ideally lives within RiskMetrics or a dedicated Regime Detector
        # For now, implementing it here based on available metrics.
        self.logger.debug("Detecting market regime (simple vol-based method).")
        vol_col = f'vol_{self.params.regime_window}' # Needs RiskMetrics config alignment
        # Find a suitable vol column if exact window not present
        if vol_col not in metrics_df.columns:
            # Try long-term vol as fallback
            vol_col = 'vol_long' if 'vol_long' in metrics_df.columns else 'vol_medium'
            if vol_col not in metrics_df.columns:
                self.logger.warning(f"Cannot detect regime: Suitable volatility column not found in {metrics_df.columns}. Defaulting to 'normal'.")
                return pd.Series('normal', index=metrics_df.index)
            self.logger.warning(f"Regime detection using fallback volatility column: {vol_col}")

        volatility = metrics_df[vol_col]
        vol_ratio = volatility / self.params.base_volatility

        low_thresh, high_thresh = self.params.regime_thresholds

        regime = pd.Series('normal', index=metrics_df.index)
        regime[vol_ratio < low_thresh] = 'low_vol'
        regime[vol_ratio > high_thresh] = 'high_vol'

        if not regime.empty:
             self._current_regime = regime.iloc[-1] # Store last detected regime

        self.logger.info(f"Regime detection results (last: {self._current_regime}):\n{regime.value_counts()}")
        return regime

    def calculate_position_sizes(self, df: pd.DataFrame,
                                 base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> pd.Series:
        \"\"\"Calculate position size multipliers using regime-based adaptation.\n\n        Args:
            df: The main intraday DataFrame.
            base_returns_or_metrics: BaseReturns object or DataFrame of metrics.
\n        Returns:
            pd.Series: Position size multipliers indexed like df.
        \"\"\"
        try:
            self.logger.info("Calculating position sizes using Adaptive Regime Targeting.")
            metrics_df = self._get_metrics_df(base_returns_or_metrics)
            if metrics_df is None:
                self.logger.warning("Could not obtain valid metrics. Returning neutral scaling (1.0).")
                return pd.Series(1.0, index=df.index)

            # Detect regime based on metrics
            # Note: Regime detection ideally uses a consistent window (e.g., long-term)
            daily_regimes = self._detect_regime(metrics_df)

            # Use a reference volatility (e.g., medium term) for scaling magnitude
            ref_vol_col = 'vol_medium'
            if ref_vol_col not in metrics_df.columns:
                 self.logger.warning(f"Reference volatility '{ref_vol_col}' not found. Using base volatility for scaling.")
                 ref_vol = pd.Series(self.params.base_volatility, index=metrics_df.index)
            else:
                 ref_vol = metrics_df[ref_vol_col].fillna(self.params.base_volatility)
                 ref_vol = ref_vol.replace(0, self.params.base_volatility).mask(ref_vol < 0, self.params.base_volatility)

            # --- Calculate Daily Scaling based on Regime --- #
            daily_scaling = pd.Series(1.0, index=metrics_df.index)
            target_vol = self.params.base_volatility # Use base vol as the target

            # Adjust target volatility or scaling factor based on regime
            # Example: Reduce size in high vol, increase in low vol
            scale_factor = pd.Series(1.0, index=metrics_df.index)
            if 'high_vol' in daily_regimes.values:
                 scale_factor[daily_regimes == 'high_vol'] = 0.75 # Reduce size by 25% in high vol
            if 'low_vol' in daily_regimes.values:
                 scale_factor[daily_regimes == 'low_vol'] = 1.25 # Increase size by 25% in low vol
            # Normal regime uses factor 1.0

            # Combine vol targeting with regime factor
            # Scaling = (Target Vol / Reference Vol) * Regime Scale Factor
            combined_scaling = (target_vol / ref_vol) * scale_factor

            # Apply adaptation (smoothing)
            if self.params.adaptation_rate > 0 and self.params.adaptation_rate < 1:
                 combined_scaling = combined_scaling.ewm(alpha=self.params.adaptation_rate, adjust=False).mean()

            # --- Apply Limits & Forward Fill --- #
            daily_scaling_clipped = combined_scaling.clip(
                lower=self.params.min_scaling,
                upper=self.params.max_scaling
            ).fillna(1.0)

            if not daily_scaling_clipped.empty:
                 self.current_scaling = daily_scaling_clipped.iloc[-1]

            # Forward fill to intraday
            if not isinstance(metrics_df.index, pd.DatetimeIndex):
                 self.logger.error("Metrics index is not DatetimeIndex, cannot map to intraday.")
                 return pd.Series(1.0, index=df.index)

            scaling_map = pd.Series(daily_scaling_clipped.values, index=metrics_df.index.date)
            intraday_scaling = df.index.date.map(scaling_map)
            intraday_scaling = intraday_scaling.fillna(method='bfill').fillna(1.0)

            self.logger.info(f"\nAdaptive Position Sizing Multipliers (Last Regime: {self._current_regime}):")
            self.logger.info(f"Average daily scaling (clipped): {daily_scaling_clipped.mean():.2f}")
            self.logger.info(f"Final intraday scaling stats - mean: {intraday_scaling.mean():.2f}, min: {intraday_scaling.min():.2f}, max: {intraday_scaling.max():.2f}")

            intraday_scaling.index = df.index
            return intraday_scaling

        except Exception as e:
            self.logger.error(f"Adaptive position size calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=df.index)

    def get_risk_summary(self) -> dict:
        \"\"\"Get current risk metrics summary for Adaptive Manager.\"\"\"
        return {
            'manager_type': 'Adaptive',
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'current_regime': self._current_regime,
            'current_scaling': self.current_scaling if self.current_scaling is not None else 'N/A',
            'config_params': self.params.__dict__,
            'risk_limits': self.risk_limits.__dict__
        }

class CombinedRiskManager(RiskManager):
    \"\"\"Combines multiple risk management approaches using weighted averaging.\"\"\"

    def __init__(self, config: Config, risk_metrics: RiskMetrics,
                 managers: List[RiskManager], weights: List[float]):
        super().__init__(config, risk_metrics)
        if not managers:
             raise ValueError("CombinedRiskManager requires at least one manager instance.")
        if len(managers) != len(weights):
             raise ValueError("Number of managers must match number of weights.")
        if not np.isclose(sum(weights), 1.0):
            self.logger.warning(f"Provided weights {weights} do not sum to 1.0. Normalizing.")
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight != 0 else [1.0/len(weights)] * len(weights)

        self.managers = managers
        self.weights = np.array(weights) # Use numpy array for easier math

        # Get optional parameters for dynamic weighting (currently unused in calc)
        combined_params = config.config.get('combined_params', {}) if hasattr(config, 'config') else {}
        self.adaptation_rate = combined_params.get('adaptation_rate', 0.0) # Default 0: no dynamic weighting
        self.performance_window = combined_params.get('performance_window', 63)

        self.logger.info(f"Initialized CombinedRiskManager with {len(managers)} managers.")
        manager_types = [type(m).__name__ for m in self.managers]
        self.logger.info(f"Manager types: {manager_types}")
        self.logger.info(f"Initial weights: {self.weights.tolist()}")
        self.logger.info(f"Dynamic weight adaptation rate: {self.adaptation_rate}")


    def calculate_position_sizes(self, df: pd.DataFrame,
                                 base_returns_or_metrics: Union[BaseReturns, pd.DataFrame]) -> pd.Series:
        \"\"\"Calculate position size multipliers by combining multiple managers.\n\n        Args:
            df: The main intraday DataFrame.
            base_returns_or_metrics: BaseReturns object or DataFrame of metrics.
\n        Returns:
            pd.Series: Combined position size multipliers indexed like df.
        \"\"\"
        try:
            self.logger.info("Calculating position sizes using Combined Risk Manager.")
            # Note: Dynamic weight updates based on performance are complex and not implemented here.
            # This implementation uses the fixed initial weights.

            # Get scaling from each manager
            individual_scalings = []
            manager_types = []
            for manager in self.managers:
                 manager_type = type(manager).__name__
                 self.logger.debug(f"Calculating scaling for manager: {manager_type}")
                 scaling = manager.calculate_position_sizes(df, base_returns_or_metrics)
                 individual_scalings.append(scaling)
                 manager_types.append(manager_type)

            # Combine scalings using weighted average
            if not individual_scalings:
                self.logger.warning("No individual scalings generated. Returning neutral scaling.")
                return pd.Series(1.0, index=df.index)

            # Ensure all scaling series have the same index as df
            aligned_scalings = [s.reindex(df.index, method='ffill').fillna(1.0) for s in individual_scalings]

            # Weighted average: sum(weight_i * scaling_i)
            weighted_scalings = np.array(aligned_scalings).T * self.weights # Transpose for broadcasting
            combined_scaling = np.sum(weighted_scalings, axis=1)

            combined_scaling_series = pd.Series(combined_scaling, index=df.index)

            # Apply final global scaling limits (from RiskParameters if available)
            # These limits act on the combined multiplier
            min_scalar = getattr(self.config, 'risk_params', {}).get('min_scalar', 0.1) # Example default
            max_scalar = getattr(self.config, 'risk_params', {}).get('max_scalar', 3.0) # Example default
            final_scaling = combined_scaling_series.clip(
                lower=min_scalar,
                upper=max_scalar
            )

            self.logger.info(f"\nCombined Position Sizing Multipliers:")
            for i, mgr_type in enumerate(manager_types):
                 self.logger.info(f"  - Avg scaling from {mgr_type} (weight {self.weights[i]:.2f}): {aligned_scalings[i].mean():.2f}")
            self.logger.info(f"Avg combined scaling (before global limits): {combined_scaling_series.mean():.2f}")
            self.logger.info(f"Final combined scaling stats - mean: {final_scaling.mean():.2f}, min: {final_scaling.min():.2f}, max: {final_scaling.max():.2f}")

            return final_scaling

        except Exception as e:
            self.logger.error(f"Combined position size calculation failed: {str(e)}", exc_info=True)
            return pd.Series(1.0, index=df.index) # Fallback

    def _calculate_manager_performance(self, manager: RiskManager,
                                       daily_performance: pd.DataFrame,
                                       risk_metrics_df: pd.DataFrame) -> float:
        \"\"\"Calculate a performance score for a manager (Example implementation - unused).\n           Requires careful definition of 'performance'.
        \"\"\"
        self.logger.debug(f"Calculating performance for {type(manager).__name__} (currently unused).")
        if daily_performance.empty or risk_metrics_df.empty:
             return 0.0

        # Requires daily_performance to have 'returns' column corresponding to the manager's sizing
        # This is complex as we only have the combined result.
        # Placeholder logic using base returns and manager's summary:
        summary = manager.get_risk_summary()
        window_perf = daily_performance['base_returns'].tail(self.performance_window)
        if window_perf.empty:
            return 0.0

        # Example score: Rolling Sharpe over the window
        vol = window_perf.std() * np.sqrt(252)
        mean_ret = (1 + window_perf).prod() ** (252.0 / len(window_perf)) - 1
        rf = self.risk_metrics.risk_free_rate
        sharpe = (mean_ret - rf) / vol if vol > 1e-9 else 0.0

        # Could incorporate drawdown from summary
        # performance = sharpe * (1 - abs(summary.get('current_drawdown', 0))) 
        performance = sharpe

        return max(performance, 0)  # Ensure non-negative score

    def _update_weights(self, daily_performance: pd.DataFrame, risk_metrics_df: pd.DataFrame) -> None:
        \"\"\"Update manager weights based on performance (complex, unused in current calc).\"\"\"
        if self.adaptation_rate <= 0:
             self.logger.debug("Dynamic weight adaptation disabled (rate=0).")
             return # No dynamic updates

        self.logger.info("Attempting to update manager weights based on performance...")
        performances = []
        for manager in self.managers:
            # Calculating individual manager performance retroactively is hard
            # Needs simulation of each manager independently or a different approach
            # Using placeholder calculation for now
            perf = self._calculate_manager_performance(manager, daily_performance, risk_metrics_df)
            performances.append(perf)

        if not performances or sum(performances) <= 0:
            self.logger.warning("Could not calculate positive performance scores for dynamic weighting.")
            return

        # Calculate new target weights based on relative performance
        total_perf = sum(performances)
        target_weights = np.array([p / total_perf for p in performances])

        # Apply gradual adaptation (Exponential Moving Average)
        new_weights = self.weights + self.adaptation_rate * (target_weights - self.weights)

        # Normalize weights to ensure they sum to 1
        self.weights = new_weights / np.sum(new_weights)

        self.logger.info(f"Updated manager weights (Adapt Rate: {self.adaptation_rate}): {self.weights.tolist()}")

    def get_risk_summary(self) -> dict:
        \"\"\"Get current risk metrics summary for Combined Manager.\"\"\"
        manager_summaries = {type(m).__name__: m.get_risk_summary() for m in self.managers}
        return {
            'manager_type': 'Combined',
            'current_drawdown': self.current_drawdown, # Overall drawdown
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'weights': self.weights.tolist(),
            'manager_summaries': manager_summaries,
            'risk_limits': self.risk_limits.__dict__ # Global limits
        } 