import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Tuple, Optional, Union

from .base import Strategy # Import base class
from ..config import Config, LoggingConfig
from ..models import StrategyParameters, ExitReason, ContractSpecification

class IntradayMomentumStrategy(Strategy):
    \"\"\"Intraday momentum strategy implementation based on noise boundaries.\"\"\"

    def __init__(self, config: Config, params: StrategyParameters, invert_signals: bool = False):
        \"\"\"Initialize intraday momentum strategy.

        Args:
            config: System configuration
            params: Strategy-specific parameters (e.g., lookback, multiplier)
            invert_signals: Whether to invert strategy signals (True for counter-trend)
        \"\"\"
        # Call parent class constructor first
        super().__init__(name="IntradayMomentum", config=config)

        if not isinstance(params, StrategyParameters):
             raise TypeError("'params' must be an instance of StrategyParameters")
        self.params = params
        self.invert_signals = invert_signals
        self._validate_parameters()
        self.logger.info(f"Invert signals set to: {self.invert_signals}")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate trading signals based on noise area boundaries.\"\"\"
        try:
            self.logger.info("Generating signals for IntradayMomentumStrategy...")
            if self.contract_spec is None:
                 raise ValueError("Cannot generate signals without valid ContractSpecification.")

            df = df.copy()
            df = self.calculate_noise_area(df)

            # Define valid trading times based on strategy params and contract specs
            if not self.params.entry_times:
                 self.logger.warning("No entry times specified in StrategyParameters. No signals will be generated.")
                 df['signal'] = 0
                 df['stop'] = np.nan # Initialize stop column
                 return df

            valid_entry_minutes = set(self.params.entry_times)
            market_open = self.contract_spec.market_open
            last_entry = self.contract_spec.last_entry

            valid_trading_times = (
                (df.index.minute.isin(valid_entry_minutes)) &
                (df.index.time >= market_open) &
                (df.index.time <= last_entry)
            )

            # Initialize signals
            df['signal'] = 0

            # Check if bounds were calculated
            if 'upper_bound' not in df.columns or 'lower_bound' not in df.columns:
                 self.logger.error("Noise boundaries ('upper_bound', 'lower_bound') not found. Cannot generate signals.")
                 df['stop'] = np.nan
                 return df

            # Generate base signals (breakouts)
            long_signals = valid_trading_times & df['upper_bound'].notna() & (df['Close'] > df['upper_bound'])
            short_signals = valid_trading_times & df['lower_bound'].notna() & (df['Close'] < df['lower_bound'])

            # Apply signals based on invert_signals flag
            if self.invert_signals:
                self.logger.info("Inverting signals (counter-trend mode).")
                df.loc[long_signals, 'signal'] = -1 # Sell breakouts
                df.loc[short_signals, 'signal'] = 1  # Buy breakdowns
            else:
                self.logger.info("Generating standard momentum signals (trend mode).")
                df.loc[long_signals, 'signal'] = 1  # Buy breakouts
                df.loc[short_signals, 'signal'] = -1 # Sell breakdowns

            # Calculate trailing stops (required for exit logic)
            df = self._calculate_stops(df)
            self._log_signal_statistics(df)

            return df

        except Exception as e:
            self.logger.error("Signal generation failed in IntradayMomentumStrategy", exc_info=True)
            # Return df with zero signals and NaN stops on error
            df['signal'] = 0
            df['stop'] = np.nan
            return df

    def check_exit_conditions(self, bar: pd.Series, position: int,
                          entry_price: float, current_time: pd.Timestamp) -> Tuple[bool, str]:
        \"\"\"Check exit conditions: market close or stop loss hit.\"\"\"
        try:
            if self.contract_spec is None:
                 self.logger.error("Cannot check exits: ContractSpecification not available.")
                 return False, "" # Don't exit if spec is missing

            # 1. Market close check
            market_close_time = self.contract_spec.market_close
            # Compare time components only
            if current_time.time() >= market_close_time:
                # Log this specific exit
                # self.logger.debug(f"Exit triggered: Market Close at {current_time} (>= {market_close_time})")
                return True, ExitReason.MARKET_CLOSE.value

            # No position to exit
            if position == 0:
                return False, ""

            # 2. Stop loss check (based on 'stop' column calculated in generate_signals)
            if 'stop' not in bar or pd.isna(bar['stop']):
                # self.logger.warning(f"Stop loss level not available or NaN for bar at {current_time}. Cannot check stop exit.")
                return False, "" # Cannot exit if stop level is missing

            stop_price = bar['stop']
            current_price = bar['Close']
            stop_hit = False
            exit_reason = ""

            if position > 0:  # Long position: exit if price drops below stop
                if current_price <= stop_price:
                    stop_hit = True
                    exit_reason = ExitReason.BOUNDARY_STOP.value # Assuming stop is boundary related
                    # self.logger.debug(f"Exit triggered: Stop hit (Long) at {current_time}. Price {current_price} <= Stop {stop_price}")
            elif position < 0: # Short position: exit if price rises above stop
                if current_price >= stop_price:
                    stop_hit = True
                    exit_reason = ExitReason.BOUNDARY_STOP.value
                    # self.logger.debug(f"Exit triggered: Stop hit (Short) at {current_time}. Price {current_price} >= Stop {stop_price}")

            return stop_hit, exit_reason

        except Exception as e:
            self.logger.error(f"Error checking exit conditions at {current_time}: {str(e)}", exc_info=True)
            return False, ""  # Conservative approach: don't force exit on error

    def _validate_parameters(self) -> None:
        \"\"\"Validate strategy-specific parameters.\"\"\"
        try:
            if self.params.lookback_days <= 0:
                raise ValueError("Lookback days must be positive")
            if self.params.volatility_multiplier <= 0:
                raise ValueError("Volatility multiplier must be positive")
            if not self.params.entry_times or not isinstance(self.params.entry_times, list):
                raise ValueError("Entry times must be a non-empty list")
            if any(not isinstance(t, int) or not (0 <= t < 60) for t in self.params.entry_times):
                 raise ValueError("Entry times must be integers between 0 and 59 (minutes).")

            # Validate trading hours consistency if contract_spec is loaded
            if self.contract_spec:
                if not (self.contract_spec.market_open <= self.contract_spec.last_entry <= self.contract_spec.market_close):
                    raise ValueError("Invalid trading hours sequence in ContractSpecification: market_open <= last_entry <= market_close must hold.")
            else:
                 self.logger.warning("Cannot validate trading hours sequence: ContractSpecification not loaded.")

            self.logger.debug(f"IntradayMomentumStrategy parameters validated successfully: {self.params}")

        except Exception as e:
            self.logger.error(f"Parameter validation failed for IntradayMomentumStrategy: {str(e)}", exc_info=True)
            raise # Re-raise validation errors

    def calculate_noise_area(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate noise area boundaries based on historical average move.

        Adds 'upper_bound' and 'lower_bound' columns to the DataFrame.
        \"\"\"
        try:
            df = df.copy()
            if self.contract_spec is None:
                 raise ValueError("Cannot calculate noise area without valid ContractSpecification.")

            market_open = self.contract_spec.market_open
            market_close = self.contract_spec.market_close

            # Ensure required columns from DataManager are present
            required_cols = ['day_open', 'prev_close', 'minute_of_day', 'Close']
            if not all(col in df.columns for col in required_cols):
                 missing = set(required_cols) - set(df.columns)
                 raise ValueError(f"Missing required columns for noise area calculation: {missing}")

            # Create market hours mask
            market_hours = (
                    (df.index.time >= market_open) &
                    (df.index.time <= market_close)
            )

            # Calculate absolute percentage move from open during market hours only
            df['pct_move_from_open'] = np.nan
            market_hours_valid_open = market_hours & df['day_open'].notna() & (df['day_open'] != 0)
            df.loc[market_hours_valid_open, 'pct_move_from_open'] = (
                    abs(df.loc[market_hours_valid_open, 'Close'] - df.loc[market_hours_valid_open, 'day_open']) /
                    df.loc[market_hours_valid_open, 'day_open']
            )

            # Calculate rolling average move for each minute of the day
            # Ensure sufficient periods for rolling calculation
            min_periods_lookback = max(5, self.params.lookback_days // 4)
            df['avg_move'] = df.groupby('minute_of_day')['pct_move_from_open'].transform(
                lambda x: x.rolling(
                    window=self.params.lookback_days,
                    min_periods=min_periods_lookback # Use a minimum number of days
                ).mean()
            )
            # Forward fill avg_move to handle initial NaNs if needed for boundaries
            df['avg_move'] = df['avg_move'].ffill()

            # Calculate reference prices for bounds (handle potential NaNs)
            df['ref_price_high'] = df[['day_open', 'prev_close']].max(axis=1)
            df['ref_price_low'] = df[['day_open', 'prev_close']].min(axis=1)

            # Calculate boundaries during market hours only, where avg_move is valid
            df['upper_bound'] = np.nan
            df['lower_bound'] = np.nan

            mult = self.params.volatility_multiplier
            valid_bounds_mask = market_hours & df['avg_move'].notna() & df['ref_price_high'].notna() & df['ref_price_low'].notna()

            df.loc[valid_bounds_mask, 'upper_bound'] = (
                    df.loc[valid_bounds_mask, 'ref_price_high'] *
                    (1 + mult * df.loc[valid_bounds_mask, 'avg_move'])
            )
            df.loc[valid_bounds_mask, 'lower_bound'] = (
                    df.loc[valid_bounds_mask, 'ref_price_low'] *
                    (1 - mult * df.loc[valid_bounds_mask, 'avg_move'])
            )

            # Ensure lower bound is not above upper bound
            if 'upper_bound' in df.columns and 'lower_bound' in df.columns:
                invalid_order_mask = valid_bounds_mask & (df['lower_bound'] > df['upper_bound'])
                if invalid_order_mask.any():
                    self.logger.warning(f"Found {invalid_order_mask.sum()} instances where lower_bound > upper_bound. Swapping them.")
                    lower_vals = df.loc[invalid_order_mask, 'lower_bound'].copy()
                    upper_vals = df.loc[invalid_order_mask, 'upper_bound'].copy()
                    df.loc[invalid_order_mask, 'upper_bound'] = lower_vals
                    df.loc[invalid_order_mask, 'lower_bound'] = upper_vals

            self.logger.debug("Noise area calculation complete.")
            return df

        except Exception as e:
            self.logger.error(f"Noise area calculation failed: {str(e)}", exc_info=True)
            # Ensure bound columns exist even on error, filled with NaN
            df['upper_bound'] = np.nan
            df['lower_bound'] = np.nan
            raise # Re-raise the exception after adding columns

    def _calculate_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Calculate stop loss levels. For this strategy, use VWAP as the stop.\"\"\"
        try:
            df = df.copy()
            if self.contract_spec is None:
                 raise ValueError("Cannot calculate stops without ContractSpecification.")

            market_open = self.contract_spec.market_open
            market_close = self.contract_spec.market_close
            market_hours = (
                    (df.index.time >= market_open) &
                    (df.index.time <= market_close)
            )

            # Initialize stop column
            df['stop'] = np.nan

            # Check if VWAP is available
            if 'vwap' not in df.columns:
                 self.logger.error("VWAP column not found. Cannot set VWAP as stop loss.")
                 return df # Return df without stops if VWAP missing

            # Set VWAP as stop level during market hours
            df.loc[market_hours, 'stop'] = df.loc[market_hours, 'vwap']

            # Optional: Add logging for stop calculation
            self.logger.debug("Stop loss levels calculated using VWAP.")
            # self.logger.debug(f"Stop levels sample:\n{df['stop'].dropna().head()}")

            return df

        except Exception as e:
            self.logger.error("Stop calculation failed", exc_info=True)
            df['stop'] = np.nan # Ensure stop column exists with NaN on error
            raise # Re-raise

    def _log_signal_statistics(self, df: pd.DataFrame) -> None:
        \"\"\"Log statistics about generated signals and boundaries.\"\"\"
        try:
            if 'signal' not in df.columns:
                self.logger.warning("Cannot log signal stats: 'signal' column missing.")
                return

            if self.contract_spec is None:
                 self.logger.warning("Cannot log signal stats: ContractSpecification missing.")
                 return

            # Calculate valid trading times again for context
            valid_entry_minutes = set(self.params.entry_times or [])
            market_open = self.contract_spec.market_open
            last_entry = self.contract_spec.last_entry
            valid_trading_times_mask = (
                (df.index.minute.isin(valid_entry_minutes)) &
                (df.index.time >= market_open) &
                (df.index.time <= last_entry)
            )

            # Count signals by type within valid times
            signals_in_valid_time = df.loc[valid_trading_times_mask, 'signal']
            long_signals = (signals_in_valid_time == 1).sum()
            short_signals = (signals_in_valid_time == -1).sum()
            total_valid_bars = valid_trading_times_mask.sum()

            self.logger.info("\n--- IntradayMomentum Signal Statistics ---")
            self.logger.info(f"Total bars considered for entry: {total_valid_bars:,}")
            self.logger.info(f"Long entry signals generated: {long_signals:,}")
            self.logger.info(f"Short entry signals generated: {short_signals:,}")

            if total_valid_bars > 0:
                signal_rate = ((long_signals + short_signals) / total_valid_bars * 100)
                self.logger.info(f"Signal generation rate (within valid times): {signal_rate:.2f}%")
            else:
                 self.logger.info("Signal generation rate: N/A (no valid bars for entry)")

            # Daily signal distribution
            if not df.empty:
                daily_data = pd.DataFrame({
                    'date': df.index.date,
                    'signals': df['signal'].abs()
                })
                daily_signals = daily_data.groupby('date')['signals'].apply(lambda x: (x != 0).sum())

                if not daily_signals.empty:
                    self.logger.info("\nDaily Signal Distribution:")
                    self.logger.info(f"  Days with signals: {(daily_signals > 0).sum()} / {len(daily_signals)} days")
                    self.logger.info(f"  Average signals per day (with signals): {daily_signals[daily_signals > 0].mean():.2f}")
                    self.logger.info(f"  Max signals in one day: {daily_signals.max():.0f}")

            # Boundary statistics (if columns exist)
            if 'upper_bound' in df.columns and 'lower_bound' in df.columns and 'ref_price_high' in df.columns:
                self.logger.info("\nBoundary Statistics (Avg during market hours):")
                market_hour_mask = (df.index.time >= market_open) & (df.index.time <= self.contract_spec.market_close)
                bounds_df = df[market_hour_mask].copy()
                bounds_df = bounds_df.dropna(subset=['upper_bound', 'lower_bound', 'Close', 'ref_price_high'])
                if not bounds_df.empty:
                     avg_noise_width_pct = ((bounds_df['upper_bound'] - bounds_df['lower_bound']) / bounds_df['ref_price_high']).mean()
                     avg_dist_upper_pct = ((bounds_df['upper_bound'] - bounds_df['Close']) / bounds_df['Close']).mean()
                     avg_dist_lower_pct = ((bounds_df['Close'] - bounds_df['lower_bound']) / bounds_df['Close']).mean()
                     self.logger.info(f"Average noise band width: {avg_noise_width_pct:.4%}")
                     self.logger.info(f"Average distance Close to Upper Bound: {avg_dist_upper_pct:.4%}")
                     self.logger.info(f"Average distance Close to Lower Bound: {avg_dist_lower_pct:.4%}")
                else:
                     self.logger.info("Could not calculate boundary stats (no valid data).")
            self.logger.info("--- End Signal Statistics ---")

        except Exception as e:
            self.logger.warning(f"Failed to log signal statistics: {str(e)}", exc_info=True) 