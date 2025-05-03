from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, time

from ..config import Config, LoggingConfig
from ..models import ContractSpecification, StrategyParameters, ExitReason, Trade, BaseReturns, LeveredReturns

class Strategy(ABC):
    \"\"\"
    Abstract base class for all trading strategies.

    This class defines the interface for trading strategies, focusing on signal generation
    and exit conditions. Trade execution and position sizing are handled by TradeManager
    and RiskManager respectively.
    \"\"\"

    def __init__(self, name: str, config: Config):
        \"\"\"
        Initialize strategy with basic parameters.

        Args:
            name: Strategy name
            config: System configuration containing all required parameters
        \"\"\"
        self.logger = LoggingConfig.get_logger(f"{__name__}.{name}")
        self.name = name
        self.config = config
        self.contract_spec = self._get_contract_spec()
        if not self.contract_spec:
             # Log error if spec not found, subsequent operations might fail
             self.logger.error(f"Failed to initialize ContractSpecification for symbol '{config.symbol}' in strategy '{name}'")
             # Consider raising an error or setting a flag
             # raise ValueError("Strategy initialization failed: Could not load ContractSpecification.")

        self.logger.info(f"Initialized '{name}' strategy for symbol '{config.symbol or "N/A"}'")

    @staticmethod
    def _create_timezone_aware_index(df: pd.DataFrame, dates: List[datetime.date]) -> Optional[pd.DatetimeIndex]:
        \"\"\"
        Create timezone-aware index preserving DataFrame's timezone.

        Args:
            df: DataFrame with timezone-aware index
            dates: List of dates to create index for

        Returns:
            DatetimeIndex with preserved timezone, or None if error.

        Raises:
            ValueError: If input DataFrame lacks timezone information
        \"\"\"
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
             # Logger might not be initialized here if called statically without instance
             # logging.getLogger(__name__).error("Input df must be DataFrame with DatetimeIndex.")
             return None

        if df.index.tz is None:
            # logging.getLogger(__name__).error("Input DataFrame must have timezone-aware index.")
            raise ValueError("Input DataFrame must have timezone-aware index")
            # return None

        try:
            return pd.DatetimeIndex([
                # Create timezone-naive timestamp first, then localize to the target timezone
                pd.Timestamp(date).tz_localize(None).tz_localize(df.index.tz)
                for date in dates
            ])
        except Exception as e:
             # logging.getLogger(__name__).error(f"Error creating timezone-aware index: {e}")
             return None

    def _get_contract_spec(self) -> Optional[ContractSpecification]:
        \"\"\"Get contract specification from config safely.\"\"\"
        # Consolidating this logic from RiskManager as well
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

        required_spec_keys = {'tick_size', 'multiplier', 'margin'}
        required_hour_keys = {'market_open', 'market_close', 'last_entry'}
        if not required_spec_keys.issubset(spec_data.keys()):
             self.logger.error(f"Missing keys in contract_specs for {self.config.symbol}: {required_spec_keys - set(spec_data.keys())}")
             return None
        if not required_hour_keys.issubset(hours_data.keys()):
             self.logger.error(f"Missing keys in trading_hours for {self.config.symbol}: {required_hour_keys - set(hours_data.keys())}")
             return None

        try:
            # Use helper to parse time strings
            market_open_time = self._parse_time(hours_data['market_open'])
            market_close_time = self._parse_time(hours_data['market_close'])
            last_entry_time = self._parse_time(hours_data['last_entry'])

            if None in [market_open_time, market_close_time, last_entry_time]:
                 raise ValueError("Failed to parse one or more trading times.")

            return ContractSpecification(
                symbol=self.config.symbol,
                tick_size=float(spec_data['tick_size']),
                multiplier=float(spec_data['multiplier']),
                margin=float(spec_data['margin']),
                market_open=market_open_time,
                market_close=market_close_time,
                last_entry=last_entry_time
            )
        except (ValueError, TypeError) as e:
             self.logger.error(f"Error creating ContractSpecification for {self.config.symbol}: {e}")
             return None

    def _parse_time(self, time_str: Union[str, time]) -> Optional[time]:
        \"\"\"Parse time string (HH:MM) or time object to time object safely.\"\"\"
        if isinstance(time_str, time):
            return time_str
        if isinstance(time_str, str):
            try:
                return datetime.strptime(time_str, '%H:%M').time()
            except ValueError:
                self.logger.error(f"Invalid time format '{time_str}'. Expected HH:MM.")
                return None
        self.logger.error(f"Cannot parse time: Expected str or time object, got {type(time_str)}.")
        return None

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Generate trading signals based on strategy logic.

        Args:
            df: DataFrame with market data (should include required indicators/columns).

        Returns:
            DataFrame with added 'signal' column (-1 for short, 0 for neutral, 1 for long).
            Should also add any strategy-specific columns needed for exits (e.g., 'stop').
        \"\"\"
        pass

    @abstractmethod
    def check_exit_conditions(self, bar: pd.Series, position: int, entry_price: float,
                              current_time: pd.Timestamp) -> Tuple[bool, str]:
        \"\"\"
        Check if current position should be exited based on the current bar data.

        Args:
            bar: Current price bar data (pd.Series).
            position: Current position (-1 for short, 1 for long, 0 for flat).
            entry_price: Position entry price.
            current_time: Current bar timestamp.

        Returns:
            Tuple of (should_exit: bool, exit_reason: str from ExitReason enum or custom string).
            Return (False, "") if no exit condition is met.
        \"\"\"
        pass

    def aggregate_to_daily(self, df_base: pd.DataFrame, df_levered: pd.DataFrame,
                           base_trades: List[Trade], final_trades: List[Trade]) -> pd.DataFrame:
        \"\"\"
        Aggregate intraday data to daily metrics, preserving timezone.

        Args:
            df_base: DataFrame with base strategy execution tracking.
            df_levered: DataFrame with levered strategy execution tracking.
            base_trades: List of base strategy trades.
            final_trades: List of levered strategy trades.

        Returns:
            DataFrame with daily aggregated metrics (indexed by date, tz-aware).
        \"\"\"
        try:
            # Get timezone from input data - assume they are consistent
            data_tz = df_base.index.tz
            if data_tz is None:
                # Fallback if base is naive, check levered
                data_tz = df_levered.index.tz
                if data_tz is None:
                     self.logger.warning("Input DataFrames for aggregation are timezone-naive. Proceeding with naive daily index.")
                else:
                     self.logger.warning("Base DataFrame is naive, Levered DataFrame is aware. Using Levered TZ for daily index.")
            else:
                 if df_levered.index.tz is not None and df_base.index.tz != df_levered.index.tz:
                     self.logger.warning(f"Timezone mismatch between base ({df_base.index.tz}) and levered ({df_levered.index.tz}) DataFrames. Using base TZ.")


            # Create daily index (use unique dates from both base and levered)
            all_dates = sorted(list(set(df_base.index.date) | set(df_levered.index.date)))
            if not all_dates:
                 self.logger.warning("No dates found for daily aggregation.")
                 return pd.DataFrame() # Return empty if no dates

            if data_tz:
                 daily_index = self._create_timezone_aware_index(df_base if df_base.index.tz else df_levered, all_dates)
                 if daily_index is None:
                      self.logger.error("Failed to create timezone-aware daily index. Aborting aggregation.")
                      return pd.DataFrame()
            else:
                 # Create naive daily index if inputs were naive
                 daily_index = pd.to_datetime(all_dates)

            # Initialize daily DataFrame with the correct index
            daily_data = pd.DataFrame(index=daily_index)

            # Initialize equity tracking
            base_equity = self.config.initial_equity
            final_equity = self.config.initial_equity

            # Process each trading day
            for date in all_dates:
                # Create date masks for both dataframes
                day_mask_base = df_base.index.date == date
                day_mask_levered = df_levered.index.date == date

                # Get day's data
                base_day_data = df_base[day_mask_base]
                levered_day_data = df_levered[day_mask_levered]

                # Get timestamp for current day (use index from daily_data)
                try:
                    # Match the date object to the potentially tz-aware daily_index
                    daily_timestamp = daily_data.index[daily_data.index.date == date][0]
                except IndexError:
                     self.logger.error(f"Could not find timestamp for date {date} in daily_index. Skipping aggregation for this day.")
                     continue

                # 1. Price Data (use base data as reference)
                if not base_day_data.empty:
                    self._aggregate_price_data(
                        daily_data=daily_data,
                        day_data=base_day_data,
                        timestamp=daily_timestamp
                    )
                else:
                     # Handle days with no base data (e.g., only levered trades? unlikely)
                     self.logger.warning(f"No base data for date {date}. Price aggregation skipped.")
                     for col in ['open', 'high', 'low', 'close']: daily_data.loc[daily_timestamp, col] = np.nan

                # 2. Position Metrics
                self._aggregate_position_metrics(
                    daily_data=daily_data,
                    base_data=base_day_data,
                    levered_data=levered_day_data,
                    timestamp=daily_timestamp
                )

                # 3. Process Base Strategy Trades
                base_day_trades = [t for t in base_trades if t.exit_time.date() == date]
                daily_pnl_base = sum(t.pnl for t in base_day_trades)
                daily_costs_base = sum(t.costs for t in base_day_trades)
                win_count_base = sum(1 for t in base_day_trades if t.pnl > t.costs)
                num_base_trades = len(base_day_trades)

                base_metrics = {
                    'base_pnl': daily_pnl_base,
                    'base_costs': daily_costs_base,
                    'base_trades': num_base_trades,
                    'base_win_rate': win_count_base / num_base_trades if num_base_trades > 0 else 0.0
                }
                for key, value in base_metrics.items():
                    daily_data.loc[daily_timestamp, key] = value

                # Update base equity (use PnL before costs for equity curve, but PnL - Costs for returns?)
                # Conventionally, equity curve includes costs.
                base_equity += daily_pnl_base - daily_costs_base
                daily_data.loc[daily_timestamp, 'base_equity'] = base_equity

                # 4. Process Levered Strategy Trades
                final_day_trades = [t for t in final_trades if t.exit_time.date() == date]
                daily_pnl_final = sum(t.pnl for t in final_day_trades)
                daily_costs_final = sum(t.costs for t in final_day_trades)
                win_count_final = sum(1 for t in final_day_trades if t.pnl > t.costs)
                num_final_trades = len(final_day_trades)

                final_metrics = {
                    'pnl': daily_pnl_final,
                    'costs': daily_costs_final,
                    'trades': num_final_trades,
                    'win_rate': win_count_final / num_final_trades if num_final_trades > 0 else 0.0
                }
                for key, value in final_metrics.items():
                    daily_data.loc[daily_timestamp, key] = value

                # Update final equity
                final_equity += daily_pnl_final - daily_costs_final
                daily_data.loc[daily_timestamp, 'equity'] = final_equity

            # Ensure base equity starts at initial equity if first day had no trades
            if not daily_data.empty and pd.isna(daily_data['base_equity'].iloc[0]):
                 daily_data['base_equity'].iloc[0] = self.config.initial_equity
                 daily_data['equity'].iloc[0] = self.config.initial_equity
                 # Forward fill equity for days with no activity
                 daily_data['base_equity'] = daily_data['base_equity'].ffill()
                 daily_data['equity'] = daily_data['equity'].ffill()

            # 5. Calculate Returns and Drawdowns (vectorized after loop)
            if not daily_data.empty:
                 # Returns calculations (handle potential division by zero if equity hits 0)
                 # Shift equity to get previous day's equity for return calculation
                 prev_base_equity = daily_data['base_equity'].shift(1).fillna(self.config.initial_equity)
                 prev_equity = daily_data['equity'].shift(1).fillna(self.config.initial_equity)

                 daily_data['base_returns'] = np.where(prev_base_equity != 0, (daily_data['base_equity'] / prev_base_equity) - 1, 0.0)
                 daily_data['returns'] = np.where(prev_equity != 0, (daily_data['equity'] / prev_equity) - 1, 0.0)
                 # Ensure first day return is zero
                 daily_data['base_returns'].iloc[0] = 0.0
                 daily_data['returns'].iloc[0] = 0.0


                 # High water mark and drawdown calculations
                 daily_data['base_high_water_mark'] = daily_data['base_equity'].cummax()
                 daily_data['high_water_mark'] = daily_data['equity'].cummax()
                 # Avoid division by zero for drawdown if HWM is zero
                 daily_data['base_drawdown'] = np.where(daily_data['base_high_water_mark'] != 0, (daily_data['base_equity'] / daily_data['base_high_water_mark']) - 1, 0.0)
                 daily_data['drawdown'] = np.where(daily_data['high_water_mark'] != 0, (daily_data['equity'] / daily_data['high_water_mark']) - 1, 0.0)
            else:
                 # If empty, create columns with appropriate types
                 cols = ['base_returns', 'returns', 'base_high_water_mark', 'high_water_mark', 'base_drawdown', 'drawdown']
                 for col in cols: daily_data[col] = pd.Series(dtype=float)


            # Log aggregation summary
            self._log_aggregation_summary(daily_data)

            return daily_data

        except Exception as e:
            self.logger.error(f"Daily aggregation failed: {str(e)}", exc_info=True)
            # Return empty DataFrame on failure
            return pd.DataFrame()

    def _aggregate_price_data(self, daily_data: pd.DataFrame, day_data: pd.DataFrame,
                              timestamp: pd.Timestamp) -> None:
        \"\"\"Aggregate OHLC price data for a single day.\"\"\"
        # Assumes day_data is not empty (checked before calling)
        try:
            daily_data.loc[timestamp, 'close'] = day_data['Close'].iloc[-1]
            daily_data.loc[timestamp, 'open'] = day_data['Open'].iloc[0]
            daily_data.loc[timestamp, 'high'] = day_data['High'].max()
            daily_data.loc[timestamp, 'low'] = day_data['Low'].min()
        except Exception as e:
            self.logger.error(f"Price data aggregation failed for timestamp {timestamp}: {str(e)}")
            # Set NaNs on error for this timestamp
            for col in ['open', 'high', 'low', 'close']: daily_data.loc[timestamp, col] = np.nan
            # Do not raise, allow aggregation to continue if possible

    def _aggregate_position_metrics(self, daily_data: pd.DataFrame, base_data: pd.DataFrame,
                                    levered_data: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        \"\"\"Aggregate position metrics for a single day.\"\"\"
        try:
            # Base strategy position metrics
            if not base_data.empty and 'base_position_size' in base_data.columns:
                base_positions = base_data['base_position_size']
                base_metrics = {
                    'base_position': base_positions.iloc[-1],
                    'max_base_position': abs(base_positions).max(),
                    'avg_base_position': abs(base_positions[base_positions != 0]).mean() if (base_positions != 0).any() else 0.0
                }
            else:
                base_metrics = {'base_position': 0.0, 'max_base_position': 0.0, 'avg_base_position': 0.0}

            for key, value in base_metrics.items():
                daily_data.loc[timestamp, key] = value

            # Levered strategy position metrics
            if not levered_data.empty and 'position_size' in levered_data.columns:
                levered_positions = levered_data['position_size']
                levered_metrics = {
                    'position': levered_positions.iloc[-1],
                    'max_position': abs(levered_positions).max(),
                    'avg_position': abs(levered_positions[levered_positions != 0]).mean() if (levered_positions != 0).any() else 0.0
                }
            else:
                levered_metrics = {'position': 0.0, 'max_position': 0.0, 'avg_position': 0.0}

            for key, value in levered_metrics.items():
                daily_data.loc[timestamp, key] = value

        except Exception as e:
            self.logger.error(f"Position metrics aggregation failed for timestamp {timestamp}: {str(e)}")
            # Set default values on error
            for prefix in ['', 'base_']:
                 for suffix in ['position', 'max_position', 'avg_position']:
                      daily_data.loc[timestamp, f"{prefix}{suffix}"] = 0.0
            # Do not raise, allow aggregation to continue

    def _log_aggregation_summary(self, daily_data: pd.DataFrame) -> None:
        \"\"\"Log summary of daily aggregation results.\"\"\"
        try:
            if daily_data.empty:
                self.logger.info("\nDaily Aggregation Summary: No data to summarize.")
                return

            self.logger.info("\n--- Daily Aggregation Summary ---")
            self.logger.info(f"Period: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
            self.logger.info(f"Total trading days aggregated: {len(daily_data)}")

            # Performance Summary
            self.logger.info("\nPerformance Summary:")
            if 'base_equity' in daily_data.columns and not daily_data['base_equity'].empty:
                base_return = (daily_data['base_equity'].iloc[-1] / daily_data['base_equity'].iloc[0] - 1) * 100
                self.logger.info(f"Base strategy total return: {base_return:.2f}%")
            if 'equity' in daily_data.columns and not daily_data['equity'].empty:
                total_return = (daily_data['equity'].iloc[-1] / daily_data['equity'].iloc[0] - 1) * 100
                self.logger.info(f"Levered strategy total return: {total_return:.2f}%")

            # Risk Metrics Summary
            self.logger.info("\nRisk Metrics (Daily):")
            if 'base_drawdown' in daily_data.columns:
                max_base_dd = daily_data['base_drawdown'].min() * 100
                self.logger.info(f"Max base drawdown: {max_base_dd:.2f}%")
            if 'drawdown' in daily_data.columns:
                max_dd = daily_data['drawdown'].min() * 100
                self.logger.info(f"Max levered drawdown: {max_dd:.2f}%")

            # Trade Statistics Summary
            self.logger.info("\nTrade Statistics (Daily Aggregation):")
            if 'base_trades' in daily_data.columns:
                total_base_trades = daily_data['base_trades'].sum()
                self.logger.info(f"Total base trades: {total_base_trades:,.0f}")
            if 'trades' in daily_data.columns:
                total_trades = daily_data['trades'].sum()
                self.logger.info(f"Total levered trades: {total_trades:,.0f}")

            self.logger.info("--- End Daily Aggregation Summary ---")

        except Exception as e:
            self.logger.error(f"Failed to log aggregation summary: {str(e)}") 