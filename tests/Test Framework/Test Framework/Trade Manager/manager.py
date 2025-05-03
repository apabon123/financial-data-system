import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Tuple, Optional, Union, Dict
import logging # Keep basic logging for potential standalone use or init issues

from ..config import Config, LoggingConfig
from ..models import Trade, ExitReason, ContractSpecification, TradingResults, BaseReturns, LeveredReturns, StrategyParameters
from ..Data Manager.manager import DataManager # Adjusted import path
from ..Strategy.base import Strategy # Adjusted import path
from ..Risk Manager.manager import RiskManager # Adjusted import path


class TradeManagerError(Exception):
    """Base exception for TradeManager errors."""
    pass

class TradeExecutionError(TradeManagerError):
    """Error during trade execution simulation."""
    pass

class ResultsGenerationError(TradeManagerError):
    """Error generating final trading results."""
    pass


class TradeManager:
    """
    Manages trade execution simulation based on strategy signals and risk management rules.
    Coordinates interaction between DataManager, Strategy, and RiskManager.
    Calculates PnL, costs, and maintains equity curve.
    """

    def __init__(self, config: Config, data_manager: DataManager,
                 strategy: Strategy, risk_manager: RiskManager):
        """
        Initialize TradeManager.

        Args:
            config: System configuration object.
            data_manager: Initialized DataManager instance.
            strategy: Initialized Strategy instance.
            risk_manager: Initialized RiskManager instance.
        """
        self.logger = LoggingConfig.get_logger(__name__)
        self.logger.info("Initializing TradeManager")

        if not all([config, data_manager, strategy, risk_manager]):
             msg = "Config, DataManager, Strategy, and RiskManager must be provided."
             self.logger.error(msg)
             raise ValueError(msg)

        self.config = config
        self.data_manager = data_manager
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbol = self.config.symbol
        self.contract_spec = self._get_contract_spec()

        if not self.contract_spec:
             # Critical failure if contract spec can't be loaded
             msg = f"TradeManager failed to load ContractSpecification for symbol '{self.symbol}'."
             self.logger.error(msg)
             raise TradeManagerError(msg)

        self.initial_equity = self.config.initial_equity
        self.transaction_costs = self._get_transaction_costs()

        self.logger.info(f"TradeManager initialized for symbol '{self.symbol}' with initial equity {self.initial_equity:,.2f}")


    def _get_contract_spec(self) -> Optional[ContractSpecification]:
        """Helper to get contract spec via the Strategy instance (as it validates it)."""
        # Leverage the validation already done in Strategy's init
        return self.strategy.contract_spec


    def _get_transaction_costs(self) -> Dict:
        """Get transaction cost settings from config."""
        # Default costs if not specified
        default_costs = {'commission_rate': 0.0001, 'slippage_rate': 0.00005, 'min_commission': 1.0}
        costs = self.config.transaction_costs.get(self.symbol, default_costs) if hasattr(self.config, 'transaction_costs') else default_costs

        # Validate cost types
        for key, val in costs.items():
            if not isinstance(val, (int, float)) or val < 0:
                 self.logger.warning(f"Invalid transaction cost '{key}' ({val}). Using default 0.")
                 costs[key] = 0.0 # Fallback to zero if invalid

        self.logger.debug(f"Transaction costs loaded: {costs}")
        return costs

    def run_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> TradingResults:
        """
        Run the full backtest simulation.

        Args:
            start_date: Optional start date string (YYYY-MM-DD).
            end_date: Optional end date string (YYYY-MM-DD).

        Returns:
            TradingResults object containing all backtest outputs.

        Raises:
            TradeManagerError: If critical steps like data loading or simulation fail.
        """
        try:
            self.logger.info(f"
--- Starting Backtest for Symbol: {self.symbol} ---")
            self.logger.info(f"Strategy: {self.strategy.name}")
            self.logger.info(f"Risk Manager: {type(self.risk_manager).__name__}")
            self.logger.info(f"Analysis Period: {self.config.days_to_analyze} days (plus {self.config.lookback_buffer} buffer)")

            # 1. Prepare Data
            self.logger.info("Preparing data for analysis...")
            df = self.data_manager.prepare_data_for_analysis(
                symbol=self.symbol,
                days_to_analyze=self.config.days_to_analyze,
                lookback_buffer=self.config.lookback_buffer
            )
            if df.empty:
                 raise TradeManagerError("Data preparation returned an empty DataFrame.")
            self.logger.info(f"Data prepared: {len(df)} rows, Index from {df.index.min()} to {df.index.max()}")

            # 2. Generate Strategy Signals
            self.logger.info("Generating strategy signals...")
            df = self.strategy.generate_signals(df)
            if 'signal' not in df.columns:
                raise TradeManagerError("Strategy failed to generate 'signal' column.")
            self.logger.info("Signal generation complete.")

            # 3. Prepare Base Positions (Fixed size based on initial equity)
            self.logger.info("Preparing base strategy positions...")
            df = self.risk_manager.prepare_base_strategy_positions(df)
            if 'base_position_size' not in df.columns:
                raise TradeManagerError("RiskManager failed to prepare 'base_position_size' column.")
            self.logger.info("Base position preparation complete.")

            # --- Intermediate Step: Base Strategy Simulation (Optional but useful) ---
            # Simulate trading with base_position_size to get BaseReturns
            self.logger.info("Simulating base strategy trading loop...")
            base_trades, df_with_base_results = self._simulate_trading_loop(
                df, position_column='base_position_size', equity_column='base_equity'
            )
            self.logger.info(f"Base strategy simulation complete. Trades: {len(base_trades)}")
            if 'base_equity' not in df_with_base_results.columns:
                 raise TradeManagerError("Base simulation failed to produce 'base_equity'.")

            # Create BaseReturns object
            # Calculate daily base returns from the simulated equity curve
            # Note: Daily aggregation happens later, need daily returns for RiskManager now.
            # We might need a temporary daily aggregation here or adjust RiskManager input.
            # For now, let's assume RiskManager needs daily metrics or BaseReturns object.
            # Let's perform a *temporary* daily aggregation for base returns needed by RiskManager.

            temp_daily_base = self._temp_aggregate_daily_returns(df_with_base_results, 'base_equity')
            base_returns_obj = BaseReturns(returns=temp_daily_base['base_returns'])


            # 4. Calculate Position Size Multipliers (Leverage)
            self.logger.info("Calculating position size multipliers (leverage)...")
            # RiskManager needs metrics or base returns. Pass BaseReturns object.
            position_size_multipliers = self.risk_manager.calculate_position_sizes(
                df, base_returns_or_metrics=base_returns_obj # Pass the BaseReturns object
            )
            if position_size_multipliers is None or position_size_multipliers.empty:
                raise TradeManagerError("RiskManager failed to calculate position size multipliers.")
            self.logger.info("Position size multiplier calculation complete.")


            # 5. Apply Position Sizing to get final target sizes
            self.logger.info("Applying position sizing...")
            # Initialize current_equity before applying sizing
            df_with_base_results['current_equity'] = self.initial_equity # Start equity for levered sim
            df_with_sized_results = self.risk_manager.apply_position_sizing(
                df_with_base_results, position_size_multipliers
            )
            if 'position_size' not in df_with_sized_results.columns:
                raise TradeManagerError("RiskManager failed to apply position sizing.")
            self.logger.info("Position sizing application complete.")


            # 6. Simulate Final Levered Trading Loop
            self.logger.info("Simulating final (levered) strategy trading loop...")
            final_trades, df_final_results = self._simulate_trading_loop(
                df_with_sized_results, position_column='position_size', equity_column='final_equity'
            )
            if 'final_equity' not in df_final_results.columns:
                 raise TradeManagerError("Final simulation failed to produce 'final_equity'.")
            self.logger.info(f"Final strategy simulation complete. Trades: {len(final_trades)}")


            # 7. Aggregate Results to Daily
            self.logger.info("Aggregating results to daily...")
            # Consolidate results into one DataFrame before aggregation
            # Columns needed: base_equity, base_position_size, final_equity, position_size, signals, OHLC etc.
            # Ensure all relevant columns are present in df_final_results
            if 'base_equity' not in df_final_results.columns:
                 df_final_results['base_equity'] = df_with_base_results['base_equity'] # Add from earlier step if missing
            if 'base_position_size' not in df_final_results.columns:
                 df_final_results['base_position_size'] = df_with_base_results['base_position_size']

            # Rename 'final_equity' to 'equity' for consistency with models.py
            df_final_results.rename(columns={'final_equity': 'equity'}, inplace=True)

            # Use the strategy's aggregation method
            daily_performance = self.strategy.aggregate_to_daily(
                df_final_results, # Pass the combined dataframe
                df_final_results, # Pass it again for levered part (contains 'equity', 'position_size')
                base_trades,
                final_trades
            )
            if daily_performance.empty:
                 raise ResultsGenerationError("Daily aggregation returned an empty DataFrame.")
            self.logger.info("Daily aggregation complete.")


            # 8. Create TradingResults object
            self.logger.info("Packaging final results...")
            results = TradingResults(
                symbol=self.symbol,
                strategy_name=self.strategy.name,
                base_trades=base_trades,
                final_trades=final_trades,
                trade_metrics=[], # To be populated by analysis layer if needed
                daily_performance=daily_performance,
                execution_data=df_final_results, # Store the final detailed DataFrame
                contract_spec=self.contract_spec,
                timestamp=pd.Timestamp.now(tz='UTC'),
                # Let TradingResults __post_init__ handle Base/LeveredReturns creation from daily_performance
            )
            self.logger.info("Results packaging complete.")
            self.logger.info(f"--- Backtest for Symbol: {self.symbol} Finished ---")

            return results

        except (DataManagerError, TradeManagerError, ResultsGenerationError) as e:
            self.logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            raise # Re-raise specific errors
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during backtest: {str(e)}", exc_info=True)
            raise TradeManagerError(f"Unexpected backtest error: {e}") # Wrap unexpected errors


    def _simulate_trading_loop(self, df: pd.DataFrame, position_column: str, equity_column: str) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Core loop for simulating trades bar-by-bar.

        Args:
            df: DataFrame with OHLC, signals, and target position size column.
            position_column: Name of the column holding the target position size ('base_position_size' or 'position_size').
            equity_column: Name of the column to store the resulting equity curve.

        Returns:
            Tuple: (List of executed Trade objects, DataFrame with added PnL and equity columns)
        """
        self.logger.info(f"Starting simulation loop using position column: '{position_column}' -> equity column: '{equity_column}'")
        df_sim = df.copy()
        trades = []

        # Initialize state variables
        current_position = 0.0
        entry_price = 0.0
        equity = self.initial_equity
        start_bar_index = df_sim.index.get_loc(df_sim[df_sim['is_trading_day']].index[0]) if 'is_trading_day' in df_sim.columns else 0


        # Initialize columns
        df_sim[equity_column] = np.nan
        df_sim['pnl'] = 0.0 # Per-bar PnL
        df_sim['trade_cost'] = 0.0 # Per-bar cost
        df_sim['position'] = 0.0 # Actual position held during the bar

        # Pre-calculate masks for efficiency
        #trading_day_mask = df_sim['is_trading_day'] if 'is_trading_day' in df_sim.columns else pd.Series(True, index=df_sim.index)

        # Iterate through bars where trading is allowed (consider `is_trading_day` if available)
        for i in range(start_bar_index, len(df_sim)):
            prev_i = i - 1 if i > 0 else i # Handle first bar
            current_time = df_sim.index[i]
            bar = df_sim.iloc[i]
            prev_bar = df_sim.iloc[prev_i]

            # Update equity from previous bar's PnL
            if i > start_bar_index:
                 equity += df_sim.loc[df_sim.index[prev_i], 'pnl'] - df_sim.loc[df_sim.index[prev_i], 'trade_cost']

            # Store starting equity for the bar
            df_sim.loc[current_time, equity_column] = equity

            # 1. Check for Exits FIRST
            should_exit = False
            exit_reason = ""
            if current_position != 0:
                should_exit, exit_reason = self.strategy.check_exit_conditions(
                    bar, int(np.sign(current_position)), entry_price, current_time
                )

            if should_exit:
                exit_price = bar['Open'] # Assume exit at next bar's open
                pnl, cost = self._calculate_trade_result(current_position, entry_price, exit_price)

                trade = self._record_trade(
                    entry_time=entry_time, # Need to store this when entering
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=current_position, # Use the actual position size
                    pnl=pnl,
                    costs=cost,
                    exit_reason=exit_reason,
                    base_size=bar['base_position_size'] if 'base_position_size' in bar else np.nan, # Add base size info
                    final_size=current_position, # Final size is the exiting position
                    strategy_name=self.strategy.name,
                    symbol=self.symbol,
                    contract_spec=self.contract_spec
                )
                trades.append(trade)
                self.logger.debug(f"EXIT recorded: {trade.direction} {abs(trade.quantity):.2f} at {trade.exit_price:.2f} ({trade.exit_reason}). PnL: {trade.pnl-trade.costs:.2f}")

                # Update PnL for the bar where the *exit* occurs (previous bar)
                df_sim.loc[df_sim.index[prev_i], 'pnl'] += pnl
                df_sim.loc[df_sim.index[prev_i], 'trade_cost'] += cost
                equity += pnl - cost # Update equity immediately after exit PnL realization

                # Reset position state
                current_position = 0.0
                entry_price = 0.0
                entry_time = None


            # 2. Check for Entries (only if flat)
            signal = bar['signal']
            target_position = bar[position_column] # Target size from RiskManager/base

            if current_position == 0 and signal != 0 and target_position != 0:
                # Check if target position aligns with signal direction
                if np.sign(signal) == np.sign(target_position):
                    entry_price = bar['Open'] # Enter at next bar's open
                    current_position = target_position # Set position to the target size
                    entry_time = current_time # Record entry time
                    self.logger.debug(f"ENTRY: {np.sign(current_position)} {abs(current_position):.2f} at {entry_price:.2f} based on signal {signal}")
                else:
                     self.logger.warning(f"Signal {signal} and target position {target_position:.2f} directions mismatch at {current_time}. No entry.")

            # 3. Update Ongoing PnL (if position held) & store position for the bar
            if current_position != 0:
                 # Calculate PnL change *during* the current bar (Close - Open)
                 bar_pnl = self._calculate_pnl(current_position, bar['Open'], bar['Close'])
                 df_sim.loc[current_time, 'pnl'] = bar_pnl # Store this bar's PnL contribution
                 # Note: Costs are only added on entry/exit, not per bar holding

            df_sim.loc[current_time, 'position'] = current_position # Record position held *during* this bar

        # Fill NaN equity values at the beginning
        df_sim[equity_column] = df_sim[equity_column].fillna(self.initial_equity)
        # Forward fill equity for bars with no PnL change (e.g., weekends, holidays)
        df_sim[equity_column] = df_sim[equity_column].ffill()

        self.logger.info(f"Simulation loop completed. Total trades: {len(trades)}")
        # Log final equity and summary stats
        if not df_sim.empty:
             self.logger.info(f"Final {equity_column}: {df_sim[equity_column].iloc[-1]:,.2f}")
             self.logger.info(f"Total PnL ({equity_column}): {df_sim['pnl'].sum():,.2f}")
             self.logger.info(f"Total Costs ({equity_column}): {df_sim['trade_cost'].sum():,.2f}")
             self.logger.info(f"Net PnL ({equity_column}): {(df_sim['pnl'].sum() - df_sim['trade_cost'].sum()):,.2f}")

        return trades, df_sim


    def _calculate_pnl(self, quantity: float, entry_price: float, exit_price: float) -> float:
        """Calculate PnL for a single trade leg (no costs)."""
        if self.contract_spec is None:
            self.logger.error("Cannot calculate PnL: ContractSpecification missing.")
            return 0.0
        return quantity * (exit_price - entry_price) * self.contract_spec.multiplier

    def _calculate_costs(self, quantity: float, entry_price: float) -> float:
        """Calculate transaction costs for a single trade leg (entry or exit)."""
        if self.contract_spec is None:
            self.logger.error("Cannot calculate costs: ContractSpecification missing.")
            return 0.0

        trade_value = abs(quantity) * entry_price * self.contract_spec.multiplier
        commission = abs(quantity) * self.transaction_costs.get('commission_rate', 0) * entry_price # Commission per contract often fixed or based on price
        min_commission = self.transaction_costs.get('min_commission', 0)
        commission = max(commission, abs(quantity) * min_commission) # Apply min commission per contract

        slippage = trade_value * self.transaction_costs.get('slippage_rate', 0)

        return commission + slippage

    def _calculate_trade_result(self, quantity: float, entry_price: float, exit_price: float) -> Tuple[float, float]:
        """Calculate PnL and total costs for a round trip trade."""
        pnl = self._calculate_pnl(quantity, entry_price, exit_price)
        # Costs are calculated per leg (entry and exit)
        entry_cost = self._calculate_costs(quantity, entry_price)
        exit_cost = self._calculate_costs(quantity, exit_price)
        total_cost = entry_cost + exit_cost
        return pnl, total_cost

    def _record_trade(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                      entry_price: float, exit_price: float, quantity: float,
                      pnl: float, costs: float, exit_reason: str, base_size: float,
                      final_size: float, strategy_name: str, symbol: str,
                      contract_spec: ContractSpecification) -> Trade:
        """Create a Trade object."""
        if contract_spec is None:
             # Should not happen if validated earlier, but defensive check
             raise ValueError("Cannot record trade without ContractSpecification.")

        # Calculate base and final returns for the trade
        base_return = 0.0
        final_return = 0.0
        if entry_price != 0:
             direction = np.sign(quantity)
             price_change_pct = (exit_price / entry_price) - 1
             base_return = direction * price_change_pct # Simple return based on price change
             # Final return considers PnL relative to margin or notional value - simple PnL for now
             # We might need equity context here for a return % calculation.
             # Using simple PnL / (abs(final_size) * entry_price * multiplier) ? Needs thought.
             # Let's stick to PnL as the primary measure for now.

        # Ensure timestamps have timezone info if possible
        if entry_time.tzinfo is None and exit_time.tzinfo is not None:
            entry_time = entry_time.tz_localize(exit_time.tzinfo)
        elif exit_time.tzinfo is None and entry_time.tzinfo is not None:
            exit_time = exit_time.tz_localize(entry_time.tzinfo)

        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=final_size, # Use final executed size
            direction=int(np.sign(quantity)),
            pnl=pnl,
            costs=costs,
            exit_reason=exit_reason,
            base_size=base_size,
            final_size=final_size,
            base_return=base_return, # Store calculated base return
            final_return=pnl / (abs(final_size) * entry_price * contract_spec.multiplier) if abs(final_size) * entry_price != 0 else 0.0, # Simplified return %
            strategy_name=strategy_name,
            symbol=symbol,
            contract_spec=contract_spec
        )

    def _temp_aggregate_daily_returns(self, df_results: pd.DataFrame, equity_column: str) -> pd.DataFrame:
         """Temporary helper to get daily returns needed for RiskManager before final aggregation."""
         self.logger.debug(f"Performing temporary daily aggregation for column '{equity_column}'")
         if equity_column not in df_results.columns or df_results[equity_column].isna().all():
              self.logger.warning(f"Equity column '{equity_column}' missing or all NaN in temporary aggregation. Returning empty DataFrame.")
              return pd.DataFrame(columns=[f'{equity_column.replace("_equity", "")}_returns'])

         # Ensure equity starts correctly
         df_results[equity_column] = df_results[equity_column].ffill().fillna(self.config.initial_equity)

         # Resample to daily, taking the last equity value of the day
         daily_equity = df_results[equity_column].resample('D').last().dropna()

         if daily_equity.empty:
              self.logger.warning("No daily equity values after resampling in temporary aggregation.")
              return pd.DataFrame(columns=[f'{equity_column.replace("_equity", "")}_returns'])

         # Calculate daily returns
         daily_returns = daily_equity.pct_change().fillna(0.0)

         # Rename column for clarity (e.g., 'base_returns' or 'final_returns')
         return_col_name = f'{equity_column.replace("_equity", "")}_returns'
         temp_daily_df = pd.DataFrame({return_col_name: daily_returns})

         self.logger.debug(f"Temporary daily aggregation complete. {len(temp_daily_df)} days.")
         return temp_daily_df 