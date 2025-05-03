import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Optional
import duckdb
import pytz

from ..config import Config, LoggingConfig  # Relative import
# Note: Need to define these error classes or import them
class DataManagerError(Exception):
    pass
class DataLoadError(DataManagerError):
    pass
class DataValidationError(DataManagerError):
    pass
class TimeZoneError(DataManagerError):
    pass

class DataManager:
    """DataManager handles data operations using specified sources (e.g., DuckDB).

    Focuses on providing clean, validated market data ready for strategy implementation.
    Handles loading, cleaning, validation, and feature calculation.
    """

    # Class-level constants for validation
    REQUIRED_SOURCE_COLUMNS = {
        'TimeStamp', 'Open', 'High', 'Low', 'Close',
        'Volume'
    }

    REQUIRED_SPECS = {
        'tick_size', 'multiplier', 'margin'
    }

    def __init__(self, config: Config):
        """Initialize DataManager with configuration."""
        self.logger = LoggingConfig.get_logger(__name__)
        self.logger.info("Initializing DataManager")
        self.config = config
        self._data_cache = {}

    def prepare_data_for_analysis(self, symbol: str, days_to_analyze: int,
                                  lookback_buffer: int) -> pd.DataFrame:
        """Prepare clean market data for analysis from the configured data source."""
        try:
            cache_key = f"{symbol}_{days_to_analyze}_{lookback_buffer}"
            if cache_key in self._data_cache:
                self.logger.info(f"Using cached data for {symbol}")
                return self._data_cache[cache_key].copy()

            self.logger.info(f"\nPreparing data for {symbol} from data source...")

            # Get symbol-specific configuration including data source info
            symbol_specs = self.config.get_symbol_specs(symbol)
            data_source_info = symbol_specs['data_source']
            contract_spec_info = symbol_specs['contract_specs']

            # Load data using the new method
            df = self.load_and_validate_data(
                data_source_info=data_source_info,
                contract_spec=contract_spec_info
            )

            if df.empty:
                 raise DataManagerError(f"Loaded data for {symbol} is empty after validation.")

            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("Index is not DatetimeIndex after loading, attempting conversion.")
                df.index = pd.to_datetime(df.index)
                if not isinstance(df.index, pd.DatetimeIndex):
                     raise DataManagerError("Failed to convert index to DatetimeIndex.")

            # --- Feature Calculation and Filtering ---
            self.logger.info("Calculating basic market data features...")
            df = self._calculate_basic_market_data(df)

            self.logger.info("Filtering analysis period...")
            df = self._filter_analysis_period(df, days_to_analyze, lookback_buffer)

            self.logger.info("Adding trading session markers...")
            df = self._add_trading_markers(df, symbol)

            self.logger.info("Validating final prepared data...")
            self._validate_prepared_data(df)

            # Cache and log summary
            self._data_cache[cache_key] = df.copy()
            self._log_data_preparation_summary(df)
            self.logger.info(f"Data preparation complete for {symbol}.")
            return df

        except Exception as e:
            self.logger.error(f"Data preparation failed for symbol {symbol}: {str(e)}", exc_info=True)
            # Return empty DataFrame on failure? Or re-raise?
            # Re-raising provides more context upstream.
            raise DataManagerError(f"Data preparation failed for {symbol}") from e

    def _calculate_daily_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily values with proper error handling."""
        try:
            self.logger.debug("Calculating daily values")
            if df.index.tz is None:
                 self.logger.warning("Calculating daily values on timezone-naive data. Ensure this is intended.")
                 df_proc = df.copy()
            else:
                 # Use timezone-aware index directly if possible, depends on aggregation needs
                 # For simple daily aggregation, converting to naive might be safer
                 df_proc = df.copy()
                 df_proc.index = df_proc.index.tz_localize(None)

            # Calculate daily values using proper grouping
            daily_values = df_proc.groupby(df_proc.index.date).agg({
                'Open': 'first',
                'Close': 'last'
            }).rename(columns={'Open': 'day_open', 'Close': 'day_close'})

            daily_values['prev_close'] = daily_values['day_close'].shift(1)
            return daily_values

        except Exception as e:
            self.logger.error("Daily value calculation failed", exc_info=True)
            # Raise specific error?
            raise DataManagerError("Failed to calculate daily values") from e

    def load_and_validate_data(self, data_source_info: Dict, contract_spec: Dict) -> pd.DataFrame:
        """Load and validate market data from the specified source (e.g., DuckDB)."""
        try:
            self.logger.info(f"Loading data from source: {data_source_info}")
            self._validate_data_source(data_source_info)

            # Load data from DuckDB
            df = self._load_data_from_duckdb(data_source_info['db_path'], data_source_info['table_name'])

            self.logger.info("Validating loaded columns...")
            self._validate_columns(df) # Validate against REQUIRED_SOURCE_COLUMNS

            self.logger.info("Processing timestamp index...")
            df = self._process_timestamp_index(df)

            self.logger.info("Handling timezones...")
            df = self._handle_timezones(df)

            self.logger.info("Validating OHLC data...")
            df = self._validate_ohlc_data(df)

            self.logger.info("Removing price anomalies...")
            df = self._remove_price_anomalies(df)

            self.logger.info("Adding contract specifications...")
            df = self._add_contract_specs(df, contract_spec)

            # Log summary after loading and validation
            self._log_data_summary(df)
            self.logger.info(f"Data loaded and validated successfully for table '{data_source_info['table_name']}'.")

            return df

        except Exception as e:
            self.logger.error(f"Data loading and validation failed: {str(e)}", exc_info=True)
            # Consider returning empty df or raising specific error
            raise DataLoadError(f"Failed loading/validating data from {data_source_info}") from e

    def _validate_data_source(self, data_source_info: Dict) -> None:
        """Validate DuckDB data source details."""
        db_path = data_source_info.get('db_path')
        table_name = data_source_info.get('table_name')

        if not db_path or not table_name:
            msg = f"Invalid data source info: db_path or table_name missing. Info: {data_source_info}"
            self.logger.error(msg)
            raise DataLoadError(msg)

        # Check if the *directory* for the DB file exists
        db_file = Path(db_path)
        if not db_file.parent.is_dir():
            msg = f"Database directory does not exist: {db_file.parent}"
            self.logger.error(msg)
            raise DataLoadError(msg)

        # Optional: Could add a check to see if the DB file exists and the table is present
        # However, DuckDB can create the file, and checking table requires connection.
        # We will handle connection errors in _load_data_from_duckdb.
        self.logger.debug(f"Data source validation passed for {db_path} / {table_name}")

    def _load_data_from_duckdb(self, db_path: str, table_name: str) -> pd.DataFrame:
        """Load data from DuckDB table with error handling."""
        self.logger.info(f"Connecting to DuckDB: {db_path}")
        con = None
        try:
            # Connect to the database (read-only mode is safer if applicable)
            con = duckdb.connect(database=db_path, read_only=True)
            self.logger.info(f"Querying table: {table_name}")

            # Construct the query safely (avoiding SQL injection if table_name were dynamic)
            # Use parameterized query if possible, though table names usually aren't parameters.
            # For safety, basic validation of table_name could be added.
            query = f"SELECT TimeStamp, Open, High, Low, Close, Volume FROM \"{table_name}\" ORDER BY TimeStamp;"
            # Fetch data into a pandas DataFrame
            df = con.execute(query).fetchdf()

            if df.empty:
                 self.logger.warning(f"Loaded empty DataFrame from table '{table_name}' in {db_path}")
            else:
                 self.logger.info(f"Successfully loaded {len(df)} rows from table '{table_name}'")
                 # Ensure Timestamp is datetime type after loading
                 if 'TimeStamp' in df.columns:
                     df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                 else:
                      raise DataLoadError("'TimeStamp' column not found in loaded data.")
            return df

        except duckdb.CatalogException:
             msg = f"Table '{table_name}' not found in database '{db_path}'."
             self.logger.error(msg)
             raise DataLoadError(msg)
        except duckdb.IOException as e:
             msg = f"Failed to read DuckDB file '{db_path}': {str(e)}"
             self.logger.error(msg, exc_info=True)
             raise DataLoadError(msg)
        except Exception as e:
            msg = f"Failed to load data from DuckDB table '{table_name}': {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise DataLoadError(msg)
        finally:
            if con:
                con.close()
                self.logger.debug("Closed DuckDB connection.")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required source columns exist in DataFrame."""
        missing_columns = self.REQUIRED_SOURCE_COLUMNS - set(df.columns)
        if missing_columns:
            msg = f"Missing required source columns: {missing_columns}. Available: {list(df.columns)}"
            self.logger.error(msg)
            raise DataValidationError(msg)
        self.logger.debug("Source columns validation passed.")

    def _process_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamp index."""
        try:
            if 'TimeStamp' not in df.columns:
                 raise DataValidationError("'TimeStamp' column not found for setting index.")

            # Ensure Timestamp is the correct type before setting index
            if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
                 self.logger.warning("'TimeStamp' column is not datetime type, attempting conversion.")
                 df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                 if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
                      raise DataValidationError("Failed to convert 'TimeStamp' column to datetime.")

            df = df.set_index('TimeStamp')
            # df.index = pd.to_datetime(df.index) # Already done above potentially

            if not df.index.is_monotonic_increasing:
                self.logger.info("Sorting index by timestamp.")
                df = df.sort_index()

            duplicates = df.index.duplicated()
            if duplicates.any():
                num_duplicates = duplicates.sum()
                self.logger.warning(f"Removing {num_duplicates} duplicate timestamps (keeping first). Consider cleaning source data.")
                df = df[~df.index.duplicated(keep='first')]

            self.logger.debug("Timestamp index processed successfully.")
            return df

        except Exception as e:
            msg = f"Failed to process timestamp index: {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise DataValidationError(msg)

    def _handle_timezones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle timezone conversion using the timezone object from Config."""
        try:
            # Get configured timezone object from config
            target_tz = getattr(self.config, 'timezone_obj', None)
            if target_tz is None:
                # Fallback if timezone_obj wasn't created in Config (shouldn't happen if Config init worked)
                self.logger.warning("Timezone object not found in config, falling back to parsing string.")
                target_tz_str = getattr(self.config, 'timezone', 'UTC')
                target_tz = pytz.timezone(target_tz_str or 'UTC')

            # If data has no timezone, assume UTC (or potentially config source TZ?)
            # Assuming source data in DuckDB is stored as UTC is common.
            if df.index.tz is None:
                self.logger.info(f"Localizing naive timestamps to UTC before converting to target {target_tz}.")
                df.index = df.index.tz_localize('UTC')

            # Convert to target timezone if different
            if df.index.tz != target_tz:
                self.logger.info(f"Converting data timezone from {df.index.tz} to {target_tz}")
                df.index = df.index.tz_convert(target_tz)
            else:
                self.logger.info(f"Data timezone is already target {target_tz}. No conversion needed.")

            return df

        except pytz.exceptions.UnknownTimeZoneError:
            # This error should ideally be caught during Config initialization
            msg = f"Unknown timezone encountered during conversion: {getattr(self.config, 'timezone', 'N/A')}"
            self.logger.error(msg)
            raise TimeZoneError(msg)
        except Exception as e:
            msg = f"Timezone handling failed: {str(e)}"
            self.logger.error(msg, exc_info=True)
            raise TimeZoneError(msg)

    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLC data."""
        df = df.copy()
        invalid_rows = pd.DataFrame(index=df.index)
        invalid_rows['high_low'] = df['High'] < df['Low']
        invalid_rows['open_range'] = (df['Open'] > df['High']) | (df['Open'] < df['Low'])
        invalid_rows['close_range'] = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        issues_found = {
            'High < Low': invalid_rows['high_low'].sum(),
            'Open outside range': invalid_rows['open_range'].sum(),
            'Close outside range': invalid_rows['close_range'].sum()
        }
        if any(v > 0 for v in issues_found.values()):
             total_issues = sum(issues_found.values())
             self.logger.warning(f"Found {total_issues} OHLC inconsistencies. Attempting to fix.")
             for issue, count in issues_found.items():
                 if count > 0: self.logger.warning(f"  - {issue}: {count} instances")
             df = self._fix_ohlc_issues(df, invalid_rows)
        else: self.logger.debug("OHLC data consistency check passed.") # Debug level
        return df

    def _fix_ohlc_issues(self, df: pd.DataFrame, invalid_rows: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC data issues."""
        if invalid_rows['high_low'].any():
            mask = invalid_rows['high_low']
            self.logger.info(f"Fixing {mask.sum()} High < Low inversions by swapping.")
            low_vals = df.loc[mask, 'Low'].copy()
            df.loc[mask, 'Low'] = df.loc[mask, 'High']
            df.loc[mask, 'High'] = low_vals
        for col, mask_col in [('Open', 'open_range'), ('Close', 'close_range')]:
            mask = invalid_rows[mask_col]
            if mask.any():
                self.logger.info(f"Clipping {mask.sum()} {col} values to High/Low range.")
                df.loc[mask, col] = df.loc[mask].apply(lambda x: np.clip(x[col], x['Low'], x['High']), axis=1)
        return df

    def _remove_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove anomalous price movements using percentage change."""
        returns = df['Close'].pct_change()
        roll_std = returns.rolling(window=20, min_periods=5).std()
        threshold = 5 * roll_std # 5 std deviations threshold
        outliers = abs(returns) > threshold
        outliers = outliers & threshold.notna()
        if outliers.any():
            num_outliers = outliers.sum()
            self.logger.warning(f"Removing {num_outliers} anomalous price movements (>{threshold.mean()*100:.2f}% approx change)")
            df = df[~outliers]
        else: self.logger.debug("No significant price anomalies detected.") # Debug level
        return df

    def _add_contract_specs(self, df: pd.DataFrame, contract_spec: Dict) -> pd.DataFrame:
        """Add required contract specifications with validation."""
        missing_specs = self.REQUIRED_SPECS - set(contract_spec.keys())
        if missing_specs:
            msg = f"Missing required contract specifications: {missing_specs}"
            self.logger.error(msg)
            raise DataValidationError(msg)
        for spec, value in contract_spec.items():
             if spec in self.REQUIRED_SPECS: df[spec] = value
        self.logger.debug("Added required contract specifications to DataFrame.") # Debug level
        return df

    def _calculate_basic_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market data columns from source data (incl. Volume, VWAP)."""
        try:
            df = df.copy()
            if self.config.symbol not in self.config.trading_hours:
                 raise ValueError(f"Trading hours not found for symbol '{self.config.symbol}'")
            symbol_hours = self.config.trading_hours[self.config.symbol]
            market_open = symbol_hours['market_open']
            market_close = symbol_hours['market_close']

            self.logger.debug(f"Processing market data between {market_open} and {market_close}")

            df['minute_of_day'] = df.index.hour * 60 + df.index.minute
            df['trading_date'] = df.index.date
            market_hours = (df.index.time >= market_open) & (df.index.time <= market_close)

            # Get daily open/close
            daily_vals = self._calculate_daily_values(df)
            df = df.merge(daily_vals, left_on='trading_date', right_index=True, how='left')

            # Fill first day's prev_close with its open if available
            first_date = df['trading_date'].iloc[0]
            if pd.isna(df['prev_close'].iloc[0]) and not pd.isna(df['day_open'].iloc[0]):
                df['prev_close'].iloc[0] = df['day_open'].iloc[0]
                self.logger.debug(f"Filled first day's prev_close for {first_date} using day_open.")

            # Calculate move from open (percentage)
            df['move_from_open'] = np.nan
            market_hours_valid_open = market_hours & df['day_open'].notna() & (df['day_open'] != 0)
            df.loc[market_hours_valid_open, 'move_from_open'] = (
                abs(df.loc[market_hours_valid_open, 'Close'] - df.loc[market_hours_valid_open, 'day_open']) /
                df.loc[market_hours_valid_open, 'day_open']
            )

            # Calculate VWAP using single Volume column
            df['vwap'] = np.nan
            if 'Volume' not in df.columns:
                 self.logger.warning("'Volume' column missing. Using Close price as VWAP fallback.")
                 df['vwap'] = df['Close']
            else:
                market_data = df[market_hours].copy()
                if not market_data.empty:
                    volume = market_data['Volume']
                    price_volume = market_data['Close'] * volume
                    cumulative_price_volume = price_volume.groupby(market_data['trading_date']).cumsum()
                    cumulative_volume = volume.groupby(market_data['trading_date']).cumsum()
                    valid_volume_mask = cumulative_volume > 0
                    # Assign VWAP back to the original df index positions
                    df.loc[market_hours & valid_volume_mask, 'vwap'] = (
                        cumulative_price_volume[valid_volume_mask] / cumulative_volume[valid_volume_mask]
                    )
                    # Fill VWAP NaNs (outside market hours or zero volume) with Close
                    df['vwap'].fillna(df['Close'], inplace=True)
                else:
                     self.logger.warning("No data within market hours for VWAP calculation. Using Close fallback.")
                     df['vwap'] = df['Close']

            # Drop temporary columns
            df = df.drop(columns=['trading_date', 'day_close'], errors='ignore')

            self.logger.debug("Basic market data calculation complete.")
            return df

        except Exception as e:
            self.logger.error(f"Failed to calculate basic market data: {str(e)}", exc_info=True)
            raise DataManagerError("Failed basic market data calculation") from e

    def _filter_analysis_period(self, df: pd.DataFrame, days_to_analyze: int,
                                lookback_buffer: int) -> pd.DataFrame:
        """Filter data to analysis period, marking the trading segment."""
        try:
            trading_days = pd.Series(df.index.date).unique()
            trading_days.sort()
            if len(trading_days) == 0:
                 self.logger.warning("No trading days found to filter analysis period.")
                 df['is_trading_period'] = False
                 return df

            total_days = len(trading_days)
            required_days = days_to_analyze + lookback_buffer
            if total_days < required_days:
                 self.logger.warning(f"Data has {total_days} days, less than required {required_days}. Using all available data.")
                 start_idx = 0
                 trading_start_idx = min(lookback_buffer, total_days - 1) if total_days > 0 else 0
            else:
                start_idx = max(total_days - required_days, 0)
                trading_start_idx = min(start_idx + lookback_buffer, total_days - 1)

            start_date = trading_days[start_idx]
            trading_start_date = trading_days[trading_start_idx]

            self.logger.info(f"Filtering data from analysis start date: {start_date}")
            self.logger.info(f"Marking trading period from: {trading_start_date}")

            df_filtered = df[df.index.date >= start_date].copy()
            df_filtered['is_trading_period'] = df_filtered.index.date >= trading_start_date

            return df_filtered

        except Exception as e:
            self.logger.error(f"Period filtering failed: {str(e)}", exc_info=True)
            raise DataManagerError("Failed filtering analysis period") from e

    def _add_trading_markers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add trading session markers (trading_hour, pre_market, post_market, can_enter)."""
        try:
            if symbol not in self.config.trading_hours:
                 raise ValueError(f"Trading hours not defined for symbol '{symbol}' in config.")
            hours_spec = self.config.trading_hours[symbol]
            market_open = hours_spec.get('market_open')
            market_close = hours_spec.get('market_close')
            last_entry = hours_spec.get('last_entry')

            if not all([isinstance(t, time) for t in [market_open, market_close, last_entry]]):
                 raise ValueError(f"Trading hours for {symbol} are not valid time objects.")

            df = df.copy()
            df['trading_hour'] = (df.index.time >= market_open) & (df.index.time <= market_close)
            df['pre_market'] = df.index.time < market_open
            df['post_market'] = df.index.time > market_close
            df['can_enter'] = (df.index.time >= market_open) & (df.index.time <= last_entry)

            self.logger.debug(f"Trading Hours Markers Added for {symbol}") # Debug level
            return df

        except Exception as e:
            self.logger.error(f"Adding trading markers failed: {str(e)}", exc_info=True)
            raise DataManagerError("Failed adding trading markers") from e

    def _log_data_preparation_summary(self, df: pd.DataFrame) -> None:
        """Log summary of prepared data."""
        try:
            self.logger.info("\n--- Prepared Data Summary ---")
            if df.empty: self.logger.info("DataFrame is empty after preparation."); return
            self.logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            self.logger.info(f"Total rows: {len(df):,}")
            unique_dates = pd.Series(df.index.date).unique()
            self.logger.info(f"Unique trading days in prepared data: {len(unique_dates)}")
            if 'trading_hour' in df.columns: self.logger.info(f"Trading hour bars: {df['trading_hour'].sum():,}")
            if 'is_trading_period' in df.columns: self.logger.info(f"Analysis period rows ('is_trading_period'): {df['is_trading_period'].sum():,}")
            self._log_data_quality_stats(df)
            self.logger.info("--- End Prepared Data Summary ---")
        except Exception as e: self.logger.error(f"Failed to log prepared data summary: {str(e)}")

    def _log_data_quality_stats(self, df: pd.DataFrame) -> None:
        """Log data quality statistics."""
        try:
            self.logger.info("\nData Quality Statistics:")
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                self.logger.info(f"\nMissing Values ({total_nulls} total):")
                for col, count in null_counts[null_counts > 0].items(): self.logger.info(f"  {col}: {count:,} ({count / len(df):.2%})")
            else: self.logger.info("No missing values detected.")

            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                 if col in df.columns:
                    stats = df[col].describe()
                    self.logger.info(f"\n{col} Statistics: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
                    zero_count = (df[col] <= 0).sum()
                    if zero_count > 0: self.logger.warning(f"Found {zero_count:,} zero or negative values in {col}")

            if 'Volume' in df.columns:
                vol_stats = df['Volume'].describe()
                self.logger.info(f"\nVolume Statistics: Mean={vol_stats['mean']:.0f}, Std={vol_stats['std']:.0f}, Max={vol_stats['max']:.0f}")
                zero_volume = (df['Volume'] == 0).sum()
                if zero_volume > 0: self.logger.warning(f"Found {zero_volume:,} zero volume bars ({zero_volume/len(df):.2%})")

            if all(c in df.columns for c in price_cols):
                inconsistent = ((df['High'] < df['Low']) | (df['Open'] > df['High']) | (df['Open'] < df['Low']) | (df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
                if inconsistent > 0: self.logger.warning(f"Found {inconsistent:,} bars with inconsistent OHLC values after fixes.")
                else: self.logger.info("OHLC consistency checks passed.")

            if 'trading_hour' in df.columns:
                trading_bars = df['trading_hour'].sum()
                coverage = trading_bars / len(df) if len(df) > 0 else 0
                self.logger.info(f"Trading Hours Coverage: {trading_bars:,} bars ({coverage:.2%})")
        except Exception as e: self.logger.error(f"Failed to log data quality stats: {str(e)}")

    def _log_data_summary(self, df: pd.DataFrame) -> None:
        """Log detailed data statistics after initial load and validation."""
        try:
            self.logger.info("\n--- Initial Loaded Data Summary ---")
            if df.empty: self.logger.info("DataFrame is empty after loading."); return
            self.logger.info(f"Total rows loaded: {len(df):,}")
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Index Timezone: {df.index.tz}")

            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    stats = df[col].describe()
                    self.logger.info(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

            if 'Volume' in df.columns:
                 vol_stats = df['Volume'].describe()
                 self.logger.info(f"Volume: mean={vol_stats['mean']:.0f}, std={vol_stats['std']:.0f}, min={vol_stats['min']:.0f}, max={vol_stats['max']:.0f}")

            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0: self.logger.warning(f"\nNull values found in loaded data ({total_nulls} total):\n{null_counts[null_counts > 0]}")
            else: self.logger.info("No null values found in initially loaded data.")
            self.logger.info("--- End Initial Loaded Data Summary ---")
        except Exception as e: self.logger.error(f"Failed to log initial data summary: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        self.logger.info("Data cache cleared") 