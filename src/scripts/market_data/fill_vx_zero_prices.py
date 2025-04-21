#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fills zero prices in VXc1-VXc5 contracts for 2004-2007 period using:
1. Interpolation from nearby days (if possible)
2. Data from other contracts on the same day
3. VIX price on that day with a calculated ratio

Marks inserted data with changed=TRUE and source='DERIVED_ZERO_FILLED'.
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse

# --- Configuration ---
DEFAULT_DB_PATH = "data/financial_data.duckdb"
TARGET_SYMBOLS = ['VXc1', 'VXc2', 'VXc3', 'VXc4', 'VXc5']
REFERENCE_SYMBOL = '$VIX.X'  
FILL_START_DATE = '2004-01-01'
FILL_END_DATE = '2007-12-31'
RATIO_CALC_DAYS = 60  # Number of trading days to use for ratio calculation
DERIVED_SOURCE_TAG = 'DERIVED_ZERO_FILLED'
MAX_INTERP_DAYS = 5  # Maximum days to interpolate across

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Operations ---
def connect_db(db_path):
    """Connects to the DuckDB database (read-write)."""
    try:
        conn = duckdb.connect(database=db_path, read_only=False)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except duckdb.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        sys.exit(1)

def load_data_for_period(conn, symbol, start_date, end_date):
    """Loads OHLC and settle prices for a symbol within a date range."""
    logger.debug(f"Loading data for {symbol} from {start_date} to {end_date}")
    try:
        query = f"""
            SELECT timestamp, open, high, low, close, settle
            FROM market_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        df = conn.execute(query, [symbol, start_date, end_date]).fetchdf()
        if df.empty:
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
        else:
            # Ensure timestamp is datetime object
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} entries for {symbol} from {start_date} to {end_date}")
        return df.set_index('timestamp')
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def load_all_vx_data(conn, start_date, end_date):
    """Loads data for all VX continuous contracts and VIX for the period."""
    symbols = TARGET_SYMBOLS + ['VXc1', REFERENCE_SYMBOL]
    data_dict = {}
    for symbol in symbols:
        data_dict[symbol] = load_data_for_period(conn, symbol, start_date, end_date)
    return data_dict

# --- Zero Detection ---
def detect_zero_prices(df):
    """Detects rows where settle price is zero."""
    if df.empty:
        return pd.DataFrame()
    
    # Consider a price zero if it's exactly 0.0 or very close to zero
    zero_mask = np.isclose(df['settle'], 0.0) | df['settle'].isna()
    zero_rows = df[zero_mask].copy()
    return zero_rows

def is_market_holiday(conn, date):
    """
    Determines if a date is a market holiday by checking if any symbols have valid data.
    Returns True if the date appears to be a holiday (no valid data across all symbols).
    """
    query = f"""
    SELECT COUNT(*) as data_count
    FROM market_data
    WHERE timestamp::DATE = '{date.strftime('%Y-%m-%d')}'
      AND settle IS NOT NULL
      AND settle != 0.0
    """
    
    data_count = conn.execute(query).fetchone()[0]
    
    # If no valid data points found, it's likely a holiday
    return data_count == 0

# --- Fill Methods ---
def interpolate_prices(df, max_gap=MAX_INTERP_DAYS):
    """
    Interpolates prices for rows with zero values using nearby data.
    Only fills gaps up to max_gap trading days.
    """
    if df.empty:
        return df.copy()
    
    filled_df = df.copy()
    
    # Mark rows with zero settle prices
    zero_mask = np.isclose(filled_df['settle'], 0.0) | filled_df['settle'].isna()
    
    # Skip if no zeros found
    if not zero_mask.any():
        return filled_df
    
    # Get the indices of zero rows
    zero_indices = filled_df.index[zero_mask]
    
    # Create a DataFrame to store fill information
    fill_info = []
    
    for col in ['open', 'high', 'low', 'close', 'settle']:
        # Skip columns that don't exist
        if col not in filled_df.columns:
            continue
            
        # Create a temporary series with NaN instead of zeros for interpolation
        temp_series = filled_df[col].copy()
        temp_series[np.isclose(temp_series, 0.0)] = np.nan
        
        # Perform linear interpolation limited to max_gap days
        # This only connects known values, not extrapolates
        interpolated = temp_series.interpolate(method='time', limit=max_gap, limit_area='inside')
        
        # Only update the zero values with interpolated values
        filled_values = interpolated[zero_mask & ~interpolated.isna()]
        
        for idx, value in filled_values.items():
            filled_df.loc[idx, col] = value
            fill_info.append({
                'timestamp': idx,
                'column': col,
                'method': 'interpolation',
                'value': value
            })
    
    if fill_info:
        logger.info(f"Interpolated {len(fill_info)} values across {len(set([i['timestamp'] for i in fill_info]))} timestamps")
    
    return filled_df

def calculate_ratio_to_vxc1(vxc1_data, target_data, reference_period=RATIO_CALC_DAYS):
    """Calculates the historical ratio between target contract and VXc1."""
    # Align series by timestamp and drop rows where either value is missing or zero
    df_merged = pd.merge(
        target_data['settle'], 
        vxc1_data['settle'], 
        left_index=True, 
        right_index=True, 
        how='inner', 
        suffixes=('_target', '_vxc1')
    )
    
    # Filter out zeros and NaNs
    df_merged = df_merged[
        (~np.isclose(df_merged['settle_target'], 0.0)) & 
        (~np.isclose(df_merged['settle_vxc1'], 0.0)) &
        (~df_merged['settle_target'].isna()) & 
        (~df_merged['settle_vxc1'].isna())
    ]
    
    if len(df_merged) < 10:  # Need at least 10 data points for reliable ratio
        logger.warning("Not enough valid data points to calculate reliable ratio.")
        return None
    
    # Limit to the most recent reference_period points
    df_merged = df_merged.iloc[-reference_period:]
    
    df_merged['ratio'] = df_merged['settle_target'] / df_merged['settle_vxc1']
    average_ratio = df_merged['ratio'].mean()
    
    logger.info(f"Calculated average ratio: {average_ratio:.4f} from {len(df_merged)} data points")
    return average_ratio

def calculate_ratio_to_vix(vix_data, target_data, reference_period=RATIO_CALC_DAYS):
    """Calculates the historical ratio between target contract and VIX index."""
    # Align series by timestamp and drop rows where either value is missing or zero
    df_merged = pd.merge(
        target_data['settle'], 
        vix_data['settle'], 
        left_index=True, 
        right_index=True, 
        how='inner', 
        suffixes=('_target', '_vix')
    )
    
    # Filter out zeros and NaNs
    df_merged = df_merged[
        (~np.isclose(df_merged['settle_target'], 0.0)) & 
        (~np.isclose(df_merged['settle_vix'], 0.0)) &
        (~df_merged['settle_target'].isna()) & 
        (~df_merged['settle_vix'].isna())
    ]
    
    if len(df_merged) < 10:  # Need at least 10 data points for reliable ratio
        logger.warning("Not enough valid data points to calculate reliable ratio.")
        return None
    
    # Limit to the most recent reference_period points
    df_merged = df_merged.iloc[-reference_period:]
    
    df_merged['ratio'] = df_merged['settle_target'] / df_merged['settle_vix']
    average_ratio = df_merged['ratio'].mean()
    
    logger.info(f"Calculated average ratio: {average_ratio:.4f} from {len(df_merged)} data points")
    return average_ratio

def fill_from_other_contracts(data_dict, zero_rows_idx, symbol):
    """
    Fill zero prices using data from other contracts on the same day.
    Returns a dictionary with filled values and the method used.
    """
    filled_data = {}
    
    # Skip if no zeros to fill
    if len(zero_rows_idx) == 0:
        return filled_data
    
    # Get the positions of the target symbol in the contract hierarchy
    target_position = int(symbol.replace('VXc', ''))
    
    # Loop through each zero timestamp
    for idx in zero_rows_idx:
        # For VXc1, we'll use VIX data directly with a ratio
        if symbol == 'VXc1':
            if idx in data_dict[REFERENCE_SYMBOL].index and not np.isclose(data_dict[REFERENCE_SYMBOL].loc[idx, 'settle'], 0.0):
                # Calculate the ratio between VXc1 and VIX
                vix_ratio = None
                
                # First try to use calculated ratio if we have one
                ratio_to_vix = calculate_ratio_to_vix(
                    data_dict[REFERENCE_SYMBOL],
                    data_dict[symbol],
                    RATIO_CALC_DAYS
                )
                
                if ratio_to_vix is not None:
                    vix_ratio = ratio_to_vix
                    method = 'vix_ratio'
                else:
                    # Fallback to typical VXc1 to VIX ratio of ~1.1
                    vix_ratio = 1.1
                    method = 'vix_estimate'
                
                # Calculate all price fields based on the ratio
                for field in ['open', 'high', 'low', 'settle']:
                    vix_field = field if field != 'close' else 'settle'  # VIX may use settle instead of close
                    if vix_field in data_dict[REFERENCE_SYMBOL].columns and not np.isclose(data_dict[REFERENCE_SYMBOL].loc[idx, vix_field], 0.0):
                        value = data_dict[REFERENCE_SYMBOL].loc[idx, vix_field] * vix_ratio
                        if idx not in filled_data:
                            filled_data[idx] = {'method': method, 'values': {}}
                        filled_data[idx]['values'][field] = value
                        
                # Also set the close field equal to settle for consistency
                if 'settle' in filled_data.get(idx, {}).get('values', {}):
                    filled_data[idx]['values']['close'] = filled_data[idx]['values']['settle']
            
        # For VXc2-VXc5, first try to use VXc1 if it's not zero
        elif idx in data_dict['VXc1'].index and not np.isclose(data_dict['VXc1'].loc[idx, 'settle'], 0.0):
            # Use the VXc1 data with a ratio
            vxc1_ratio = None
            
            # First try to use calculated ratio if we have one
            ratio_to_vxc1 = calculate_ratio_to_vxc1(
                data_dict['VXc1'], 
                data_dict[symbol],
                RATIO_CALC_DAYS
            )
            
            if ratio_to_vxc1 is not None:
                vxc1_ratio = ratio_to_vxc1
                method = 'vxc1_ratio'
            else:
                # Fallback to position-based estimate
                # VXc2 is typically ~1.05x VXc1, VXc3 is ~1.08x VXc1, etc.
                vxc1_ratio = 1.0 + (target_position - 1) * 0.03
                method = 'position_estimate'
            
            # Calculate all price fields based on the ratio
            for field in ['open', 'high', 'low', 'close', 'settle']:
                if field in data_dict['VXc1'].columns and not np.isclose(data_dict['VXc1'].loc[idx, field], 0.0):
                    value = data_dict['VXc1'].loc[idx, field] * vxc1_ratio
                    if idx not in filled_data:
                        filled_data[idx] = {'method': method, 'values': {}}
                    filled_data[idx]['values'][field] = value
                    
        # If VXc1 is not available or is zero, try VIX
        elif idx in data_dict[REFERENCE_SYMBOL].index and not np.isclose(data_dict[REFERENCE_SYMBOL].loc[idx, 'settle'], 0.0):
            # Calculate the ratio between the target and VIX
            vix_ratio = None
            
            # First try to use calculated ratio if we have one
            ratio_to_vix = calculate_ratio_to_vix(
                data_dict[REFERENCE_SYMBOL],
                data_dict[symbol],
                RATIO_CALC_DAYS
            )
            
            if ratio_to_vix is not None:
                vix_ratio = ratio_to_vix
                method = 'vix_ratio'
            else:
                # Fallback to position-based estimate
                # VXc1 is typically ~1.1x VIX, VXc2 is ~1.15x VIX, etc.
                vix_ratio = 1.1 + (target_position - 1) * 0.05
                method = 'vix_estimate'
            
            # Calculate all price fields based on the ratio
            for field in ['open', 'high', 'low', 'settle']:
                vix_field = field if field != 'close' else 'settle'  # VIX may use settle instead of close
                if vix_field in data_dict[REFERENCE_SYMBOL].columns and not np.isclose(data_dict[REFERENCE_SYMBOL].loc[idx, vix_field], 0.0):
                    value = data_dict[REFERENCE_SYMBOL].loc[idx, vix_field] * vix_ratio
                    if idx not in filled_data:
                        filled_data[idx] = {'method': method, 'values': {}}
                    filled_data[idx]['values'][field] = value
                    
            # Also set the close field equal to settle for consistency if it wasn't set
            if 'settle' in filled_data.get(idx, {}).get('values', {}) and 'close' not in filled_data.get(idx, {}).get('values', {}):
                filled_data[idx]['values']['close'] = filled_data[idx]['values']['settle']
    
    return filled_data

# --- Main Execution ---
def main(args_dict=None):
    """Main execution function."""
    # --- Parse Arguments ---
    if args_dict is None:
        parser = argparse.ArgumentParser(description='Fill zero prices in VXc1-VXc5 for 2004-2007 period.')
        parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='Path to the DuckDB database file.')
        parser.add_argument('--start-date', type=str, default=FILL_START_DATE, help='Start date for filling zeros (YYYY-MM-DD).')
        parser.add_argument('--end-date', type=str, default=FILL_END_DATE, help='End date for filling zeros (YYYY-MM-DD).')
        parsed_args = parser.parse_args()
        effective_args = vars(parsed_args)
        logger.info("Running fill_vx_zero_prices from command line.")
    else:
        effective_args = args_dict
        logger.info("Running fill_vx_zero_prices programmatically.")

    # --- Connect to Database ---
    conn = connect_db(effective_args.get('db_path', DEFAULT_DB_PATH))
    start_date = effective_args.get('start_date', FILL_START_DATE)
    end_date = effective_args.get('end_date', FILL_END_DATE)
    
    try:
        # --- Load All Relevant Data ---
        logger.info(f"Loading data for VX contracts and VIX from {start_date} to {end_date}")
        data_dict = load_all_vx_data(conn, start_date, end_date)
        
        # List to store all derived data for insertion
        all_derived_data = []
        
        # --- Process Each Target Symbol ---
        for symbol in TARGET_SYMBOLS:
            logger.info(f"Processing {symbol} for zero price filling")
            
            # Skip if no data available
            if symbol not in data_dict or data_dict[symbol].empty:
                logger.warning(f"No data found for {symbol}. Skipping.")
                continue
            
            # --- Detect Zero Prices ---
            zero_rows = detect_zero_prices(data_dict[symbol])
            if zero_rows.empty:
                logger.info(f"No zero prices found for {symbol}. Skipping.")
                continue
            
            logger.info(f"Found {len(zero_rows)} timestamps with zero prices for {symbol}")
            
            # --- Check for Market Holidays ---
            # Filter out zeros on known market holidays
            holiday_zeros = []
            non_holiday_zeros = []
            
            for idx in zero_rows.index:
                date = idx.date()
                if is_market_holiday(conn, date):
                    holiday_zeros.append(idx)
                else:
                    non_holiday_zeros.append(idx)
            
            if holiday_zeros:
                logger.info(f"Skipping {len(holiday_zeros)} zeros on market holidays")
            
            # If all zeros are on holidays, skip this symbol
            if len(non_holiday_zeros) == 0:
                logger.info(f"All zeros for {symbol} are on market holidays. No filling needed.")
                continue
            
            # Update zero_rows to include only non-holiday zeros
            non_holiday_mask = zero_rows.index.isin(non_holiday_zeros)
            zero_rows = zero_rows[non_holiday_mask]
            
            # --- Attempt Interpolation ---
            interpolated_df = interpolate_prices(data_dict[symbol])
            
            # Check which zeros are still present after interpolation
            remaining_zeros = detect_zero_prices(interpolated_df)
            
            # Filter remaining zeros to exclude holidays
            non_holiday_mask = remaining_zeros.index.isin(non_holiday_zeros)
            remaining_zeros = remaining_zeros[non_holiday_mask]
            
            # Store which rows were fixed by interpolation
            fixed_by_interp = set(non_holiday_zeros) - set(remaining_zeros.index)
            if fixed_by_interp:
                logger.info(f"Fixed {len(fixed_by_interp)} zero prices with interpolation")
            
            # --- Fill Remaining Zeros from Other Contracts ---
            filled_data = fill_from_other_contracts(
                data_dict, 
                remaining_zeros.index,
                symbol
            )
            
            if filled_data:
                logger.info(f"Filled {len(filled_data)} additional zero prices using other contracts")
            
            # --- Prepare Data for Database ---
            rows_to_insert = []
            
            # Add interpolated data
            for idx in fixed_by_interp:
                row_data = {
                    'timestamp': idx,
                    'symbol': symbol,
                    'open': interpolated_df.loc[idx, 'open'] if 'open' in interpolated_df.columns else None,
                    'high': interpolated_df.loc[idx, 'high'] if 'high' in interpolated_df.columns else None,
                    'low': interpolated_df.loc[idx, 'low'] if 'low' in interpolated_df.columns else None,
                    'close': interpolated_df.loc[idx, 'close'] if 'close' in interpolated_df.columns else None,
                    'settle': interpolated_df.loc[idx, 'settle'],
                    'volume': None,
                    'open_interest': None,
                    'interval_value': 1,
                    'interval_unit': 'day',
                    'source': f"{DERIVED_SOURCE_TAG}_INTERP",
                    'changed': True,
                    'adjusted': False,
                    'quality': 80  # Lower quality score for derived data
                }
                rows_to_insert.append(row_data)
            
            # Add data filled from other contracts
            for idx, fill_info in filled_data.items():
                values = fill_info['values']
                row_data = {
                    'timestamp': idx,
                    'symbol': symbol,
                    'open': values.get('open'),
                    'high': values.get('high'),
                    'low': values.get('low'),
                    'close': values.get('close'),
                    'settle': values.get('settle'),
                    'volume': None,
                    'open_interest': None,
                    'interval_value': 1,
                    'interval_unit': 'day',
                    'source': f"{DERIVED_SOURCE_TAG}_{fill_info['method'].upper()}",
                    'changed': True,
                    'adjusted': False,
                    'quality': 75  # Lower quality score for derived data
                }
                rows_to_insert.append(row_data)
            
            if rows_to_insert:
                df_to_insert = pd.DataFrame(rows_to_insert)
                all_derived_data.append(df_to_insert)
                logger.info(f"Prepared {len(df_to_insert)} rows for insertion for {symbol}")
        
        # --- Insert All Derived Data ---
        if not all_derived_data:
            logger.info("No derived data generated. Nothing to insert.")
        else:
            df_to_insert = pd.concat(all_derived_data, ignore_index=True)
            logger.info(f"Attempting to insert {len(df_to_insert)} derived rows into market_data...")
            
            try:
                # Check if UnderlyingSymbol column exists
                try:
                    conn.execute("SELECT UnderlyingSymbol FROM market_data LIMIT 1")
                    df_to_insert['UnderlyingSymbol'] = None
                except Exception:
                    logger.debug("UnderlyingSymbol column not found, not adding to derived data.")
                
                conn.register('df_derived_view', df_to_insert)
                cols = df_to_insert.columns
                col_names_db = ", ".join([f'"{c}"' for c in cols])
                col_names_view = ", ".join([f'v."{c}"' for c in cols])
                
                sql = f"""
                    INSERT OR REPLACE INTO market_data ({col_names_db})
                    SELECT {col_names_view}
                    FROM df_derived_view v
                """
                logger.debug(f"Executing INSERT OR REPLACE SQL: {sql}")
                conn.execute(sql)
                conn.commit()
                logger.info(f"Successfully inserted {len(df_to_insert)} derived rows.")
            
            except Exception as e:
                logger.error(f"Error inserting derived data: {e}")
                conn.rollback()
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    
    finally:
        # Close database connection
        if conn:
            try:
                if not getattr(conn, 'closed', False):
                    conn.close()
                    logger.info("Database connection closed.")
            except Exception as close_e:
                logger.error(f"Error closing database connection: {close_e}")

# Add ability to be called programmatically
def fill_zero_prices_main(args_dict=None):
    return main(args_dict)

if __name__ == "__main__":
    main() 