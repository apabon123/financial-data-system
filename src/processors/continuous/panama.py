#!/usr/bin/env python
"""
Panama (Price-Adjusted) Continuous Futures Generator

This module implements the Panama method for generating continuous futures contracts.
The Panama method adjusts historical prices to ensure a continuous price series
without gaps at contract rollovers.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import ContinuousContractBuilder, ContinuousContractError, ContractRollover
from ...core.database import Database

logger = logging.getLogger(__name__)

class PanamaContractBuilder(ContinuousContractBuilder):
    """Panama (price-adjusted) continuous contract builder."""
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        """
        Initialize the Panama continuous contract builder.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        super().__init__(db, config)
        
        # Panama method specific settings
        self.adjustment_method = config.get('adjustment_method', 'ratio')  # 'ratio' or 'additive'
        self.cumulative_factor = 1.0  # Starting adjustment factor (multiplicative)
        self.cumulative_offset = 0.0  # Starting adjustment offset (additive)
    
    def build_continuous_series(self, root_symbol: str, continuous_symbol: str,
                               interval_unit: str = 'daily', interval_value: int = 1,
                               start_date: str = None, end_date: str = None,
                               force: bool = False) -> pd.DataFrame:
        """
        Build a Panama continuous series for the given parameters.
        
        Args:
            root_symbol: The root symbol (e.g., ES, VX)
            continuous_symbol: Target continuous symbol (e.g., @ES=102XC)
            interval_unit: Time interval unit
            interval_value: Time interval value
            start_date: Start date for the series (None for all available)
            end_date: End date for the series (None for current date)
            force: Whether to force rebuild the entire series
            
        Returns:
            DataFrame with the continuous series data
            
        Raises:
            ContinuousContractError: If continuous series generation fails
        """
        try:
            # Parse the continuous symbol
            parsed = self.parse_continuous_symbol(continuous_symbol)
            if parsed['adjustment'] != 'panama':
                logger.warning(f"Symbol {continuous_symbol} is not configured for Panama adjustment")
            
            # Clean existing data if forced
            if force:
                self.delete_continuous_data(continuous_symbol, interval_unit, interval_value)
            
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            # Determine if we're updating or creating
            # For Panama method, we need all history for correct adjustments
            # because adjustments are cumulative
            existing_data = self.get_existing_continuous_data(
                continuous_symbol, interval_unit, interval_value
            )
            
            if not existing_data.empty and not force:
                # We have existing data, so we need to build the new data
                # and apply the cumulative adjustment from the last point
                last_date = existing_data['timestamp'].max().strftime('%Y-%m-%d')
                last_contract = existing_data.iloc[-1]['underlying_symbol']
                
                # For Panama method, we need the cumulative factor from the last point
                if self.adjustment_method == 'ratio':
                    if 'panama_factor' in existing_data.columns:
                        self.cumulative_factor = existing_data.iloc[-1]['panama_factor']
                    else:
                        # Try to calculate the factor by comparing adjusted vs. raw prices
                        if last_contract:
                            raw_contract = self.load_contract_data(
                                last_contract,
                                last_date,
                                last_date,
                                interval_unit,
                                interval_value
                            )
                            if not raw_contract.empty:
                                adjusted_close = existing_data.iloc[-1]['close']
                                raw_close = raw_contract.iloc[0]['close']
                                if adjusted_close is not None and raw_close is not None and raw_close != 0:
                                    self.cumulative_factor = adjusted_close / raw_close
                                    logger.info(f"Calculated cumulative Panama factor from data: {self.cumulative_factor}")
                
                # Update start_date to the day after the last date
                start_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Skip if we're already up to date
                if pd.Timestamp(start_date) > pd.Timestamp(end_date):
                    logger.info(f"Continuous data for {continuous_symbol} is already up to date")
                    return existing_data
                    
                logger.info(f"Updating {continuous_symbol} from {start_date} to {end_date}, "
                          f"cumulative factor: {self.cumulative_factor}")
            else:
                # If no existing data or force rebuild, we're building from scratch
                # Set a reasonable default start date if not provided
                if start_date is None:
                    metadata_query_specific = '''
                        SELECT start_date
                        FROM symbol_metadata
                        WHERE symbol = ? 
                        LIMIT 1
                    '''
                    metadata_df = self.db.query_to_df(metadata_query_specific, [continuous_symbol])

                    if metadata_df.empty or metadata_df.iloc[0]['start_date'] is None:
                        logger.info(f"No start_date in symbol_metadata for specific '{continuous_symbol}'. Trying base_symbol '{root_symbol}'.")
                        metadata_query_root = '''
                            SELECT start_date
                            FROM symbol_metadata
                            WHERE base_symbol = ?
                            LIMIT 1 
                        '''
                        metadata_df = self.db.query_to_df(metadata_query_root, [root_symbol])

                    if not metadata_df.empty and metadata_df.iloc[0]['start_date'] is not None:
                        # Ensure start_date is a string in 'YYYY-MM-DD' format
                        retrieved_start_date = pd.to_datetime(metadata_df.iloc[0]['start_date'])
                        start_date = retrieved_start_date.strftime('%Y-%m-%d')
                        logger.info(f"Using start_date '{start_date}' from symbol_metadata for {continuous_symbol} (derived from root/specific query).")
                    else:
                        default_start_date = '2000-01-01'
                        logger.warning(f"No start_date found in symbol_metadata for {continuous_symbol} or its root {root_symbol}. Using default: {default_start_date}")
                        start_date = default_start_date
                        
                logger.info(f"Building new continuous series for {continuous_symbol} "
                          f"from {start_date} to {end_date}")
            
            # Get roll dates for the period
            rollovers = self.get_roll_dates(root_symbol, start_date, end_date)
            
            # If we didn't find any rollovers, we may need to look further back
            # to find the contract we should be starting with
            if not rollovers and start_date > '2000-01-01':
                logger.info(f"No roll dates found in date range, looking for earlier rollovers")
                earlier_rollovers = self.get_roll_dates(root_symbol, '2000-01-01', start_date)
                if earlier_rollovers:
                    # Use the last rollover before our start date to determine the active contract
                    last_rollover = earlier_rollovers[-1]
                    active_contract = last_rollover.to_contract
                    logger.info(f"Found active contract at start date: {active_contract}")
                    
                    # Add this contract to our processing
                    full_active_contract_symbol = f"{root_symbol}{active_contract}"
                    logger.info(f"Attempting to load data for full symbol: {full_active_contract_symbol}")
                    contract_data = self.load_contract_data(
                        full_active_contract_symbol, start_date, end_date, interval_unit, interval_value
                    )
                    
                    continuous_data = self._build_continuous_from_contracts(
                        [contract_data], [], continuous_symbol, interval_unit, interval_value
                    )
                else:
                    # If we still don't have any rollovers, try to find the oldest contract
                    query = f"""
                    SELECT symbol, MIN(timestamp) as first_date
                    FROM {self.market_data_table}
                    WHERE symbol LIKE '{root_symbol}%'
                    AND interval_unit = ?
                    AND interval_value = ?
                    AND timestamp::DATE >= ?
                    AND timestamp::DATE <= ?
                    GROUP BY symbol
                    ORDER BY first_date
                    LIMIT 1
                    """
                    params = [interval_unit, interval_value, start_date, end_date]
                    df = self.db.query_to_df(query, params)
                    
                    if not df.empty:
                        # df.iloc[0]['symbol'] here is already the full symbol like 'ESH03'
                        active_contract_full_symbol = df.iloc[0]['symbol'] 
                        logger.info(f"Found oldest contract: {active_contract_full_symbol}")
                        
                        # Add this contract to our processing
                        contract_data = self.load_contract_data(
                            active_contract_full_symbol, start_date, end_date, interval_unit, interval_value
                        )
                        
                        continuous_data = self._build_continuous_from_contracts(
                            [contract_data], [], continuous_symbol, interval_unit, interval_value
                        )
                    else:
                        logger.warning(f"No contracts found for {root_symbol} in date range")
                        # If no contracts found, _build_continuous_from_contracts will likely receive empty list
                        # and continuous_data will be empty.
                        continuous_data = pd.DataFrame() # Ensure continuous_data is defined
            else:
                # Process each contract segment defined by the roll dates
                contract_segments = []
                active_contract = None
                
                # If we have rollovers, we need to determine the active contract at the start date
                if rollovers:
                    # The active contract at start_date is the to_contract from the last rollover before start_date
                    # or the from_contract of the first rollover if no earlier rollover exists
                    active_contract = rollovers[0].from_contract
                    
                    # Check if we need to get earlier rollovers to find the active contract at start_date
                    if pd.Timestamp(start_date) >= rollovers[0].date:
                        # Find the rollover that precedes our start date
                        for i, rollover in enumerate(rollovers):
                            if pd.Timestamp(start_date) < rollover.date:
                                active_contract = rollovers[i-1].to_contract if i > 0 else rollover.from_contract
                                break
                            elif i == len(rollovers) - 1:
                                # If we've reached the last rollover, the active contract is its to_contract
                                active_contract = rollover.to_contract
                    
                    logger.info(f"Active contract at start date: {active_contract}")
                    
                    # Load data for each segment
                    for i, rollover in enumerate(rollovers):
                        # Determine the segment's date range
                        segment_start = start_date if i == 0 else rollovers[i-1].date.strftime('%Y-%m-%d')
                        segment_end = rollover.date.strftime('%Y-%m-%d')
                        
                        # Only process segments that overlap with our date range
                        if pd.Timestamp(segment_end) >= pd.Timestamp(start_date) and pd.Timestamp(segment_start) <= pd.Timestamp(end_date):
                            segment_contract_code = rollover.from_contract # This is just 'H03', 'M03' etc.
                            
                            # Load contract data for this segment
                            full_segment_contract_symbol = f"{root_symbol}{segment_contract_code}"
                            logger.info(f"Attempting to load segment data for full symbol: {full_segment_contract_symbol}")
                            contract_data = self.load_contract_data(
                                full_segment_contract_symbol, segment_start, segment_end, interval_unit, interval_value
                            )
                            
                            if not contract_data.empty:
                                contract_segments.append(contract_data)
                                logger.info(f"Loaded {len(contract_data)} rows for {segment_contract_code}")
                    
                    # Add the final segment after the last rollover
                    if rollovers:
                        final_start = rollovers[-1].date.strftime('%Y-%m-%d')
                        final_contract = rollovers[-1].to_contract
                        
                        if pd.Timestamp(final_start) <= pd.Timestamp(end_date):
                            # Load contract data for the final segment
                            final_data = self.load_contract_data(
                                final_contract, final_start, end_date, interval_unit, interval_value
                            )
                            
                            if not final_data.empty:
                                contract_segments.append(final_data)
                                logger.info(f"Loaded {len(final_data)} rows for final segment {final_contract}")
                
                # Generate the continuous series
                if not contract_segments:
                    logger.warning(f"No contract segments found for {root_symbol} after processing rollovers.")
                    continuous_data = pd.DataFrame()
                else:
                    continuous_data = self._build_continuous_from_contracts(
                        contract_segments, rollovers, continuous_symbol, interval_unit, interval_value
                    )
            
            # At this point, continuous_data contains the newly built series (potentially empty)

            final_data_to_store = pd.DataFrame()

            if force:
                logger.info(f"Force rebuild: using newly generated data for {continuous_symbol}")
                final_data_to_store = continuous_data
            else: # Not forcing, so consider existing_data
                if existing_data.empty:
                    logger.info(f"No existing data found for {continuous_symbol}. Using newly generated data.")
                    final_data_to_store = continuous_data
                else:
                    # We have existing_data, try to append new portions of continuous_data
                    new_data_to_append = pd.DataFrame()
                    if not continuous_data.empty and 'timestamp' in continuous_data.columns and not continuous_data['timestamp'].empty:
                        if not existing_data['timestamp'].empty:
                            try:
                                # Filter continuous_data for timestamps strictly greater than the max in existing_data
                                new_data_to_append = continuous_data[continuous_data['timestamp'] > existing_data['timestamp'].max()].copy()
                            except Exception as e: # Catch any error during filtering, though primarily expecting type issues if any
                                logger.error(f"Error filtering new data against existing for {continuous_symbol}: {e}")
                                new_data_to_append = pd.DataFrame() # Default to no new data on error
                        else:
                            # existing_data has no timestamps, so all of continuous_data is "new"
                            new_data_to_append = continuous_data.copy()
                    else:
                        logger.info(f"Newly generated continuous_data is empty or lacks 'timestamp' column for {continuous_symbol}. No new data to filter.")
                    
                    if not new_data_to_append.empty:
                        logger.info(f"Appending {len(new_data_to_append)} new rows to {len(existing_data)} existing rows for {continuous_symbol}.")
                        final_data_to_store = pd.concat([existing_data, new_data_to_append], ignore_index=True)
                        # Ensure unique timestamps, keeping the latest if any duplicates arose (should not happen with > filter)
                        final_data_to_store = final_data_to_store.sort_values(by='timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
                    else:
                        logger.info(f"No new data to append to existing data for {continuous_symbol}.")
                        final_data_to_store = existing_data
            
            # Store the final data
            if not final_data_to_store.empty:
                # Sort before storing, especially if concatenated
                final_data_to_store = final_data_to_store.sort_values(by='timestamp').reset_index(drop=True)
                
                rows_stored = self.store_continuous_data(continuous_symbol, final_data_to_store)
                logger.info(f"Successfully stored {rows_stored} rows for {continuous_symbol}.")
                return final_data_to_store
            else:
                logger.warning(f"No data to store for {continuous_symbol} after processing.")
                return pd.DataFrame() # Return empty if nothing was stored

        except ContinuousContractError: # Re-raise specific errors
            raise
        except Exception as e:
            logger.error(f"Error building continuous series for {continuous_symbol}: {e}", exc_info=True)
            raise ContinuousContractError(f"Failed to build continuous series: {e}")
    
    def _build_continuous_from_contracts(self, contract_segments: List[pd.DataFrame],
                                       rollovers: List[ContractRollover],
                                       continuous_symbol: str,
                                       interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Build a continuous series from contract segments using the Panama method.
        
        Args:
            contract_segments: List of DataFrames with contract data
            rollovers: List of ContractRollover objects
            continuous_symbol: Target continuous symbol
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            DataFrame with the continuous series data
        """
        if not contract_segments:
            logger.warning(f"No contract segments provided for {continuous_symbol}")
            return pd.DataFrame()
            
        # Combine contract segments
        combined_data = pd.concat(contract_segments, ignore_index=True)
        
        if combined_data.empty:
            logger.warning(f"Combined data is empty for {continuous_symbol}")
            return pd.DataFrame()
            
        # Sort by timestamp
        combined_data = combined_data.sort_values('timestamp')
        
        # Create a copy to build the continuous series
        continuous_data = combined_data.copy()
        
        # Apply Panama adjustment
        if self.adjustment_method == 'ratio':
            # Apply multiplicative adjustment
            continuous_data, self.cumulative_factor = self._apply_panama_ratio_adjustment(
                continuous_data, rollovers, self.cumulative_factor
            )
        else:
            # Apply additive adjustment
            continuous_data, self.cumulative_offset = self._apply_panama_additive_adjustment(
                continuous_data, rollovers, self.cumulative_offset
            )
        
        # Add continuous symbol
        continuous_data['symbol'] = continuous_symbol
        
        # Add adjusted flag
        continuous_data['adjusted'] = True
        
        # Add built_by column
        continuous_data['built_by'] = self.__class__.__name__
        
        return continuous_data
    
    def _apply_panama_ratio_adjustment(self, data: pd.DataFrame,
                                     rollovers: List[ContractRollover],
                                     start_factor: float = 1.0) -> Tuple[pd.DataFrame, float]:
        """
        Apply Panama ratio adjustment to the continuous series.
        
        Args:
            data: DataFrame with combined contract data
            rollovers: List of ContractRollover objects
            start_factor: Starting cumulative adjustment factor
            
        Returns:
            Tuple of (adjusted DataFrame, final cumulative factor)
        """
        if data.empty:
            return data, start_factor
            
        # Create a copy to avoid modifying the input
        adjusted_data = data.copy()
        
        # Add a column to track the adjustment factor
        adjusted_data['panama_factor'] = start_factor
        
        current_factor = start_factor
        
        # Process each rollover
        for rollover in rollovers:
            rollover_date = rollover.date
            
            # Find the to_contract's closing price on the rollover date
            to_contract_price = adjusted_data[
                (adjusted_data['symbol'] == rollover.to_contract) & 
                (adjusted_data['timestamp'].dt.date == rollover_date.date())
            ]['close'].values
            
            if len(to_contract_price) == 0:
                logger.warning(f"No price found for {rollover.to_contract} on {rollover_date.date()}")
                continue
                
            to_price = to_contract_price[0]
            
            # Find the from_contract's closing price on the rollover date
            from_contract_price = adjusted_data[
                (adjusted_data['symbol'] == rollover.from_contract) & 
                (adjusted_data['timestamp'].dt.date == rollover_date.date())
            ]['close'].values
            
            if len(from_contract_price) == 0:
                logger.warning(f"No price found for {rollover.from_contract} on {rollover_date.date()}")
                continue
                
            from_price = from_contract_price[0]
            
            # Calculate the adjustment factor for this rollover
            if from_price == 0:
                logger.warning(f"From price is zero for {rollover.from_contract} on {rollover_date.date()}")
                continue
                
            # Calculate adjustment factor for this rollover
            rollover_factor = to_price / from_price
            
            # Calculate new cumulative factor for the incoming contract
            new_factor = current_factor * rollover_factor
            
            # Apply the new factor to the incoming contract and all later contracts
            mask = (adjusted_data['timestamp'] >= rollover_date) & (adjusted_data['symbol'] == rollover.to_contract)
            adjusted_data.loc[mask, 'panama_factor'] = new_factor
            
            # Update the current factor for the next rollover
            current_factor = new_factor
            
            logger.info(f"Panama adjustment at {rollover_date.date()}: "
                       f"{rollover.from_contract} -> {rollover.to_contract}, "
                       f"factor: {rollover_factor:.6f}, cumulative: {current_factor:.6f}")
        
        # Apply the cumulative factor to adjust prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in adjusted_data.columns:
                adjusted_data[col] = adjusted_data[col] / adjusted_data['panama_factor']
        
        return adjusted_data, current_factor
    
    def _apply_panama_additive_adjustment(self, data: pd.DataFrame,
                                        rollovers: List[ContractRollover],
                                        start_offset: float = 0.0) -> Tuple[pd.DataFrame, float]:
        """
        Apply Panama additive adjustment to the continuous series.
        
        Args:
            data: DataFrame with combined contract data
            rollovers: List of ContractRollover objects
            start_offset: Starting cumulative adjustment offset
            
        Returns:
            Tuple of (adjusted DataFrame, final cumulative offset)
        """
        if data.empty:
            return data, start_offset
            
        # Create a copy to avoid modifying the input
        adjusted_data = data.copy()
        
        # Add a column to track the adjustment offset
        adjusted_data['panama_offset'] = start_offset
        
        current_offset = start_offset
        
        # Process each rollover
        for rollover in rollovers:
            rollover_date = rollover.date
            
            # Find the to_contract's closing price on the rollover date
            to_contract_price = adjusted_data[
                (adjusted_data['symbol'] == rollover.to_contract) & 
                (adjusted_data['timestamp'].dt.date == rollover_date.date())
            ]['close'].values
            
            if len(to_contract_price) == 0:
                logger.warning(f"No price found for {rollover.to_contract} on {rollover_date.date()}")
                continue
                
            to_price = to_contract_price[0]
            
            # Find the from_contract's closing price on the rollover date
            from_contract_price = adjusted_data[
                (adjusted_data['symbol'] == rollover.from_contract) & 
                (adjusted_data['timestamp'].dt.date == rollover_date.date())
            ]['close'].values
            
            if len(from_contract_price) == 0:
                logger.warning(f"No price found for {rollover.from_contract} on {rollover_date.date()}")
                continue
                
            from_price = from_contract_price[0]
            
            # Calculate the adjustment offset for this rollover
            rollover_offset = from_price - to_price
            
            # Calculate new cumulative offset for the incoming contract
            new_offset = current_offset + rollover_offset
            
            # Apply the new offset to the incoming contract and all later contracts
            mask = (adjusted_data['timestamp'] >= rollover_date) & (adjusted_data['symbol'] == rollover.to_contract)
            adjusted_data.loc[mask, 'panama_offset'] = new_offset
            
            # Update the current offset for the next rollover
            current_offset = new_offset
            
            logger.info(f"Panama adjustment at {rollover_date.date()}: "
                       f"{rollover.from_contract} -> {rollover.to_contract}, "
                       f"offset: {rollover_offset:.2f}, cumulative: {current_offset:.2f}")
        
        # Apply the cumulative offset to adjust prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in adjusted_data.columns:
                adjusted_data[col] = adjusted_data[col] + adjusted_data['panama_offset']
        
        return adjusted_data, current_offset


# Register the builder
from .base import ContinuousContractRegistry
ContinuousContractRegistry.register("panama", PanamaContractBuilder)