#!/usr/bin/env python
"""
Unadjusted Continuous Futures Generator

This module implements the unadjusted method for generating continuous futures contracts.
The unadjusted method simply concatenates contract segments without adjusting prices,
resulting in gaps at contract rollovers.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import ContinuousContractBuilder, ContinuousContractError, ContractRollover
from ...core.database import Database

logger = logging.getLogger(__name__)

class UnadjustedContractBuilder(ContinuousContractBuilder):
    """Unadjusted continuous contract builder."""
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        """
        Initialize the unadjusted continuous contract builder.
        
        Args:
            db: Database instance
            config: Configuration dictionary
        """
        super().__init__(db, config)
    
    def build_continuous_series(self, root_symbol: str, continuous_symbol: str,
                               interval_unit: str = 'daily', interval_value: int = 1,
                               start_date: str = None, end_date: str = None,
                               force: bool = False) -> pd.DataFrame:
        """
        Build an unadjusted continuous series for the given parameters.
        
        Args:
            root_symbol: The root symbol (e.g., ES, VX)
            continuous_symbol: Target continuous symbol (e.g., @ES=101XN)
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
            if parsed['adjustment'] != 'unadjusted':
                logger.warning(f"Symbol {continuous_symbol} is not configured for unadjusted method")
            
            # Clean existing data if forced
            if force:
                self.delete_continuous_data(continuous_symbol, interval_unit, interval_value)
            
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            # Determine if we're updating or creating
            existing_data = self.get_existing_continuous_data(
                continuous_symbol, interval_unit, interval_value
            )
            
            if not existing_data.empty and not force:
                # We have existing data, so we need to build from the last date
                last_date = existing_data['timestamp'].max().strftime('%Y-%m-%d')
                
                # Update start_date to the day after the last date
                start_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Skip if we're already up to date
                if pd.Timestamp(start_date) > pd.Timestamp(end_date):
                    logger.info(f"Continuous data for {continuous_symbol} is already up to date")
                    return existing_data
                    
                logger.info(f"Updating {continuous_symbol} from {start_date} to {end_date}")
            else:
                # If no existing data or force rebuild, we're building from scratch
                # Set a reasonable default start date if not provided
                if start_date is None:
                    # Look for a symbol_metadata record for this symbol
                    query = """
                    SELECT start_date 
                    FROM symbol_metadata 
                    WHERE base_symbol = ? 
                    LIMIT 1
                    """
                    df = self.db.query_to_df(query, [root_symbol])
                    if not df.empty and df.iloc[0]['start_date'] is not None:
                        start_date = df.iloc[0]['start_date']
                    else:
                        # Fall back to a reasonable default
                        start_date = '2000-01-01'
                        
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
                    contract_data = self.load_contract_data(
                        active_contract, start_date, end_date, interval_unit, interval_value
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
                        active_contract = df.iloc[0]['symbol']
                        logger.info(f"Found oldest contract: {active_contract}")
                        
                        # Add this contract to our processing
                        contract_data = self.load_contract_data(
                            active_contract, start_date, end_date, interval_unit, interval_value
                        )
                        
                        continuous_data = self._build_continuous_from_contracts(
                            [contract_data], [], continuous_symbol, interval_unit, interval_value
                        )
                    else:
                        logger.warning(f"No contracts found for {root_symbol} in date range")
                        return pd.DataFrame()
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
                            segment_contract = rollover.from_contract
                            
                            # Load contract data for this segment
                            contract_data = self.load_contract_data(
                                segment_contract, segment_start, segment_end, interval_unit, interval_value
                            )
                            
                            if not contract_data.empty:
                                contract_segments.append(contract_data)
                                logger.info(f"Loaded {len(contract_data)} rows for {segment_contract}")
                    
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
                continuous_data = self._build_continuous_from_contracts(
                    contract_segments, rollovers, continuous_symbol, interval_unit, interval_value
                )
            
            # If we have existing data, append the new data
            if not existing_data.empty and not force:
                # Filter out any overlap with existing data
                new_data = continuous_data[
                    continuous_data['timestamp'] > existing_data['timestamp'].max()
                ]
                
                if not new_data.empty:
                    logger.info(f"Appending {len(new_data)} new rows to existing {len(existing_data)} rows")
                    continuous_data = pd.concat([existing_data, new_data], ignore_index=True)
                else:
                    logger.info(f"No new data to append")
                    continuous_data = existing_data
            
            # Store the continuous data
            if not continuous_data.empty:
                self.store_continuous_data(continuous_symbol, continuous_data)
                
            return continuous_data
            
        except Exception as e:
            logger.error(f"Error building continuous series for {continuous_symbol}: {e}")
            raise ContinuousContractError(f"Failed to build continuous series: {e}")
    
    def _build_continuous_from_contracts(self, contract_segments: List[pd.DataFrame],
                                       rollovers: List[ContractRollover],
                                       continuous_symbol: str,
                                       interval_unit: str, interval_value: int) -> pd.DataFrame:
        """
        Build a continuous series from contract segments using the unadjusted method.
        
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
            
        # For unadjusted method, we simply concatenate the segments
        # and ensure they don't overlap
        result_segments = []
        
        for i, segment in enumerate(contract_segments):
            if segment.empty:
                continue
                
            # Keep track of the original symbol for each row
            segment['underlying_symbol'] = segment['symbol']
            
            # If this is not the first segment, ensure it doesn't overlap with previous segments
            if i > 0 and rollovers and i - 1 < len(rollovers):
                # Use the roll date as the cutoff
                roll_date = rollovers[i-1].date
                
                # Filter to only include rows on or after the roll date
                filtered_segment = segment[segment['timestamp'] >= roll_date].copy()
                
                if not filtered_segment.empty:
                    result_segments.append(filtered_segment)
            else:
                result_segments.append(segment.copy())
        
        if not result_segments:
            logger.warning(f"No valid segments after filtering for {continuous_symbol}")
            return pd.DataFrame()
            
        # Combine the filtered segments
        continuous_data = pd.concat(result_segments, ignore_index=True)
        
        # Sort by timestamp
        continuous_data = continuous_data.sort_values('timestamp')
        
        # Remove duplicate timestamps (should not happen, but just in case)
        continuous_data = continuous_data.drop_duplicates(subset=['timestamp'])
        
        # Set the continuous symbol
        continuous_data['symbol'] = continuous_symbol
        
        # Set adjusted flag to False (unadjusted)
        continuous_data['adjusted'] = False
        
        # Add built_by column
        continuous_data['built_by'] = self.__class__.__name__
        
        return continuous_data


# Register the builder
from .base import ContinuousContractRegistry
ContinuousContractRegistry.register("unadjusted", UnadjustedContractBuilder)