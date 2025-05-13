"""
Data cleaning pipeline for orchestrating multiple data cleaners.

This module provides a pipeline for applying multiple data cleaners in sequence,
with proper tracking of all modifications made to the raw data.

The pipeline ensures that:
1. Raw data is preserved in separate tables
2. All modifications are logged with timestamp, reason, and values
3. Cleaners are applied in the correct order based on priority
4. The process is transactional and can be rolled back if needed
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import time
from datetime import datetime

from .base import DataCleanerBase

# Logger for this module
logger = logging.getLogger(__name__)

class DataCleaningPipeline:
    """
    Pipeline for orchestrating and applying multiple data cleaners.
    
    The pipeline manages the cleaning process from start to finish, including:
    - Loading raw data from the database
    - Applying cleaners in order of priority
    - Tracking all modifications
    - Saving cleaned data back to the database
    - Maintaining audit logs of cleaning operations
    """
    
    def __init__(
        self,
        name: str,
        db_connector = None,
        cleaners: List[DataCleanerBase] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the data cleaning pipeline.
        
        Args:
            name: Name of the pipeline
            db_connector: Database connector instance
            cleaners: List of data cleaner instances
            config: Configuration options
        """
        self.name = name
        self.db_connector = db_connector
        self.cleaners = cleaners or []
        self.config = config or {}
        
        # Sort cleaners by priority
        self._sort_cleaners()
        
        # Configure tracking
        self._track_performance = self.config.get('track_performance', True)
        self._track_modifications = self.config.get('track_modifications', True)
        self._save_summary = self.config.get('save_summary', True)
        
        # Internal tracking
        self._run_count = 0
        self._total_records_cleaned = 0
        self._total_modifications = 0
        self._performance_history = []
        self._recent_run_summary = None
    
    def add_cleaner(self, cleaner: DataCleanerBase) -> None:
        """
        Add a cleaner to the pipeline.
        
        Args:
            cleaner: Data cleaner instance to add
        """
        if not isinstance(cleaner, DataCleanerBase):
            logger.error(f"Cannot add cleaner: {cleaner} is not an instance of DataCleanerBase")
            return
        
        self.cleaners.append(cleaner)
        self._sort_cleaners()
        
        logger.debug(f"Added cleaner '{cleaner.name}' to pipeline '{self.name}'")
    
    def remove_cleaner(self, cleaner_name: str) -> bool:
        """
        Remove a cleaner from the pipeline by name.
        
        Args:
            cleaner_name: Name of the cleaner to remove
            
        Returns:
            True if the cleaner was removed, False otherwise
        """
        for i, cleaner in enumerate(self.cleaners):
            if cleaner.name == cleaner_name:
                del self.cleaners[i]
                logger.debug(f"Removed cleaner '{cleaner_name}' from pipeline '{self.name}'")
                return True
        
        logger.warning(f"Cleaner '{cleaner_name}' not found in pipeline '{self.name}'")
        return False
    
    def _sort_cleaners(self) -> None:
        """Sort cleaners by priority (lower number = higher priority)."""
        self.cleaners.sort(key=lambda c: c.priority)
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply all cleaners to a DataFrame and track modifications.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Tuple of (cleaned DataFrame, summary dictionary)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to cleaning pipeline. Returning unmodified.")
            return df, {'status': 'skipped', 'reason': 'empty_dataframe'}
        
        # Make a copy of the original data for tracking changes
        original_df = df.copy()
        
        # Track overall run metrics
        self._run_count += 1
        run_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._run_count}"
        
        start_time = time.time()
        records_count = len(df)
        self._total_records_cleaned += records_count
        
        # Summary of this run
        run_summary = {
            'run_id': run_id,
            'pipeline_name': self.name,
            'start_time': datetime.now(),
            'records_count': records_count,
            'cleaners_applied': [],
            'total_modifications': 0,
            'elapsed_time': 0,
            'fields_modified': set(),
            'status': 'success'
        }
        
        try:
            # Apply each cleaner in order
            for cleaner in self.cleaners:
                if not cleaner.enabled:
                    logger.debug(f"Skipping disabled cleaner: {cleaner.name}")
                    continue
                
                # Track cleaner performance
                cleaner_start = time.time()
                
                # Reset cleaner stats
                original_mods_count = cleaner._modifications_count
                
                # Apply the cleaner
                logger.debug(f"Applying cleaner: {cleaner.name}")
                df = cleaner.clean(df)
                
                # Calculate modifications made by this cleaner
                cleaner_mods = cleaner._modifications_count - original_mods_count
                self._total_modifications += cleaner_mods
                run_summary['total_modifications'] += cleaner_mods
                
                # Get the fields modified by this cleaner from recent modifications
                modified_fields = set()
                for mod in cleaner.get_recent_modifications():
                    modified_fields.add(mod['field'])
                
                run_summary['fields_modified'].update(modified_fields)
                
                # Record cleaner performance
                cleaner_elapsed = time.time() - cleaner_start
                cleaner_summary = {
                    'name': cleaner.name,
                    'elapsed_time': cleaner_elapsed,
                    'modifications': cleaner_mods,
                    'fields_modified': list(modified_fields)
                }
                
                run_summary['cleaners_applied'].append(cleaner_summary)
                
                logger.debug(f"Applied cleaner: {cleaner.name} "
                           f"({cleaner_mods} modifications in {cleaner_elapsed:.2f}s)")
            
            # Finalize summary
            run_summary['elapsed_time'] = time.time() - start_time
            run_summary['fields_modified'] = list(run_summary['fields_modified'])
            run_summary['end_time'] = datetime.now()
            
            logger.info(f"Cleaning pipeline '{self.name}' completed: "
                      f"{run_summary['total_modifications']} modifications "
                      f"across {len(run_summary['fields_modified'])} fields "
                      f"in {run_summary['elapsed_time']:.2f}s")
            
            # Store performance history if enabled
            if self._track_performance:
                self._performance_history.append(run_summary)
                # Limit history size
                max_history = self.config.get('max_history', 10)
                if len(self._performance_history) > max_history:
                    self._performance_history = self._performance_history[-max_history:]
            
            # Store most recent run summary
            self._recent_run_summary = run_summary
            
            # Log summary to database if configured
            if self._save_summary and self.db_connector:
                self._log_run_summary(run_summary)
            
            return df, run_summary
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline '{self.name}': {e}", exc_info=True)
            
            run_summary['status'] = 'error'
            run_summary['error'] = str(e)
            run_summary['elapsed_time'] = time.time() - start_time
            run_summary['end_time'] = datetime.now()
            
            self._recent_run_summary = run_summary
            
            # Log error summary to database if configured
            if self._save_summary and self.db_connector:
                self._log_run_summary(run_summary)
            
            # Return original data if there's an error
            return original_df, run_summary
    
    def process_symbol(self, 
                     symbol: str, 
                     start_date: Optional[Union[str, datetime]] = None,
                     end_date: Optional[Union[str, datetime]] = None,
                     interval_unit: str = 'daily',
                     interval_value: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """
        Load data for a symbol from the database, clean it, and save it back.
        
        Args:
            symbol: Symbol to clean data for
            start_date: Optional start date
            end_date: Optional end date
            interval_unit: Time interval unit
            interval_value: Time interval value
            
        Returns:
            Tuple of (success flag, summary dictionary)
        """
        if not self.db_connector:
            logger.error("No database connector provided. Cannot process symbol from database.")
            return False, {'status': 'error', 'reason': 'no_db_connector'}
        
        try:
            # Determine raw and clean table names based on symbol
            table_prefix = self._get_table_prefix_for_symbol(symbol)
            raw_table = f"{table_prefix}_raw"
            clean_table = table_prefix
            
            # Build query parameters
            params = [symbol, interval_unit, interval_value]
            date_conditions = []
            
            if start_date:
                date_conditions.append("timestamp >= ?")
                if isinstance(start_date, datetime):
                    params.append(start_date.strftime('%Y-%m-%d'))
                else:
                    params.append(start_date)
            
            if end_date:
                date_conditions.append("timestamp <= ?")
                if isinstance(end_date, datetime):
                    params.append(end_date.strftime('%Y-%m-%d'))
                else:
                    params.append(end_date)
            
            # Build date condition string
            date_condition = ""
            if date_conditions:
                date_condition = f"AND {' AND '.join(date_conditions)}"
            
            # Query to get raw data
            query = f"""
                SELECT *
                FROM {raw_table}
                WHERE symbol = ?
                  AND interval_unit = ?
                  AND interval_value = ?
                  {date_condition}
                ORDER BY timestamp ASC
            """
            
            # Load raw data
            raw_df = self.db_connector.query(query, params)
            
            if raw_df.empty:
                logger.warning(f"No data found for {symbol} in {raw_table}")
                return False, {'status': 'skipped', 'reason': 'no_data'}
            
            # Process the data through the pipeline
            cleaned_df, summary = self.process_dataframe(raw_df)
            
            # If no modifications or errors, we're done
            if summary['status'] != 'success' or summary['total_modifications'] == 0:
                logger.info(f"No modifications needed for {symbol}")
                return True, summary
            
            # Start a transaction
            self.db_connector.begin_transaction()
            
            try:
                # Delete existing clean data for this range
                delete_query = f"""
                    DELETE FROM {clean_table}
                    WHERE symbol = ?
                      AND interval_unit = ?
                      AND interval_value = ?
                      {date_condition}
                """
                self.db_connector.execute(delete_query, params)
                
                # Insert cleaned data
                self.db_connector.insert_dataframe(clean_table, cleaned_df)
                
                # Commit the transaction
                self.db_connector.commit_transaction()
                
                logger.info(f"Successfully saved cleaned data for {symbol} to {clean_table}")
                return True, summary
                
            except Exception as e:
                # Rollback on error
                self.db_connector.rollback_transaction()
                logger.error(f"Error saving cleaned data for {symbol}: {e}")
                
                summary['status'] = 'error'
                summary['error'] = str(e)
                
                return False, summary
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
            return False, {'status': 'error', 'reason': str(e)}
    
    def process_table(self,
                    table_name: str,
                    symbols: Optional[List[str]] = None,
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    batch_size: int = 1000) -> Dict[str, Any]:
        """
        Process all data in a table, optionally filtered by symbols and date range.
        
        Args:
            table_name: Raw table name to process
            symbols: Optional list of symbols to filter by
            start_date: Optional start date
            end_date: Optional end date
            batch_size: Number of symbols to process in each batch
            
        Returns:
            Summary dictionary
        """
        if not self.db_connector:
            logger.error("No database connector provided. Cannot process table.")
            return {'status': 'error', 'reason': 'no_db_connector'}
        
        # Create clean table name
        if table_name.endswith('_raw'):
            clean_table = table_name[:-4]  # Remove _raw suffix
        else:
            clean_table = table_name
            table_name = f"{table_name}_raw"  # Add _raw suffix
        
        try:
            # Get all symbols from the table
            symbol_query = f"""
                SELECT DISTINCT symbol
                FROM {table_name}
            """
            
            if symbols:
                placeholders = ", ".join(["?" for _ in symbols])
                symbol_query += f" WHERE symbol IN ({placeholders})"
            
            symbol_df = self.db_connector.query(symbol_query, symbols or [])
            
            if symbol_df.empty:
                logger.warning(f"No symbols found in {table_name}")
                return {'status': 'skipped', 'reason': 'no_symbols'}
            
            all_symbols = symbol_df['symbol'].tolist()
            total_symbols = len(all_symbols)
            
            # Start overall processing
            overall_start = time.time()
            overall_summary = {
                'pipeline_name': self.name,
                'start_time': datetime.now(),
                'table_name': table_name,
                'total_symbols': total_symbols,
                'processed_symbols': 0,
                'successful_symbols': 0,
                'skipped_symbols': 0,
                'error_symbols': 0,
                'total_modifications': 0,
                'elapsed_time': 0,
                'symbol_summaries': {}
            }
            
            # Process symbols in batches
            for i in range(0, total_symbols, batch_size):
                batch_symbols = all_symbols[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_symbols-1)//batch_size + 1} "
                          f"({len(batch_symbols)} symbols)")
                
                for symbol in batch_symbols:
                    success, summary = self.process_symbol(
                        symbol, start_date, end_date, 
                        # Default interval, override if needed for specific tables
                        interval_unit='daily', interval_value=1
                    )
                    
                    overall_summary['processed_symbols'] += 1
                    
                    if success:
                        if summary['status'] == 'skipped':
                            overall_summary['skipped_symbols'] += 1
                        else:
                            overall_summary['successful_symbols'] += 1
                            overall_summary['total_modifications'] += summary.get('total_modifications', 0)
                    else:
                        overall_summary['error_symbols'] += 1
                    
                    # Store symbol summary
                    overall_summary['symbol_summaries'][symbol] = {
                        'status': summary.get('status', 'unknown'),
                        'modifications': summary.get('total_modifications', 0),
                        'elapsed_time': summary.get('elapsed_time', 0),
                        'error': summary.get('error', None)
                    }
            
            # Finalize overall summary
            overall_summary['elapsed_time'] = time.time() - overall_start
            overall_summary['end_time'] = datetime.now()
            
            # Log performance
            symbols_per_second = overall_summary['processed_symbols'] / overall_summary['elapsed_time'] if overall_summary['elapsed_time'] > 0 else 0
            logger.info(f"Processed {overall_summary['successful_symbols']}/{overall_summary['processed_symbols']} symbols "
                      f"with {overall_summary['total_modifications']} modifications "
                      f"in {overall_summary['elapsed_time']:.2f}s "
                      f"({symbols_per_second:.2f} symbols/s)")
            
            return overall_summary
            
        except Exception as e:
            logger.error(f"Error processing table {table_name}: {e}", exc_info=True)
            return {'status': 'error', 'reason': str(e)}
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """
        Get the performance history of the pipeline.
        
        Returns:
            List of run summaries
        """
        return self._performance_history
    
    def get_latest_run_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get the summary of the most recent pipeline run.
        
        Returns:
            Run summary dictionary or None if no runs have been performed
        """
        return self._recent_run_summary
    
    def get_cleaner_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all cleaners in the pipeline.
        
        Returns:
            List of cleaner statistics dictionaries
        """
        return [cleaner.get_modification_stats() for cleaner in self.cleaners]
    
    def reset_stats(self) -> None:
        """Reset all performance tracking statistics."""
        self._run_count = 0
        self._total_records_cleaned = 0
        self._total_modifications = 0
        self._performance_history = []
        self._recent_run_summary = None
        
        # Also reset each cleaner's stats
        for cleaner in self.cleaners:
            cleaner.reset_stats()
    
    def _log_run_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Log a run summary to the database.
        
        Args:
            summary: Run summary dictionary
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self.db_connector:
            return False
        
        try:
            # Prepare data for logging
            log_data = {
                'run_id': summary['run_id'],
                'pipeline_name': summary['pipeline_name'],
                'start_time': summary['start_time'],
                'end_time': summary.get('end_time', datetime.now()),
                'records_count': summary['records_count'],
                'modifications_count': summary['total_modifications'],
                'elapsed_time': summary['elapsed_time'],
                'status': summary['status'],
                'error': summary.get('error', None),
                'cleaners_applied': ','.join([c['name'] for c in summary['cleaners_applied']]),
                'fields_modified': ','.join(summary['fields_modified']),
                'timestamp': datetime.now()
            }
            
            # Insert into log table
            self.db_connector.insert_record('data_cleaning_runs', log_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging run summary: {e}")
            return False
    
    def _get_table_prefix_for_symbol(self, symbol: str) -> str:
        """
        Get the appropriate table prefix for a symbol.
        
        Args:
            symbol: Symbol to determine table prefix for
            
        Returns:
            Table prefix (e.g., 'market_data', 'cboe_data', 'continuous_contracts')
        """
        # Logic to determine the appropriate table prefix
        # This is a simplified example and should be adapted based on your system
        
        # Check for special symbol prefixes
        if symbol.startswith('@'):
            return 'continuous_contracts'
        elif symbol.startswith('$'):
            return 'cboe_data'
        else:
            return 'market_data'