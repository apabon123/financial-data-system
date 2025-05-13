"""
Unit tests for data cleaning pipeline.

These tests verify that the data cleaning pipeline correctly processes data,
tracks modifications, and logs all changes to the raw data.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import modules to test
from src.processors.cleaners.base import DataCleanerBase
from src.processors.cleaners.pipeline import DataCleaningPipeline
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner

class SimpleTestCleaner(DataCleanerBase):
    """Simple test cleaner for unit testing."""
    
    def __init__(self, db_connector=None, enabled=True):
        """Initialize test cleaner."""
        super().__init__(
            name="test_cleaner",
            description="A simple test cleaner",
            db_connector=db_connector,
            fields_to_clean=['close', 'volume'],
            enabled=enabled,
            priority=100,
            config={'test_param': 123}
        )
    
    def clean(self, df):
        """Test cleaning that replaces zero values with 1."""
        if df.empty:
            return df
        
        result = df.copy()
        
        # Replace zeros with 1 in close and volume columns
        for field in ['close', 'volume']:
            if field in result.columns:
                zeros_mask = (result[field] == 0) | result[field].isna()
                zero_indices = result.index[zeros_mask]
                
                for idx in zero_indices:
                    # Get timestamp and symbol for logging
                    timestamp = result.loc[idx, 'timestamp']
                    symbol = result.loc[idx, 'symbol']
                    old_value = result.loc[idx, field]
                    
                    # Replace with 1
                    result.loc[idx, field] = 1.0
                    
                    # Log the modification
                    self.log_modification(
                        timestamp=timestamp,
                        symbol=symbol,
                        field=field,
                        old_value=old_value,
                        new_value=1.0,
                        reason="Zero value replacement",
                        details="Test cleaner replaced zero with 1"
                    )
        
        return result

class AnotherTestCleaner(DataCleanerBase):
    """Another test cleaner for unit testing with different priority."""
    
    def __init__(self, db_connector=None, enabled=True):
        """Initialize test cleaner."""
        super().__init__(
            name="another_cleaner",
            description="Another test cleaner",
            db_connector=db_connector,
            fields_to_clean=['high', 'low'],
            enabled=enabled,
            priority=50,  # Higher priority (lower number)
            config={'test_param': 456}
        )
    
    def clean(self, df):
        """Test cleaning that ensures high >= low."""
        if df.empty:
            return df
        
        result = df.copy()
        
        # Ensure high >= low
        if 'high' in result.columns and 'low' in result.columns:
            invalid_mask = result['high'] < result['low']
            invalid_indices = result.index[invalid_mask]
            
            for idx in invalid_indices:
                # Get timestamp and symbol for logging
                timestamp = result.loc[idx, 'timestamp']
                symbol = result.loc[idx, 'symbol']
                old_high = result.loc[idx, 'high']
                old_low = result.loc[idx, 'low']
                
                # Swap high and low
                result.loc[idx, 'high'] = old_low
                result.loc[idx, 'low'] = old_high
                
                # Log the modifications
                self.log_modification(
                    timestamp=timestamp,
                    symbol=symbol,
                    field='high',
                    old_value=old_high,
                    new_value=old_low,
                    reason="High < Low correction",
                    details="Swapped high and low values"
                )
                
                self.log_modification(
                    timestamp=timestamp,
                    symbol=symbol,
                    field='low',
                    old_value=old_low,
                    new_value=old_high,
                    reason="High < Low correction",
                    details="Swapped high and low values"
                )
        
        return result

class TestDataCleaningPipeline(unittest.TestCase):
    """Test cases for data cleaning pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock database connector
        self.mock_db = MagicMock()
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'VXG23',
            'open': np.linspace(20.0, 25.0, len(dates)),
            'high': np.linspace(21.0, 26.0, len(dates)),
            'low': np.linspace(19.0, 24.0, len(dates)),
            'close': np.linspace(20.5, 25.5, len(dates)),
            'volume': np.linspace(1000, 1500, len(dates)).astype(int),
            'source': 'test'
        })
        
        # Add some problematic data for testing
        # Zero values
        self.test_data.loc[2, 'close'] = 0.0
        self.test_data.loc[5, 'volume'] = 0
        
        # High < Low inversion
        self.test_data.loc[3, 'high'] = 18.0  # Lower than the low value
        
        # Create cleaners
        self.cleaner1 = SimpleTestCleaner(db_connector=self.mock_db)
        self.cleaner2 = AnotherTestCleaner(db_connector=self.mock_db)
        
        # Create pipeline
        self.pipeline = DataCleaningPipeline(
            name="test_pipeline",
            db_connector=self.mock_db,
            cleaners=[self.cleaner1, self.cleaner2],
            config={'track_performance': True}
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.name, "test_pipeline")
        self.assertEqual(len(self.pipeline.cleaners), 2)
        
        # Cleaners should be sorted by priority (lower number first)
        self.assertEqual(self.pipeline.cleaners[0].name, "another_cleaner")
        self.assertEqual(self.pipeline.cleaners[1].name, "test_cleaner")
    
    def test_process_dataframe(self):
        """Test processing a DataFrame through the pipeline."""
        # Process the test data
        result_df, summary = self.pipeline.process_dataframe(self.test_data)
        
        # Verify summary
        self.assertEqual(summary['status'], 'success')
        self.assertGreater(summary['total_modifications'], 0)
        self.assertEqual(len(summary['cleaners_applied']), 2)
        
        # Verify that zero close and volume values were fixed
        self.assertNotEqual(result_df.loc[2, 'close'], 0.0)
        self.assertNotEqual(result_df.loc[5, 'volume'], 0)
        
        # Verify that high-low inversion was fixed
        self.assertGreaterEqual(result_df.loc[3, 'high'], result_df.loc[3, 'low'])
        
        # Check that modifications were tracked
        self.assertGreater(self.cleaner1._modifications_count, 0)
        self.assertGreater(self.cleaner2._modifications_count, 0)
    
    def test_cleaner_order(self):
        """Test that cleaners are applied in the correct order."""
        # Create a new test DataFrame with a specific test case
        test_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'symbol': ['TEST'],
            'open': [10.0],
            'high': [9.0],  # Inverted (high < low)
            'low': [11.0],
            'close': [0.0],  # Zero
            'volume': [0]    # Zero
        })
        
        # Process with a fresh pipeline
        pipeline = DataCleaningPipeline(
            name="order_test",
            db_connector=self.mock_db,
            cleaners=[self.cleaner1, self.cleaner2]
        )
        
        result_df, summary = pipeline.process_dataframe(test_df)
        
        # Cleaner2 (priority 50) should run before Cleaner1 (priority 100)
        # So high and low should be swapped first
        # Get the order of operations from the modifications
        cleaner1_modifications = self.cleaner1.get_recent_modifications()
        cleaner2_modifications = self.cleaner2.get_recent_modifications()
        
        # Both cleaners should have modifications
        self.assertGreater(len(cleaner1_modifications), 0)
        self.assertGreater(len(cleaner2_modifications), 0)
        
        # Determine execution order from summary
        cleaner_order = [c['name'] for c in summary['cleaners_applied']]
        self.assertEqual(cleaner_order, ['another_cleaner', 'test_cleaner'])
    
    def test_disabled_cleaner(self):
        """Test that disabled cleaners are skipped."""
        # Disable the first cleaner
        self.cleaner1.enabled = False
        
        # Process data
        result_df, summary = self.pipeline.process_dataframe(self.test_data)
        
        # Only the second cleaner should have been applied
        self.assertEqual(len(summary['cleaners_applied']), 1)
        self.assertEqual(summary['cleaners_applied'][0]['name'], 'another_cleaner')
        
        # Zero values should still be present (since cleaner1 is disabled)
        self.assertEqual(result_df.loc[2, 'close'], 0.0)
        self.assertEqual(result_df.loc[5, 'volume'], 0)
        
        # But high-low inversion should be fixed
        self.assertGreaterEqual(result_df.loc[3, 'high'], result_df.loc[3, 'low'])
    
    def test_modification_logging(self):
        """Test that modifications are properly logged."""
        # Create mock execute for the database connector
        self.mock_db.execute = MagicMock()
        
        # Process data
        result_df, summary = self.pipeline.process_dataframe(self.test_data)
        
        # Check that log_modification calls resulted in db.execute calls
        self.assertTrue(self.mock_db.execute.called)
        
        # Get all calls to execute
        calls = self.mock_db.execute.call_args_list
        
        # Should have at least one call for each modification
        self.assertGreaterEqual(len(calls), summary['total_modifications'])
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Create a cleaner that raises an exception
        error_cleaner = MagicMock()
        error_cleaner.name = "error_cleaner"
        error_cleaner.enabled = True
        error_cleaner.priority = 10  # Highest priority
        error_cleaner._modifications_count = 0
        error_cleaner.clean.side_effect = Exception("Test error")
        
        # Create a new pipeline with the error cleaner
        pipeline = DataCleaningPipeline(
            name="error_test",
            db_connector=self.mock_db,
            cleaners=[error_cleaner, self.cleaner1, self.cleaner2]
        )
        
        # Process data - should return original data on error
        result_df, summary = pipeline.process_dataframe(self.test_data)
        
        # Summary should indicate error
        self.assertEqual(summary['status'], 'error')
        self.assertIn('error', summary)
        
        # Result should be equal to original data
        pd.testing.assert_frame_equal(result_df, self.test_data)
    
    def test_performance_tracking(self):
        """Test that performance is properly tracked."""
        # Process data multiple times
        for _ in range(3):
            result_df, summary = self.pipeline.process_dataframe(self.test_data)
        
        # Check performance history
        performance_history = self.pipeline.get_performance_history()
        self.assertEqual(len(performance_history), 3)
        
        # Each record should have timing information
        for record in performance_history:
            self.assertIn('elapsed_time', record)
            self.assertGreater(record['elapsed_time'], 0)
            self.assertIn('cleaners_applied', record)
    
    def test_add_remove_cleaner(self):
        """Test adding and removing cleaners."""
        # Create a new cleaner
        new_cleaner = SimpleTestCleaner(db_connector=self.mock_db)
        new_cleaner.name = "new_cleaner"
        
        # Add to pipeline
        self.pipeline.add_cleaner(new_cleaner)
        
        # Should now have 3 cleaners
        self.assertEqual(len(self.pipeline.cleaners), 3)
        
        # Remove a cleaner
        result = self.pipeline.remove_cleaner("test_cleaner")
        self.assertTrue(result)
        
        # Should now have 2 cleaners
        self.assertEqual(len(self.pipeline.cleaners), 2)
        
        # Try to remove a non-existent cleaner
        result = self.pipeline.remove_cleaner("nonexistent")
        self.assertFalse(result)
        
        # Should still have 2 cleaners
        self.assertEqual(len(self.pipeline.cleaners), 2)
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Process data to generate some stats
        result_df, summary = self.pipeline.process_dataframe(self.test_data)
        
        # Should have stats now
        self.assertGreater(self.pipeline._run_count, 0)
        self.assertIsNotNone(self.pipeline._recent_run_summary)
        self.assertGreater(len(self.pipeline._performance_history), 0)
        
        # Reset stats
        self.pipeline.reset_stats()
        
        # Stats should be reset
        self.assertEqual(self.pipeline._run_count, 0)
        self.assertIsNone(self.pipeline._recent_run_summary)
        self.assertEqual(len(self.pipeline._performance_history), 0)
        
        # Cleaner stats should also be reset
        self.assertEqual(self.cleaner1._modifications_count, 0)
        self.assertEqual(self.cleaner2._modifications_count, 0)

if __name__ == '__main__':
    unittest.main()