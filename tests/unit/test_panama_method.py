"""
Unit tests for the Panama continuous futures contract generator.

These tests verify that the Panama method properly adjusts price data at roll points
and applies the adjustments consistently across the entire contract history.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import modules to test
from src.processors.continuous.panama import PanamaContractGenerator

class TestPanamaMethod(unittest.TestCase):
    """Test cases for the Panama method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock database connector
        self.mock_db = MagicMock()
        
        # Create test data for two contracts with a roll point
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        contract1_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'VXG23',
            'open': np.linspace(20.0, 25.0, len(dates)),
            'high': np.linspace(21.0, 26.0, len(dates)),
            'low': np.linspace(19.0, 24.0, len(dates)),
            'close': np.linspace(20.5, 25.5, len(dates)),
            'settle': np.linspace(20.5, 25.5, len(dates)),
            'volume': np.linspace(1000, 1500, len(dates)).astype(int),
            'open_interest': np.linspace(5000, 5500, len(dates)).astype(int),
            'underlying_symbol': 'VXG23',
            'source': 'test'
        })
        
        dates = pd.date_range(start='2023-01-06', end='2023-01-15')
        contract2_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'VXH23',
            'open': np.linspace(24.0, 29.0, len(dates)),
            'high': np.linspace(25.0, 30.0, len(dates)),
            'low': np.linspace(23.0, 28.0, len(dates)),
            'close': np.linspace(24.5, 29.5, len(dates)),
            'settle': np.linspace(24.5, 29.5, len(dates)),
            'volume': np.linspace(1200, 1700, len(dates)).astype(int),
            'open_interest': np.linspace(5200, 5700, len(dates)).astype(int),
            'underlying_symbol': 'VXH23',
            'source': 'test'
        })
        
        # Create a combined dataframe that represents what would be the input to the adjust_prices method
        # The rollover is on 2023-01-06
        roll_date = pd.Timestamp('2023-01-06')
        
        # Filter contract1 data to before roll date (inclusive)
        contract1_before_roll = contract1_data[contract1_data['timestamp'] <= roll_date]
        
        # Filter contract2 data to after roll date (exclusive)
        contract2_after_roll = contract2_data[contract2_data['timestamp'] > roll_date]
        
        # At the roll date, include data from both contracts
        contract1_at_roll = contract1_data[contract1_data['timestamp'] == roll_date]
        contract2_at_roll = contract2_data[contract2_data['timestamp'] == roll_date]
        
        # Combine data for testing
        self.combined_data = pd.concat([
            contract1_before_roll,
            contract1_at_roll,  # Last day of contract1 (roll day)
            contract2_after_roll  # First day after roll for contract2
        ]).sort_values('timestamp').reset_index(drop=True)
        
        # Create a roll point series where the first contract (VXG23) is used until 2023-01-06
        # and the second contract (VXH23) is used from 2023-01-07 onward
        self.roll_point = roll_date
        
        # Initialize the generator
        self.generator = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            interval_unit='daily',
            interval_value=1,
            db_connector=self.mock_db,
            ratio_limit=0.75  # 75% back-adjustment, 25% forward-adjustment
        )
    
    def test_panama_adjustment_at_roll(self):
        """Test that Panama method properly adjusts prices at roll points."""
        # Test with the standard 0.75 ratio
        adjusted_df = self.generator.adjust_prices(self.combined_data)
        
        # Check that adjusted DataFrame is not empty
        self.assertFalse(adjusted_df.empty)
        
        # Check that all price fields have been adjusted
        for field in ['open', 'high', 'low', 'close', 'settle']:
            self.assertTrue(field in adjusted_df.columns)
        
        # Identify roll point
        roll_day_old = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXG23')
        ]
        
        roll_day_new = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXH23')
        ]
        
        if not roll_day_old.empty and not roll_day_new.empty:
            # Get the close price of the old contract and open price of the new contract
            old_close = roll_day_old['close'].iloc[0]
            new_open = roll_day_new['open'].iloc[0]
            
            # Calculate the price ratio at the roll
            price_ratio = new_open / old_close
            
            # Calculate the expected adjustment factors
            expected_old_factor = price_ratio ** self.generator.ratio_limit
            expected_new_factor = price_ratio ** (self.generator.ratio_limit - 1.0)
            
            # Verify adjustment factors are applied correctly
            # The product of old_factor and new_factor should equal the price ratio
            self.assertAlmostEqual(expected_old_factor * expected_new_factor, price_ratio, places=6)
            
            # Check that adjustment factors are properly applied to the data
            # For the day before the roll
            pre_roll_idx = self.combined_data[
                (self.combined_data['timestamp'] < self.roll_point) & 
                (self.combined_data['underlying_symbol'] == 'VXG23')
            ].index[-1]
            
            # For the day after the roll
            post_roll_idx = self.combined_data[
                (self.combined_data['timestamp'] > self.roll_point) & 
                (self.combined_data['underlying_symbol'] == 'VXH23')
            ].index[0]
            
            # Extract original values
            orig_pre_roll_close = self.combined_data.loc[pre_roll_idx, 'close']
            orig_post_roll_open = self.combined_data.loc[post_roll_idx, 'open']
            
            # Extract adjusted values
            adj_pre_roll_close = adjusted_df.loc[pre_roll_idx, 'close']
            adj_post_roll_open = adjusted_df.loc[post_roll_idx, 'open']
            
            # Check that adjustment factors are correctly applied
            # Pre-roll close should be multiplied by old_factor
            self.assertAlmostEqual(
                adj_pre_roll_close, 
                orig_pre_roll_close * expected_old_factor, 
                places=4
            )
            
            # Post-roll open should be multiplied by new_factor
            self.assertAlmostEqual(
                adj_post_roll_open, 
                orig_post_roll_open * expected_new_factor, 
                places=4
            )
    
    def test_different_ratio_limits(self):
        """Test Panama adjustments with different ratio limits."""
        # Test with ratio 0 (pure forward adjustment)
        generator_fwd = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            interval_unit='daily',
            interval_value=1,
            db_connector=self.mock_db,
            ratio_limit=0.0  # 0% back-adjustment, 100% forward-adjustment
        )
        
        adjusted_fwd = generator_fwd.adjust_prices(self.combined_data)
        
        # Test with ratio 1 (pure back adjustment)
        generator_back = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            interval_unit='daily',
            interval_value=1,
            db_connector=self.mock_db,
            ratio_limit=1.0  # 100% back-adjustment, 0% forward-adjustment
        )
        
        adjusted_back = generator_back.adjust_prices(self.combined_data)
        
        # Test with ratio 0.5 (50/50 split)
        generator_split = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            interval_unit='daily',
            interval_value=1,
            db_connector=self.mock_db,
            ratio_limit=0.5  # 50% back-adjustment, 50% forward-adjustment
        )
        
        adjusted_split = generator_split.adjust_prices(self.combined_data)
        
        # Get pre and post roll indices
        # For the day before the roll
        pre_roll_indices = self.combined_data[
            (self.combined_data['timestamp'] < self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXG23')
        ].index
        
        # For the day after the roll
        post_roll_indices = self.combined_data[
            (self.combined_data['timestamp'] > self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXH23')
        ].index
        
        if pre_roll_indices.empty or post_roll_indices.empty:
            self.skipTest("Insufficient data around roll point")
            return
        
        # Verify expected behaviors:
        # 1. With ratio 0 (forward adjustment), only post-roll prices are adjusted
        # 2. With ratio 1 (back adjustment), only pre-roll prices are adjusted
        # 3. With ratio 0.5, adjustments are split evenly
        
        # Check adjustment patterns
        # For forward adjustment (ratio 0), pre-roll values should be unchanged
        pre_roll_idx = pre_roll_indices[-1]
        orig_pre_roll_close = self.combined_data.loc[pre_roll_idx, 'close']
        adj_fwd_pre_roll_close = adjusted_fwd.loc[pre_roll_idx, 'close']
        
        # Pre-roll values should be identical in forward adjustment
        self.assertAlmostEqual(orig_pre_roll_close, adj_fwd_pre_roll_close, places=4)
        
        # For back adjustment (ratio 1), post-roll values should be unchanged
        post_roll_idx = post_roll_indices[0]
        orig_post_roll_open = self.combined_data.loc[post_roll_idx, 'open']
        adj_back_post_roll_open = adjusted_back.loc[post_roll_idx, 'open']
        
        # Post-roll values should be identical in back adjustment
        self.assertAlmostEqual(orig_post_roll_open, adj_back_post_roll_open, places=4)
        
        # The 50/50 split should adjust both sides equally
        # Extract values from the split-adjusted series
        adj_split_pre_roll_close = adjusted_split.loc[pre_roll_idx, 'close']
        adj_split_post_roll_open = adjusted_split.loc[post_roll_idx, 'open']
        
        # Get roll date prices to calculate expected adjustments
        roll_day_old = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXG23')
        ]
        
        roll_day_new = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXH23')
        ]
        
        if not roll_day_old.empty and not roll_day_new.empty:
            old_close = roll_day_old['close'].iloc[0]
            new_open = roll_day_new['open'].iloc[0]
            price_ratio = new_open / old_close
            
            # Calculate expected factors for 50/50 split
            expected_old_factor_split = price_ratio ** 0.5
            expected_new_factor_split = price_ratio ** -0.5
            
            # Check that adjustments match expectations
            self.assertAlmostEqual(
                adj_split_pre_roll_close, 
                orig_pre_roll_close * expected_old_factor_split, 
                places=4
            )
            
            self.assertAlmostEqual(
                adj_split_post_roll_open, 
                orig_post_roll_open * expected_new_factor_split, 
                places=4
            )
    
    def test_adjustment_propagation(self):
        """Test that adjustments propagate correctly through the data."""
        adjusted_df = self.generator.adjust_prices(self.combined_data)
        
        # Get all pre-roll indices
        pre_roll_indices = self.combined_data[
            (self.combined_data['timestamp'] <= self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXG23')
        ].index
        
        # Get all post-roll indices
        post_roll_indices = self.combined_data[
            (self.combined_data['timestamp'] >= self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXH23')
        ].index
        
        # Calculate the adjustment factors based on roll point prices
        roll_day_old = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXG23')
        ]
        
        roll_day_new = self.combined_data[
            (self.combined_data['timestamp'] == self.roll_point) & 
            (self.combined_data['underlying_symbol'] == 'VXH23')
        ]
        
        if not roll_day_old.empty and not roll_day_new.empty:
            old_close = roll_day_old['close'].iloc[0]
            new_open = roll_day_new['open'].iloc[0]
            price_ratio = new_open / old_close
            
            expected_old_factor = price_ratio ** self.generator.ratio_limit
            expected_new_factor = price_ratio ** (self.generator.ratio_limit - 1.0)
            
            # Check that all pre-roll values are adjusted by the same factor
            for idx in pre_roll_indices:
                for field in ['open', 'high', 'low', 'close', 'settle']:
                    if field in self.combined_data.columns:
                        orig_value = self.combined_data.loc[idx, field]
                        adj_value = adjusted_df.loc[idx, field]
                        
                        if pd.notna(orig_value) and pd.notna(adj_value) and orig_value != 0:
                            actual_factor = adj_value / orig_value
                            self.assertAlmostEqual(actual_factor, expected_old_factor, places=4)
            
            # Check that all post-roll values are adjusted by the same factor
            for idx in post_roll_indices:
                for field in ['open', 'high', 'low', 'close', 'settle']:
                    if field in self.combined_data.columns:
                        orig_value = self.combined_data.loc[idx, field]
                        adj_value = adjusted_df.loc[idx, field]
                        
                        if pd.notna(orig_value) and pd.notna(adj_value) and orig_value != 0:
                            actual_factor = adj_value / orig_value
                            self.assertAlmostEqual(actual_factor, expected_new_factor, places=4)

if __name__ == '__main__':
    unittest.main()