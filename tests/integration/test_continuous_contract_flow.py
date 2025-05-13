"""
Integration tests for the continuous contract generation workflow.

These tests verify that the complete flow from raw market data to
continuous contract generation with data cleaning works correctly.
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
from src.processors.continuous.base import ContinuousContractBase
from src.processors.continuous.panama import PanamaContractGenerator
from src.processors.continuous.registry import get_registry
from src.processors.continuous.roll_calendar import RollCalendar
from src.processors.cleaners.vx_zero_prices import VXZeroPricesCleaner
from src.processors.cleaners.pipeline import DataCleaningPipeline

class TestContinuousContractFlow(unittest.TestCase):
    """Integration tests for continuous contract generation workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Create test data directory if it doesn't exist
        cls.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create test database connector
        cls.mock_db = MagicMock()
        
        # Prepare test data for multiple contracts
        cls.generate_test_contract_data()
        
        # Set up mock database queries
        cls.setup_mock_db_queries()
    
    @classmethod
    def generate_test_contract_data(cls):
        """Generate test data for multiple contracts with realistic relationships."""
        # Create 3 contracts with overlapping dates
        # VXF23 (Jan futures, expires in Jan)
        # VXG23 (Feb futures, expires in Feb)
        # VXH23 (Mar futures, expires in Mar)
        
        # Contract 1 (VXF23) - Jan 1 to Jan 20
        dates1 = pd.date_range(start='2023-01-01', end='2023-01-20')
        contract1_data = pd.DataFrame({
            'timestamp': dates1,
            'symbol': 'VXF23',
            'open': np.linspace(20.0, 22.0, len(dates1)),
            'high': np.linspace(21.0, 23.0, len(dates1)),
            'low': np.linspace(19.0, 21.0, len(dates1)),
            'close': np.linspace(20.5, 22.5, len(dates1)),
            'settle': np.linspace(20.5, 22.5, len(dates1)),
            'volume': np.linspace(1000, 500, len(dates1)).astype(int),  # Decreasing volume
            'open_interest': np.linspace(5000, 2500, len(dates1)).astype(int),  # Decreasing OI
            'underlying_symbol': 'VXF23',
            'source': 'test'
        })
        
        # Contract 2 (VXG23) - Jan 10 to Feb 20
        dates2 = pd.date_range(start='2023-01-10', end='2023-02-20')
        contract2_data = pd.DataFrame({
            'timestamp': dates2,
            'symbol': 'VXG23',
            'open': np.linspace(21.5, 23.5, len(dates2)),
            'high': np.linspace(22.5, 24.5, len(dates2)),
            'low': np.linspace(20.5, 22.5, len(dates2)),
            'close': np.linspace(22.0, 24.0, len(dates2)),
            'settle': np.linspace(22.0, 24.0, len(dates2)),
            'volume': np.concatenate([
                np.linspace(600, 1500, 11),  # Jan 10-20: Increasing volume
                np.linspace(1500, 600, len(dates2) - 11)  # Jan 21-Feb 20: Decreasing volume
            ]).astype(int),
            'open_interest': np.concatenate([
                np.linspace(3000, 6000, 11),  # Jan 10-20: Increasing OI
                np.linspace(6000, 3000, len(dates2) - 11)  # Jan 21-Feb 20: Decreasing OI
            ]).astype(int),
            'underlying_symbol': 'VXG23',
            'source': 'test'
        })
        
        # Contract 3 (VXH23) - Feb 10 to Mar 20
        dates3 = pd.date_range(start='2023-02-10', end='2023-03-20')
        contract3_data = pd.DataFrame({
            'timestamp': dates3,
            'symbol': 'VXH23',
            'open': np.linspace(23.0, 25.0, len(dates3)),
            'high': np.linspace(24.0, 26.0, len(dates3)),
            'low': np.linspace(22.0, 24.0, len(dates3)),
            'close': np.linspace(23.5, 25.5, len(dates3)),
            'settle': np.linspace(23.5, 25.5, len(dates3)),
            'volume': np.concatenate([
                np.linspace(700, 1600, 11),  # Feb 10-20: Increasing volume
                np.linspace(1600, 800, len(dates3) - 11)  # Feb 21-Mar 20: Decreasing volume
            ]).astype(int),
            'open_interest': np.concatenate([
                np.linspace(3500, 6500, 11),  # Feb 10-20: Increasing OI
                np.linspace(6500, 3500, len(dates3) - 11)  # Feb 21-Mar 20: Decreasing OI
            ]).astype(int),
            'underlying_symbol': 'VXH23',
            'source': 'test'
        })
        
        # Add some test anomalies for data cleaning to fix
        
        # Zero prices in first contract
        contract1_data.loc[5, 'close'] = 0.0
        contract1_data.loc[5, 'settle'] = 0.0
        
        # Zero volume in second contract
        contract2_data.loc[15, 'volume'] = 0
        
        # Price inversion in third contract (high < low)
        contract3_data.loc[10, 'high'] = 21.0  # Less than low
        
        # Roll dates based on volume/OI crossover
        # Should be around Jan 15 (VXF23 -> VXG23) and Feb 15 (VXG23 -> VXH23)
        
        # Store the test data
        cls.contract1_data = contract1_data
        cls.contract2_data = contract2_data
        cls.contract3_data = contract3_data
        
        # Combined data for testing
        cls.all_contracts_data = pd.concat([
            contract1_data, contract2_data, contract3_data
        ]).sort_values('timestamp').reset_index(drop=True)
    
    @classmethod
    def setup_mock_db_queries(cls):
        """Set up mock database queries to return test data."""
        
        def mock_query(query, params=None):
            """Mock query implementation."""
            # Return contract data based on symbol in WHERE clause
            if "symbol = ?" in query or "symbol LIKE ?" in query:
                symbol_param = params[0] if params else None
                
                if symbol_param == 'VXF23':
                    return cls.contract1_data
                elif symbol_param == 'VXG23':
                    return cls.contract2_data
                elif symbol_param == 'VXH23':
                    return cls.contract3_data
                elif symbol_param == 'VX%' or symbol_param is None:
                    return cls.all_contracts_data
            
            # Return roll calendar data
            if "futures_roll_calendar" in query:
                # Create mock roll calendar
                dates = ['2023-01-15', '2023-02-15', '2023-03-15']
                contracts = ['VXF23', 'VXG23', 'VXH23']
                
                return pd.DataFrame({
                    'contract_code': contracts,
                    'last_trading_day': [pd.Timestamp(d) for d in dates],
                    'roll_date': [pd.Timestamp(d) - pd.Timedelta(days=5) for d in dates],
                    'expiration_date': [pd.Timestamp(d) + pd.Timedelta(days=1) for d in dates],
                    'roll_method': ['volume', 'volume', 'volume'],
                    'root_symbol': ['VX', 'VX', 'VX']
                })
            
            # Default empty response
            return pd.DataFrame()
        
        cls.mock_db.query = mock_query
    
    def test_roll_calendar_generation(self):
        """Test generating roll calendar from market data."""
        roll_calendar = RollCalendar(
            root_symbol='VX',
            db_connector=self.mock_db,
            roll_strategy='volume'
        )
        
        # Create from market data
        success = roll_calendar.create_from_market_data(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Check that creation was successful
        self.assertTrue(success)
        
        # Check that we have roll pairs
        roll_pairs = roll_calendar.get_roll_pairs(position=1)
        self.assertGreaterEqual(len(roll_pairs), 2)
        
        # Check that roll dates are reasonable (should be mid-month for VX)
        for roll_date, from_contract, to_contract in roll_pairs:
            # Roll date should be in the middle of the month
            self.assertTrue(10 <= roll_date.day <= 20)
            
            # Contracts should follow sequence
            if from_contract == 'VXF23':
                self.assertEqual(to_contract, 'VXG23')
            elif from_contract == 'VXG23':
                self.assertEqual(to_contract, 'VXH23')
    
    def test_panama_contract_generation(self):
        """Test generating continuous contract with Panama method."""
        # Create Panama contract generator
        generator = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            interval_unit='daily',
            interval_value=1,
            db_connector=self.mock_db,
            ratio_limit=0.75  # 75% back, 25% forward
        )
        
        # Generate continuous contract
        result_df = generator.generate(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Check that we have data for the entire date range
        self.assertFalse(result_df.empty)
        date_range = pd.date_range(start='2023-01-01', end='2023-03-20')
        expected_dates = set(pd.DatetimeIndex(date_range).normalize())
        actual_dates = set(pd.DatetimeIndex(result_df['timestamp']).normalize())
        
        # Allow for weekends/holidays
        common_dates = expected_dates.intersection(actual_dates)
        self.assertGreaterEqual(len(common_dates), len(expected_dates) - 30)
        
        # Check that prices were adjusted at roll points
        # Extract data for specific dates to examine adjustments
        
        # Before first roll (when using VXF23)
        pre_roll1 = result_df[result_df['timestamp'] < pd.Timestamp('2023-01-15')]
        # Between rolls (when using VXG23)
        between_rolls = result_df[(result_df['timestamp'] >= pd.Timestamp('2023-01-15')) &
                                (result_df['timestamp'] < pd.Timestamp('2023-02-15'))]
        # After second roll (when using VXH23)
        post_roll2 = result_df[result_df['timestamp'] >= pd.Timestamp('2023-02-15')]
        
        # Check that adjustments create continuous prices across roll points
        if not pre_roll1.empty and not between_rolls.empty:
            last_pre_roll1 = pre_roll1.iloc[-1]
            first_between = between_rolls.iloc[0]
            
            # Prices should be similar across the roll (within 0.5 price unit)
            self.assertLess(abs(last_pre_roll1['close'] - first_between['close']), 0.5)
        
        if not between_rolls.empty and not post_roll2.empty:
            last_between = between_rolls.iloc[-1]
            first_post_roll2 = post_roll2.iloc[0]
            
            # Prices should be similar across the roll (within 0.5 price unit)
            self.assertLess(abs(last_between['close'] - first_post_roll2['close']), 0.5)
        
        # Verify that adjustment factors are properly tracked
        self.assertTrue(hasattr(generator, '_adjustment_factors'))
        self.assertGreaterEqual(len(generator._adjustment_factors), 1)
    
    def test_registry_integration(self):
        """Test using the registry to create and manage generators."""
        # Get the registry
        registry = get_registry()
        
        # Check that Panama method is registered
        generators = registry.list_generators()
        self.assertIn('panama', generators)
        
        # Create a generator through the registry
        generator = registry.create(
            'panama',
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            db_connector=self.mock_db
        )
        
        # Check generator creation
        self.assertIsInstance(generator, PanamaContractGenerator)
        
        # Create from config
        config = {
            'method': 'panama',
            'root_symbol': 'VX',
            'position': 2,
            'roll_strategy': 'volume',
            'ratio_limit': 0.5,
            'db_connector': self.mock_db
        }
        
        generator2 = registry.create_from_config(config)
        
        # Check properties of the created generator
        self.assertIsInstance(generator2, PanamaContractGenerator)
        self.assertEqual(generator2.root_symbol, 'VX')
        self.assertEqual(generator2.position, 2)
        self.assertEqual(generator2.ratio_limit, 0.5)
    
    def test_data_cleaning_integration(self):
        """Test integrating data cleaning with continuous contract generation."""
        # Create the data cleaning pipeline
        pipeline = DataCleaningPipeline(
            name="test_integration_pipeline",
            db_connector=self.mock_db
        )
        
        # Add VX zero prices cleaner
        vx_cleaner = VXZeroPricesCleaner(
            db_connector=self.mock_db,
            config={'interpolation_method': 'linear'}
        )
        
        pipeline.add_cleaner(vx_cleaner)
        
        # Generate continuous contract
        generator = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            db_connector=self.mock_db,
            ratio_limit=0.75
        )
        
        # First generate without cleaning
        result_no_cleaning = generator.generate(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Now apply cleaning
        cleaned_df, summary = pipeline.process_dataframe(self.all_contracts_data)
        
        # Mock fetch_contract_data to return cleaned data
        def mock_fetch_contract_data(contracts, start_date=None, end_date=None):
            result = {}
            for contract in contracts:
                contract_data = cleaned_df[cleaned_df['symbol'] == contract].copy()
                if not contract_data.empty:
                    result[contract] = contract_data
            return result
        
        # Patch the method to use our mocked function
        with patch.object(generator, '_fetch_contract_data', side_effect=mock_fetch_contract_data):
            # Generate with cleaned data
            result_with_cleaning = generator.generate(
                start_date='2023-01-01',
                end_date='2023-03-31'
            )
        
        # Check that both generation methods produced data
        self.assertFalse(result_no_cleaning.empty)
        self.assertFalse(result_with_cleaning.empty)
        
        # Cleaned data should not have any zeros in price fields
        zero_prices = (result_with_cleaning['close'] == 0) | (result_with_cleaning['settle'] == 0)
        self.assertFalse(any(zero_prices))
        
        # Compare results - there should be differences due to cleaning
        self.assertFalse(result_no_cleaning.equals(result_with_cleaning))
    
    def test_full_workflow(self):
        """Test the complete workflow from raw data to continuous contracts."""
        # Step 1: Clean raw data
        pipeline = DataCleaningPipeline(
            name="full_workflow_pipeline",
            db_connector=self.mock_db
        )
        
        vx_cleaner = VXZeroPricesCleaner(
            db_connector=self.mock_db,
            config={'interpolation_method': 'linear'}
        )
        
        pipeline.add_cleaner(vx_cleaner)
        
        cleaned_data, cleaning_summary = pipeline.process_dataframe(self.all_contracts_data)
        
        # Step 2: Generate roll calendar
        roll_calendar = RollCalendar(
            root_symbol='VX',
            db_connector=self.mock_db,
            roll_strategy='volume'
        )
        
        roll_calendar.create_from_market_data(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Step 3: Set up mock contract for Panama generation
        def mock_fetch_contract_data(contracts, start_date=None, end_date=None):
            result = {}
            for contract in contracts:
                contract_data = cleaned_data[cleaned_data['symbol'] == contract].copy()
                if not contract_data.empty:
                    result[contract] = contract_data
            return result
        
        # Step 4: Generate continuous contract with Panama method
        generator = PanamaContractGenerator(
            root_symbol='VX',
            position=1,
            roll_strategy='volume',
            db_connector=self.mock_db,
            ratio_limit=0.75
        )
        
        # Mock methods to use our data
        with patch.object(generator, '_fetch_contract_data', side_effect=mock_fetch_contract_data):
            with patch.object(generator, '_get_calendar_roll_pairs', return_value=roll_calendar.get_roll_pairs(1)):
                # Generate the continuous contract
                continuous_contract = generator.generate(
                    start_date='2023-01-01',
                    end_date='2023-03-31'
                )
        
        # Verify results
        self.assertFalse(continuous_contract.empty)
        
        # Check essential attributes of the continuous contract
        self.assertTrue('symbol' in continuous_contract.columns)
        self.assertTrue('timestamp' in continuous_contract.columns)
        self.assertTrue('interval_unit' in continuous_contract.columns)
        self.assertTrue('interval_value' in continuous_contract.columns)
        
        # Check for price columns
        for field in ['open', 'high', 'low', 'close', 'settle']:
            self.assertTrue(field in continuous_contract.columns)
        
        # Verify price continuity: no large gaps at roll points
        continuous_contract = continuous_contract.sort_values('timestamp')
        price_diff = continuous_contract['close'].diff()
        
        # Calculate percentile for typical price moves
        normal_move = np.percentile(abs(price_diff.dropna()), 75)
        
        # Check for anomalous price jumps (more than 5x the normal daily move)
        large_moves = price_diff.abs() > 5 * normal_move
        
        # Should be very few large moves (ideally only non-roll related)
        large_move_count = sum(large_moves)
        self.assertLessEqual(large_move_count, 3)
        
        # Adjustment factors should be available
        self.assertTrue(hasattr(generator, '_adjustment_factors'))
        self.assertGreater(len(generator._adjustment_factors), 0)

if __name__ == '__main__':
    unittest.main()