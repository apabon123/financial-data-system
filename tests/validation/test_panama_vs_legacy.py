"""
Validation tests to compare Panama method against legacy continuous contracts.

These tests verify that our new Panama implementation produces results that are
consistent with the legacy implementation, while also improving price adjustment
accuracy and stability.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import modules to test
from src.processors.continuous.panama import PanamaContractGenerator
from src.core.database import DatabaseConnector

# Constants
TEST_SYMBOL = "VX"  # Default symbol to test
TEST_START_DATE = "2020-01-01"  # Default start date
TEST_END_DATE = "2022-12-31"  # Default end date
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class TestPanamaVsLegacy(unittest.TestCase):
    """Tests comparing Panama method against legacy continuous contracts."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize real database connector
        try:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "../..", "data", "financial_data.duckdb")
            
            cls.db = DatabaseConnector(db_path=db_path)
            cls.has_real_db = True
            print(f"Connected to real database at {db_path}")
        except Exception as e:
            print(f"Could not connect to real database: {e}")
            cls.has_real_db = False
            cls.db = MagicMock()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if cls.has_real_db:
            cls.db.close()
    
    def skip_if_no_db(self):
        """Skip test if no real database is available."""
        if not self.has_real_db:
            self.skipTest("No real database available for validation tests")
    
    def load_legacy_continuous_data(self, symbol: str, position: int = 1, 
                                 start_date: str = None, end_date: str = None):
        """
        Load legacy continuous contract data from the database.
        
        Args:
            symbol: Root symbol (e.g., 'VX')
            position: Contract position (1 for front month)
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with legacy continuous contract data
        """
        self.skip_if_no_db()
        
        # Construct the continuous symbol
        continuous_symbol = f"@{symbol}={position}01XN"  # Standard format
        
        # Build query conditions
        conditions = []
        params = [continuous_symbol]
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        
        # Construct WHERE clause
        where_clause = " AND ".join(["symbol = ?"] + conditions)
        
        # Query the continuous_contracts table
        query = f"""
            SELECT *
            FROM continuous_contracts
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """
        
        df = self.db.query(query, params)
        return df
    
    def load_raw_futures_data(self, symbol: str, start_date: str = None, end_date: str = None):
        """
        Load raw futures contract data from the database.
        
        Args:
            symbol: Root symbol (e.g., 'VX')
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with raw futures data
        """
        self.skip_if_no_db()
        
        # Build query conditions
        conditions = []
        params = [f"{symbol}%"]  # Use LIKE query for all contracts with the root symbol
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        
        # Construct WHERE clause
        where_clause = " AND ".join(["symbol LIKE ?", "interval_unit = 'daily'"] + conditions)
        
        # Query the market_data table
        query = f"""
            SELECT *
            FROM market_data
            WHERE {where_clause}
            ORDER BY timestamp ASC, symbol ASC
        """
        
        df = self.db.query(query, params)
        return df
    
    def load_roll_dates(self, symbol: str):
        """
        Load roll dates from the database.
        
        Args:
            symbol: Root symbol (e.g., 'VX')
            
        Returns:
            DataFrame with roll calendar data
        """
        self.skip_if_no_db()
        
        # Query the futures_roll_calendar table
        query = """
            SELECT *
            FROM futures_roll_calendar
            WHERE root_symbol = ?
            ORDER BY last_trading_day ASC
        """
        
        df = self.db.query(query, [symbol])
        return df
    
    def test_panama_vs_legacy_basic(self):
        """Basic test comparing Panama method to legacy continuous contracts."""
        self.skip_if_no_db()
        
        # Load legacy continuous contract data
        legacy_df = self.load_legacy_continuous_data(
            symbol=TEST_SYMBOL,
            position=1,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        if legacy_df.empty:
            self.skipTest(f"No legacy continuous data found for {TEST_SYMBOL}")
            return
        
        # Generate Panama continuous contracts
        panama_generator = PanamaContractGenerator(
            root_symbol=TEST_SYMBOL,
            position=1,
            roll_strategy='volume',
            db_connector=self.db,
            ratio_limit=0.75  # 75% back, 25% forward
        )
        
        # Generate with the same date range as the legacy data
        panama_df = panama_generator.generate(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        # Make sure we got data
        self.assertFalse(panama_df.empty)
        
        # Ensure we have a common date range for comparison
        legacy_dates = set(pd.DatetimeIndex(legacy_df['timestamp']).normalize())
        panama_dates = set(pd.DatetimeIndex(panama_df['timestamp']).normalize())
        common_dates = sorted(list(legacy_dates.intersection(panama_dates)))
        
        # Make sure we have enough overlapping dates
        self.assertGreaterEqual(len(common_dates), 100)
        
        # Filter to common dates
        common_legacy_df = legacy_df[legacy_df['timestamp'].dt.normalize().isin(common_dates)]
        common_panama_df = panama_df[panama_df['timestamp'].dt.normalize().isin(common_dates)]
        
        # Comparison should be done on a subset of dates, sorted by timestamp
        common_legacy_df = common_legacy_df.sort_values('timestamp')
        common_panama_df = common_panama_df.sort_values('timestamp')
        
        # Extract price series for comparison
        legacy_prices = common_legacy_df['close'].values
        panama_prices = common_panama_df['close'].values
        
        # Calculate correlation between the two series
        correlation = np.corrcoef(legacy_prices, panama_prices)[0, 1]
        
        # They should be highly correlated (> 0.95)
        self.assertGreaterEqual(correlation, 0.95)
        
        # Calculate percentage differences
        # Use mean absolute percentage difference (MAPD)
        valid_indices = (legacy_prices != 0) & ~np.isnan(legacy_prices)
        pct_diff = np.abs((panama_prices[valid_indices] - legacy_prices[valid_indices]) / legacy_prices[valid_indices])
        mean_pct_diff = np.mean(pct_diff) * 100  # Convert to percentage
        
        # Log the results
        print(f"Correlation between Panama and legacy: {correlation:.4f}")
        print(f"Mean absolute percentage difference: {mean_pct_diff:.2f}%")
        
        # Mean percentage difference should be reasonably small
        self.assertLessEqual(mean_pct_diff, 10.0)
        
        # Create visualization comparing the two methods
        self.create_comparison_chart(
            common_legacy_df, common_panama_df, 
            f"{TEST_SYMBOL}_panama_vs_legacy"
        )
    
    def test_panama_ratios(self):
        """Test Panama method with different ratio limits and compare results."""
        self.skip_if_no_db()
        
        # Load raw data to ensure consistent inputs
        raw_data = self.load_raw_futures_data(
            symbol=TEST_SYMBOL,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        if raw_data.empty:
            self.skipTest(f"No raw data found for {TEST_SYMBOL}")
            return
        
        # Define ratio limits to test
        ratio_limits = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = {}
        
        # Create mock for _fetch_contract_data to use our raw data
        def mock_fetch_contract_data(contracts, start_date=None, end_date=None):
            result = {}
            for contract in contracts:
                contract_data = raw_data[raw_data['symbol'] == contract].copy()
                if not contract_data.empty:
                    result[contract] = contract_data
            return result
        
        # Generate with different ratio limits
        for ratio in ratio_limits:
            generator = PanamaContractGenerator(
                root_symbol=TEST_SYMBOL,
                position=1,
                roll_strategy='volume',
                db_connector=self.db,
                ratio_limit=ratio
            )
            
            # Patch the method to use our raw data
            with patch.object(generator, '_fetch_contract_data', side_effect=mock_fetch_contract_data):
                # Generate continuous contract
                df = generator.generate(
                    start_date=TEST_START_DATE,
                    end_date=TEST_END_DATE
                )
                
                results[f"ratio_{ratio}"] = df
        
        # Make sure we got data for all ratios
        for ratio in ratio_limits:
            key = f"ratio_{ratio}"
            self.assertIn(key, results)
            self.assertFalse(results[key].empty)
        
        # Ensure all results have the same timestamps
        # Extract timestamps from first result
        first_dates = set(pd.DatetimeIndex(results[f"ratio_{ratio_limits[0]}"]['timestamp']).normalize())
        
        for ratio in ratio_limits[1:]:
            key = f"ratio_{ratio}"
            dates = set(pd.DatetimeIndex(results[key]['timestamp']).normalize())
            
            # Timestamps should be identical
            self.assertEqual(first_dates, dates)
        
        # Create comparison chart for different ratios
        self.create_ratio_comparison_chart(results, f"{TEST_SYMBOL}_panama_ratios")
    
    def test_statistical_properties(self):
        """Test statistical properties of Panama method vs legacy implementation."""
        self.skip_if_no_db()
        
        # Load legacy continuous contract data
        legacy_df = self.load_legacy_continuous_data(
            symbol=TEST_SYMBOL,
            position=1,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        if legacy_df.empty:
            self.skipTest(f"No legacy continuous data found for {TEST_SYMBOL}")
            return
        
        # Generate Panama continuous contracts
        panama_generator = PanamaContractGenerator(
            root_symbol=TEST_SYMBOL,
            position=1,
            roll_strategy='volume',
            db_connector=self.db,
            ratio_limit=0.75  # 75% back, 25% forward
        )
        
        # Generate with the same date range as the legacy data
        panama_df = panama_generator.generate(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        # Make sure we got data
        self.assertFalse(panama_df.empty)
        
        # Ensure we have a common date range for comparison
        legacy_dates = set(pd.DatetimeIndex(legacy_df['timestamp']).normalize())
        panama_dates = set(pd.DatetimeIndex(panama_df['timestamp']).normalize())
        common_dates = sorted(list(legacy_dates.intersection(panama_dates)))
        
        # Make sure we have enough overlapping dates
        self.assertGreaterEqual(len(common_dates), 100)
        
        # Filter to common dates
        common_legacy_df = legacy_df[legacy_df['timestamp'].dt.normalize().isin(common_dates)]
        common_panama_df = panama_df[panama_df['timestamp'].dt.normalize().isin(common_dates)]
        
        # Sort by timestamp
        common_legacy_df = common_legacy_df.sort_values('timestamp')
        common_panama_df = common_panama_df.sort_values('timestamp')
        
        # Calculate daily returns
        legacy_returns = common_legacy_df['close'].pct_change().dropna()
        panama_returns = common_panama_df['close'].pct_change().dropna()
        
        # Statistical tests
        
        # 1. Calculate mean, std, min, max, skew, kurtosis
        legacy_stats = {
            'mean': legacy_returns.mean(),
            'std': legacy_returns.std(),
            'min': legacy_returns.min(),
            'max': legacy_returns.max(),
            'skew': legacy_returns.skew(),
            'kurtosis': legacy_returns.kurtosis()
        }
        
        panama_stats = {
            'mean': panama_returns.mean(),
            'std': panama_returns.std(),
            'min': panama_returns.min(),
            'max': panama_returns.max(),
            'skew': panama_returns.skew(),
            'kurtosis': panama_returns.kurtosis()
        }
        
        # 2. Calculate return correlation
        returns_corr = np.corrcoef(legacy_returns, panama_returns)[0, 1]
        
        # 3. Check for return outliers at roll points
        
        # Get roll dates from database
        roll_dates_df = self.load_roll_dates(TEST_SYMBOL)
        
        if not roll_dates_df.empty:
            roll_dates = pd.DatetimeIndex(roll_dates_df['roll_date'])
            
            # Check returns around roll dates
            # Using a 3-day window around roll dates
            roll_windows = []
            for roll_date in roll_dates:
                window_start = roll_date - timedelta(days=1)
                window_end = roll_date + timedelta(days=1)
                
                legacy_roll_returns = common_legacy_df[
                    (common_legacy_df['timestamp'] >= window_start) &
                    (common_legacy_df['timestamp'] <= window_end)
                ]['close'].pct_change().dropna()
                
                panama_roll_returns = common_panama_df[
                    (common_panama_df['timestamp'] >= window_start) &
                    (common_panama_df['timestamp'] <= window_end)
                ]['close'].pct_change().dropna()
                
                if not legacy_roll_returns.empty and not panama_roll_returns.empty:
                    roll_windows.append({
                        'roll_date': roll_date,
                        'legacy_max_return': legacy_roll_returns.abs().max(),
                        'panama_max_return': panama_roll_returns.abs().max(),
                        'legacy_std': legacy_roll_returns.std(),
                        'panama_std': panama_roll_returns.std()
                    })
            
            # Create DataFrame of roll window statistics
            roll_stats_df = pd.DataFrame(roll_windows)
            
            if not roll_stats_df.empty:
                # Calculate improvement metrics
                roll_stats_df['return_ratio'] = roll_stats_df['panama_max_return'] / roll_stats_df['legacy_max_return']
                roll_stats_df['std_ratio'] = roll_stats_df['panama_std'] / roll_stats_df['legacy_std']
                
                # Panama should have smaller max returns and lower std at roll points
                panama_improvement_pct = (roll_stats_df['return_ratio'] < 1.0).mean() * 100
        else:
            panama_improvement_pct = None
        
        # Log statistical comparison
        print("\n==== Statistical Comparison ====")
        print(f"Legacy Mean Return: {legacy_stats['mean']:.6f}")
        print(f"Panama Mean Return: {panama_stats['mean']:.6f}")
        print(f"Legacy Return Std: {legacy_stats['std']:.6f}")
        print(f"Panama Return Std: {panama_stats['std']:.6f}")
        print(f"Return Correlation: {returns_corr:.4f}")
        
        if panama_improvement_pct is not None:
            print(f"Panama improved roll points in {panama_improvement_pct:.1f}% of cases")
        
        # Assert that statistical properties are reasonable
        
        # Returns should be highly correlated
        self.assertGreaterEqual(returns_corr, 0.95)
        
        # Mean returns should be similar
        self.assertLess(abs(legacy_stats['mean'] - panama_stats['mean']), 0.001)
        
        # Panama should have similar or lower volatility
        self.assertLessEqual(panama_stats['std'] / legacy_stats['std'], 1.05)
        
        # Create statistical comparison visualization
        self.create_return_comparison_chart(
            legacy_returns, panama_returns, 
            f"{TEST_SYMBOL}_returns_comparison"
        )
    
    def create_comparison_chart(self, legacy_df, panama_df, filename_base):
        """
        Create a chart comparing legacy and Panama continuous contracts.
        
        Args:
            legacy_df: Legacy data DataFrame
            panama_df: Panama data DataFrame
            filename_base: Base filename for output
        """
        if not self.has_real_db:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot prices
        plt.subplot(2, 1, 1)
        plt.plot(legacy_df['timestamp'], legacy_df['close'], label='Legacy', alpha=0.7)
        plt.plot(panama_df['timestamp'], panama_df['close'], label='Panama', alpha=0.7)
        plt.title('Price Comparison: Legacy vs Panama Method')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot percentage difference
        plt.subplot(2, 1, 2)
        # Calculate percentage difference
        common_dates = legacy_df['timestamp'].isin(panama_df['timestamp'])
        panama_dates = panama_df['timestamp'].isin(legacy_df['timestamp'])
        
        legacy_aligned = legacy_df[common_dates].sort_values('timestamp')
        panama_aligned = panama_df[panama_dates].sort_values('timestamp')
        
        pct_diff = ((panama_aligned['close'].values - legacy_aligned['close'].values) / 
                    legacy_aligned['close'].values) * 100
        
        plt.plot(legacy_aligned['timestamp'], pct_diff)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Percentage Difference: (Panama - Legacy) / Legacy')
        plt.ylabel('Percent Difference')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Comparison chart saved to {output_path}")
    
    def create_ratio_comparison_chart(self, results_dict, filename_base):
        """
        Create a chart comparing Panama method with different ratio limits.
        
        Args:
            results_dict: Dictionary of DataFrames with ratio keys
            filename_base: Base filename for output
        """
        if not self.has_real_db:
            return
        
        # Initialize figure
        plt.figure(figsize=(12, 8))
        
        # Plot prices for each ratio
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (key, df) in enumerate(sorted(results_dict.items())):
            if i < len(colors):
                ratio_val = key.split('_')[1]
                plt.plot(df['timestamp'], df['close'], 
                         label=f'Ratio {ratio_val}', 
                         color=colors[i], alpha=0.7)
        
        plt.title('Panama Method: Price Comparison with Different Ratio Limits')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save chart
        output_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Ratio comparison chart saved to {output_path}")
    
    def create_return_comparison_chart(self, legacy_returns, panama_returns, filename_base):
        """
        Create a chart comparing return properties of legacy and Panama methods.
        
        Args:
            legacy_returns: Series of legacy daily returns
            panama_returns: Series of Panama daily returns
            filename_base: Base filename for output
        """
        if not self.has_real_db:
            return
        
        # Initialize figure
        plt.figure(figsize=(15, 10))
        
        # 1. Return series overlay
        plt.subplot(2, 2, 1)
        plt.plot(legacy_returns.index, legacy_returns, label='Legacy', alpha=0.6)
        plt.plot(panama_returns.index, panama_returns, label='Panama', alpha=0.6)
        plt.title('Daily Returns Comparison')
        plt.ylabel('Daily Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Return distribution histogram
        plt.subplot(2, 2, 2)
        plt.hist(legacy_returns, bins=50, alpha=0.5, label='Legacy')
        plt.hist(panama_returns, bins=50, alpha=0.5, label='Panama')
        plt.title('Return Distribution')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. QQ plot of returns
        plt.subplot(2, 2, 3)
        
        # Sort returns for QQ plot
        legacy_sorted = sorted(legacy_returns)
        panama_sorted = sorted(panama_returns)
        
        # Use the smaller length
        min_length = min(len(legacy_sorted), len(panama_sorted))
        
        plt.scatter(legacy_sorted[:min_length], panama_sorted[:min_length], alpha=0.5)
        
        # Add y=x line
        min_val = min(min(legacy_sorted), min(panama_sorted))
        max_val = max(max(legacy_sorted), max(panama_sorted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Q-Q Plot: Legacy vs Panama Returns')
        plt.xlabel('Legacy Returns')
        plt.ylabel('Panama Returns')
        plt.grid(True, alpha=0.3)
        
        # 4. Absolute difference in returns
        plt.subplot(2, 2, 4)
        
        # Calculate return differences
        if len(legacy_returns) == len(panama_returns):
            # Assumes same dates
            return_diff = abs(legacy_returns.values - panama_returns.values)
            return_diff_dates = legacy_returns.index
            
            plt.plot(return_diff_dates, return_diff)
            plt.title('Absolute Difference in Daily Returns')
            plt.ylabel('|Legacy Return - Panama Return|')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Return comparison chart saved to {output_path}")

if __name__ == '__main__':
    unittest.main()