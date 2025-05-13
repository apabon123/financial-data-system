#!/usr/bin/env python3
"""
Agent Name: Analysis Agent
Purpose: Perform analysis on financial data
Author: Claude
Date: 2025-04-02

Description:
    This agent performs various analyses on financial data stored in the DuckDB database.
    It supports technical analysis, portfolio analysis, correlation analysis, and performance metrics.
    The agent provides visualization data and statistical metrics to help with financial decision-making.

Usage:
    uv run analysis_agent.py -d ./path/to/database.duckdb -q "analyze performance of AAPL over the last 6 months"
    uv run analysis_agent.py -d ./path/to/database.duckdb -q "calculate correlation between AAPL and MSFT"
    uv run analysis_agent.py -d ./path/to/database.duckdb -f ./queries/portfolio_analysis.json -v
"""

import os
import sys
import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

import typer
import duckdb
import pandas as pd
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)
logger = logging.getLogger("Analysis Agent")

# Setup console
console = Console()

# Agent configuration
AGENT_NAME = "Analysis Agent"
AGENT_VERSION = "0.1.0"
AGENT_DESCRIPTION = "Perform analysis on financial data and generate insights"

# Analysis types
class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    PORTFOLIO = "portfolio"
    CORRELATION = "correlation"
    PERFORMANCE = "performance"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    DISTRIBUTION = "distribution"
    RISK = "risk"
    CUSTOM = "custom"

# Main CLI application
app = typer.Typer(help=AGENT_DESCRIPTION)

class AnalysisAgent:
    """Agent for performing analysis on financial data.
    
    This agent analyzes financial data stored in the DuckDB database,
    performing various analyses such as technical analysis, portfolio
    analysis, correlation analysis, and performance metrics.
    
    Attributes:
        database_path (str): Path to the DuckDB database file
        verbose (bool): Whether to enable verbose logging
        compute_loops (int): Number of reasoning iterations to perform
        conn (duckdb.DuckDBPyConnection): Database connection
    """
    
    def __init__(
        self, 
        database_path: str,
        verbose: bool = False,
        compute_loops: int = 3
    ):
        """Initialize the agent.
        
        Args:
            database_path: Path to DuckDB database
            verbose: Enable verbose output
            compute_loops: Number of reasoning iterations
        """
        self.database_path = database_path
        self.verbose = verbose
        self.compute_loops = compute_loops
        self.conn = None
        
        # Set logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._connect_database()
    
    def _connect_database(self) -> None:
        """Connect to DuckDB database.
        
        Establishes a connection to the DuckDB database and validates
        that the connection is successful. If the database file doesn't
        exist, it will be created.
        
        Raises:
            SystemExit: If the database connection fails
        """
        try:
            self.conn = duckdb.connect(self.database_path)
            logger.debug(f"Connected to database: {self.database_path}")
            
            # Check if the database connection is valid
            test_query = "SELECT 1"
            result = self.conn.execute(test_query).fetchone()
            
            if result and result[0] == 1:
                logger.debug("Database connection validated")
            else:
                logger.error("Database connection validation failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query to extract parameters.
        
        Args:
            query: Natural language query to parse
            
        Returns:
            Dictionary containing extracted parameters
            
        Example:
            Input: "analyze performance of AAPL over the last 6 months"
            Output: {
                "analysis_type": "performance",
                "symbols": ["AAPL"],
                "start_date": datetime(2023, 10, 1),
                "end_date": datetime(2023, 3, 31),
                "timeframe": "daily"
            }
        """
        query = query.lower()
        
        # Default parameters
        params = {
            "analysis_type": None,
            "symbols": [],
            "start_date": None,
            "end_date": None,
            "timeframe": "daily",
            "metrics": [],
            "benchmark": None,
            "window": 20,  # Default window for rolling calculations
            "custom_params": {}
        }
        
        # Determine analysis type
        if "technical" in query or "indicator" in query:
            params["analysis_type"] = AnalysisType.TECHNICAL
        elif "portfolio" in query or "holdings" in query:
            params["analysis_type"] = AnalysisType.PORTFOLIO
        elif "correlation" in query or "correlate" in query:
            params["analysis_type"] = AnalysisType.CORRELATION
        elif "performance" in query or "return" in query:
            params["analysis_type"] = AnalysisType.PERFORMANCE
        elif "volatility" in query or "variance" in query:
            params["analysis_type"] = AnalysisType.VOLATILITY
        elif "drawdown" in query:
            params["analysis_type"] = AnalysisType.DRAWDOWN
        elif "distribution" in query or "histogram" in query:
            params["analysis_type"] = AnalysisType.DISTRIBUTION
        elif "risk" in query or "sharpe" in query or "sortino" in query:
            params["analysis_type"] = AnalysisType.RISK
        else:
            params["analysis_type"] = AnalysisType.CUSTOM
        
        # Extract symbols
        symbols = []
        # Look for ticker symbols (typically 1-5 uppercase letters)
        import re
        ticker_matches = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        if ticker_matches:
            symbols.extend(ticker_matches)
        
        # Check for "symbol/ticker X, Y, Z" pattern
        symbol_patterns = ["symbol", "ticker", "stock", "etf"]
        for pattern in symbol_patterns:
            if pattern in query:
                pattern_idx = query.index(pattern) + len(pattern)
                symbol_part = query[pattern_idx:].strip()
                
                # Check for multiple symbols separated by commas or "and"
                if "," in symbol_part or " and " in symbol_part:
                    symbol_part = symbol_part.replace(" and ", ",")
                    symbol_candidates = [s.strip().upper() for s in symbol_part.split(",")]
                    for symbol in symbol_candidates:
                        if symbol and symbol.isalpha() and len(symbol) <= 5:
                            symbols.append(symbol)
                # Single symbol
                else:
                    symbol = symbol_part.split()[0].strip().upper()
                    if symbol and symbol.isalpha() and len(symbol) <= 5:
                        symbols.append(symbol)
        
        # Remove duplicates and clean up
        params["symbols"] = list(set([s for s in symbols if s.isalpha()]))
        
        # Extract timeframe
        timeframes = {
            "minute": "minute",
            "hour": "hour",
            "daily": "daily",
            "day": "daily",
            "weekly": "weekly",
            "week": "weekly",
            "monthly": "monthly",
            "month": "monthly"
        }
        
        for key, value in timeframes.items():
            if key in query:
                params["timeframe"] = value
                break
        
        # Extract date range
        # Check for "last X days/weeks/months/years" pattern
        time_units = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365
        }
        
        for unit, days in time_units.items():
            pattern = f"last (\\d+) {unit}s?"
            match = re.search(pattern, query)
            if match:
                num_units = int(match.group(1))
                days_back = num_units * days
                params["end_date"] = datetime.now()
                params["start_date"] = params["end_date"] - timedelta(days=days_back)
                break
        
        # Extract date range from explicit dates
        if "from" in query and "to" in query:
            try:
                from_index = query.index("from") + 5
                to_index = query.index("to")
                start_date_str = query[from_index:to_index].strip()
                
                # Assuming date format of YYYY-MM-DD
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                params["start_date"] = start_date
                
                end_date_str = query[to_index + 3:].strip().split()[0]
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                params["end_date"] = end_date
            except (ValueError, IndexError):
                # If date parsing fails, ignore and continue
                pass
        
        # Set default date range if not extracted
        if not params["start_date"]:
            params["end_date"] = datetime.now()
            params["start_date"] = params["end_date"] - timedelta(days=180)  # Default to last 6 months
        
        # Extract benchmark for performance comparison
        if "benchmark" in query or "against" in query or "compare" in query:
            benchmark_symbols = ["SPY", "QQQ", "DIA", "IWM", "^GSPC", "^DJI", "^IXIC"]
            for symbol in benchmark_symbols:
                if symbol in query.upper():
                    params["benchmark"] = symbol
                    break
            
            # If no specific benchmark found but benchmark analysis requested
            if not params["benchmark"] and ("benchmark" in query or "against" in query or "compare" in query):
                params["benchmark"] = "SPY"  # Default to S&P 500 ETF
        
        # Extract window size for rolling calculations
        window_match = re.search(r"window of (\d+)", query)
        if window_match:
            params["window"] = int(window_match.group(1))
        
        # Extract specific metrics if mentioned
        metric_keywords = {
            "return": "return",
            "cagr": "cagr",
            "sharpe": "sharpe_ratio",
            "sortino": "sortino_ratio",
            "drawdown": "max_drawdown",
            "volatility": "volatility",
            "beta": "beta",
            "alpha": "alpha",
            "r squared": "r_squared",
            "correlation": "correlation",
            "var": "value_at_risk",
            "value at risk": "value_at_risk"
        }
        
        for keyword, metric in metric_keywords.items():
            if keyword in query:
                params["metrics"].append(metric)
        
        # If no specific metrics mentioned, but analysis type is specified
        if not params["metrics"] and params["analysis_type"]:
            if params["analysis_type"] == AnalysisType.PERFORMANCE:
                params["metrics"] = ["return", "cagr"]
            elif params["analysis_type"] == AnalysisType.VOLATILITY:
                params["metrics"] = ["volatility", "value_at_risk"]
            elif params["analysis_type"] == AnalysisType.DRAWDOWN:
                params["metrics"] = ["max_drawdown", "drawdown_duration"]
            elif params["analysis_type"] == AnalysisType.RISK:
                params["metrics"] = ["sharpe_ratio", "sortino_ratio", "beta", "alpha"]
        
        logger.debug(f"Parsed parameters: {params}")
        return params
    
    def fetch_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str = "daily") -> pd.DataFrame:
        """Fetch market data for the specified symbols and date range.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (minute, hour, daily, weekly, monthly)
            
        Returns:
            DataFrame containing market data
        """
        logger.debug(f"Fetching {timeframe} market data for {symbols} from {start_date} to {end_date}")
        
        # Map timeframe to appropriate table/view
        timeframe_mapping = {
            "minute": "minute_bars",
            "hour": "market_data WHERE interval_unit = 'hour'",
            "daily": "daily_bars",
            "weekly": "weekly_bars",
            "monthly": "monthly_bars"
        }
        
        table_name = timeframe_mapping.get(timeframe, "daily_bars")
        
        # Format the symbol list for SQL query
        symbols_str = ",".join([f"'{symbol}'" for symbol in symbols])
        
        # Construct and execute the query
        query = f"""
            SELECT 
                timestamp, 
                symbol, 
                open, 
                high, 
                low, 
                close, 
                volume
            FROM {table_name}
            WHERE 
                symbol IN ({symbols_str}) AND
                timestamp BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY 
                symbol, 
                timestamp
        """
        
        try:
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning(f"No data found for {symbols} in the specified date range")
                return pd.DataFrame()
            
            logger.debug(f"Retrieved {len(df)} rows of market data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise
    
    def calculate_returns(self, data: pd.DataFrame, period: str = "daily") -> pd.DataFrame:
        """Calculate returns for the given market data.
        
        Args:
            data: DataFrame containing market data with close prices
            period: Return calculation period (daily, weekly, monthly, annual)
            
        Returns:
            DataFrame with calculated returns
        """
        logger.debug(f"Calculating {period} returns")
        
        if data.empty:
            return pd.DataFrame()
        
        # Make a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Group by symbol to calculate returns separately for each
        symbols = result["symbol"].unique()
        returns_dfs = []
        
        for symbol in symbols:
            symbol_data = result[result["symbol"] == symbol].copy()
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values("timestamp")
            
            # Calculate returns
            if period == "daily":
                symbol_data["return"] = symbol_data["close"].pct_change()
            elif period == "weekly":
                symbol_data["return"] = symbol_data["close"].pct_change(5)  # Assuming 5 trading days per week
            elif period == "monthly":
                symbol_data["return"] = symbol_data["close"].pct_change(21)  # Assuming 21 trading days per month
            elif period == "annual":
                symbol_data["return"] = symbol_data["close"].pct_change(252)  # Assuming 252 trading days per year
            else:
                symbol_data["return"] = symbol_data["close"].pct_change()
            
            # Calculate cumulative returns
            symbol_data["cumulative_return"] = (1 + symbol_data["return"].fillna(0)).cumprod() - 1
            
            # Calculate log returns for statistical analysis
            symbol_data["log_return"] = np.log(symbol_data["close"] / symbol_data["close"].shift(1))
            
            returns_dfs.append(symbol_data)
        
        if returns_dfs:
            result = pd.concat(returns_dfs)
        
        return result
    
    def calculate_performance_metrics(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each symbol in the returns data.
        
        Args:
            returns_data: DataFrame with calculated returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            
        Returns:
            Dictionary of performance metrics by symbol
        """
        logger.debug("Calculating performance metrics")
        
        if returns_data.empty:
            return {}
        
        metrics = {}
        symbols = returns_data["symbol"].unique()
        
        for symbol in symbols:
            symbol_data = returns_data[returns_data["symbol"] == symbol].copy()
            
            # Skip if not enough data
            if len(symbol_data) < 2:
                continue
            
            # Select returns and clean data
            returns = symbol_data["return"].fillna(0)
            log_returns = symbol_data["log_return"].fillna(0)
            
            # Total return
            total_return = symbol_data["cumulative_return"].iloc[-1]
            
            # Calculate CAGR (Compound Annual Growth Rate)
            days = (symbol_data["timestamp"].max() - symbol_data["timestamp"].min()).days
            years = days / 365
            cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
            
            # Volatility (annualized)
            daily_vol = returns.std()
            annualized_vol = daily_vol * (252 ** 0.5)  # Assuming 252 trading days per year
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Sharpe Ratio
            daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
            excess_returns = returns - daily_risk_free
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5) if excess_returns.std() > 0 else 0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * (252 ** 0.5)
            sortino_ratio = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = returns.quantile(0.05)
            
            # Cumulative return at various percentiles
            return_percentiles = {
                "p10": returns.quantile(0.1),
                "p25": returns.quantile(0.25),
                "p50": returns.quantile(0.5),
                "p75": returns.quantile(0.75),
                "p90": returns.quantile(0.9)
            }
            
            # Skewness and kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Positive days ratio
            positive_days = (returns > 0).sum() / len(returns)
            
            # Store metrics
            metrics[symbol] = {
                "total_return": total_return,
                "cagr": cagr,
                "annualized_volatility": annualized_vol,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "value_at_risk_95": var_95,
                "positive_days_ratio": positive_days,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "average_daily_return": returns.mean(),
                "median_daily_return": returns.median(),
                "return_percentiles": return_percentiles,
                "days": days
            }
        
        return metrics
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between symbols.
        
        Args:
            returns_data: DataFrame with calculated returns
            
        Returns:
            Correlation matrix DataFrame
        """
        logger.debug("Calculating correlation matrix")
        
        if returns_data.empty:
            return pd.DataFrame()
        
        # Pivot the data to have symbols as columns and dates as rows
        pivot_data = returns_data.pivot(index="timestamp", columns="symbol", values="return").fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = pivot_data.corr()
        
        return correlation_matrix
    
    def calculate_beta_alpha(self, returns_data: pd.DataFrame, benchmark: str = "SPY") -> Dict[str, Dict[str, float]]:
        """Calculate beta and alpha for each symbol relative to a benchmark.
        
        Args:
            returns_data: DataFrame with calculated returns
            benchmark: Benchmark symbol (default: SPY)
            
        Returns:
            Dictionary containing beta and alpha values by symbol
        """
        logger.debug(f"Calculating beta and alpha against {benchmark}")
        
        if returns_data.empty:
            return {}
        
        # Check if benchmark is in the data
        if benchmark not in returns_data["symbol"].unique():
            logger.warning(f"Benchmark {benchmark} not found in returns data")
            return {}
        
        # Extract benchmark returns
        benchmark_returns = returns_data[returns_data["symbol"] == benchmark]["return"].fillna(0)
        benchmark_dates = returns_data[returns_data["symbol"] == benchmark]["timestamp"]
        benchmark_df = pd.DataFrame({"timestamp": benchmark_dates, "benchmark_return": benchmark_returns}).reset_index(drop=True)
        
        # Calculate beta and alpha for each symbol
        beta_alpha = {}
        symbols = [s for s in returns_data["symbol"].unique() if s != benchmark]
        
        for symbol in symbols:
            symbol_data = returns_data[returns_data["symbol"] == symbol][["timestamp", "return"]].rename(columns={"return": "symbol_return"})
            
            # Merge symbol returns with benchmark returns
            merged_data = pd.merge(symbol_data, benchmark_df, on="timestamp", how="inner")
            
            if len(merged_data) < 2:
                continue
            
            # Calculate beta (covariance / variance)
            covariance = merged_data["symbol_return"].cov(merged_data["benchmark_return"])
            variance = merged_data["benchmark_return"].var()
            beta = covariance / variance if variance > 0 else 0
            
            # Calculate alpha (Jensen's alpha)
            symbol_avg_return = merged_data["symbol_return"].mean() * 252  # Annualized
            benchmark_avg_return = merged_data["benchmark_return"].mean() * 252  # Annualized
            alpha = symbol_avg_return - (0.02 + beta * (benchmark_avg_return - 0.02))  # Assuming 2% risk-free rate
            
            # Calculate R-squared
            correlation = merged_data["symbol_return"].corr(merged_data["benchmark_return"])
            r_squared = correlation ** 2
            
            beta_alpha[symbol] = {
                "beta": beta,
                "alpha": alpha,
                "r_squared": r_squared
            }
        
        return beta_alpha
    
    def calculate_rolling_metrics(self, returns_data: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
        """Calculate rolling performance metrics.
        
        Args:
            returns_data: DataFrame with calculated returns
            window: Window size for rolling calculations
            
        Returns:
            Dictionary of DataFrames with rolling metrics by symbol
        """
        logger.debug(f"Calculating rolling metrics with window={window}")
        
        if returns_data.empty:
            return {}
        
        rolling_metrics = {}
        symbols = returns_data["symbol"].unique()
        
        for symbol in symbols:
            symbol_data = returns_data[returns_data["symbol"] == symbol].copy()
            
            # Skip if not enough data
            if len(symbol_data) <= window:
                continue
            
            # Calculate rolling metrics
            metrics_df = pd.DataFrame(index=symbol_data.index)
            metrics_df["timestamp"] = symbol_data["timestamp"]
            metrics_df["symbol"] = symbol
            
            # Rolling volatility
            metrics_df["rolling_volatility"] = symbol_data["return"].rolling(window=window).std() * (252 ** 0.5)
            
            # Rolling return
            metrics_df["rolling_return"] = symbol_data["return"].rolling(window=window).sum()
            
            # Rolling Sharpe ratio (assuming 0% risk-free rate for simplicity)
            metrics_df["rolling_sharpe"] = (
                symbol_data["return"].rolling(window=window).mean() / 
                symbol_data["return"].rolling(window=window).std()
            ) * (252 ** 0.5)
            
            # Skip records with NaN values from rolling window startup
            metrics_df = metrics_df.dropna()
            
            rolling_metrics[symbol] = metrics_df
        
        return rolling_metrics
    
    def calculate_drawdowns(self, returns_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate drawdown periods for each symbol.
        
        Args:
            returns_data: DataFrame with calculated returns
            
        Returns:
            Dictionary of DataFrames with drawdown data by symbol
        """
        logger.debug("Calculating drawdowns")
        
        if returns_data.empty:
            return {}
        
        drawdowns = {}
        symbols = returns_data["symbol"].unique()
        
        for symbol in symbols:
            symbol_data = returns_data[returns_data["symbol"] == symbol].copy()
            
            # Skip if not enough data
            if len(symbol_data) < 2:
                continue
            
            # Calculate drawdowns
            cumulative_returns = (1 + symbol_data["return"].fillna(0)).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns / peak) - 1
            
            # Create drawdown DataFrame
            drawdown_df = pd.DataFrame({
                "timestamp": symbol_data["timestamp"],
                "symbol": symbol,
                "cumulative_return": cumulative_returns.values,
                "peak": peak.values,
                "drawdown": drawdown.values
            })
            
            # Find drawdown periods
            is_drawdown = drawdown_df["drawdown"] < 0
            drawdown_start = (is_drawdown & ~is_drawdown.shift(1).fillna(False)).astype(int)
            drawdown_end = (~is_drawdown & is_drawdown.shift(1).fillna(False)).astype(int)
            
            drawdown_df["drawdown_start"] = drawdown_start
            drawdown_df["drawdown_end"] = drawdown_end
            
            drawdowns[symbol] = drawdown_df
        
        return drawdowns
    
    def analyze_portfolio(self, returns_data: pd.DataFrame, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze portfolio performance based on symbol returns.
        
        Args:
            returns_data: DataFrame with calculated returns
            weights: Dictionary of symbol weights (if None, equal weights are used)
            
        Returns:
            Dictionary with portfolio analysis results
        """
        logger.debug("Analyzing portfolio performance")
        
        if returns_data.empty:
            return {}
        
        symbols = returns_data["symbol"].unique()
        
        # If no weights provided, use equal weighting
        if not weights:
            weight_value = 1.0 / len(symbols)
            weights = {symbol: weight_value for symbol in symbols}
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Pivot returns data to have symbols as columns
        pivot_data = returns_data.pivot(index="timestamp", columns="symbol", values="return").fillna(0)
        
        # Filter to only include symbols with weights
        pivot_data = pivot_data[[s for s in pivot_data.columns if s in weights]]
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=pivot_data.index)
        for symbol in pivot_data.columns:
            if symbol in weights:
                portfolio_returns += pivot_data[symbol] * weights[symbol]
        
        # Calculate portfolio cumulative returns
        portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Calculate portfolio metrics
        days = (portfolio_returns.index.max() - portfolio_returns.index.min()).days
        years = days / 365
        
        # Total return
        total_return = portfolio_cumulative_returns.iloc[-1]
        
        # CAGR
        cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
        
        # Volatility
        volatility = portfolio_returns.std() * (252 ** 0.5)
        
        # Maximum drawdown
        portfolio_value = (1 + portfolio_returns).cumprod()
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value / peak) - 1
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * (252 ** 0.5) if portfolio_returns.std() > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = portfolio_returns.quantile(0.05)
        
        # Contribution to performance by symbol
        performance_contribution = {}
        for symbol in pivot_data.columns:
            if symbol in weights:
                symbol_contrib = (pivot_data[symbol] * weights[symbol]).sum() / portfolio_returns.sum()
                performance_contribution[symbol] = symbol_contrib
        
        # Correlation matrix
        correlation_matrix = pivot_data.corr()
        
        # Format portfolio returns for output
        portfolio_return_data = pd.DataFrame({
            "timestamp": portfolio_returns.index,
            "return": portfolio_returns.values,
            "cumulative_return": portfolio_cumulative_returns.values
        })
        
        portfolio_analysis = {
            "weights": weights,
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "value_at_risk_95": var_95,
            "performance_contribution": performance_contribution,
            "correlation_matrix": correlation_matrix.to_dict(),
            "portfolio_returns": portfolio_return_data.to_dict(orient="records")
        }
        
        return portfolio_analysis
    
    def analyze_technical(self, market_data: pd.DataFrame, indicators: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Analyze technical indicators for the given market data.
        
        Args:
            market_data: DataFrame containing market data
            indicators: List of technical indicators to calculate
            
        Returns:
            Dictionary of DataFrames with technical indicators by symbol
        """
        logger.debug(f"Analyzing technical indicators: {indicators}")
        
        if market_data.empty:
            return {}
        
        # Default indicators if none provided
        if not indicators:
            indicators = ["sma", "ema", "rsi", "macd", "bollinger"]
        
        technical_data = {}
        symbols = market_data["symbol"].unique()
        
        for symbol in symbols:
            symbol_data = market_data[market_data["symbol"] == symbol].copy()
            
            # Skip if not enough data
            if len(symbol_data) < 30:  # Need enough data for meaningful indicators
                continue
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values("timestamp")
            
            # Create DataFrame for indicators
            indicators_df = pd.DataFrame({
                "timestamp": symbol_data["timestamp"],
                "symbol": symbol,
                "close": symbol_data["close"],
                "volume": symbol_data["volume"]
            })
            
            # Calculate each requested indicator
            for indicator in indicators:
                if indicator == "sma":
                    indicators_df["sma_20"] = symbol_data["close"].rolling(window=20).mean()
                    indicators_df["sma_50"] = symbol_data["close"].rolling(window=50).mean()
                    indicators_df["sma_200"] = symbol_data["close"].rolling(window=200).mean()
                
                elif indicator == "ema":
                    indicators_df["ema_12"] = symbol_data["close"].ewm(span=12, adjust=False).mean()
                    indicators_df["ema_26"] = symbol_data["close"].ewm(span=26, adjust=False).mean()
                
                elif indicator == "rsi":
                    delta = symbol_data["close"].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    rs = avg_gain / avg_loss
                    indicators_df["rsi_14"] = 100 - (100 / (1 + rs))
                
                elif indicator == "macd":
                    ema_12 = symbol_data["close"].ewm(span=12, adjust=False).mean()
                    ema_26 = symbol_data["close"].ewm(span=26, adjust=False).mean()
                    
                    indicators_df["macd"] = ema_12 - ema_26
                    indicators_df["macd_signal"] = indicators_df["macd"].ewm(span=9, adjust=False).mean()
                    indicators_df["macd_histogram"] = indicators_df["macd"] - indicators_df["macd_signal"]
                
                elif indicator == "bollinger":
                    sma_20 = symbol_data["close"].rolling(window=20).mean()
                    std_20 = symbol_data["close"].rolling(window=20).std()
                    
                    indicators_df["bollinger_mid"] = sma_20
                    indicators_df["bollinger_upper"] = sma_20 + (std_20 * 2)
                    indicators_df["bollinger_lower"] = sma_20 - (std_20 * 2)
                
                elif indicator == "volume_sma":
                    indicators_df["volume_sma_20"] = symbol_data["volume"].rolling(window=20).mean()
            
            # Drop rows with NaN values from rolling calculations
            indicators_df = indicators_df.dropna()
            
            technical_data[symbol] = indicators_df
        
        return technical_data
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query.
        
        This is the main entry point for agent functionality. It parses
        the query, performs the requested operation, and returns results.
        
        Args:
            query: Natural language query to process
            
        Returns:
            Dict containing the results and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Parse query to extract parameters
        params = self._parse_query(query)
        
        # Execute query processing based on extracted parameters
        results = self._execute_compute_loops(params)
        
        # Compile and return results
        return {
            "query": query,
            "parameters": params,
            "results": results,
            "success": len(results.get("errors", [])) == 0,
            "errors": results.get("errors", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a query from a JSON file.
        
        Args:
            file_path: Path to JSON file containing query parameters
            
        Returns:
            Dict containing the results and metadata
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            # Execute query processing based on extracted parameters
            results = self._execute_compute_loops(params)
            
            # Compile and return results
            return {
                "file": file_path,
                "parameters": params,
                "results": results,
                "success": len(results.get("errors", [])) == 0,
                "errors": results.get("errors", []),
                "timestamp": datetime.now().isoformat()
            }
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {e}")
            raise
    
    def _execute_compute_loops(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning iterations.
        
        This method implements the multi-step reasoning process for the agent.
        Each loop builds on the results of the previous loop to iteratively
        refine the results.
        
        Args:
            params: Query parameters extracted from natural language query
            
        Returns:
            Processed results as a dictionary
            
        Loop 1: Validate parameters
        Loop 2: Retrieve data
        Loop 3: Perform analysis
        """
        # Default result structure
        result = {
            "analysis_type": params.get("analysis_type", "unknown"),
            "success": False,
            "errors": [],
            "warnings": [],
            "data": None,
            "metrics": {},
            "charts": {},
            "metadata": {
                "compute_loops": self.compute_loops,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "execution_time_ms": 0
            }
        }
        
        start_time = datetime.now()
        
        # Execute compute loops
        for i in range(self.compute_loops):
            loop_start = datetime.now()
            logger.debug(f"Compute loop {i+1}/{self.compute_loops}")
            
            try:
                if i == 0:
                    # Loop 1: Parameter validation
                    self._loop_validate_parameters(params, result)
                elif i == 1:
                    # Loop 2: Data retrieval
                    self._loop_retrieve_data(params, result)
                elif i == 2:
                    # Loop 3: Analysis
                    self._loop_perform_analysis(params, result)
                else:
                    # Additional loops if compute_loops > 3
                    pass
            except Exception as e:
                logger.error(f"Error in compute loop {i+1}: {e}")
                result["errors"].append(f"Error in compute loop {i+1}: {str(e)}")
                result["success"] = False
                break
            
            loop_end = datetime.now()
            loop_duration = (loop_end - loop_start).total_seconds() * 1000
            logger.debug(f"Loop {i+1} completed in {loop_duration:.2f}ms")
        
        # Calculate total execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Update metadata
        result["metadata"]["end_time"] = end_time.isoformat()
        result["metadata"]["execution_time_ms"] = execution_time
        
        # Set success flag if no errors occurred
        if not result["errors"]:
            result["success"] = True
        
        return result
    
    def _loop_validate_parameters(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """First compute loop: Validate input parameters.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
            
        Raises:
            ValueError: If parameters are invalid
        """
        logger.debug("Validating parameters")
        
        # Validate analysis type
        analysis_type = params.get("analysis_type")
        if not analysis_type:
            raise ValueError("Analysis type is required")
        
        # Validate symbols
        symbols = params.get("symbols", [])
        if not symbols:
            # Try to retrieve most actively traded symbols if none provided
            try:
                symbols_query = """
                    SELECT DISTINCT symbol 
                    FROM market_data 
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '90' DAY
                    ORDER BY symbol
                    LIMIT 5
                """
                symbols_df = self.conn.execute(symbols_query).fetchdf()
                if not symbols_df.empty:
                    params["symbols"] = symbols_df["symbol"].tolist()
                    result["warnings"].append(f"No symbols provided, using top symbols: {params['symbols']}")
                else:
                    raise ValueError("No symbols provided and no market data available")
            except Exception as e:
                raise ValueError(f"No symbols provided: {e}")
        
        # Validate date range
        if params.get("start_date") and params.get("end_date"):
            if params["start_date"] > params["end_date"]:
                raise ValueError("Start date cannot be after end date")
        
        # For portfolio analysis, validate weights
        if analysis_type == AnalysisType.PORTFOLIO and "weights" in params:
            weights = params["weights"]
            if sum(weights.values()) == 0:
                raise ValueError("Portfolio weights sum to zero")
        
        # For correlation analysis, ensure at least 2 symbols
        if analysis_type == AnalysisType.CORRELATION and len(params["symbols"]) < 2:
            raise ValueError("At least 2 symbols are required for correlation analysis")
        
        # Ensure benchmark is valid if provided
        if params.get("benchmark") and params["benchmark"] not in params["symbols"]:
            # Add benchmark to symbols list for data retrieval
            params["symbols"].append(params["benchmark"])
        
        logger.debug("Parameters validated successfully")
    
    def _loop_retrieve_data(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Second compute loop: Retrieve necessary data.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Retrieving data")
        
        symbols = params.get("symbols", [])
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        timeframe = params.get("timeframe", "daily")
        
        # Fetch market data
        try:
            market_data = self.fetch_market_data(symbols, start_date, end_date, timeframe)
            
            if market_data.empty:
                raise ValueError(f"No market data found for {symbols} from {start_date} to {end_date}")
            
            result["data"] = {
                "market_data": market_data.to_dict(orient="records"),
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timeframe": timeframe
            }
            
            # Calculate returns for further analysis
            returns_data = self.calculate_returns(market_data, timeframe)
            result["data"]["returns_data"] = returns_data.to_dict(orient="records")
            
        except Exception as e:
            result["errors"].append(f"Error retrieving data: {e}")
            raise
        
        logger.debug("Data retrieval completed")
    
    def _loop_perform_analysis(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Third compute loop: Perform analysis based on parameters.
        
        Args:
            params: Query parameters
            result: Result dictionary to update
        """
        logger.debug("Performing analysis")
        
        analysis_type = params.get("analysis_type")
        metrics = params.get("metrics", [])
        
        # Check if we have the necessary data
        if not result.get("data") or not result["data"].get("market_data"):
            raise ValueError("No market data available for analysis")
        
        # Convert data dictionaries back to DataFrames
        market_data = pd.DataFrame(result["data"]["market_data"])
        returns_data = pd.DataFrame(result["data"]["returns_data"])
        
        # Perform analysis based on type
        if analysis_type == AnalysisType.PERFORMANCE:
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(returns_data)
            result["metrics"]["performance"] = performance_metrics
            
            # Calculate rolling metrics
            window = params.get("window", 20)
            rolling_metrics = self.calculate_rolling_metrics(returns_data, window)
            result["metrics"]["rolling"] = {
                symbol: metrics_df.to_dict(orient="records") 
                for symbol, metrics_df in rolling_metrics.items()
            }
            
            # Add benchmark comparison if requested
            if params.get("benchmark"):
                benchmark = params["benchmark"]
                beta_alpha = self.calculate_beta_alpha(returns_data, benchmark)
                result["metrics"]["benchmark_comparison"] = beta_alpha
        
        elif analysis_type == AnalysisType.PORTFOLIO:
            # Get weights if provided, otherwise use equal weights
            weights = params.get("weights")
            if not weights:
                symbols = params.get("symbols", [])
                weight_value = 1.0 / len(symbols)
                weights = {symbol: weight_value for symbol in symbols}
            
            # Analyze portfolio
            portfolio_analysis = self.analyze_portfolio(returns_data, weights)
            result["metrics"]["portfolio"] = portfolio_analysis
        
        elif analysis_type == AnalysisType.CORRELATION:
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(returns_data)
            result["metrics"]["correlation"] = correlation_matrix.to_dict()
        
        elif analysis_type == AnalysisType.VOLATILITY:
            # Calculate volatility metrics
            performance_metrics = self.calculate_performance_metrics(returns_data)
            volatility_metrics = {
                symbol: {
                    "volatility": metrics["annualized_volatility"],
                    "value_at_risk_95": metrics["value_at_risk_95"],
                    "skewness": metrics["skewness"],
                    "kurtosis": metrics["kurtosis"]
                }
                for symbol, metrics in performance_metrics.items()
            }
            result["metrics"]["volatility"] = volatility_metrics
            
            # Calculate rolling volatility
            window = params.get("window", 20)
            rolling_metrics = self.calculate_rolling_metrics(returns_data, window)
            result["metrics"]["rolling_volatility"] = {
                symbol: [
                    {"timestamp": row["timestamp"], "volatility": row["rolling_volatility"]} 
                    for row in metrics_df.to_dict(orient="records")
                ]
                for symbol, metrics_df in rolling_metrics.items()
            }
        
        elif analysis_type == AnalysisType.DRAWDOWN:
            # Calculate drawdowns
            drawdowns = self.calculate_drawdowns(returns_data)
            result["metrics"]["drawdowns"] = {
                symbol: df.to_dict(orient="records") 
                for symbol, df in drawdowns.items()
            }
        
        elif analysis_type == AnalysisType.DISTRIBUTION:
            # Calculate return distribution statistics
            distribution_metrics = {}
            symbols = params.get("symbols", [])
            
            for symbol in symbols:
                symbol_returns = returns_data[returns_data["symbol"] == symbol]["return"].dropna()
                
                if len(symbol_returns) < 2:
                    continue
                
                # Calculate distribution statistics
                distribution_metrics[symbol] = {
                    "mean": symbol_returns.mean(),
                    "median": symbol_returns.median(),
                    "std": symbol_returns.std(),
                    "skewness": symbol_returns.skew(),
                    "kurtosis": symbol_returns.kurtosis(),
                    "min": symbol_returns.min(),
                    "max": symbol_returns.max(),
                    "percentiles": {
                        "p1": symbol_returns.quantile(0.01),
                        "p5": symbol_returns.quantile(0.05),
                        "p10": symbol_returns.quantile(0.1),
                        "p25": symbol_returns.quantile(0.25),
                        "p50": symbol_returns.quantile(0.5),
                        "p75": symbol_returns.quantile(0.75),
                        "p90": symbol_returns.quantile(0.9),
                        "p95": symbol_returns.quantile(0.95),
                        "p99": symbol_returns.quantile(0.99)
                    }
                }
            
            result["metrics"]["distribution"] = distribution_metrics
        
        elif analysis_type == AnalysisType.RISK:
            # Calculate risk metrics
            performance_metrics = self.calculate_performance_metrics(returns_data)
            risk_metrics = {
                symbol: {
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "sortino_ratio": metrics["sortino_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "volatility": metrics["annualized_volatility"],
                    "value_at_risk_95": metrics["value_at_risk_95"]
                }
                for symbol, metrics in performance_metrics.items()
            }
            result["metrics"]["risk"] = risk_metrics
            
            # Add benchmark comparison if requested
            if params.get("benchmark"):
                benchmark = params["benchmark"]
                beta_alpha = self.calculate_beta_alpha(returns_data, benchmark)
                result["metrics"]["benchmark_comparison"] = beta_alpha
        
        elif analysis_type == AnalysisType.TECHNICAL:
            # Calculate technical indicators
            technical_indicators = self.analyze_technical(market_data)
            result["metrics"]["technical"] = {
                symbol: df.to_dict(orient="records") 
                for symbol, df in technical_indicators.items()
            }
        
        elif analysis_type == AnalysisType.CUSTOM:
            # Perform custom analysis based on requested metrics
            if "performance" in metrics:
                performance_metrics = self.calculate_performance_metrics(returns_data)
                result["metrics"]["performance"] = performance_metrics
            
            if "correlation" in metrics:
                correlation_matrix = self.calculate_correlation_matrix(returns_data)
                result["metrics"]["correlation"] = correlation_matrix.to_dict()
            
            if "beta" in metrics or "alpha" in metrics:
                benchmark = params.get("benchmark", "SPY")
                beta_alpha = self.calculate_beta_alpha(returns_data, benchmark)
                result["metrics"]["benchmark_comparison"] = beta_alpha
            
            if "drawdown" in metrics:
                drawdowns = self.calculate_drawdowns(returns_data)
                result["metrics"]["drawdowns"] = {
                    symbol: df.to_dict(orient="records") 
                    for symbol, df in drawdowns.items()
                }
            
            if "technical" in metrics:
                technical_indicators = self.analyze_technical(market_data)
                result["metrics"]["technical"] = {
                    symbol: df.to_dict(orient="records") 
                    for symbol, df in technical_indicators.items()
                }
        
        logger.debug("Analysis completed successfully")
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display analysis results in a user-friendly format.
        
        Args:
            results: Results dictionary from process_query or process_file
        """
        if not results["success"]:
            console.print(Panel(f"[bold red]Analysis Failed![/]", title=AGENT_NAME))
            for error in results.get("errors", []):
                console.print(f"[red]- {error}[/]")
            return
        
        analysis_type = results.get("parameters", {}).get("analysis_type", "unknown")
        symbols = results.get("parameters", {}).get("symbols", [])
        
        console.print(Panel(f"[bold green]Analysis Completed: {analysis_type.title()}[/]", title=AGENT_NAME))
        
        # Display results based on analysis type
        metrics = results.get("results", {}).get("metrics", {})
        
        if analysis_type == AnalysisType.PERFORMANCE:
            if "performance" in metrics:
                perf_metrics = metrics["performance"]
                
                # Create performance summary table
                table = Table(title=f"Performance Summary")
                table.add_column("Symbol", style="cyan")
                table.add_column("Total Return", justify="right")
                table.add_column("CAGR", justify="right")
                table.add_column("Volatility", justify="right")
                table.add_column("Sharpe", justify="right")
                table.add_column("Max DD", justify="right")
                
                for symbol, metric in perf_metrics.items():
                    total_return = metric.get("total_return", 0)
                    cagr = metric.get("cagr", 0)
                    volatility = metric.get("annualized_volatility", 0)
                    sharpe = metric.get("sharpe_ratio", 0)
                    max_drawdown = metric.get("max_drawdown", 0)
                    
                    return_color = "green" if total_return > 0 else "red"
                    cagr_color = "green" if cagr > 0 else "red"
                    sharpe_color = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
                    
                    table.add_row(
                        symbol,
                        f"[{return_color}]{total_return:.2%}[/{return_color}]",
                        f"[{cagr_color}]{cagr:.2%}[/{cagr_color}]",
                        f"{volatility:.2%}",
                        f"[{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]",
                        f"{max_drawdown:.2%}"
                    )
                
                console.print(table)
                
                # Show benchmark comparison if available
                if "benchmark_comparison" in metrics:
                    benchmark = results.get("parameters", {}).get("benchmark", "SPY")
                    benchmark_table = Table(title=f"Comparison to {benchmark}")
                    benchmark_table.add_column("Symbol", style="cyan")
                    benchmark_table.add_column("Beta", justify="right")
                    benchmark_table.add_column("Alpha", justify="right")
                    benchmark_table.add_column("R²", justify="right")
                    
                    for symbol, metric in metrics["benchmark_comparison"].items():
                        beta = metric.get("beta", 0)
                        alpha = metric.get("alpha", 0)
                        r_squared = metric.get("r_squared", 0)
                        
                        alpha_color = "green" if alpha > 0 else "red"
                        beta_color = "yellow" if 0.8 <= beta <= 1.2 else "cyan"
                        
                        benchmark_table.add_row(
                            symbol,
                            f"[{beta_color}]{beta:.2f}[/{beta_color}]",
                            f"[{alpha_color}]{alpha:.2%}[/{alpha_color}]",
                            f"{r_squared:.2f}"
                        )
                    
                    console.print(benchmark_table)
        
        elif analysis_type == AnalysisType.PORTFOLIO:
            if "portfolio" in metrics:
                portfolio = metrics["portfolio"]
                
                # Create portfolio summary table
                table = Table(title=f"Portfolio Performance Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", justify="right")
                
                metrics_to_display = [
                    ("Total Return", portfolio.get("total_return", 0), "{:.2%}"),
                    ("CAGR", portfolio.get("cagr", 0), "{:.2%}"),
                    ("Volatility", portfolio.get("volatility", 0), "{:.2%}"),
                    ("Sharpe Ratio", portfolio.get("sharpe_ratio", 0), "{:.2f}"),
                    ("Max Drawdown", portfolio.get("max_drawdown", 0), "{:.2%}"),
                    ("Value at Risk (95%)", portfolio.get("value_at_risk_95", 0), "{:.2%}")
                ]
                
                for name, value, fmt in metrics_to_display:
                    color = ""
                    if name in ["Total Return", "CAGR"]:
                        color = "green" if value > 0 else "red"
                    elif name == "Sharpe Ratio":
                        color = "green" if value > 1 else "yellow" if value > 0 else "red"
                    
                    formatted_value = fmt.format(value)
                    if color:
                        formatted_value = f"[{color}]{formatted_value}[/{color}]"
                    
                    table.add_row(name, formatted_value)
                
                console.print(table)
                
                # Show portfolio weights
                weights = portfolio.get("weights", {})
                if weights:
                    weights_table = Table(title="Portfolio Weights")
                    weights_table.add_column("Symbol", style="cyan")
                    weights_table.add_column("Weight", justify="right")
                    
                    for symbol, weight in weights.items():
                        weights_table.add_row(symbol, f"{weight:.2%}")
                    
                    console.print(weights_table)
        
        elif analysis_type == AnalysisType.CORRELATION:
            if "correlation" in metrics:
                correlation = metrics["correlation"]
                
                # Create correlation matrix table
                table = Table(title="Correlation Matrix")
                
                # Add column headers
                table.add_column("Symbol", style="cyan")
                for symbol in symbols:
                    table.add_column(symbol, justify="right")
                
                # Add rows
                for symbol1 in symbols:
                    row_values = [symbol1]
                    for symbol2 in symbols:
                        if symbol1 in correlation and symbol2 in correlation[symbol1]:
                            corr_value = correlation[symbol1][symbol2]
                            
                            # Color-code correlation values
                            if corr_value > 0.8:
                                color = "red"  # Strong positive correlation
                            elif corr_value > 0.5:
                                color = "yellow"  # Moderate positive correlation
                            elif corr_value > 0.2:
                                color = "green"  # Weak positive correlation
                            elif corr_value > -0.2:
                                color = "white"  # No meaningful correlation
                            elif corr_value > -0.5:
                                color = "green"  # Weak negative correlation
                            elif corr_value > -0.8:
                                color = "yellow"  # Moderate negative correlation
                            else:
                                color = "cyan"  # Strong negative correlation
                            
                            row_values.append(f"[{color}]{corr_value:.2f}[/{color}]")
                        else:
                            row_values.append("N/A")
                    
                    table.add_row(*row_values)
                
                console.print(table)
        
        elif analysis_type == AnalysisType.VOLATILITY:
            if "volatility" in metrics:
                volatility = metrics["volatility"]
                
                # Create volatility table
                table = Table(title="Volatility Metrics")
                table.add_column("Symbol", style="cyan")
                table.add_column("Annualized Volatility", justify="right")
                table.add_column("Value at Risk (95%)", justify="right")
                table.add_column("Skewness", justify="right")
                table.add_column("Kurtosis", justify="right")
                
                for symbol, metric in volatility.items():
                    vol = metric.get("volatility", 0)
                    var = metric.get("value_at_risk_95", 0)
                    skew = metric.get("skewness", 0)
                    kurt = metric.get("kurtosis", 0)
                    
                    # Color-code volatility
                    vol_color = "green" if vol < 0.2 else "yellow" if vol < 0.3 else "red"
                    var_color = "green" if var > -0.02 else "yellow" if var > -0.03 else "red"
                    
                    table.add_row(
                        symbol,
                        f"[{vol_color}]{vol:.2%}[/{vol_color}]",
                        f"[{var_color}]{var:.2%}[/{var_color}]",
                        f"{skew:.2f}",
                        f"{kurt:.2f}"
                    )
                
                console.print(table)
        
        elif analysis_type == AnalysisType.RISK:
            if "risk" in metrics:
                risk = metrics["risk"]
                
                # Create risk metrics table
                table = Table(title="Risk Metrics")
                table.add_column("Symbol", style="cyan")
                table.add_column("Sharpe Ratio", justify="right")
                table.add_column("Sortino Ratio", justify="right")
                table.add_column("Max Drawdown", justify="right")
                table.add_column("Volatility", justify="right")
                
                for symbol, metric in risk.items():
                    sharpe = metric.get("sharpe_ratio", 0)
                    sortino = metric.get("sortino_ratio", 0)
                    max_dd = metric.get("max_drawdown", 0)
                    vol = metric.get("volatility", 0)
                    
                    # Color-code metrics
                    sharpe_color = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
                    sortino_color = "green" if sortino > 1 else "yellow" if sortino > 0 else "red"
                    
                    table.add_row(
                        symbol,
                        f"[{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]",
                        f"[{sortino_color}]{sortino:.2f}[/{sortino_color}]",
                        f"{max_dd:.2%}",
                        f"{vol:.2%}"
                    )
                
                console.print(table)
                
                # Show benchmark comparison if available
                if "benchmark_comparison" in metrics:
                    benchmark = results.get("parameters", {}).get("benchmark", "SPY")
                    benchmark_table = Table(title=f"Risk Relative to {benchmark}")
                    benchmark_table.add_column("Symbol", style="cyan")
                    benchmark_table.add_column("Beta", justify="right")
                    benchmark_table.add_column("Alpha", justify="right")
                    benchmark_table.add_column("R²", justify="right")
                    
                    for symbol, metric in metrics["benchmark_comparison"].items():
                        beta = metric.get("beta", 0)
                        alpha = metric.get("alpha", 0)
                        r_squared = metric.get("r_squared", 0)
                        
                        alpha_color = "green" if alpha > 0 else "red"
                        
                        benchmark_table.add_row(
                            symbol,
                            f"{beta:.2f}",
                            f"[{alpha_color}]{alpha:.2%}[/{alpha_color}]",
                            f"{r_squared:.2f}"
                        )
                    
                    console.print(benchmark_table)
        
        elif analysis_type == AnalysisType.TECHNICAL:
            # Show summary of technical indicators
            console.print(f"[cyan]Technical analysis completed for {', '.join(symbols)}[/]")
            console.print("[yellow]Technical indicators calculated:[/]")
            
            if "technical" in metrics and metrics["technical"]:
                first_symbol = list(metrics["technical"].keys())[0]
                if metrics["technical"][first_symbol]:
                    first_record = metrics["technical"][first_symbol][0]
                    indicators = [k for k in first_record.keys() if k not in ["timestamp", "symbol", "close", "volume"]]
                    
                    for indicator in indicators:
                        console.print(f"- {indicator}")
        
        elif analysis_type == AnalysisType.DRAWDOWN:
            if "drawdowns" in metrics:
                drawdowns = metrics["drawdowns"]
                
                for symbol, dd_data in drawdowns.items():
                    # Find maximum drawdown
                    max_dd = min([record.get("drawdown", 0) for record in dd_data], default=0)
                    max_dd_date = None
                    
                    for record in dd_data:
                        if record.get("drawdown") == max_dd:
                            max_dd_date = record.get("timestamp")
                            break
                    
                    console.print(f"[cyan]{symbol}[/] - Maximum Drawdown: [red]{max_dd:.2%}[/]" + 
                                  (f" on {max_dd_date}" if max_dd_date else ""))
        
        elif analysis_type == AnalysisType.DISTRIBUTION:
            if "distribution" in metrics:
                distribution = metrics["distribution"]
                
                # Create distribution metrics table
                table = Table(title="Return Distribution")
                table.add_column("Symbol", style="cyan")
                table.add_column("Mean", justify="right")
                table.add_column("Median", justify="right")
                table.add_column("Std Dev", justify="right")
                table.add_column("Skewness", justify="right")
                table.add_column("Min", justify="right")
                table.add_column("Max", justify="right")
                
                for symbol, metric in distribution.items():
                    mean = metric.get("mean", 0)
                    median = metric.get("median", 0)
                    std = metric.get("std", 0)
                    skew = metric.get("skewness", 0)
                    min_val = metric.get("min", 0)
                    max_val = metric.get("max", 0)
                    
                    mean_color = "green" if mean > 0 else "red"
                    
                    table.add_row(
                        symbol,
                        f"[{mean_color}]{mean:.2%}[/{mean_color}]",
                        f"{median:.2%}",
                        f"{std:.2%}",
                        f"{skew:.2f}",
                        f"{min_val:.2%}",
                        f"{max_val:.2%}"
                    )
                
                console.print(table)
                
                # Show percentiles table
                percentiles_table = Table(title="Return Percentiles")
                percentiles_table.add_column("Symbol", style="cyan")
                percentiles_table.add_column("1%", justify="right")
                percentiles_table.add_column("5%", justify="right")
                percentiles_table.add_column("25%", justify="right")
                percentiles_table.add_column("50%", justify="right")
                percentiles_table.add_column("75%", justify="right")
                percentiles_table.add_column("95%", justify="right")
                percentiles_table.add_column("99%", justify="right")
                
                for symbol, metric in distribution.items():
                    if "percentiles" in metric:
                        p = metric["percentiles"]
                        
                        percentiles_table.add_row(
                            symbol,
                            f"{p.get('p1', 0):.2%}",
                            f"{p.get('p5', 0):.2%}",
                            f"{p.get('p25', 0):.2%}",
                            f"{p.get('p50', 0):.2%}",
                            f"{p.get('p75', 0):.2%}",
                            f"{p.get('p95', 0):.2%}",
                            f"{p.get('p99', 0):.2%}"
                        )
                
                console.print(percentiles_table)
        
        # Show any warnings
        for warning in results.get("results", {}).get("warnings", []):
            console.print(f"[yellow]Warning: {warning}[/]")
    
    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

@app.command()
def query(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    query_str: str = typer.Option(None, "--query", "-q", help="Natural language query"),
    file: str = typer.Option(None, "--file", "-f", help="Path to JSON query file"),
    compute_loops: int = typer.Option(3, "--compute_loops", "-c", help="Number of reasoning iterations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save results (JSON format)"),
):
    """
    Perform analysis on financial data.
    
    Examples:
        uv run analysis_agent.py -d ./financial_data.duckdb -q "analyze performance of AAPL over the last 6 months"
        uv run analysis_agent.py -d ./financial_data.duckdb -q "calculate correlation between AAPL and MSFT"
    """
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")
    
    if not query_str and not file:
        console.print("[bold red]Error:[/] Either --query or --file must be specified")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = AnalysisAgent(
            database_path=database,
            verbose=verbose,
            compute_loops=compute_loops
        )
        
        # Process query or file
        if query_str:
            console.print(f"Processing query: [italic]{query_str}[/]")
            result = agent.process_query(query_str)
        else:
            console.print(f"Processing file: [italic]{file}[/]")
            result = agent.process_file(file)
        
        # Display results
        agent.display_results(result)
        
        # Save results to file if specified
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"Results saved to [bold]{output}[/]")
        
        # Clean up
        agent.close()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

@app.command()
def version():
    """Display version information."""
    console.print(f"[bold]{AGENT_NAME}[/] v{AGENT_VERSION}")

@app.command()
def test_connection(
    database: str = typer.Option(..., "--database", "-d", help="Path to DuckDB database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Test database connection."""
    try:
        agent = AnalysisAgent(database_path=database, verbose=verbose)
        console.print("[bold green]Connection successful![/]")
        
        # Test market data availability
        try:
            test_query = """
                SELECT COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM market_data
            """
            result = agent.conn.execute(test_query).fetchone()
            
            if result and result[0] > 0:
                console.print(f"[green]Market data available: {result[0]} records from {result[1]} to {result[2]}[/]")
            else:
                console.print("[yellow]No market data found in database[/]")
        except Exception as e:
            console.print(f"[yellow]Error checking market data: {e}[/]")
        
        agent.close()
    except Exception as e:
        console.print(f"[bold red]Connection test failed:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()