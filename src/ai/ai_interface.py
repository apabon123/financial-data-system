#!/usr/bin/env python
"""
AI Interface for Financial Data System

This module provides a natural language interface to the financial data system.
It routes user queries to the appropriate agents and scripts based on LLM interpretation.
"""

import os
import sys
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import argparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import pandas as pd

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Typer app and Rich console
app = typer.Typer(help="AI-powered command router for financial data system")
console = Console()

# Define available agents and scripts
TOOLS = {
    # Agents
    "analysis": {
        "path": "src/agents/analysis_agent.py",
        "description": "Performs analysis on financial data including technical, portfolio, correlation, and performance metrics",
        "example": "analyze performance of AAPL over the last 6 months"
    },
    "data_retrieval": {
        "path": "src/agents/data_retrieval_agent.py",
        "description": "Retrieves financial data from various sources",
        "example": "download historical data for TSLA from 2020 to 2022"
    },
    "data_validation": {
        "path": "src/agents/data_validation_agent.py",
        "description": "Validates and cleans financial data",
        "example": "validate the quality of SPY data"
    },
    "derived_indicators": {
        "path": "src/agents/derived_indicators_agent.py",
        "description": "Calculates technical indicators and derived metrics",
        "example": "calculate RSI for SPY"
    },
    "economic_data": {
        "path": "src/agents/economic_data_api_agent.py",
        "description": "Retrieves economic data from various APIs",
        "example": "get the latest GDP data"
    },
    "market_data": {
        "path": "src/agents/tradestation_market_data_agent.py",
        "description": "Retrieves and processes market data from TradeStation",
        "example": "download ES futures data for the last month"
    },
    "account_data": {
        "path": "src/agents/tradestation_account_data_agent.py",
        "description": "Retrieves and processes account data from TradeStation",
        "example": "get my account balance and positions"
    },
    "web_scraping": {
        "path": "src/agents/web_scraping_agent.py",
        "description": "Scrapes financial data from websites",
        "example": "scrape earnings data for AAPL"
    },
    "schema_management": {
        "path": "src/agents/schema_management_agent.py",
        "description": "Manages database schemas and data structures",
        "example": "create a new table for options data"
    },
    "data_normalization": {
        "path": "src/agents/data_normalization_agent.py",
        "description": "Normalizes financial data for consistency",
        "example": "normalize price data for different exchanges"
    },
    "duckdb_write": {
        "path": "src/agents/duckdb_write_agent.py",
        "description": "Writes data to DuckDB database",
        "example": "save the downloaded data to the database"
    },
    
    # Scripts
    "continuous_contract": {
        "path": "src/scripts/generate_continuous_contract.py",
        "description": "Generates continuous contracts for futures",
        "example": "generate continuous contract for ES futures"
    },
    "check_market_data": {
        "path": "src/scripts/check_market_data.py",
        "description": "Checks the quality and completeness of market data",
        "example": "check if ES futures data is complete"
    },
    "fetch_market_data": {
        "path": "src/scripts/fetch_market_data.py",
        "description": "Fetches market data from various sources",
        "example": "fetch SPY data for the last 30 days"
    },
    "cleanup_market_data": {
        "path": "src/scripts/cleanup_market_data.py",
        "description": "Cleans up market data in the database",
        "example": "clean up duplicate entries in the market data"
    },
    "generate_futures_symbols": {
        "path": "src/scripts/generate_futures_symbols.py",
        "description": "Generates futures symbols for various contracts",
        "example": "generate symbols for ES futures contracts"
    },
    "visualize_data": {
        "path": "../src/scripts/visualize_data.py",
        "description": "Visualizes financial data with charts and graphs",
        "example": "plot SPY price for the last 30 days"
    },
    "check_db": {
        "path": "check_db.py",
        "description": "Checks the database for data quality and completeness",
        "example": "check the database for missing data"
    },
    "generate_continuous_contract": {
        "path": "src/scripts/generate_continuous_contract.py",
        "description": "Generates or rebuilds continuous contracts for futures",
        "example": "rebuild continuous contracts for ES"
    }
}

def route_command(query: str) -> Dict[str, Any]:
    """
    Route a natural language command to the appropriate tool.
    
    Args:
        query: Natural language query
        
    Returns:
        Dictionary with routing information
    """
    # In a real implementation, this would use an LLM to determine which tool to use
    # For now, we'll use a simple keyword-based approach
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for rebuild continuous contract requests
    if "rebuild continuous contract" in query_lower:
        # Extract symbol from the query
        words = query.split()
        symbol = None
        rollover_method = "volume"  # Default to volume-based rollover
        
        # Look for the symbol after "for"
        for i, word in enumerate(words):
            if word.lower() == "for" and i + 1 < len(words):
                symbol = words[i + 1]
                break
        
        # If no symbol found, use the last word as a fallback
        if not symbol and words:
            symbol = words[-1]
        
        # Check for rollover method specification
        if "fixed" in query_lower or "expiry" in query_lower:
            rollover_method = "fixed"
        
        # Build the command with proper arguments
        cmd = [
            "python", "src/scripts/generate_continuous_contract.py",
            "--symbol", symbol,
            "--output", f"{symbol}_backadj",
            "--rollover-method", rollover_method,
            "--force"
        ]
        
        return {
            "tool": "generate_continuous_contract",
            "command": " ".join(cmd),
            "params": {}
        }
    
    # Check for visualization requests
    if any(word in query_lower for word in ["plot", "chart", "graph", "visualize", "show"]):
        # Extract symbol from the query
        # This is a simple implementation - in a real system, this would use NLP
        words = query.split()
        symbol = None
        interval = "daily"  # Default interval
        start_date = "2020-01-01"  # Default start date
        end_date = datetime.now().strftime("%Y-%m-%d")  # Default end date (today)
        
        # Look for the symbol after the visualization command
        for i, word in enumerate(words):
            if word.lower() in ["plot", "chart", "graph", "visualize", "show"] and i + 1 < len(words):
                symbol = words[i + 1]
                break
        
        # If no symbol found, use the last word as a fallback
        if not symbol and words:
            symbol = words[-1]
        
        # Look for time period indicators
        for i, word in enumerate(words):
            if word.lower() in ["last", "past", "previous"] and i + 1 < len(words):
                try:
                    period = int(words[i + 1])
                    if i + 2 < len(words) and words[i + 2].lower() in ["days", "months", "years"]:
                        unit = words[i + 2].lower()
                        if unit == "days":
                            start_date = (datetime.now() - pd.Timedelta(days=period)).strftime("%Y-%m-%d")
                        elif unit == "months":
                            start_date = (datetime.now() - pd.Timedelta(days=period*30)).strftime("%Y-%m-%d")
                        elif unit == "years":
                            start_date = (datetime.now() - pd.Timedelta(days=period*365)).strftime("%Y-%m-%d")
                except ValueError:
                    pass
        
        # Set parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return {
            "tool": "visualize_data",
            "command": query,
            "params": params
        }
    
    # Check for continuous contract requests
    if "continuous contract" in query_lower or "backadjusted" in query_lower:
        return {
            "tool": "continuous_contract",
            "command": query,
            "params": {}
        }
    
    # Check for data fetching requests
    if any(word in query_lower for word in ["fetch", "download", "get", "retrieve"]):
        if "market data" in query_lower or "price" in query_lower:
            return {
                "tool": "fetch_market_data",
                "command": query,
                "params": {}
            }
        elif "economic" in query_lower:
            return {
                "tool": "economic_data",
                "command": query,
                "params": {}
            }
        else:
            return {
                "tool": "data_retrieval",
                "command": query,
                "params": {}
            }
    
    # Check for analysis requests
    if any(word in query_lower for word in ["analyze", "analysis", "calculate", "compute"]):
        if any(word in query_lower for word in ["rsi", "macd", "indicator", "technical"]):
            return {
                "tool": "derived_indicators",
                "command": query,
                "params": {}
            }
        else:
            return {
                "tool": "analysis",
                "command": query,
                "params": {}
            }
    
    # Check for database operations
    if any(word in query_lower for word in ["check", "verify", "validate", "database"]):
        if "market data" in query_lower:
            return {
                "tool": "check_market_data",
                "command": query,
                "params": {}
            }
        else:
            return {
                "tool": "check_db",
                "command": query,
                "params": {}
            }
    
    # Default to data retrieval if no specific match
    return {
        "tool": "data_retrieval",
        "command": query,
        "params": {}
    }

def display_tools():
    """Display available tools in a formatted table."""
    table = Table(title="Available Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Example", style="yellow")
    
    for tool_name, tool_info in TOOLS.items():
        table.add_row(
            tool_name,
            tool_info["description"],
            tool_info["example"]
        )
    
    console.print(table)

def main():
    """Main entry point for the AI interface."""
    parser = argparse.ArgumentParser(description='AI Interface for Financial Data System')
    parser.add_argument('query', nargs='?', help='Natural language query')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-tools', '-l', action='store_true', help='List available tools')
    args = parser.parse_args()

    if args.list_tools:
        display_tools()
        return

    if not args.query:
        print("Please provide a query. Use --list-tools to see available commands.")
        return

    # Route the command
    routing = route_command(args.query)
    
    if args.verbose:
        print(f"Routing to: {routing['tool']}")
        print(f"Executing: {routing['command']}")
    
    # Execute the command
    try:
        result = subprocess.run(routing['command'], shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 