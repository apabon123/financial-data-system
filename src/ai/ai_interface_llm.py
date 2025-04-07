#!/usr/bin/env python
"""
Advanced AI Interface for Financial Data System

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

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    "generate_continuous_contract": {
        "path": "src/scripts/generate_continuous_contract.py",
        "description": "Generates continuous contracts for futures",
        "example": "rebuild continuous contracts for ES",
        "params": {
            "symbol": "Base symbol (e.g., ES, NQ)",
            "output": "Output symbol (e.g., ES_backadj, NQ_backadj)",
            "rollover_method": "Method to determine rollover dates (volume or fixed)",
            "force": "Force rebuild of existing continuous contract"
        }
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
        "path": "src/visualize_data.py",
        "description": "Visualizes financial data with charts and graphs",
        "example": "plot SPY price for the last 30 days"
    },
    "check_db": {
        "path": "check_db.py",
        "description": "Checks the database for data quality and completeness",
        "example": "check the database for missing data"
    }
}

def get_llm_client():
    """
    Get the LLM client based on environment variables.
    
    Returns:
        LLM client object
    """
    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            # Use the new client initialization format
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return "openai", client
        except ImportError:
            logger.warning("OpenAI package not installed. Falling back to keyword-based routing.")
    
    # Check for Anthropic API key
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return "anthropic", client
        except ImportError:
            logger.warning("Anthropic package not installed. Falling back to keyword-based routing.")
    
    # No LLM available
    return None, None

def route_command_with_llm(query: str, llm_type: str, llm_client: Any) -> Dict[str, Any]:
    """
    Route a natural language command to the appropriate tool using an LLM.
    
    Args:
        query: Natural language query
        llm_type: Type of LLM client ("openai" or "anthropic")
        llm_client: LLM client object
        
    Returns:
        Dictionary with routing information
    """
    # Create a prompt for the LLM
    tools_json = json.dumps(TOOLS, indent=2)
    
    if llm_type == "openai":
        prompt = f"""
        Based on the following query, determine which tool should handle it.
        Available tools:
        {tools_json}
        
        Query: {query}
        
        For the continuous contract generator, you must return these exact parameters:
        - symbol: The base symbol (e.g., ES, NQ)
        - output: The output symbol (e.g., ES_backadj, NQ_backadj)
        - rollover_method: The rollover method (volume or fixed)
        - force: Whether to force rebuild (true or false)
        
        Return a JSON object with:
        1. tool: The name of the tool to use
        2. command: The command to pass to the tool
        3. params: Any specific parameters to include
        
        Example response for continuous contract:
        {{
            "tool": "generate_continuous_contract",
            "command": "rebuild continuous contracts for ES",
            "params": {{
                "symbol": "ES",
                "output": "ES_backadj",
                "rollover_method": "volume",
                "force": true
            }}
        }}
        
        Example response for visualization:
        {{
            "tool": "visualize_data",
            "command": "plot SPY for the last 30 days",
            "params": {{
                "symbol": "SPY",
                "days": 30
            }}
        }}
        """
        
        try:
            logger.info("Sending request to OpenAI API")
            # Use the new API format for OpenAI v1.0+
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using gpt-3.5-turbo as it's more widely available
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            logger.info("Received response from OpenAI API")
            content = response.choices[0].message.content
            logger.info(f"Response content: {content}")
            
            routing = json.loads(content)
            logger.info(f"Parsed routing: {routing}")
            return routing
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {"error": str(e)}
    
    elif llm_type == "anthropic":
        prompt = f"""
        Based on the following query, determine which tool should handle it.
        Available tools:
        {tools_json}
        
        Query: {query}
        
        Return a JSON object with:
        1. tool: The name of the tool to use
        2. command: The command to pass to the tool
        3. params: Any specific parameters to include
        
        Example response:
        {{
            "tool": "visualize_data",
            "command": "plot SPY for the last 30 days",
            "params": {{
                "symbol": "SPY",
                "days": 30
            }}
        }}
        """
        
        try:
            logger.info("Sending request to Anthropic API")
            response = llm_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are a helpful assistant that routes commands to the appropriate tools.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            logger.info("Received response from Anthropic API")
            content = response.content[0].text
            logger.info(f"Response content: {content}")
            
            # Extract JSON from the response
            json_str = content[content.find("{"):content.rfind("}")+1]
            routing = json.loads(json_str)
            logger.info(f"Parsed routing: {routing}")
            return routing
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return {"error": str(e)}
    
    else:
        logger.error(f"Unsupported LLM type: {llm_type}")
        return {"error": f"Unsupported LLM type: {llm_type}"}

def route_command_keyword(query: str) -> Dict[str, Any]:
    """
    Route a natural language command to the appropriate tool using keyword matching.
    
    Args:
        query: Natural language query
        
    Returns:
        Dictionary with routing information
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for visualization requests
    if any(word in query_lower for word in ["plot", "chart", "graph", "visualize", "show"]):
        return {
            "tool": "visualize_data",
            "command": query,
            "params": {}
        }
    
    # Check for continuous contract requests
    if "continuous contract" in query_lower or "backadjusted" in query_lower:
        return {
            "tool": "generate_continuous_contract",
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

def route_command(query: str) -> Dict[str, Any]:
    """
    Route a natural language command to the appropriate tool.
    
    Args:
        query: Natural language query
        
    Returns:
        Dictionary with routing information
    """
    # Get the LLM client
    llm_type, llm_client = get_llm_client()
    
    # If LLM is available, use it for routing
    if llm_type and llm_client:
        logger.info(f"Using {llm_type} for command routing")
        return route_command_with_llm(query, llm_type, llm_client)
    
    # Otherwise, use keyword-based routing
    logger.info("Using keyword-based routing")
    return route_command_keyword(query)

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

@app.command()
def run(
    query: str = typer.Argument(None, help="Natural language query"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    list_tools: bool = typer.Option(False, "--list", "-l", help="List available tools"),
    use_keyword: bool = typer.Option(False, "--keyword", "-k", help="Use keyword-based routing instead of LLM")
):
    """
    Run a natural language query through the appropriate tool.
    """
    if list_tools:
        display_tools()
        return
    
    if not query:
        console.print("[red]Error: Missing query argument. Please provide a natural language query.[/red]")
        console.print("Example: ai-llm.bat \"plot SPY for the last 30 days\"")
        console.print("For a list of available tools: ai-llm.bat --list")
        return
    
    console.print(Panel(f"Processing: {query}", title="AI Interface", border_style="blue"))
    
    # Determine which tool to use
    if use_keyword:
        logger.info("Using keyword-based routing")
        routing = route_command_keyword(query)
    else:
        routing = route_command(query)
    
    if "error" in routing:
        console.print(f"[red]Error: {routing['error']}[/red]")
        return
    
    tool_name = routing.get("tool")
    if not tool_name:
        console.print("[red]Error: No tool specified in routing[/red]")
        return
        
    if tool_name not in TOOLS:
        console.print(f"[red]Error: Unknown tool '{tool_name}'[/red]")
        return
    
    tool_info = TOOLS[tool_name]
    tool_path = tool_info["path"]
    
    # Get the project root directory (parent of src directory)
    project_root = str(Path(__file__).parent.parent.parent)
    
    # Build command for the tool using Windows path format
    cmd = [".\\venv\\Scripts\\python.exe", tool_path]
    
    # Add any additional parameters
    params = routing.get("params", {})
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.extend([f"--{key}"])
        elif not isinstance(value, bool):
            # Special handling for rollover-method parameter
            if key == "rollover_method":
                cmd.extend([f"--rollover-method", str(value)])
            else:
                cmd.extend([f"--{key}", str(value)])
    
    if verbose:
        console.print(f"[yellow]Running command: {' '.join(cmd)}[/yellow]")
    
    # Execute the command from the project root directory with real-time output
    try:
        logger.info(f"Executing command from {project_root}: {' '.join(cmd)}")
        
        # Use Popen to get real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd=project_root
        )
        
        # Function to handle output streams
        def handle_output(pipe, is_error=False):
            for line in pipe:
                line = line.strip()
                if line:
                    if is_error:
                        if verbose:
                            console.print(f"[red]{line}[/red]")
                        else:
                            logger.error(line)
                    else:
                        console.print(line)
        
        # Create threads to handle stdout and stderr
        import threading
        stdout_thread = threading.Thread(target=handle_output, args=(process.stdout,))
        stderr_thread = threading.Thread(target=handle_output, args=(process.stderr, True))
        
        # Start the threads
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for the process to complete
        return_code = process.wait()
        
        # Wait for the threads to complete
        stdout_thread.join()
        stderr_thread.join()
        
        # Close the pipes
        process.stdout.close()
        process.stderr.close()
        
        if return_code != 0:
            console.print(f"[red]Command failed with return code {return_code}[/red]")
            
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        console.print(f"[red]Error executing tool: {e}[/red]")

if __name__ == "__main__":
    app() 