#!/usr/bin/env python3
"""
API Connection Test Script

This script tests the connection to various APIs used in the financial data system:
- TradeStation API
- FRED API
- OpenAI API
- Anthropic API

Usage:
    python test_api_connections.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import requests
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
import openai
from anthropic import Anthropic

# Get the project root directory and add it to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Load environment variables from the .env file
env_file = os.path.join(project_root, ".env")
load_dotenv(env_file)

# Print environment info for debugging
print(f"Python executable: {sys.executable}")
print(f"Project root: {project_root}")
print(f"Env file: {env_file}")
print(f"Current working directory: {os.getcwd()}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("API Test")

# Setup console
console = Console()

def test_tradestation_api() -> Dict[str, Any]:
    """Test TradeStation API connection."""
    result = {"success": False, "message": "", "details": {}}
    
    try:
        # Get API credentials
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        refresh_token = os.getenv("REFRESH_TOKEN")
        
        if not all([client_id, client_secret, refresh_token]):
            result["message"] = "Missing TradeStation API credentials"
            return result
        
        # Test authentication
        url = "https://signin.tradestation.com/oauth/token"
        payload = f"grant_type=refresh_token&client_id={client_id}&client_secret={client_secret}&refresh_token={refresh_token}"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            result["message"] = "TradeStation API authentication successful"
            result["details"] = {
                "token_expires_in": data.get("expires_in"),
                "token_type": data.get("token_type")
            }
        else:
            result["message"] = f"TradeStation API authentication failed: {response.text}"
            
    except Exception as e:
        result["message"] = f"TradeStation API test failed: {str(e)}"
    
    return result

def test_fred_api() -> Dict[str, Any]:
    """Test FRED API connection."""
    result = {"success": False, "message": "", "details": {}}
    
    try:
        # Get API key
        api_key = os.getenv("FRED_API_KEY")
        
        if not api_key:
            result["message"] = "Missing FRED API key"
            return result
        
        # Test API by fetching GDP data
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "GDP",
            "api_key": api_key,
            "file_type": "json",
            "limit": 1
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            result["message"] = "FRED API connection successful"
            result["details"] = {
                "series_id": "GDP",
                "observations_count": len(data.get("observations", []))
            }
        else:
            result["message"] = f"FRED API test failed: {response.text}"
            
    except Exception as e:
        result["message"] = f"FRED API test failed: {str(e)}"
    
    return result

def test_openai_api() -> Dict[str, Any]:
    """Test OpenAI API connection."""
    result = {"success": False, "message": "", "details": {}}
    
    try:
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            result["message"] = "Missing OpenAI API key"
            return result
        
        # Check API key format
        if not api_key.startswith("sk-"):
            result["message"] = "Invalid OpenAI API key format. Key should start with 'sk-'"
            result["details"] = {"key_format": "invalid", "key_prefix": api_key[:10] + "..."}
            return result
        
        # Test API with a simple completion using direct API call
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            result["message"] = "OpenAI API connection successful"
            result["details"] = {
                "model": data["model"],
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"]
                }
            }
        else:
            result["message"] = f"OpenAI API test failed: {response.text}"
            
    except Exception as e:
        result["message"] = f"OpenAI API test failed: {str(e)}"
    
    return result

def test_anthropic_api() -> Dict[str, Any]:
    """Test Anthropic API connection."""
    result = {"success": False, "message": "", "details": {}}
    
    try:
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            result["message"] = "Missing Anthropic API key"
            return result
        
        # Test API with a simple completion using direct API call
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            result["message"] = "Anthropic API connection successful"
            result["details"] = {
                "model": data["model"],
                "usage": {
                    "input_tokens": data["usage"]["input_tokens"],
                    "output_tokens": data["usage"]["output_tokens"]
                }
            }
        else:
            result["message"] = f"Anthropic API test failed: {response.text}"
            
    except Exception as e:
        result["message"] = f"Anthropic API test failed: {str(e)}"
    
    return result

def main():
    """Main function to test all API connections."""
    console.print(Panel("Testing API Connections", style="bold blue"))
    
    # Test TradeStation API
    console.print("\n[bold]Testing TradeStation API...[/]")
    ts_result = test_tradestation_api()
    if ts_result["success"]:
        console.print("[green]✓ TradeStation API: Connected successfully[/]")
        console.print(f"  Token expires in: {ts_result['details']['token_expires_in']} seconds")
    else:
        console.print(f"[red]✗ TradeStation API: {ts_result['message']}[/]")
    
    # Test FRED API
    console.print("\n[bold]Testing FRED API...[/]")
    fred_result = test_fred_api()
    if fred_result["success"]:
        console.print("[green]✓ FRED API: Connected successfully[/]")
        console.print(f"  Tested with series: {fred_result['details']['series_id']}")
    else:
        console.print(f"[red]✗ FRED API: {fred_result['message']}[/]")
    
    # Test OpenAI API
    console.print("\n[bold]Testing OpenAI API...[/]")
    openai_result = test_openai_api()
    if openai_result["success"]:
        console.print("[green]✓ OpenAI API: Connected successfully[/]")
        console.print(f"  Tested with model: {openai_result['details']['model']}")
    else:
        console.print(f"[red]✗ OpenAI API: {openai_result['message']}[/]")
    
    # Test Anthropic API
    console.print("\n[bold]Testing Anthropic API...[/]")
    anthropic_result = test_anthropic_api()
    if anthropic_result["success"]:
        console.print("[green]✓ Anthropic API: Connected successfully[/]")
        console.print(f"  Tested with model: {anthropic_result['details']['model']}")
    else:
        console.print(f"[red]✗ Anthropic API: {anthropic_result['message']}[/]")
    
    # Summary
    console.print("\n[bold]Summary:[/]")
    all_success = all([
        ts_result["success"],
        fred_result["success"],
        openai_result["success"],
        anthropic_result["success"]
    ])
    
    if all_success:
        console.print("[green]All API connections successful![/]")
        return 0
    else:
        console.print("[red]Some API connections failed. Please check the errors above.[/]")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 