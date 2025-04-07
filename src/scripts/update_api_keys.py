#!/usr/bin/env python
"""
Update API Keys

This script helps update API keys in the .env file.
"""

import os
import sys
from pathlib import Path
import re
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Initialize Rich console
console = Console()

def get_env_file_path():
    """Get the path to the .env file."""
    # Try to find the .env file in the current directory or parent directories
    current_dir = Path.cwd()
    env_file = current_dir / ".env"
    
    if not env_file.exists():
        # Try parent directory
        parent_dir = current_dir.parent
        env_file = parent_dir / ".env"
        
        if not env_file.exists():
            # Create a new .env file in the current directory
            env_file = current_dir / ".env"
            console.print(f"[yellow]No .env file found. Creating a new one at {env_file}[/yellow]")
    
    return env_file

def read_env_file(env_file):
    """Read the .env file and return a dictionary of key-value pairs."""
    if not env_file.exists():
        return {}
    
    env_vars = {}
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split on the first equals sign
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key, value = parts
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def write_env_file(env_file, env_vars):
    """Write the environment variables to the .env file."""
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    console.print(f"[green]Updated .env file at {env_file}[/green]")

def update_api_key(env_vars, key_name, current_value=None):
    """Update an API key in the environment variables."""
    if current_value:
        # Mask the current value
        masked_value = mask_value(current_value)
        console.print(f"Current {key_name}: {masked_value}")
    
    # Ask if the user wants to update the key
    if current_value and not Confirm.ask(f"Update {key_name}?"):
        return env_vars
    
    # Get the new API key
    new_value = Prompt.ask(f"Enter new {key_name}", password=True)
    
    if new_value:
        env_vars[key_name] = new_value
        console.print(f"[green]Updated {key_name}[/green]")
    else:
        console.print(f"[yellow]No new value provided for {key_name}[/yellow]")
    
    return env_vars

def mask_value(value, visible_chars=4):
    """Mask a value, showing only the first and last few characters."""
    if not value:
        return "None"
    
    if len(value) <= visible_chars * 2:
        return "*" * len(value)
    
    return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]

def main():
    """Main function."""
    console.print(Panel("Update API Keys", title="API Key Manager", border_style="blue"))
    
    # Get the .env file path
    env_file = get_env_file_path()
    
    # Read the current environment variables
    env_vars = read_env_file(env_file)
    
    # Check for OpenAI API key
    openai_key = env_vars.get("OPENAI_API_KEY")
    if openai_key and openai_key.startswith("sk-proj-"):
        console.print("[yellow]Warning: You are using a project API key (sk-proj-).[/yellow]")
        console.print("This may not work with the OpenAI Python library.")
        console.print("Consider using a regular API key (sk-) from https://platform.openai.com/account/api-keys")
    
    # Update OpenAI API key
    env_vars = update_api_key(env_vars, "OPENAI_API_KEY", openai_key)
    
    # Update Anthropic API key
    anthropic_key = env_vars.get("ANTHROPIC_API_KEY")
    env_vars = update_api_key(env_vars, "ANTHROPIC_API_KEY", anthropic_key)
    
    # Update FRED API key
    fred_key = env_vars.get("FRED_API_KEY")
    env_vars = update_api_key(env_vars, "FRED_API_KEY", fred_key)
    
    # Update TradeStation API keys
    ts_client_id = env_vars.get("TRADESTATION_CLIENT_ID")
    env_vars = update_api_key(env_vars, "TRADESTATION_CLIENT_ID", ts_client_id)
    
    ts_client_secret = env_vars.get("TRADESTATION_CLIENT_SECRET")
    env_vars = update_api_key(env_vars, "TRADESTATION_CLIENT_SECRET", ts_client_secret)
    
    ts_refresh_token = env_vars.get("TRADESTATION_REFRESH_TOKEN")
    env_vars = update_api_key(env_vars, "TRADESTATION_REFRESH_TOKEN", ts_refresh_token)
    
    # Write the updated environment variables to the .env file
    write_env_file(env_file, env_vars)
    
    console.print("\n[green]API keys updated successfully![/green]")
    console.print("You can now run the test_api_connections.py script to verify the connections.")

if __name__ == "__main__":
    main() 