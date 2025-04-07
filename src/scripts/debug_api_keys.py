#!/usr/bin/env python3
"""
API Keys Debug Script

This script helps debug issues with API keys by:
1. Checking package versions
2. Verifying environment variables are loaded correctly
3. Testing API keys with direct requests
4. Checking .env file location and content
"""

import os
import sys
import requests
import openai
import anthropic
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup console
console = Console()

def mask_key(key, visible_chars=4):
    """Mask an API key for safe display."""
    if not key:
        return "Not set"
    return f"{key[:visible_chars]}{'*' * (len(key) - visible_chars)}"

def check_env_file():
    """Check the .env file location and content."""
    console.print("\n[bold]Checking .env file:[/]")
    
    # Find .env file
    env_path = find_dotenv()
    if env_path:
        console.print(f"[green]✓ Found .env file at: {env_path}[/]")
        
        # Check if file is readable
        try:
            with open(env_path, 'r') as f:
                env_content = f.read()
                console.print(f"[green]✓ .env file is readable[/]")
                
                # Count variables
                var_count = sum(1 for line in env_content.splitlines() 
                               if line.strip() and not line.startswith('#') and '=' in line)
                console.print(f"  Contains {var_count} environment variables")
                
                # Check for required variables
                required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FRED_API_KEY", 
                                "CLIENT_ID", "CLIENT_SECRET", "REFRESH_TOKEN"]
                missing_vars = [var for var in required_vars 
                               if not any(line.startswith(f"{var}=") for line in env_content.splitlines())]
                
                if missing_vars:
                    console.print(f"[yellow]! Missing variables in .env: {', '.join(missing_vars)}[/]")
                else:
                    console.print("[green]✓ All required variables are present in .env[/]")
        except Exception as e:
            console.print(f"[red]✗ Error reading .env file: {str(e)}[/]")
    else:
        console.print("[red]✗ No .env file found in current directory or parent directories[/]")
        
        # Check current directory
        current_dir = Path.cwd()
        console.print(f"\nChecking current directory: {current_dir}")
        env_files = list(current_dir.glob(".env*"))
        if env_files:
            console.print(f"[yellow]! Found potential .env files: {', '.join(str(f) for f in env_files)}[/]")
        else:
            console.print("[red]✗ No .env files found in current directory[/]")

def main():
    """Main function to debug API keys."""
    console.print(Panel("API Keys Debug Information", style="bold blue"))
    
    # Check package versions
    console.print("\n[bold]Package Versions:[/]")
    console.print(f"OpenAI: {openai.__version__}")
    console.print(f"Anthropic: {anthropic.__version__}")
    
    # Check .env file
    check_env_file()
    
    # Load environment variables
    console.print("\n[bold]Loading environment variables...[/]")
    load_dotenv()
    
    # Check environment variables
    console.print("\n[bold]Environment Variables:[/]")
    env_vars = {
        "CLIENT_ID": os.getenv("CLIENT_ID"),
        "CLIENT_SECRET": os.getenv("CLIENT_SECRET"),
        "REFRESH_TOKEN": os.getenv("REFRESH_TOKEN"),
        "FRED_API_KEY": os.getenv("FRED_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
    }
    
    # Create a table to display environment variables
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Length", style="yellow")
    
    for var, value in env_vars.items():
        masked_value = mask_key(value)
        length = len(value) if value else 0
        table.add_row(var, masked_value, str(length))
    
    console.print(table)
    
    # Test OpenAI API key directly
    console.print("\n[bold]Testing OpenAI API key directly...[/]")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        # Check API key format
        if openai_key.startswith("sk-proj-"):
            console.print("[yellow]Warning: You are using a project API key (sk-proj-).[/yellow]")
            console.print("This format is not compatible with the OpenAI Python library.")
            console.print("Please get a regular API key (sk-) from https://platform.openai.com/account/api-keys")
            console.print("Project API keys are for specific projects and may have limited access.")
        elif not openai_key.startswith("sk-"):
            console.print("[red]Error: Invalid OpenAI API key format.[/red]")
            console.print("The key should start with 'sk-' (not 'sk-proj-').")
            console.print("Get a valid API key from https://platform.openai.com/account/api-keys")
        else:
            console.print("[green]OpenAI API key format is valid (sk-).[/green]")
        
        try:
            # Try using the OpenAI client library with the latest API format (v1.0+)
            client = openai.OpenAI(api_key=openai_key)
            try:
                # Try a simple API call using the new format
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                console.print("[green]✓ OpenAI API key is valid! (using client library)[/]")
                console.print(f"  Response received: {response.choices[0].message.content}")
            except Exception as client_error:
                console.print(f"[yellow]Client library test failed: {str(client_error)}[/]")
                console.print("Trying direct API call...")
                
                # Fallback to direct API call
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                }
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers=headers
                )
                
                if response.status_code == 200:
                    console.print("[green]✓ OpenAI API key is valid! (using direct API call)[/]")
                    models = response.json().get("data", [])
                    console.print(f"  Available models: {len(models)}")
                    if models:
                        console.print(f"  First model: {models[0].get('id')}")
                else:
                    console.print(f"[red]✗ OpenAI API key test failed: {response.status_code}[/]")
                    console.print(f"  Error: {response.text}")
                    
                    # Additional debugging for project API keys
                    if openai_key.startswith("sk-proj-"):
                        console.print("\n[yellow]Note: This is a project API key (sk-proj-).[/]")
                        console.print("Make sure:")
                        console.print("1. The project is active in your OpenAI account")
                        console.print("2. The API key has the necessary permissions")
                        console.print("3. You're using the latest version of the OpenAI Python package")
                        console.print("4. The project has sufficient credits/quota")
                        
                        # Try to get more information about the project
                        try:
                            console.print("\n[bold]Attempting to get project information...[/]")
                            project_response = requests.get(
                                "https://api.openai.com/v1/projects",
                                headers=headers
                            )
                            if project_response.status_code == 200:
                                projects = project_response.json().get("data", [])
                                console.print(f"[green]✓ Found {len(projects)} projects[/]")
                                for project in projects:
                                    console.print(f"  - {project.get('id')}: {project.get('name')}")
                            else:
                                console.print(f"[red]✗ Failed to get project information: {project_response.status_code}[/]")
                                console.print(f"  Error: {project_response.text}")
                        except Exception as e:
                            console.print(f"[red]✗ Error getting project information: {str(e)}[/]")
        except Exception as e:
            console.print(f"[red]✗ OpenAI API key test error: {str(e)}[/]")
    else:
        console.print("[red]✗ OpenAI API key not found in environment variables[/]")
    
    # Test Anthropic API key directly
    console.print("\n[bold]Testing Anthropic API key directly...[/]")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            # Try using the Anthropic client library
            client = anthropic.Anthropic(api_key=anthropic_key)
            try:
                # Try a simple API call
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                console.print("[green]✓ Anthropic API key is valid! (using client library)[/]")
                console.print(f"  Response received: {response.content[0].text}")
            except Exception as client_error:
                console.print(f"[yellow]Client library test failed: {str(client_error)}[/]")
                console.print("Trying direct API call...")
                
                # Fallback to direct API call
                headers = {
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                response = requests.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers
                )
                
                if response.status_code == 200:
                    console.print("[green]✓ Anthropic API key is valid![/]")
                    models = response.json().get("models", [])
                    console.print(f"  Available models: {len(models)}")
                    if models:
                        console.print(f"  First model: {models[0].get('id')}")
                else:
                    console.print(f"[red]✗ Anthropic API key test failed: {response.status_code}[/]")
                    console.print(f"  Error: {response.text}")
        except Exception as e:
            console.print(f"[red]✗ Anthropic API key test error: {str(e)}[/]")
    else:
        console.print("[red]✗ Anthropic API key not found in environment variables[/]")
    
    # Check for special characters in API keys
    console.print("\n[bold]Checking for special characters in API keys...[/]")
    for var, value in env_vars.items():
        if value:
            # Check for newlines
            if "\n" in value:
                console.print(f"[red]✗ {var} contains newlines![/]")
            # Check for spaces
            if " " in value:
                console.print(f"[red]✗ {var} contains spaces![/]")
            # Check for tabs
            if "\t" in value:
                console.print(f"[red]✗ {var} contains tabs![/]")
            # Check for invisible characters
            invisible_chars = [c for c in value if not c.isprintable()]
            if invisible_chars:
                console.print(f"[red]✗ {var} contains {len(invisible_chars)} invisible characters![/]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 