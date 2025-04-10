#!/usr/bin/env python
"""
List OpenAI Models

This script lists all available models for the OpenAI API key.
"""

import os
import sys
from typing import Dict, Any, List
import json
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

def list_openai_models() -> List[Dict[str, Any]]:
    """
    List all available OpenAI models.
    
    Returns:
        List of model information dictionaries
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        console.print("[red]Error: OpenAI API key not found in environment variables[/red]")
        return []
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(f"Response: {response.text}")
            return []
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return []

def main():
    """Main function."""
    console.print(Panel("OpenAI Models", title="Model List", border_style="blue"))
    
    # Get models
    models = list_openai_models()
    
    if not models:
        console.print("[yellow]No models found or error occurred.[/yellow]")
        return
    
    # Create a table to display the models
    table = Table(title="Available OpenAI Models")
    table.add_column("ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Owned By", style="yellow")
    table.add_column("Permission", style="magenta")
    
    # Add models to the table
    for model in models:
        table.add_row(
            model.get("id", "N/A"),
            str(model.get("created", "N/A")),
            model.get("owned_by", "N/A"),
            str(model.get("permission", []))
        )
    
    # Display the table
    console.print(table)
    
    # Print model count
    console.print(f"\nTotal models: {len(models)}")
    
    # Check for specific models
    gpt_models = [m for m in models if "gpt" in m.get("id", "").lower()]
    console.print(f"GPT models: {len(gpt_models)}")
    
    # Print GPT model IDs
    if gpt_models:
        console.print("\n[bold]GPT Models:[/bold]")
        for model in gpt_models:
            console.print(f"  - {model.get('id')}")

if __name__ == "__main__":
    main() 