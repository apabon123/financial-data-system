#!/usr/bin/env python3
"""
Check environment variables script
"""

import os
from dotenv import load_dotenv
import sys

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
env_path = os.path.join(project_root, ".env")

print(f"Project root: {project_root}")
print(f"Env path: {env_path}")
print(f"Current working directory: {os.getcwd()}")

# Load environment variables
load_dotenv(env_path)

# Print environment variables
print("\nEnvironment Variables:")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'Not found')[:10]}...")
print(f"ANTHROPIC_API_KEY: {os.getenv('ANTHROPIC_API_KEY', 'Not found')[:10]}...")
print(f"TRADESTATION_CLIENT_ID: {os.getenv('CLIENT_ID', 'Not found')[:10]}...")
print(f"FRED_API_KEY: {os.getenv('FRED_API_KEY', 'Not found')[:10]}...")

# Print Python path
print("\nPython Path:")
for path in sys.path:
    print(path) 