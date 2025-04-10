#!/usr/bin/env python3
"""
Simple script to check environment variables in both environments.
"""

import os
import sys
from dotenv import load_dotenv

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
env_file = os.path.join(project_root, ".env")

# Load environment variables from the .env file
load_dotenv(env_file)

# Print Python executable path
print(f"Python executable: {sys.executable}")

# Print environment variables
print("\nEnvironment variables:")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"ANTHROPIC_API_KEY: {os.getenv('ANTHROPIC_API_KEY')}")

# Print .env file path
print("\n.env file path:")
print(env_file)

# Print current working directory
print("\nCurrent working directory:")
print(os.getcwd())

# Print sys.path
print("\nPython path:")
for path in sys.path:
    print(path) 