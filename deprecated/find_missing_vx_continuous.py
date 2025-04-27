#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to call the find_missing_vx_continuous.py utility.
For detailed implementation, see src/scripts/analysis/find_missing_vx_continuous.py
"""

import sys

try:
    from src.scripts.analysis.find_missing_vx_continuous import main as find_missing_main
    
    # Run the script
    if __name__ == "__main__":
        find_missing_main()
        
except ImportError:
    import os
    print("Error: find_missing_vx_continuous script not found or not properly installed.")
    print("This script should be run from the project root directory.")
    print("Current directory:", os.getcwd())
    sys.exit(1) 