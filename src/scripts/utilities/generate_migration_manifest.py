#!/usr/bin/env python
"""
Generate a migration manifest for Financial Data System transformation.

This script identifies files that should be deprecated, migrated, or kept as-is
during the migration to the new architecture. It generates a CSV file with the
file paths, current functionality, migration status, and replacement components.

Usage:
    python generate_migration_manifest.py [--output FILENAME] [--dry-run]

Options:
    --output FILENAME   Output filename for the manifest [default: migration_manifest.csv]
    --dry-run           Print the manifest to stdout instead of saving to file
"""

import os
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Define categories for files
DEPRECATED = "deprecated"
MIGRATE = "migrate"
KEEP = "keep"
REMOVE = "remove"

def get_project_root() -> Path:
    """Get the project root directory."""
    script_path = Path(__file__).resolve()
    # Navigate up to the project root (src/scripts/utilities -> src/scripts -> src -> project_root)
    return script_path.parent.parent.parent.parent

def get_migration_status(file_path: str) -> Tuple[str, str, str]:
    """
    Determine the migration status for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (status, functionality, replacement)
    """
    # Default values
    status = KEEP
    functionality = "Unknown"
    replacement = "N/A"
    
    # Skip directories
    if os.path.isdir(file_path):
        return status, functionality, replacement
    
    # Skip non-code files
    if not any(file_path.endswith(ext) for ext in ['.py', '.bat', '.sh', '.sql']):
        return status, functionality, replacement
    
    # Extract relative path for better matching
    rel_path = os.path.relpath(file_path, get_project_root())
    
    # Files in deprecated directory
    if 'deprecated' in rel_path:
        status = DEPRECATED
        functionality = "Legacy functionality that has been superseded"
        replacement = "New components in src/core/*"
        
    # Old continuous futures generation
    elif 'market_data/generate_continuous_futures.py' in rel_path:
        status = MIGRATE
        functionality = "Generates continuous futures contracts using older methodology"
        replacement = "src/processors/continuous/* with Panama method"
        
    # Temporary files
    elif '/temp_' in rel_path or rel_path.startswith('temp_'):
        status = DEPRECATED
        functionality = "Temporary script for one-time operations"
        replacement = "Formal components in new architecture"
        
    # VX zero prices and gaps
    elif 'fill_vx_zero_prices.py' in rel_path or 'fill_vx_continuous_gaps.py' in rel_path:
        status = MIGRATE
        functionality = "Fixes missing or zero values in VX data"
        replacement = "src/processors/cleaners/vx_zero_prices.py and data cleaning pipeline"
        
    # Old test versions
    elif 'test_fetch_vx' in rel_path:
        status = DEPRECATED
        functionality = "Test scripts for VX data fetching"
        replacement = "Formal test suite in tests/"
    
    # Helper function to extract functionality from file
    def extract_functionality(file_path: str) -> str:
        """Extract functionality description from a file's docstring."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read just the beginning
                
                # Look for docstring
                if '"""' in content:
                    docstring = content.split('"""')[1].strip()
                    first_line = docstring.split('\n')[0]
                    return first_line
                elif "'''" in content:
                    docstring = content.split("'''")[1].strip()
                    first_line = docstring.split('\n')[0]
                    return first_line
                    
                return "Unknown"
        except Exception:
            return "Unknown"
    
    # If functionality is still unknown, try to extract from docstring
    if functionality == "Unknown" and os.path.exists(file_path):
        extracted = extract_functionality(file_path)
        if extracted != "Unknown":
            functionality = extracted
    
    return status, functionality, replacement

def generate_manifest(output_file: Optional[str] = None, dry_run: bool = False) -> None:
    """
    Generate migration manifest.
    
    Args:
        output_file: Path to output CSV file
        dry_run: If True, print to stdout instead of saving to file
    """
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    manifest = []
    
    # Walk through project directory
    for root, dirs, files in os.walk(project_root):
        # Skip node_modules, __pycache__, and similar
        if any(skip_dir in root for skip_dir in ['node_modules', '__pycache__', '.git', 'venv']):
            continue
            
        for file in files:
            full_path = os.path.join(root, file)
            status, functionality, replacement = get_migration_status(full_path)
            
            # Skip 'keep' status to focus on files needing attention
            if status == KEEP:
                continue
                
            rel_path = os.path.relpath(full_path, project_root)
            modified_time = datetime.fromtimestamp(os.path.getmtime(full_path))
            
            manifest.append({
                'file_path': rel_path,
                'status': status,
                'functionality': functionality,
                'replacement': replacement,
                'last_modified': modified_time.strftime('%Y-%m-%d'),
                'file_type': os.path.splitext(file)[1],
                'notes': ''
            })
    
    # Sort by status and then by file path
    manifest.sort(key=lambda x: (x['status'], x['file_path']))
    
    # Print or save manifest
    if dry_run:
        print("\nMigration Manifest:")
        print("-" * 100)
        for entry in manifest:
            print(f"{entry['status'].upper()}: {entry['file_path']}")
            print(f"  Functionality: {entry['functionality']}")
            print(f"  Replacement: {entry['replacement']}")
            print(f"  Modified: {entry['last_modified']}")
            print("-" * 100)
    else:
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"migration_manifest_{timestamp}.csv"
        
        output_path = os.path.join(project_root, output_file)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_path', 'status', 'functionality', 'replacement', 
                         'last_modified', 'file_type', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(manifest)
            
        print(f"Migration manifest saved to {output_path}")
        print(f"Total files identified: {len(manifest)}")
        
        # Print summary by status
        status_counts = {}
        for entry in manifest:
            status = entry['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
        print("\nSummary:")
        for status, count in status_counts.items():
            print(f"  {status.upper()}: {count} files")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate migration manifest")
    parser.add_argument('--output', help="Output filename for the manifest")
    parser.add_argument('--dry-run', action='store_true', help="Print to stdout instead of saving to file")
    
    args = parser.parse_args()
    
    generate_manifest(args.output, args.dry_run)

if __name__ == "__main__":
    main()