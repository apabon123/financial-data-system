#!/usr/bin/env python
"""
Clean up deprecated files in the Financial Data System project.

This script moves deprecated files to an archive directory, backs them up,
and can optionally remove them. It uses the migration manifest to identify
which files should be deprecated.

Usage:
    python cleanup_deprecated_files.py [--manifest FILENAME] [--archive-dir DIR] [--remove] [--dry-run]

Options:
    --manifest FILENAME     Path to migration manifest CSV [default: latest manifest]
    --archive-dir DIR       Directory to archive deprecated files [default: archive]
    --remove                Remove deprecated files instead of archiving them
    --dry-run               Show what would be done without making changes
"""

import os
import sys
import csv
import shutil
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

def get_project_root() -> Path:
    """Get the project root directory."""
    script_path = Path(__file__).resolve()
    # Navigate up to the project root (src/scripts/utilities -> src/scripts -> src -> project_root)
    return script_path.parent.parent.parent.parent

def find_latest_manifest() -> Optional[str]:
    """Find the latest migration manifest file."""
    project_root = get_project_root()
    manifest_pattern = os.path.join(project_root, "migration_manifest_*.csv")
    
    manifests = glob.glob(manifest_pattern)
    if not manifests:
        return None
        
    # Sort by modification time, newest first
    manifests.sort(key=os.path.getmtime, reverse=True)
    return manifests[0]

def load_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """
    Load migration manifest from CSV file.
    
    Args:
        manifest_path: Path to manifest CSV file
        
    Returns:
        List of manifest entries
    """
    manifest = []
    
    with open(manifest_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            manifest.append(row)
    
    return manifest

def cleanup_files(manifest_path: Optional[str] = None, 
                archive_dir: str = "archive",
                remove: bool = False,
                dry_run: bool = False) -> None:
    """
    Clean up deprecated files.
    
    Args:
        manifest_path: Path to manifest CSV file
        archive_dir: Directory to archive deprecated files
        remove: If True, remove files instead of archiving
        dry_run: If True, show what would be done without making changes
    """
    project_root = get_project_root()
    
    # Find latest manifest if not specified
    if not manifest_path:
        manifest_path = find_latest_manifest()
        if not manifest_path:
            print("No migration manifest found. Please run generate_migration_manifest.py first.")
            return
    
    print(f"Using manifest: {os.path.basename(manifest_path)}")
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    
    # Filter for deprecated files
    deprecated_files = [entry for entry in manifest if entry['status'] == 'deprecated']
    
    if not deprecated_files:
        print("No deprecated files found in manifest.")
        return
    
    print(f"Found {len(deprecated_files)} deprecated files to process.")
    
    # Create archive directory if needed
    archive_path = os.path.join(project_root, archive_dir)
    if not dry_run and not remove:
        os.makedirs(archive_path, exist_ok=True)
    
    # Process files
    processed_count = 0
    skipped_count = 0
    
    for entry in deprecated_files:
        file_path = os.path.join(project_root, entry['file_path'])
        
        # Skip if file doesn't exist
        if not os.path.isfile(file_path):
            print(f"Skipping (not found): {entry['file_path']}")
            skipped_count += 1
            continue
        
        if remove:
            # Remove file
            print(f"{'Would remove' if dry_run else 'Removing'}: {entry['file_path']}")
            if not dry_run:
                os.remove(file_path)
        else:
            # Archive file
            # Preserve directory structure in archive
            rel_path = entry['file_path']
            archive_file_path = os.path.join(archive_path, rel_path)
            archive_dir_path = os.path.dirname(archive_file_path)
            
            print(f"{'Would archive' if dry_run else 'Archiving'}: {entry['file_path']}")
            if not dry_run:
                # Create directory structure
                os.makedirs(archive_dir_path, exist_ok=True)
                
                # Copy file to archive
                shutil.copy2(file_path, archive_file_path)
                
                # Remove original
                os.remove(file_path)
        
        processed_count += 1
    
    # Create manifest in archive directory
    archive_manifest_path = os.path.join(archive_path, "deprecated_files_manifest.csv")
    if not dry_run and not remove and deprecated_files:
        with open(archive_manifest_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(deprecated_files[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(deprecated_files)
    
    # Summary
    print("\nSummary:")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    
    if not dry_run and not remove:
        print(f"\nDeprecated files have been archived to: {archive_path}")
        print(f"Manifest of deprecated files saved to: {archive_manifest_path}")
    elif not dry_run and remove:
        print(f"\nDeprecated files have been permanently removed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up deprecated files")
    parser.add_argument('--manifest', help="Path to migration manifest CSV")
    parser.add_argument('--archive-dir', default="archive", help="Directory to archive deprecated files")
    parser.add_argument('--remove', action='store_true', help="Remove deprecated files instead of archiving")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    cleanup_files(args.manifest, args.archive_dir, args.remove, args.dry_run)

if __name__ == "__main__":
    main()