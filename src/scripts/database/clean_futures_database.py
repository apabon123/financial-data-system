#!/usr/bin/env python
"""
Clean Futures Database

This script helps maintain a clean futures database by:
1. Identifying duplicate data for the same contract
2. Keeping only the highest quality/most complete data for each contract
3. Removing placeholder entries once real data exists
4. Providing a summary of cleaning actions taken
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from datetime import datetime, date
import re
import argparse

# Database path
DB_PATH = "./data/financial_data.duckdb"

# Backup database path (will be created before cleaning)
BACKUP_DB_PATH = f"./backups/financial_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"clean_futures_database_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("clean_futures_database")

def parse_futures_symbol(symbol):
    """Parse a futures symbol into its components."""
    # Match the typical futures symbol pattern
    match = re.match(r'^([A-Z]{2,3})([FGHJKMNQUVXZ])([0-9]{2})$', symbol)
    
    if not match:
        return None
    
    base_symbol, month_code, year_code = match.groups()
    
    # Convert month code to month name
    month_map = {
        'F': 'January',
        'G': 'February', 
        'H': 'March',
        'J': 'April',
        'K': 'May',
        'M': 'June',
        'N': 'July',
        'Q': 'August',
        'U': 'September',
        'V': 'October',
        'X': 'November',
        'Z': 'December'
    }
    
    month_name = month_map.get(month_code, 'Unknown')
    
    # Convert 2-digit year to 4-digit year
    year_num = int(year_code)
    if year_num < 50:  # Assume 20xx for years less than 50
        year = 2000 + year_num
    else:  # Assume 19xx for years 50 and greater
        year = 1900 + year_num
    
    return {
        'base_symbol': base_symbol,
        'month_code': month_code,
        'month_name': month_name,
        'year_code': year_code,
        'year': year,
        'full_name': f"{base_symbol} {month_name} {year}"
    }

def get_futures_contracts(conn, base_symbols=None):
    """
    Get a list of futures contracts from the database.
    
    Args:
        conn: Database connection
        base_symbols: Optional list of base symbols to filter by (e.g., ['ES', 'NQ'])
        
    Returns:
        DataFrame with contract symbols
    """
    if base_symbols:
        # Create pattern to match specified base symbols
        pattern = f"^({'|'.join(base_symbols)})[FGHJKMNQUVXZ][0-9]{{2}}$"
        logger.info(f"Filtering contracts by pattern: {pattern}")
    else:
        # Default pattern for all futures
        pattern = "^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]{2}$"
        logger.info("Getting all futures contracts")
    
    query = f"""
        SELECT DISTINCT symbol
        FROM market_data
        WHERE regexp_matches(symbol, '{pattern}')
            AND interval_value = 1 
            AND interval_unit = 'daily'
        ORDER BY symbol
    """
    
    return conn.execute(query).fetchdf()

def find_duplicate_dates(conn, symbol):
    """
    Find duplicate dates for a given contract.
    
    Args:
        conn: Database connection
        symbol: Futures contract symbol
        
    Returns:
        DataFrame with duplicate dates information
    """
    query = f"""
        SELECT 
            timestamp,
            COUNT(*) as count,
            STRING_AGG(source, ', ') as sources
        FROM market_data
        WHERE symbol = '{symbol}'
            AND interval_value = 1 
            AND interval_unit = 'daily'
        GROUP BY timestamp
        HAVING COUNT(*) > 1
        ORDER BY timestamp
    """
    
    return conn.execute(query).fetchdf()

def find_placeholder_entries(conn, symbol):
    """
    Find placeholder entries for a given contract.
    
    Args:
        conn: Database connection
        symbol: Futures contract symbol
        
    Returns:
        DataFrame with placeholder entries
    """
    query = f"""
        SELECT 
            *
        FROM market_data
        WHERE symbol = '{symbol}'
            AND interval_value = 1 
            AND interval_unit = 'daily'
            AND (
                source = 'Placeholder' OR
                (open = 0 AND high = 0 AND low = 0 AND close = 0 AND volume = 0)
            )
        ORDER BY timestamp
    """
    
    return conn.execute(query).fetchdf()

def backup_database():
    """Create a backup of the database before cleaning."""
    if not os.path.exists("./backups"):
        os.makedirs("./backups")
        logger.info("Created backups directory")
    
    try:
        import shutil
        shutil.copy2(DB_PATH, BACKUP_DB_PATH)
        logger.info(f"Created database backup at {BACKUP_DB_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        return False

def remove_duplicate_entries(conn, symbol, dry_run=True):
    """
    Remove duplicate entries for a given contract,
    keeping the entry with the most complete data.
    
    Args:
        conn: Database connection
        symbol: Futures contract symbol
        dry_run: If True, only show what would be removed
        
    Returns:
        Number of removed entries
    """
    # Find duplicate dates
    duplicates_df = find_duplicate_dates(conn, symbol)
    
    if duplicates_df.empty:
        logger.info(f"No duplicate dates found for {symbol}")
        return 0
    
    logger.info(f"Found {len(duplicates_df)} duplicate dates for {symbol}")
    
    removed_count = 0
    
    for _, row in duplicates_df.iterrows():
        timestamp = row['timestamp']
        count = row['count']
        
        logger.info(f"  {timestamp}: {count} entries ({row['sources']})")
        
        # Get all entries for this date
        query = f"""
            SELECT 
                id,
                timestamp,
                symbol,
                source,
                open,
                high,
                low,
                close,
                volume,
                CASE WHEN open IS NULL THEN 1 ELSE 0 END +
                CASE WHEN high IS NULL THEN 1 ELSE 0 END +
                CASE WHEN low IS NULL THEN 1 ELSE 0 END +
                CASE WHEN close IS NULL THEN 1 ELSE 0 END +
                CASE WHEN volume IS NULL THEN 1 ELSE 0 END as null_count
            FROM market_data
            WHERE symbol = '{symbol}'
                AND timestamp = '{timestamp}'
                AND interval_value = 1 
                AND interval_unit = 'daily'
            ORDER BY null_count, id
        """
        
        entries = conn.execute(query).fetchdf()
        
        # Keep the first entry (with least NULL values)
        keep_id = entries.iloc[0]['id']
        
        # Get IDs to remove
        remove_ids = entries[entries['id'] != keep_id]['id'].tolist()
        
        if dry_run:
            logger.info(f"    Would keep ID {keep_id} and remove {len(remove_ids)} entries")
        else:
            # Remove duplicate entries
            delete_query = f"""
                DELETE FROM market_data
                WHERE id IN ({', '.join(map(str, remove_ids))})
            """
            conn.execute(delete_query)
            logger.info(f"    Kept ID {keep_id} and removed {len(remove_ids)} entries")
        
        removed_count += len(remove_ids)
    
    return removed_count

def remove_placeholder_entries(conn, symbol, dry_run=True):
    """
    Remove placeholder entries for a contract if real data exists.
    
    Args:
        conn: Database connection
        symbol: Futures contract symbol
        dry_run: If True, only show what would be removed
        
    Returns:
        Number of removed entries
    """
    # Find placeholder entries
    placeholders_df = find_placeholder_entries(conn, symbol)
    
    if placeholders_df.empty:
        logger.info(f"No placeholder entries found for {symbol}")
        return 0
    
    # Get total number of entries for this contract
    count_query = f"""
        SELECT COUNT(*) FROM market_data
        WHERE symbol = '{symbol}'
            AND interval_value = 1 
            AND interval_unit = 'daily'
    """
    total_count = conn.execute(count_query).fetchone()[0]
    
    # If all entries are placeholders, keep them
    if total_count == len(placeholders_df):
        logger.info(f"All {total_count} entries for {symbol} are placeholders, keeping them")
        return 0
    
    # Get IDs to remove
    placeholder_ids = placeholders_df['id'].tolist()
    
    if dry_run:
        logger.info(f"Would remove {len(placeholder_ids)} placeholder entries for {symbol}")
    else:
        # Remove placeholder entries
        delete_query = f"""
            DELETE FROM market_data
            WHERE id IN ({', '.join(map(str, placeholder_ids))})
        """
        conn.execute(delete_query)
        logger.info(f"Removed {len(placeholder_ids)} placeholder entries for {symbol}")
    
    return len(placeholder_ids)

def clean_contract(conn, symbol, dry_run=True):
    """
    Clean a single contract by removing duplicates and placeholders.
    
    Args:
        conn: Database connection
        symbol: Futures contract symbol
        dry_run: If True, only show what would be removed
        
    Returns:
        Dictionary with cleaning results
    """
    logger.info(f"\nCleaning contract: {symbol}")
    
    # Get contract details before cleaning
    count_query = f"""
        SELECT COUNT(*) FROM market_data
        WHERE symbol = '{symbol}'
            AND interval_value = 1 
            AND interval_unit = 'daily'
    """
    count_before = conn.execute(count_query).fetchone()[0]
    
    # Parse the symbol for better logging
    parsed = parse_futures_symbol(symbol)
    full_name = parsed['full_name'] if parsed else symbol
    
    logger.info(f"Contract {symbol} ({full_name}) has {count_before} entries before cleaning")
    
    # Remove duplicates
    duplicates_removed = remove_duplicate_entries(conn, symbol, dry_run)
    
    # Remove placeholders
    placeholders_removed = remove_placeholder_entries(conn, symbol, dry_run)
    
    # Get count after cleaning
    count_after = count_before
    if not dry_run:
        count_after = conn.execute(count_query).fetchone()[0]
    
    results = {
        'symbol': symbol,
        'full_name': full_name,
        'count_before': count_before,
        'count_after': count_after if not dry_run else count_before - duplicates_removed - placeholders_removed,
        'duplicates_removed': duplicates_removed,
        'placeholders_removed': placeholders_removed
    }
    
    return results

def main():
    """Main function to clean the futures database."""
    parser = argparse.ArgumentParser(description='Clean futures database')
    parser.add_argument('--symbols', nargs='+', help='Base symbols to clean (e.g., ES NQ)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    args = parser.parse_args()
    
    logger.info(f"Starting database cleaning {'(DRY RUN)' if args.dry_run else ''}")
    
    base_symbols = args.symbols if args.symbols else None
    
    # Create a backup before making changes
    if not args.dry_run:
        if not backup_database():
            logger.error("Aborting due to backup failure")
            return
    
    try:
        # Connect to the database
        conn = duckdb.connect(DB_PATH, read_only=False)
        
        # Get all contracts to clean
        contracts_df = get_futures_contracts(conn, base_symbols)
        
        if contracts_df.empty:
            logger.info("No futures contracts found in the database")
            return
        
        logger.info(f"Found {len(contracts_df)} contracts to clean")
        
        # Clean each contract
        cleaning_results = []
        
        for _, row in contracts_df.iterrows():
            symbol = row['symbol']
            results = clean_contract(conn, symbol, args.dry_run)
            cleaning_results.append(results)
        
        # Print summary
        logger.info("\n=== Cleaning Summary ===")
        
        total_before = sum(r['count_before'] for r in cleaning_results)
        total_after = sum(r['count_after'] for r in cleaning_results)
        total_duplicates = sum(r['duplicates_removed'] for r in cleaning_results)
        total_placeholders = sum(r['placeholders_removed'] for r in cleaning_results)
        
        logger.info(f"Total contracts processed: {len(cleaning_results)}")
        logger.info(f"Total rows before: {total_before}")
        logger.info(f"Total rows after: {total_after}")
        logger.info(f"Total duplicates removed: {total_duplicates}")
        logger.info(f"Total placeholders removed: {total_placeholders}")
        
        # List contracts with the most changes
        if cleaning_results:
            most_cleaned = sorted(cleaning_results, 
                                 key=lambda r: r['duplicates_removed'] + r['placeholders_removed'], 
                                 reverse=True)[:5]
            
            logger.info("\nContracts with most cleaning:")
            for r in most_cleaned:
                if r['duplicates_removed'] + r['placeholders_removed'] > 0:
                    logger.info(f"{r['symbol']} ({r['full_name']}): {r['duplicates_removed']} duplicates, "
                              f"{r['placeholders_removed']} placeholders removed")
        
        # Final message
        if args.dry_run:
            logger.info("\nThis was a dry run. No changes were made to the database.")
            logger.info(f"Run without --dry-run to apply these changes (backup will be created at {BACKUP_DB_PATH})")
        else:
            logger.info(f"\nDatabase cleaning complete. A backup was created at {BACKUP_DB_PATH}")
    
    except Exception as e:
        logger.error(f"Error during database cleaning: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 