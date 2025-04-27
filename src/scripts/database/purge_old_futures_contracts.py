#!/usr/bin/env python
"""
Purge Old Futures Contracts

This script helps maintain the database size by purging old or unnecessary futures contracts.
It allows for removing:
1. Contracts older than a specified year
2. Contracts with incomplete data (low row counts)
3. Specific contracts by pattern or list
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

# Backup database path (will be created before purging)
BACKUP_DB_PATH = f"./backups/financial_data_backup_before_purge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"purge_futures_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("purge_old_futures")

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
    Get a list of futures contracts from the database with details.
    
    Args:
        conn: Database connection
        base_symbols: Optional list of base symbols to filter by (e.g., ['ES', 'NQ'])
        
    Returns:
        DataFrame with contract details
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
        SELECT 
            symbol,
            COUNT(*) as row_count,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            DATEDIFF('day', MIN(timestamp), MAX(timestamp)) as date_span_days
        FROM market_data
        WHERE regexp_matches(symbol, '{pattern}')
            AND interval_value = 1 
            AND interval_unit = 'daily'
        GROUP BY symbol
        ORDER BY symbol
    """
    
    return conn.execute(query).fetchdf()

def backup_database():
    """Create a backup of the database before purging."""
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

def purge_contracts(conn, symbols, dry_run=True):
    """
    Purge specified contracts from the database.
    
    Args:
        conn: Database connection
        symbols: List of contract symbols to purge
        dry_run: If True, only show what would be purged
        
    Returns:
        Dictionary with purge results
    """
    if not symbols:
        logger.info("No contracts to purge")
        return {'contracts_purged': 0, 'rows_purged': 0}
    
    logger.info(f"Preparing to purge {len(symbols)} contracts")
    
    # Get row counts for each contract
    total_rows = 0
    for symbol in symbols:
        count_query = f"""
            SELECT COUNT(*) FROM market_data
            WHERE symbol = '{symbol}'
                AND interval_value = 1 
                AND interval_unit = 'daily'
        """
        row_count = conn.execute(count_query).fetchone()[0]
        total_rows += row_count
        
        # Log the contract details
        parsed = parse_futures_symbol(symbol)
        full_name = parsed['full_name'] if parsed else symbol
        logger.info(f"  {symbol} ({full_name}): {row_count} rows")
    
    if dry_run:
        logger.info(f"Would purge {len(symbols)} contracts with a total of {total_rows} rows")
    else:
        # Purge contracts
        symbols_str = "', '".join(symbols)
        purge_query = f"""
            DELETE FROM market_data
            WHERE symbol IN ('{symbols_str}')
                AND interval_value = 1 
                AND interval_unit = 'daily'
        """
        conn.execute(purge_query)
        logger.info(f"Purged {len(symbols)} contracts with a total of {total_rows} rows")
    
    return {'contracts_purged': len(symbols), 'rows_purged': total_rows}

def get_old_contracts(contracts_df, before_year):
    """
    Get contracts from before a specified year.
    
    Args:
        contracts_df: DataFrame with contract details
        before_year: Purge contracts before this year
        
    Returns:
        List of contract symbols to purge
    """
    old_contracts = []
    
    for _, row in contracts_df.iterrows():
        symbol = row['symbol']
        parsed = parse_futures_symbol(symbol)
        
        if parsed and parsed['year'] < before_year:
            old_contracts.append(symbol)
    
    return old_contracts

def get_incomplete_contracts(contracts_df, min_rows):
    """
    Get contracts with fewer than the specified number of rows.
    
    Args:
        contracts_df: DataFrame with contract details
        min_rows: Minimum number of rows required to keep a contract
        
    Returns:
        List of contract symbols to purge
    """
    incomplete_contracts = []
    
    for _, row in contracts_df.iterrows():
        if row['row_count'] < min_rows:
            incomplete_contracts.append(row['symbol'])
    
    return incomplete_contracts

def get_specific_contracts(contracts_df, pattern):
    """
    Get contracts matching a specific pattern.
    
    Args:
        contracts_df: DataFrame with contract details
        pattern: Regex pattern to match contract symbols
        
    Returns:
        List of contract symbols to purge
    """
    matching_contracts = []
    
    for _, row in contracts_df.iterrows():
        if re.match(pattern, row['symbol']):
            matching_contracts.append(row['symbol'])
    
    return matching_contracts

def main():
    """Main function to purge old futures contracts."""
    parser = argparse.ArgumentParser(description='Purge old futures contracts from database')
    parser.add_argument('--symbols', nargs='+', help='Base symbols to process (e.g., ES NQ)')
    parser.add_argument('--before-year', type=int, help='Purge contracts before this year')
    parser.add_argument('--min-rows', type=int, help='Purge contracts with fewer than this many rows')
    parser.add_argument('--pattern', type=str, help='Purge contracts matching this regex pattern')
    parser.add_argument('--contract-list', type=str, help='File with list of contracts to purge (one per line)')
    parser.add_argument('--keep-one-per-year', action='store_true', help='Keep at least one contract per year')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be purged without actually purging')
    args = parser.parse_args()
    
    # Check that at least one purge criteria is specified
    if not any([args.before_year, args.min_rows, args.pattern, args.contract_list]):
        logger.error("No purge criteria specified. Use --before-year, --min-rows, --pattern, or --contract-list")
        parser.print_help()
        sys.exit(1)
    
    logger.info(f"Starting database purge {'(DRY RUN)' if args.dry_run else ''}")
    
    base_symbols = args.symbols if args.symbols else None
    
    # Create a backup before making changes
    if not args.dry_run:
        if not backup_database():
            logger.error("Aborting due to backup failure")
            return
    
    try:
        # Connect to the database
        conn = duckdb.connect(DB_PATH, read_only=False)
        
        # Get all contracts with details
        contracts_df = get_futures_contracts(conn, base_symbols)
        
        if contracts_df.empty:
            logger.info("No futures contracts found in the database")
            return
        
        logger.info(f"Found {len(contracts_df)} contracts in the database")
        
        # Get contracts to purge based on criteria
        contracts_to_purge = set()
        
        # Process by year
        if args.before_year:
            logger.info(f"Finding contracts before year {args.before_year}")
            old_contracts = get_old_contracts(contracts_df, args.before_year)
            logger.info(f"Found {len(old_contracts)} contracts before year {args.before_year}")
            contracts_to_purge.update(old_contracts)
        
        # Process by minimum rows
        if args.min_rows:
            logger.info(f"Finding contracts with fewer than {args.min_rows} rows")
            incomplete_contracts = get_incomplete_contracts(contracts_df, args.min_rows)
            logger.info(f"Found {len(incomplete_contracts)} contracts with fewer than {args.min_rows} rows")
            contracts_to_purge.update(incomplete_contracts)
        
        # Process by pattern
        if args.pattern:
            logger.info(f"Finding contracts matching pattern '{args.pattern}'")
            pattern_contracts = get_specific_contracts(contracts_df, args.pattern)
            logger.info(f"Found {len(pattern_contracts)} contracts matching pattern '{args.pattern}'")
            contracts_to_purge.update(pattern_contracts)
        
        # Process by contract list
        if args.contract_list:
            logger.info(f"Reading contract list from {args.contract_list}")
            try:
                with open(args.contract_list, 'r') as f:
                    contract_list = [line.strip() for line in f if line.strip()]
                logger.info(f"Found {len(contract_list)} contracts in list")
                contracts_to_purge.update(contract_list)
            except Exception as e:
                logger.error(f"Error reading contract list: {e}")
        
        # Apply keep-one-per-year filter if requested
        if args.keep_one_per_year and contracts_to_purge:
            logger.info("Applying keep-one-per-year filter")
            
            # Group contracts by base symbol and year
            grouped_contracts = {}
            for contract in list(contracts_to_purge):
                parsed = parse_futures_symbol(contract)
                if parsed:
                    key = (parsed['base_symbol'], parsed['year'])
                    if key not in grouped_contracts:
                        grouped_contracts[key] = []
                    grouped_contracts[key].append(contract)
            
            # Keep only one contract per year (the one with most rows)
            contracts_to_keep = set()
            for key, contract_list in grouped_contracts.items():
                if len(contract_list) > 1:
                    base_symbol, year = key
                    
                    # Find contract with most rows
                    max_rows = 0
                    keep_contract = None
                    
                    for contract in contract_list:
                        row_count = contracts_df[contracts_df['symbol'] == contract]['row_count'].iloc[0]
                        if row_count > max_rows:
                            max_rows = row_count
                            keep_contract = contract
                    
                    if keep_contract:
                        contracts_to_keep.add(keep_contract)
                        logger.info(f"Keeping {keep_contract} for {base_symbol} {year} ({max_rows} rows)")
            
            # Remove contracts to keep from the purge list
            contracts_to_purge -= contracts_to_keep
            logger.info(f"After keep-one-per-year filter: {len(contracts_to_purge)} contracts to purge")
        
        # Convert set to list for processing
        contracts_to_purge = sorted(list(contracts_to_purge))
        
        # Purge contracts
        results = purge_contracts(conn, contracts_to_purge, args.dry_run)
        
        # Final message
        if args.dry_run:
            logger.info("\nThis was a dry run. No changes were made to the database.")
            logger.info(f"Would have purged {results['contracts_purged']} contracts with {results['rows_purged']} rows")
            logger.info(f"Run without --dry-run to apply these changes (backup will be created at {BACKUP_DB_PATH})")
        else:
            logger.info(f"\nDatabase purge complete. {results['contracts_purged']} contracts with {results['rows_purged']} rows purged.")
            logger.info(f"A backup was created at {BACKUP_DB_PATH}")
    
    except Exception as e:
        logger.error(f"Error during database purge: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 