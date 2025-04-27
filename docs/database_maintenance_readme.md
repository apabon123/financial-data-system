# Futures Database Maintenance Tools

This directory contains scripts for maintaining the financial futures database, with a focus on keeping the active ES and NQ futures contracts up to date.

## Available Tools

### 1. View Futures Contracts (`view_futures_contracts.py`)

This script displays a summary of all futures contracts in the database, with details like symbol, date range, and row counts.

**Usage:**
```bash
# View all contracts
python view_futures_contracts.py

# View only ES contracts
python view_futures_contracts.py ES

# View only NQ contracts
python view_futures_contracts.py NQ
```

### 2. Update Active ES and NQ Futures (`update_active_es_nq_futures.py`)

This script identifies and updates only the active ES and NQ futures contracts based on the CME quarterly cycle (March, June, September, December). It automatically determines which contracts are currently active and fetches data for the next two upcoming contracts per symbol.

Key features:
- Follows the proper CME futures cycle (H, M, U, Z)
- Handles contract rollovers based on the 3rd Friday expiration rule
- Pulls 90 days of historical data to ensure continuity if the script hasn't run recently
- Avoids duplicates through database primary key constraints

**Usage:**
```bash
# Update active contracts
python update_active_es_nq_futures.py
```

## Database Structure

The market data is stored in a DuckDB database with the following primary key structure:

- Primary Key: (`timestamp`, `symbol`, `interval_value`, `interval_unit`)

This composite primary key ensures that there can never be duplicate entries in the database, as the database engine enforces uniqueness on this combination of fields.

## Common Database Maintenance Tasks

### Regular Maintenance (Weekly)

1. Update active contracts:
   ```bash
   python update_active_es_nq_futures.py
   ```

2. View updated contracts:
   ```bash
   python view_futures_contracts.py ES NQ
   ```

## For Advanced Users: Special Maintenance Operations

For special one-off maintenance scenarios, there are additional scripts available in the repository:

- `clean_futures_database.py`: For removing duplicate entries (only needed in special circumstances)
- `purge_old_futures_contracts.py`: For removing old contracts to save space (only as a one-off operation)

These operations should rarely be needed as the database enforces uniqueness through primary keys and historical data is typically valuable to retain.

## Simplified Maintenance with Batch File

For Windows users, a convenient batch file is provided that offers a menu-driven interface:

```bash
# Run the maintenance menu
maintain_futures_db.bat
```

This provides easy access to viewing and updating contracts without needing to remember command-line syntax. 