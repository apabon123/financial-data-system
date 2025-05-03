#!/usr/bin/env python
"""
Utility script to test the calculate_expiration_date function from MarketDataFetcher.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd # Required for Timestamp checks if needed

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the necessary class AFTER setting the path
try:
    from src.scripts.market_data.fetch_market_data import MarketDataFetcher, DEFAULT_DB_PATH
except ImportError as e:
    print(f"Error importing MarketDataFetcher: {e}")
    print("Ensure the script is run from the project root or the path is correctly configured.")
    sys.exit(1)

# Configure basic logging for the fetcher's internal logs
logging.basicConfig(
    level=logging.WARNING, # Set to DEBUG to see detailed fetcher logs
    format='[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__) # Logger for this script itself
logger.setLevel(logging.INFO) # Set level for this script's logs

def main():
    parser = argparse.ArgumentParser(description='Test futures contract expiration date calculation.')
    parser.add_argument('symbol', help='The futures contract symbol (e.g., VXH14, ESH24)')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the market symbols configuration file (e.g., config/market_symbols.yaml)')
    parser.add_argument('--db-path', help=f'Path to database file (defaults to {DEFAULT_DB_PATH})', default=None)
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Set the logging level for this script')

    args = parser.parse_args()

    # Set logging level for this script
    logger.setLevel(getattr(logging, args.loglevel))
    # Optionally set the root logger level if you want to see fetcher debug logs too
    # logging.getLogger().setLevel(getattr(logging, args.loglevel))

    fetcher = None
    try:
        # Ensure config path is absolute or relative to project root
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)

        if not os.path.exists(config_path):
             logger.error(f"Configuration file not found: {config_path}")
             sys.exit(1)

        # Use provided db_path or the default from fetch_market_data
        db_path_to_use = args.db_path or DEFAULT_DB_PATH
        if not os.path.isabs(db_path_to_use):
             db_path_to_use = os.path.join(project_root, db_path_to_use)

        logger.info(f"Using config: {config_path}")
        logger.info(f"Using DB path: {db_path_to_use}")

        # Instantiate the fetcher. It needs the config to find the rules.
        # It might try to connect to the DB, so provide the path.
        fetcher = MarketDataFetcher(config_path=config_path, db_path=db_path_to_use)

        logger.info(f"Calculating expiration date for symbol: {args.symbol}")
        expiration_date = fetcher.calculate_expiration_date(args.symbol)

        if expiration_date:
            if isinstance(expiration_date, pd.Timestamp):
                 print(f"Calculated Expiration Date for {args.symbol}: {expiration_date.strftime('%Y-%m-%d')}")
            else:
                 print(f"Calculated Expiration Date for {args.symbol}: {expiration_date} (Type: {type(expiration_date)})")
        else:
            print(f"Could not calculate expiration date for {args.symbol}.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Close the connection if the fetcher was instantiated and has a connection
        if fetcher and hasattr(fetcher, 'conn') and fetcher.conn:
            try:
                fetcher.conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing fetcher database connection: {e_close}")

if __name__ == "__main__":
    main() 