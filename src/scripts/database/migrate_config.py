#!/usr/bin/env python
"""
Migrate Configuration Script

This script migrates the legacy single-file YAML configuration to the new
multi-file structure. It extracts sections from the legacy file and creates
specialized YAML files in the new format.

Usage:
    python migrate_config.py [--config PATH] [--output DIR] [--backup]
"""

import os
import sys
import argparse
import logging
import yaml
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.core.config import convert_legacy_to_new

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Migrate legacy configuration to new structure')
    
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root / 'config' / 'market_symbols.yaml'),
        help='Path to legacy configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=str(project_root / 'config' / 'yaml'),
        help='Directory to output new configuration files'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of legacy configuration file'
    )
    
    return parser.parse_args()

def backup_legacy_config(config_path):
    """Create a backup of the legacy configuration file."""
    if not os.path.exists(config_path):
        logger.error(f"Legacy configuration file not found: {config_path}")
        return False
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{config_path}.{timestamp}.bak"
    
    try:
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def extract_exchanges(legacy_config):
    """Extract exchanges information from legacy config."""
    exchanges = {}
    
    # Extract exchanges from futures section
    for item in legacy_config.get('futures', []):
        if 'exchange' in item and 'calendar' in item:
            exchange = item['exchange']
            calendar = item['calendar']
            
            if exchange not in exchanges:
                exchanges[exchange] = {
                    'name': exchange,
                    'calendars': {}
                }
            
            if calendar not in exchanges[exchange]['calendars']:
                exchanges[exchange]['calendars'][calendar] = {
                    'description': f"{exchange} {calendar} Trading Calendar"
                }
    
    # Extract exchanges from equities section
    for item in legacy_config.get('equities', []):
        if 'exchange' in item and 'calendar' in item:
            exchange = item['exchange']
            calendar = item['calendar']
            
            if exchange not in exchanges:
                exchanges[exchange] = {
                    'name': exchange,
                    'calendars': {}
                }
            
            if calendar not in exchanges[exchange]['calendars']:
                exchanges[exchange]['calendars'][calendar] = {
                    'description': f"{exchange} {calendar} Trading Calendar"
                }
    
    # Extract exchanges from indices section
    for item in legacy_config.get('indices', []):
        if 'exchange' in item and 'calendar' in item:
            exchange = item['exchange']
            calendar = item['calendar']
            
            if exchange not in exchanges:
                exchanges[exchange] = {
                    'name': exchange,
                    'calendars': {}
                }
            
            if calendar not in exchanges[exchange]['calendars']:
                exchanges[exchange]['calendars'][calendar] = {
                    'description': f"{exchange} {calendar} Trading Calendar"
                }
    
    return {
        'version': '1.0',
        'exchanges': exchanges
    }

def extract_futures(legacy_config):
    """Extract futures information from legacy config."""
    futures = {}
    templates = {
        'equity_index_futures': {
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['1min', '15min', 'daily']
        },
        'vix_futures': {
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['1min', '15min', 'daily']
        }
    }
    
    for item in legacy_config.get('futures', []):
        if 'base_symbol' in item:
            base_symbol = item['base_symbol']
            
            futures[base_symbol] = {
                'name': item.get('description', f"{base_symbol} Futures"),
                'description': item.get('description', f"{base_symbol} Futures"),
                'exchange': item.get('exchange', ''),
                'calendar': item.get('calendar', ''),
                'start_date': item.get('start_date', '')
            }
            
            # Determine which template to use
            if base_symbol == 'VX':
                futures[base_symbol]['inherit'] = 'vix_futures'
            elif base_symbol in ['ES', 'NQ']:
                futures[base_symbol]['inherit'] = 'equity_index_futures'
            
            # Copy contract info
            if 'historical_contracts' in item:
                futures[base_symbol]['contract_info'] = {
                    'patterns': item['historical_contracts'].get('patterns', []),
                    'start_year': item['historical_contracts'].get('start_year', 2000)
                }
                
                if 'start_month' in item['historical_contracts']:
                    futures[base_symbol]['contract_info']['start_month'] = item['historical_contracts']['start_month']
                    
                if 'exclude_contracts' in item['historical_contracts']:
                    futures[base_symbol]['contract_info']['exclude_contracts'] = item['historical_contracts']['exclude_contracts']
            
            # Copy num_active_contracts
            if 'num_active_contracts' in item:
                futures[base_symbol]['contract_info']['num_active_contracts'] = item['num_active_contracts']
            
            # Copy expiry rule
            if 'expiry_rule' in item:
                futures[base_symbol]['expiry_rule'] = item['expiry_rule']
    
    # Extract continuous contracts
    continuous_contracts = {}
    for item in legacy_config.get('futures', []):
        if 'symbol' in item and item.get('type') == 'continuous_future':
            symbol = item['symbol']
            
            # Extract root symbol from continuous symbol
            root_symbol = None
            if symbol.startswith('@'):
                parts = symbol[1:].split('=')
                if len(parts) > 0:
                    root_symbol = parts[0]
            
            if root_symbol and root_symbol in futures:
                if 'continuous_contracts' not in futures[root_symbol]:
                    futures[root_symbol]['continuous_contracts'] = []
                
                futures[root_symbol]['continuous_contracts'].append({
                    'identifier': symbol,
                    'description': item.get('description', f"{root_symbol} Continuous Contract"),
                    'type': 'continuous_future',
                    'frequencies': item.get('frequencies', ['daily']),
                    'start_date': item.get('start_date', '')
                })
    
    # Handle continuous_group if present
    for item in legacy_config.get('futures', []):
        if 'continuous_group' in item:
            group = item['continuous_group']
            identifier_base = group.get('identifier_base', '')
            
            # Extract root symbol from identifier_base
            root_symbol = None
            if identifier_base.startswith('@'):
                root_symbol = identifier_base[1:]
            
            if root_symbol and root_symbol in futures:
                if 'continuous_contracts' not in futures[root_symbol]:
                    futures[root_symbol]['continuous_contracts'] = {}
                
                futures[root_symbol]['continuous_contracts']['group'] = {
                    'identifier_base': group.get('identifier_base', ''),
                    'month_codes': group.get('month_codes', []),
                    'settings_code': group.get('settings_code', ''),
                    'description_template': group.get('description_template', ''),
                    'type': 'continuous_future',
                    'frequencies': group.get('frequencies', ['daily']),
                    'start_date': group.get('start_date', '')
                }
    
    return {
        'version': '1.0',
        'templates': templates,
        'futures': futures
    }

def extract_indices(legacy_config):
    """Extract indices information from legacy config."""
    indices = {}
    templates = {
        'equity_index': {
            'type': 'index',
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['daily']
        },
        'volatility_index': {
            'type': 'index',
            'default_source': 'cboe',
            'default_raw_table': 'market_data_cboe',
            'frequencies': ['daily']
        }
    }
    
    for item in legacy_config.get('indices', []):
        if 'symbol' in item:
            symbol = item['symbol']
            
            # Normalize the symbol for the dict key
            if symbol == '$VIX.X':
                key = 'VIX'
            elif symbol == '$SPX.X':
                key = 'SPX'
            elif symbol == '$NDX.X':
                key = 'NDX'
            else:
                key = symbol
            
            indices[key] = {
                'symbol': symbol,
                'name': item.get('description', ''),
                'description': item.get('description', ''),
                'exchange': item.get('exchange', ''),
                'calendar': item.get('calendar', ''),
                'start_date': item.get('start_date', '')
            }
            
            # Determine which template to use
            if key == 'VIX':
                indices[key]['inherit'] = 'volatility_index'
            else:
                indices[key]['inherit'] = 'equity_index'
            
            # Add frequencies
            if 'frequencies' in item:
                if isinstance(item['frequencies'], list):
                    if all(isinstance(x, str) for x in item['frequencies']):
                        # Simple list of string frequencies
                        indices[key]['frequencies'] = item['frequencies']
                    else:
                        # List of dictionaries
                        frequencies = []
                        for freq in item['frequencies']:
                            if isinstance(freq, dict) and 'name' in freq:
                                frequencies.append(freq['name'])
                        indices[key]['frequencies'] = frequencies
    
    return {
        'version': '1.0',
        'templates': templates,
        'indices': indices
    }

def extract_etfs(legacy_config):
    """Extract ETFs information from legacy config."""
    etfs = {}
    templates = {
        'equity_index_etf': {
            'type': 'ETF',
            'asset_class': 'equity',
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['1min', '15min', 'daily']
        },
        'volatility_etf': {
            'type': 'ETF',
            'asset_class': 'volatility',
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['1min', '15min', 'daily']
        }
    }
    
    for item in legacy_config.get('equities', []):
        if 'type' in item and item['type'] == 'ETF' and 'symbol' in item:
            symbol = item['symbol']
            
            etfs[symbol] = {
                'symbol': symbol,
                'name': item.get('description', ''),
                'description': item.get('description', ''),
                'exchange': item.get('exchange', ''),
                'calendar': item.get('calendar', ''),
                'start_date': item.get('start_date', '')
            }
            
            # Determine which template to use
            if symbol in ['SPY', 'QQQ']:
                etfs[symbol]['inherit'] = 'equity_index_etf'
            else:
                etfs[symbol]['inherit'] = 'equity_index_etf'  # Default
            
            # Add frequencies
            if 'frequencies' in item:
                if isinstance(item['frequencies'], list):
                    if all(isinstance(x, str) for x in item['frequencies']):
                        # Simple list of string frequencies
                        etfs[symbol]['frequencies'] = item['frequencies']
                    else:
                        # List of dictionaries
                        frequencies = []
                        for freq in item['frequencies']:
                            if isinstance(freq, dict) and 'name' in freq:
                                frequencies.append(freq['name'])
                        etfs[symbol]['frequencies'] = frequencies
    
    return {
        'version': '1.0',
        'templates': templates,
        'etfs': etfs
    }

def extract_equities(legacy_config):
    """Extract equities information from legacy config."""
    equities = {}
    templates = {
        'common_stock': {
            'type': 'Stock',
            'asset_class': 'equity',
            'default_source': 'tradestation',
            'default_raw_table': 'market_data',
            'frequencies': ['daily']
        }
    }
    
    for item in legacy_config.get('equities', []):
        if 'type' in item and item['type'] == 'Stock' and 'symbol' in item:
            symbol = item['symbol']
            
            equities[symbol] = {
                'symbol': symbol,
                'name': item.get('description', ''),
                'description': item.get('description', ''),
                'exchange': item.get('exchange', ''),
                'calendar': item.get('calendar', ''),
                'start_date': item.get('start_date', '')
            }
            
            # Always use the common_stock template
            equities[symbol]['inherit'] = 'common_stock'
            
            # Add frequencies
            if 'frequencies' in item:
                if isinstance(item['frequencies'], list):
                    if all(isinstance(x, str) for x in item['frequencies']):
                        # Simple list of string frequencies
                        equities[symbol]['frequencies'] = item['frequencies']
                    else:
                        # List of dictionaries
                        frequencies = []
                        for freq in item['frequencies']:
                            if isinstance(freq, dict) and 'name' in freq:
                                frequencies.append(freq['name'])
                        equities[symbol]['frequencies'] = frequencies
    
    return {
        'version': '1.0',
        'templates': templates,
        'equities': equities
    }

def extract_data_sources(legacy_config):
    """Extract data sources information from legacy config."""
    data_sources = {
        'tradestation': {
            'name': 'TradeStation',
            'type': 'broker_api',
            'description': 'TradeStation market data API',
            'base_url': 'https://api.tradestation.com/v3'
        },
        'cboe': {
            'name': 'CBOE',
            'type': 'exchange_data',
            'description': 'CBOE market data files',
            'base_url': 'https://cdn.cboe.com/api/global/delayed_quotes'
        }
    }
    
    return {
        'version': '1.0',
        'data_sources': data_sources
    }

def extract_cleaning_rules(legacy_config):
    """Extract cleaning rules information from legacy config."""
    # Create default cleaning rules based on asset classes
    cleaning_rules = {
        'equity': {
            'description': 'Cleaning rules for equity data',
            'enabled': True,
            'target': {
                'asset_class': 'equity'
            },
            'rules': [
                {
                    'rule': 'zero_values',
                    'enabled': True,
                    'priority': 10,
                    'parameters': {
                        'threshold': 0.0001,
                        'fields': ['open', 'high', 'low', 'close'],
                        'action': 'interpolate'
                    }
                },
                {
                    'rule': 'high_low_inversion',
                    'enabled': True,
                    'priority': 20,
                    'parameters': {
                        'fields': ['high', 'low'],
                        'action': 'swap'
                    }
                }
            ]
        },
        'vix_index': {
            'description': 'Cleaning rules for VIX index data',
            'enabled': True,
            'target': {
                'symbols': ['$VIX.X']
            },
            'rules': [
                {
                    'rule': 'zero_values',
                    'enabled': True,
                    'priority': 10,
                    'parameters': {
                        'threshold': 0.0001,
                        'fields': ['open', 'high', 'low', 'close'],
                        'action': 'interpolate'
                    }
                },
                {
                    'rule': 'minimum_value',
                    'enabled': True,
                    'priority': 50,
                    'parameters': {
                        'fields': ['open', 'high', 'low', 'close'],
                        'min_value': 9.0,
                        'action': 'replace_with_min'
                    }
                }
            ]
        },
        'vx_futures': {
            'description': 'Cleaning rules for VX futures data',
            'enabled': True,
            'target': {
                'symbols_pattern': '^VX[A-Z]\\d{2}$'
            },
            'rules': [
                {
                    'rule': 'zero_price',
                    'enabled': True,
                    'priority': 10,
                    'parameters': {
                        'fields': ['open', 'high', 'low', 'close', 'settle'],
                        'threshold': 0.05,
                        'action': 'interpolate',
                        'max_gap_days': 5
                    }
                }
            ]
        }
    }
    
    return {
        'version': '1.0',
        'templates': {
            'common_price_data_rules': {
                'priority': 100,
                'enabled': True,
                'fields': ['open', 'high', 'low', 'close'],
                'rules': [
                    {
                        'rule': 'zero_values',
                        'enabled': True,
                        'priority': 10,
                        'parameters': {
                            'threshold': 0.0001,
                            'fields': ['open', 'high', 'low', 'close'],
                            'action': 'interpolate'
                        }
                    },
                    {
                        'rule': 'high_low_inversion',
                        'enabled': True,
                        'priority': 20,
                        'parameters': {
                            'fields': ['high', 'low'],
                            'action': 'swap'
                        }
                    }
                ]
            }
        },
        'cleaning_rules': cleaning_rules
    }

def manual_migration(legacy_path, output_dir):
    """Perform manual migration from legacy to new structure."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load legacy configuration
        with open(legacy_path, 'r') as f:
            legacy_config = yaml.safe_load(f)
        
        # Extract sections
        exchanges_config = extract_exchanges(legacy_config)
        futures_config = extract_futures(legacy_config)
        indices_config = extract_indices(legacy_config)
        etfs_config = extract_etfs(legacy_config)
        equities_config = extract_equities(legacy_config)
        data_sources_config = extract_data_sources(legacy_config)
        cleaning_rules_config = extract_cleaning_rules(legacy_config)
        
        # Save to files
        save_yaml(exchanges_config, os.path.join(output_dir, 'exchanges.yaml'))
        save_yaml(futures_config, os.path.join(output_dir, 'futures.yaml'))
        save_yaml(indices_config, os.path.join(output_dir, 'indices.yaml'))
        save_yaml(etfs_config, os.path.join(output_dir, 'etfs.yaml'))
        save_yaml(equities_config, os.path.join(output_dir, 'equities.yaml'))
        save_yaml(data_sources_config, os.path.join(output_dir, 'data_sources.yaml'))
        save_yaml(cleaning_rules_config, os.path.join(output_dir, 'cleaning_rules.yaml'))
        
        logger.info(f"Successfully migrated legacy configuration to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error in manual migration: {e}")
        return False

def save_yaml(data, file_path):
    """Save data to a YAML file."""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create backup if requested
    if args.backup:
        if not backup_legacy_config(args.config):
            logger.error("Backup failed. Aborting migration.")
            return 1
    
    # Try using the ConfigLoader's built-in conversion first
    try:
        logger.info("Attempting to use ConfigLoader for migration...")
        convert_legacy_to_new(args.output)
        logger.info("ConfigLoader migration completed successfully")
        return 0
    except Exception as e:
        logger.warning(f"ConfigLoader migration failed: {e}")
        logger.info("Falling back to manual migration...")
    
    # Fall back to manual migration
    if manual_migration(args.config, args.output):
        logger.info("Manual migration completed successfully")
        return 0
    else:
        logger.error("Manual migration failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())