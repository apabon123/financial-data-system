import yaml
import logging
import logging.handlers
import sys
import pytz
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Tuple

# --- Configuration Class ---

class Config:
    """Centralized configuration management optimized for futures/options."""

    def __init__(self, config_file='config.yaml'):
        # Load configurations from YAML file
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # --- Data Source Configuration ---
        self.data_source = config_data.get('data_source', {})
        # Define default data source if not provided in YAML
        if not self.data_source:
             self.data_source = {
                 'type': 'duckdb', # Default to duckdb
                 'db_path': 'data/market_data.duckdb', # Default path relative to workspace root
                 'table_name_format': '{symbol}_1min' # Default table naming convention
             }
        elif 'db_path' not in self.data_source:
             # Use default path if section exists but path is missing
             self.data_source['db_path'] = 'data/market_data.duckdb'
        elif 'table_name_format' not in self.data_source:
            self.data_source['table_name_format'] = '{symbol}_1min'


        # --- Output Path Configuration ---
        # Base output directory, relative to workspace root
        self.base_output_dir = Path(config_data.get('base_output_dir', 'output/IntradayMomentum'))
        # Ensure base output directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Symbol-specific output paths derived from base_output_dir
        # This structure assumes output_paths in YAML is no longer used,
        # or it's used for overrides (handling overrides is not implemented here for simplicity).
        # We will generate paths dynamically when needed via get_symbol_specs.
        # self.output_paths = config_data.get('output_paths', {}) # Removed old output_paths loading

        # Basic configurations
        # self.input_files = config_data.get('input_files', {}) # Removed old input_files
        self.strategy_params = config_data.get('strategy_params', {})
        self.contract_specs = config_data.get('contract_specs', {})
        self.trading_hours = config_data.get('trading_hours', {})
        self.transaction_costs = config_data.get('transaction_costs', {})
        self.initial_equity = config_data.get('initial_equity', 100000)
        self.timezone = config_data.get('timezone', 'US/Central')

        # Risk and position management configurations
        self.risk_metrics = config_data.get('risk_metrics', {})
        self.position_params = config_data.get('position_params', {})
        self.adaptive_params = config_data.get('adaptive_params', {})
        self.volatility_params = config_data.get('volatility_params', {})
        self.sharpe_params = config_data.get('sharpe_params', {})
        self.risk_limits = config_data.get('risk_limits', {})
        self.risk_params = config_data.get('risk_params', {'risk_manager_type': 'volatility'})

        # Analysis parameters
        self.symbol = config_data.get('symbol') # Main symbol for the backtest run
        self.days_to_analyze = config_data.get('days_to_analyze', 252)
        self.lookback_buffer = config_data.get('lookback_buffer', 63)

        # Validate and process configurations
        self._validate_parameters()
        self._process_trading_hours()
        # Initialize logging configuration with the updated base output path
        LoggingConfig(log_base_dir=self.base_output_dir / 'logs').setup()


    def get_risk_manager_config(self) -> Dict:
        """Get risk manager configuration."""
        return {
            'type': self.risk_params.get('risk_manager_type', 'volatility'),
            'volatility_params': self.volatility_params,
            'sharpe_params': self.sharpe_params,
            'adaptive_params': self.adaptive_params,
            'risk_limits': self.risk_limits,
            'combined_weights': self.risk_params.get('combined_weights', [0.4, 0.3, 0.3])
        }

    def _validate_parameters(self):
        """Validate critical configuration parameters."""
        # --- Data Source Validation ---
        if not self.data_source or 'db_path' not in self.data_source or 'table_name_format' not in self.data_source:
            raise ValueError("Data source configuration ('db_path', 'table_name_format') is missing or incomplete.")
        db_file = Path(self.data_source['db_path'])
        # Check if the directory exists, not necessarily the file itself, as DuckDB can create it.
        if not db_file.parent.is_dir():
             raise FileNotFoundError(f"Data source directory not found: {db_file.parent}")

        # --- Output Path Validation ---
        # Base output dir existence is checked in __init__

        # --- Other Validations ---
        if self.initial_equity <= 0:
            raise ValueError("Initial equity must be a positive number.")
        if self.symbol is None:
            raise ValueError("Trading symbol must be specified in configuration.")

        # Validate contract specifications (ensure the main symbol has specs)
        if self.symbol not in self.contract_specs:
             raise ValueError(f"Contract specifications missing for the main symbol '{self.symbol}'")
        for symbol, specs in self.contract_specs.items():
            required_specs = ['tick_size', 'multiplier', 'margin']
            missing_specs = [spec for spec in required_specs if spec not in specs]
            if missing_specs:
                raise ValueError(f"Contract specification(s) {missing_specs} missing for symbol '{symbol}'")

        # Validate trading hours (ensure the main symbol has hours)
        if self.symbol not in self.trading_hours:
             raise ValueError(f"Trading hours missing for the main symbol '{self.symbol}'")
        for symbol, hours in self.trading_hours.items():
            required_hours = ['market_open', 'market_close', 'last_entry']
            missing_hours = [hour for hour in required_hours if hour not in hours]
            if missing_hours:
                raise ValueError(f"Trading hour(s) {missing_hours} missing for symbol '{symbol}'")

        # Validate risk parameters
        if not self.risk_metrics:
            raise ValueError("Risk metrics configuration is missing")
        if not self.risk_limits:
            raise ValueError("Risk limits configuration is missing")

    def _process_trading_hours(self):
        """Convert trading hour strings to time objects and store timezone."""
        try:
            self.timezone_obj = pytz.timezone(self.timezone) # Store the tz object
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f"Invalid timezone specified: {self.timezone}")

        for symbol, hours in self.trading_hours.items():
            processed_hours = {}
            for key in ['market_open', 'market_close', 'last_entry']:
                time_val = hours.get(key) # Use get for safety
                if isinstance(time_val, str):
                    try:
                        processed_hours[key] = datetime.strptime(time_val, '%H:%M').time()
                    except ValueError:
                        raise ValueError(
                            f"Invalid time format for '{key}' ('{time_val}') in trading hours for symbol '{symbol}'. "
                            f"Expected format 'HH:MM'."
                        )
                elif isinstance(time_val, time):
                     processed_hours[key] = time_val # Already a time object
                else:
                     # Handle cases where the key might be missing or has wrong type
                     raise ValueError(f"Invalid or missing value for '{key}' in trading hours for symbol '{symbol}'.")

            self.trading_hours[symbol] = processed_hours

    def get_symbol_specs(self, symbol: str) -> Dict:
        """Get all specifications and data source details for a given symbol."""
        if symbol not in self.contract_specs:
            raise ValueError(f"No contract specifications found for symbol '{symbol}'")
        if symbol not in self.trading_hours:
            raise ValueError(f"No trading hours found for symbol '{symbol}'")

        # Generate symbol-specific output path
        symbol_output_path = self.base_output_dir / symbol
        symbol_output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists

        # Generate data source table name
        table_name = self.data_source.get('table_name_format', '{symbol}_1min').format(symbol=symbol)

        return {
            'contract_specs': self.contract_specs[symbol],
            'trading_hours': self.trading_hours[symbol],
            'data_source': { # Provide structured data source info
                'type': self.data_source.get('type', 'duckdb'),
                'db_path': self.data_source['db_path'],
                'table_name': table_name
            },
            'output_path': symbol_output_path # Provide the derived output path
        }


# --- Logging Configuration ---

class LoggingConfig:
    """Centralized logging configuration for the trading system."""
    _instance = None
    _initialized = False
    _loggers = set()

    def __new__(cls, *args, **kwargs): # Allow passing args to __init__ via __new__
        if cls._instance is None:
            cls._instance = super(LoggingConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_base_dir: Path = Path('logs')): # Accept base dir
        # Ensure initialization happens only once and uses the provided path
        if not self._initialized:
            # Use the provided log_base_dir or default to 'logs' if called without arg first time
            self.log_dir = log_base_dir
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure log directory exists (moved here from setup)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Create separate log files within the specified directory
            self.main_log = self.log_dir / f"trading_system_{self.timestamp}.log"
            self.trade_log = self.log_dir / f"trades_{self.timestamp}.log"
            self.debug_log = self.log_dir / f"debug_{self.timestamp}.log"

            self.config = self._load_config()
            self._initialized = True
            self._setup_complete = False # Flag to track if setup has run

    def _load_config(self) -> Dict:
        """Load logging configuration from yaml file or return default config."""
        try:
            # Look for config relative to workspace root standardly
            with open('logging_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return improved default config
            return {
                'log_levels': {
                    'root': 'INFO', # Keep root INFO by default
                    'DataManager': 'INFO', # Specific module levels
                    'RiskManager': 'INFO',
                    'Strategy': 'INFO',
                    'TradeManager': 'INFO',
                    'RiskMetrics': 'INFO',
                    # Add more specific modules as needed
                },
                'formatters': {
                    'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s', # More context
                    'simple': '[%(levelname)s] %(message)s',
                    'trade': '%(asctime)s - TRADE - %(message)s' # Simplified trade format
                },
                'rotation': {
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 7
                },
                'debug_level': 'DEBUG' # Overall level for debug file/handler
            }

    def setup(self) -> None:
        """Set up logging configuration with separate handlers for different log types."""
        if self._setup_complete: # Prevent re-running setup
            return

        # Configure root logger
        root_logger = logging.getLogger()
        # Set root logger level high initially, handlers control specifics
        root_logger.setLevel(logging.DEBUG) # Set to lowest level to allow all messages to handlers
        root_logger.handlers.clear() # Clear any existing handlers

        # Create formatters
        formatters = {
            'detailed': logging.Formatter(self.config['formatters']['detailed']),
            'simple': logging.Formatter(self.config['formatters']['simple']),
            'trade': logging.Formatter(self.config['formatters']['trade'])
        }

        # Determine levels from config
        root_level_str = self.config.get('log_levels', {}).get('root', 'INFO')
        debug_level_str = self.config.get('debug_level', 'DEBUG')
        root_level = logging.getLevelName(root_level_str.upper())
        debug_level = logging.getLevelName(debug_level_str.upper())

        # --- Handlers ---
        # Main log file handler (INFO and above by default, controlled by root_level)
        main_handler = logging.handlers.TimedRotatingFileHandler(
            self.main_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        main_handler.setFormatter(formatters['detailed'])
        main_handler.setLevel(root_level) # Use root level from config

        # Trade log file handler (INFO and above, specific format)
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            self.trade_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        trade_handler.setFormatter(formatters['trade'])
        # Filter specifically for trade-related messages if needed, or just log >= INFO
        # trade_handler.addFilter(lambda record: 'TRADE' in record.getMessage().upper()) # Example filter
        trade_handler.setLevel(logging.INFO) # Typically keep trades at INFO

        # Debug log file handler (DEBUG level by default)
        debug_handler = logging.handlers.TimedRotatingFileHandler(
            self.debug_log,
            when=self.config['rotation']['when'],
            interval=self.config['rotation']['interval'],
            backupCount=self.config['rotation']['backupCount']
        )
        debug_handler.setFormatter(formatters['detailed'])
        debug_handler.setLevel(debug_level) # Use debug level from config

        # Console handler (INFO and above by default)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatters['simple'])
        console_handler.setLevel(root_level) # Match main log level for console
        # Optional: Filter console output further if needed
        # console_handler.addFilter(...)

        # Add handlers to root logger
        root_logger.addHandler(main_handler)
        root_logger.addHandler(trade_handler)
        root_logger.addHandler(debug_handler)
        root_logger.addHandler(console_handler)

        self._setup_complete = True
        logging.getLogger(__name__).info(f"Logging setup complete. Log directory: {self.log_dir}")


    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a module-level logger with proper configuration.
        Ensures setup is called if needed.
        """
        # Initialize singleton if not already done (uses default path if first call)
        if not cls._initialized:
            cls() # Initialize if not already done

        # Ensure setup is called (idempotent)
        cls._instance.setup()

        logger = logging.getLogger(name)

        # Set level from config if available for this specific logger's name
        # Use the module name part for lookup in config
        module_key = name.split('.')[-1] # e.g., 'DataManager' from 'Test_Framework.Data_Manager.manager'
        log_levels = cls._instance.config.get('log_levels', {})

        if module_key in log_levels:
            logger.setLevel(log_levels[module_key].upper())
            # print(f"Setting level for {name} ({module_key}) to {log_levels[module_key].upper()}") # Debug print
        # else: # If not specifically set, it inherits from the root logger's level
            # print(f"Logger {name} ({module_key}) inherits level from root.") # Debug print

        # No need to add handlers here, they are attached to the root logger

        if name not in cls._loggers:
             # Optional: Log first time a logger is requested
             # logger.debug(f"Logger '{name}' accessed for the first time.")
             cls._loggers.add(name)


        return logger 