# DuckDB Schema

## Main Tables

### market_data
- timestamp (TIMESTAMP) - Primary time index
- symbol (VARCHAR) - Trading symbol
- open (DOUBLE) - Opening price
- high (DOUBLE) - High price
- low (DOUBLE) - Low price
- close (DOUBLE) - Closing price
- volume (BIGINT) - Trading volume
- up_volume (BIGINT) - Optional up volume
- down_volume (BIGINT) - Optional down volume
- source (VARCHAR) - Data source identifier
- interval_value (INTEGER) - Interval value (e.g., 1, 5, 15)
- interval_unit (VARCHAR) - Interval unit (minute, daily, etc.)
- adjusted (BOOLEAN) - Whether prices are adjusted for splits/dividends
- quality (INTEGER) - Data quality/reliability indicator (0-100)

### economic_data
- timestamp (TIMESTAMP) - Primary time index
- indicator (VARCHAR) - Economic indicator name
- value (DOUBLE) - Indicator value
- source (VARCHAR) - Data source
- frequency (VARCHAR) - Data frequency (daily, weekly, monthly, quarterly)
- revision_number (INTEGER) - Revision version of the data point

### symbols
- symbol_id (VARCHAR) - Primary key
- symbol (VARCHAR) - Trading symbol 
- name (VARCHAR) - Full name of the security
- sector (VARCHAR) - Industry sector
- type (VARCHAR) - Security type (stock, ETF, option, etc.)
- active (BOOLEAN) - Whether the symbol is actively traded
- exchange (VARCHAR) - Primary exchange
- currency (VARCHAR) - Trading currency
- added_date (TIMESTAMP) - When the symbol was added to the database
- last_updated (TIMESTAMP) - When the symbol data was last updated

### data_sources
- source_id (VARCHAR) - Primary key
- name (VARCHAR) - Source name
- type (VARCHAR) - Source type (API, web scraping, etc.)
- last_updated (TIMESTAMP) - When data was last fetched
- status (VARCHAR) - Current status of the source
- priority (INTEGER) - Priority when multiple sources exist
- api_key_reference (VARCHAR) - Reference to API key in environment (not the actual key)
- rate_limit (INTEGER) - API rate limit information

### derived_indicators
- timestamp (TIMESTAMP) - Time index
- symbol (VARCHAR) - Trading symbol
- indicator_name (VARCHAR) - Name of the indicator (RSI, MACD, etc.)
- value (DOUBLE) - Calculated value
- parameters (JSON) - Parameters used in calculation
- interval_value (INTEGER) - Interval value for the indicator
- interval_unit (VARCHAR) - Interval unit for the indicator
- created_at (TIMESTAMP) - When the indicator was calculated

### metadata
- key (VARCHAR) - Setting name
- value (VARCHAR) - Setting value
- updated_at (TIMESTAMP) - Last update time
- description (VARCHAR) - Description of the setting

## Relations

### symbol_tags
- symbol_id (VARCHAR) - Reference to symbols table
- tag (VARCHAR) - Tag name
- added_at (TIMESTAMP) - When the tag was added

### indicator_metadata
- indicator_name (VARCHAR) - Name of economic indicator
- display_name (VARCHAR) - Human-readable name
- description (TEXT) - Description of the indicator
- unit (VARCHAR) - Unit of measurement
- source (VARCHAR) - Primary source

## Account Tables

### account_balances
- timestamp (TIMESTAMP) - When the balance data was recorded
- account_id (VARCHAR) - TradeStation account identifier
- cash_balance (DOUBLE) - Total cash balance
- buying_power (DOUBLE) - Available buying power
- day_trading_buying_power (DOUBLE) - Day trading buying power
- equity (DOUBLE) - Account equity value
- margin_balance (DOUBLE) - Margin account balance
- real_time_buying_power (DOUBLE) - Real-time buying power
- real_time_equity (DOUBLE) - Real-time account equity
- real_time_cost_of_positions (DOUBLE) - Real-time cost of all positions
- day_trades_count (INTEGER) - Number of day trades in the period
- day_trading_qualified (BOOLEAN) - Whether account is qualified for day trading
- source (VARCHAR) - Data source identifier
- currency (VARCHAR) - Account base currency

### positions
- timestamp (TIMESTAMP) - When the position data was recorded
- account_id (VARCHAR) - TradeStation account identifier
- symbol (VARCHAR) - Trading symbol
- quantity (DOUBLE) - Position quantity (negative for short positions)
- average_price (DOUBLE) - Average price of the position
- market_value (DOUBLE) - Current market value of the position
- cost_basis (DOUBLE) - Total cost of acquiring the position
- open_pl (DOUBLE) - Unrealized profit/loss
- open_pl_percent (DOUBLE) - Unrealized profit/loss percentage
- day_pl (DOUBLE) - Profit/loss for the day
- initial_margin (DOUBLE) - Initial margin requirement
- maintenance_margin (DOUBLE) - Maintenance margin requirement
- position_id (VARCHAR) - Unique position identifier
- source (VARCHAR) - Data source identifier

### orders
- timestamp (TIMESTAMP) - When the order was placed
- account_id (VARCHAR) - TradeStation account identifier
- order_id (VARCHAR) - Unique order identifier
- symbol (VARCHAR) - Trading symbol
- quantity (DOUBLE) - Order quantity
- order_type (VARCHAR) - Type of order (Market, Limit, Stop, etc.)
- side (VARCHAR) - Buy or Sell
- status (VARCHAR) - Status of the order (Open, Filled, Cancelled, etc.)
- limit_price (DOUBLE) - Limit price if applicable
- stop_price (DOUBLE) - Stop price if applicable
- filled_quantity (DOUBLE) - Quantity that has been filled
- remaining_quantity (DOUBLE) - Quantity remaining to be filled
- average_fill_price (DOUBLE) - Average price of fills
- duration (VARCHAR) - Time in force (Day, GTC, etc.)
- route (VARCHAR) - Order routing destination
- execution_time (TIMESTAMP) - When the order was executed
- cancellation_time (TIMESTAMP) - When the order was cancelled
- source (VARCHAR) - Data source identifier

### trades
- timestamp (TIMESTAMP) - When the trade occurred
- account_id (VARCHAR) - TradeStation account identifier
- order_id (VARCHAR) - Reference to the order
- trade_id (VARCHAR) - Unique trade identifier
- symbol (VARCHAR) - Trading symbol
- quantity (DOUBLE) - Trade quantity
- price (DOUBLE) - Trade price
- side (VARCHAR) - Buy or Sell
- commission (DOUBLE) - Commission paid
- fees (DOUBLE) - Additional fees
- trade_time (TIMESTAMP) - Exact time of trade execution
- position_effect (VARCHAR) - Opening, Closing, or Offsetting
- source (VARCHAR) - Data source identifier

## Views

### daily_bars
Aggregates market data to daily timeframes, filtering for daily data or data with an interval value of 1440 minutes.

### minute_bars
Filters market data for records with an interval unit of 'minute'.

### five_minute_bars
Aggregates 1-minute data into 5-minute bars with appropriate OHLCV calculations.

### latest_prices
Shows the most recent price for each symbol using window functions.

### weekly_bars
Aggregates daily data into weekly bars with appropriate OHLCV calculations.

### monthly_bars
Aggregates daily data into monthly bars with appropriate OHLCV calculations.

### economic_calendar
Joins economic data with indicator metadata for a comprehensive view of economic indicators.

### account_summary
Provides an overview of account performance including position metrics and balance information.

### active_positions
Shows only the most recent positions that have non-zero quantity.

### open_orders
Filters for orders with status 'Open' or 'PartiallyFilled'.

### symbol_list
Simple view that returns a distinct list of symbols from the market_data table.

## Indexes

- (timestamp, symbol, interval_value, interval_unit) on market_data
- (timestamp, indicator) on economic_data
- (symbol) on symbols
- (indicator_name, timestamp) on derived_indicators
- (timestamp, account_id) on account_balances
- (timestamp, account_id, symbol) on positions
- (order_id) on orders
- (account_id, status) on orders
- (symbol, status) on orders
- (trade_id) on trades
- (account_id, symbol) on trades

## Notes

The complete SQL implementation for all tables, views, and indexes can be found in the `init_schema.sql` file, which should be used for initializing the database in Claude Code.
