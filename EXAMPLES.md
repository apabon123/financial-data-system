# Usage Examples

## Example 1: Fetch Daily Data for Specific Symbols
\`\`\`python
# Should be able to run commands like:
fetch_market_data(symbols=[\"AAPL\", \"MSFT\", \"GOOGL\"], 
                  timeframe=\"daily\", 
                  start_date=\"2023-01-01\", 
                  end_date=\"2023-12-31\")
\`\`\`

## Example 2: Update Database with Latest Data
\`\`\`python
# Daily update process
update_all_data(date=\"latest\")
\`\`\`

## Example 3: Query Stored Data
\`\`\`python
# Get OHLC data for analysis
query_data(symbols=[\"AAPL\"], 
           timeframe=\"daily\", 
           start_date=\"2023-01-01\", 
           end_date=\"2023-12-31\")
\`\`\`