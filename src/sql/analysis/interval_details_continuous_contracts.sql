-- Get details (count, date range) for specific symbols and a specific interval
-- Modify the WHERE clause to specify symbols and interval as needed.
SELECT 
    symbol, 
    interval_value, 
    interval_unit, 
    COUNT(*) as count, 
    MIN(timestamp) as min_date, 
    MAX(timestamp) as max_date 
FROM continuous_contracts 
WHERE 
    symbol IN ('@ES=102XC', '@ES=102XN', '@NQ=102XC', '@NQ=102XN') -- Example symbols
    AND interval_value = 15 
    AND interval_unit = 'minute' 
GROUP BY symbol, interval_value, interval_unit
ORDER BY symbol; 