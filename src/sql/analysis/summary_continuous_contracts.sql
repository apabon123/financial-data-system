-- Get a summary of all data in the continuous_contracts table
SELECT 
    symbol, 
    interval_value, 
    interval_unit, 
    adjusted, 
    MIN(timestamp) as first_date, 
    MAX(timestamp) as last_date, 
    COUNT(*) as count 
FROM continuous_contracts 
GROUP BY symbol, interval_value, interval_unit, adjusted
ORDER BY symbol, interval_value, interval_unit; 