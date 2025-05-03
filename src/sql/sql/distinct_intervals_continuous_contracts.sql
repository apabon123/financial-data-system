-- List distinct symbol, interval_value, and interval_unit combinations
SELECT DISTINCT 
    symbol, 
    interval_value, 
    interval_unit 
FROM continuous_contracts
ORDER BY symbol, interval_value, interval_unit; 