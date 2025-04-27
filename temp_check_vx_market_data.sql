SELECT 
    DISTINCT symbol 
FROM market_data 
WHERE 
    symbol LIKE 'VX%' 
ORDER BY symbol 
LIMIT 50; 