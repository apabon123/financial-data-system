-- Update source field for continuous contracts with _d suffix
UPDATE continuous_contracts
SET source = 'inhouse_built'
WHERE symbol LIKE '%_d'
AND source = 'tradestation';

-- Log the number of records updated
SELECT COUNT(*) as records_updated
FROM continuous_contracts
WHERE symbol LIKE '%_d'
AND source = 'inhouse_built';

-- Delete @ES=102XN_test daily tradestation record
DELETE FROM continuous_contracts
WHERE symbol = '@ES=102XN_test'
  AND interval_value = 1
  AND interval_unit = 'daily'
  AND source = 'tradestation'; 