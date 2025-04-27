-- Standardize interval_unit to 'daily'
UPDATE market_data
SET interval_unit = 'daily'
WHERE interval_unit = 'day';      