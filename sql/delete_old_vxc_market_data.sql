-- Delete old VXc continuous contract data mistakenly inserted into market_data
DELETE FROM market_data
WHERE symbol LIKE 'VXc%'; 