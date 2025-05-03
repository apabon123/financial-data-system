-- ========================================================
-- VX CONTINUOUS CONTRACT ROLLOVER VERIFICATION
-- ========================================================
-- This query verifies the rollover logic for VX continuous contracts
-- by comparing VXc1 data with the underlying futures contracts
-- that should be used for each day.
-- ========================================================

-- 1. First Month Verification (March 26, 2004 - April 30, 2004)
-- ------------------------------------------------------------
-- This query shows the first month of VXc1 data and the underlying
-- futures contracts that should be used for each day
WITH vx_contracts AS (
    -- Get all VX futures contracts available in the database
    SELECT DISTINCT
        symbol,
        MIN(timestamp) as first_date,
        MAX(timestamp) as last_date
    FROM market_data
    WHERE symbol LIKE 'VX%'
    AND symbol NOT LIKE 'VXc%'  -- Exclude continuous contracts
    AND interval_value = 1
    AND interval_unit = 'day'
    GROUP BY symbol
),
vxc1_data AS (
    -- Get VXc1 data for the first month
    SELECT
        timestamp as date,
        open,
        high,
        low,
        close,
        volume
    FROM continuous_contracts
    WHERE symbol = 'VXc1'
    AND timestamp BETWEEN '2004-03-26' AND '2004-04-30'
    ORDER BY timestamp
),
-- Determine which contract should be active on each day
-- VX futures expire on the third Wednesday of the month
active_contracts AS (
    SELECT
        date,
        -- Find the contract that should be active on this date
        -- For simplicity, we'll use the contract with the earliest expiry date
        -- that includes this date
        (
            SELECT symbol
            FROM vx_contracts
            WHERE first_date <= date
            AND last_date >= date
            ORDER BY last_date ASC
            LIMIT 1
        ) as active_contract
    FROM (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE timestamp BETWEEN '2004-03-26' AND '2004-04-30'
        AND interval_value = 1
        AND interval_unit = 'day'
    ) dates
)
-- Join VXc1 data with the active contract information
SELECT
    v.date,
    v.open as vxc1_open,
    v.high as vxc1_high,
    v.low as vxc1_low,
    v.close as vxc1_close,
    v.volume as vxc1_volume,
    a.active_contract,
    -- Get the price data for the active contract
    (
        SELECT close
        FROM market_data
        WHERE symbol = a.active_contract
        AND timestamp = v.date
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as contract_close,
    -- Check if this is a rollover day (when the active contract changes)
    CASE
        WHEN LAG(a.active_contract) OVER (ORDER BY v.date) != a.active_contract 
             AND LAG(a.active_contract) OVER (ORDER BY v.date) IS NOT NULL
        THEN 'ROLLOVER'
        ELSE 'NORMAL'
    END as day_type
FROM vxc1_data v
JOIN active_contracts a ON v.date = a.date
ORDER BY v.date;

-- 2. Rollover Day Verification
-- ------------------------------------------------------------
-- This query specifically examines rollover days to ensure
-- the continuous contract is correctly using the next contract's data
WITH vx_contracts AS (
    -- Get all VX futures contracts available in the database
    SELECT DISTINCT
        symbol,
        MIN(timestamp) as first_date,
        MAX(timestamp) as last_date
    FROM market_data
    WHERE symbol LIKE 'VX%'
    AND symbol NOT LIKE 'VXc%'  -- Exclude continuous contracts
    AND interval_value = 1
    AND interval_unit = 'day'
    GROUP BY symbol
),
-- Determine which contract should be active on each day
active_contracts AS (
    SELECT
        date,
        -- Find the contract that should be active on this date
        (
            SELECT symbol
            FROM vx_contracts
            WHERE first_date <= date
            AND last_date >= date
            ORDER BY last_date ASC
            LIMIT 1
        ) as active_contract
    FROM (
        SELECT DISTINCT timestamp as date
        FROM market_data
        WHERE timestamp BETWEEN '2004-03-26' AND '2004-12-31'
        AND interval_value = 1
        AND interval_unit = 'day'
    ) dates
),
-- Identify rollover days
rollover_days AS (
    SELECT
        date,
        active_contract,
        LAG(active_contract) OVER (ORDER BY date) as prev_contract
    FROM active_contracts
    WHERE LAG(active_contract) OVER (ORDER BY date) != active_contract
    AND LAG(active_contract) OVER (ORDER BY date) IS NOT NULL
)
-- Examine data around rollover days
SELECT
    r.date as rollover_date,
    r.prev_contract,
    r.active_contract,
    -- VXc1 data on rollover day
    (
        SELECT close
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        AND timestamp = r.date
        LIMIT 1
    ) as vxc1_close,
    -- Previous contract data on rollover day
    (
        SELECT close
        FROM market_data
        WHERE symbol = r.prev_contract
        AND timestamp = r.date
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as prev_contract_close,
    -- New contract data on rollover day
    (
        SELECT close
        FROM market_data
        WHERE symbol = r.active_contract
        AND timestamp = r.date
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as new_contract_close,
    -- VXc1 data on day before rollover
    (
        SELECT close
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        AND timestamp = r.date - interval '1 day'
        LIMIT 1
    ) as vxc1_prev_close,
    -- Previous contract data on day before rollover
    (
        SELECT close
        FROM market_data
        WHERE symbol = r.prev_contract
        AND timestamp = r.date - interval '1 day'
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as prev_contract_prev_close,
    -- VXc1 data on day after rollover
    (
        SELECT close
        FROM continuous_contracts
        WHERE symbol = 'VXc1'
        AND timestamp = r.date + interval '1 day'
        LIMIT 1
    ) as vxc1_next_close,
    -- New contract data on day after rollover
    (
        SELECT close
        FROM market_data
        WHERE symbol = r.active_contract
        AND timestamp = r.date + interval '1 day'
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as new_contract_next_close
FROM rollover_days r
ORDER BY r.date;

-- 3. Continuous Contract Consistency Check
-- ------------------------------------------------------------
-- This query checks if VXc1 data is consistent with the
-- underlying futures contracts it should be using
WITH vx_contracts AS (
    -- Get all VX futures contracts available in the database
    SELECT DISTINCT
        symbol,
        MIN(timestamp) as first_date,
        MAX(timestamp) as last_date
    FROM market_data
    WHERE symbol LIKE 'VX%'
    AND symbol NOT LIKE 'VXc%'  -- Exclude continuous contracts
    AND interval_value = 1
    AND interval_unit = 'day'
    GROUP BY symbol
),
active_contracts AS (
    SELECT
        timestamp as date,
        -- Find the contract that should be active on this date
        (
            SELECT symbol
            FROM vx_contracts
            WHERE first_date <= timestamp
            AND last_date >= timestamp
            ORDER BY last_date ASC
            LIMIT 1
        ) as active_contract
    FROM market_data
    WHERE interval_value = 1
    AND interval_unit = 'day'
    GROUP BY timestamp
)
SELECT
    c.timestamp as date,
    c.symbol as continuous_contract,
    a.active_contract,
    c.close as continuous_close,
    -- Get the price data for the active contract
    (
        SELECT close
        FROM market_data
        WHERE symbol = a.active_contract
        AND timestamp = c.timestamp
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    ) as contract_close,
    -- Calculate the difference between continuous and active contract
    ABS(c.close - (
        SELECT close
        FROM market_data
        WHERE symbol = a.active_contract
        AND timestamp = c.timestamp
        AND interval_value = 1
        AND interval_unit = 'day'
        LIMIT 1
    )) as price_diff,
    -- Flag significant differences (>1% of continuous price)
    CASE
        WHEN ABS(c.close - (
            SELECT close
            FROM market_data
            WHERE symbol = a.active_contract
            AND timestamp = c.timestamp
            AND interval_value = 1
            AND interval_unit = 'day'
            LIMIT 1
        )) > c.close * 0.01 THEN 'WARNING'
        ELSE 'OK'
    END as consistency_check
FROM continuous_contracts c
JOIN active_contracts a ON c.timestamp = a.date
WHERE c.symbol = 'VXc1'
ORDER BY c.timestamp;