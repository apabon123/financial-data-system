-- ========================================================
-- DATABASE MAINTENANCE QUERIES
-- ========================================================
-- This file contains queries for maintaining and monitoring
-- the database health, including table statistics, space usage,
-- and performance metrics.
-- ========================================================

-- 1.1 Table Statistics
-- ------------------------------------------------------------
-- Get basic statistics about all tables
SELECT 
    table_name,
    (SELECT COUNT(*) FROM (SELECT 1 FROM {table_name})) as row_count,
    (SELECT SUM(pg_column_size(column_name)) 
     FROM information_schema.columns 
     WHERE table_name = '{table_name}') as estimated_size_bytes
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY estimated_size_bytes DESC;

-- 1.2 Index Usage Statistics
-- ------------------------------------------------------------
-- Check how often indexes are being used
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 1.3 Table Bloat Analysis
-- ------------------------------------------------------------
-- Identify tables that might need vacuuming
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - 
                  pg_relation_size(schemaname || '.' || tablename)) as index_size,
    n_dead_tup as dead_tuples,
    n_live_tup as live_tuples
FROM pg_stat_user_tables
ORDER BY dead_tuples DESC;

-- 1.4 Long-Running Queries
-- ------------------------------------------------------------
-- Find queries that have been running for a long time
SELECT 
    pid,
    now() - query_start as duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- 1.5 Table Growth Rate
-- ------------------------------------------------------------
-- Calculate the growth rate of tables over time
WITH table_sizes AS (
    SELECT 
        table_name,
        pg_total_relation_size(table_name) as size_bytes,
        now() as check_time
    FROM information_schema.tables
    WHERE table_schema = 'public'
)
SELECT 
    table_name,
    size_bytes,
    check_time,
    LAG(size_bytes) OVER (PARTITION BY table_name ORDER BY check_time) as prev_size_bytes,
    (size_bytes - LAG(size_bytes) OVER (PARTITION BY table_name ORDER BY check_time))::float / 
    NULLIF(LAG(size_bytes) OVER (PARTITION BY table_name ORDER BY check_time), 0) * 100 as growth_percent
FROM table_sizes
ORDER BY size_bytes DESC;

-- 1.6 Database Size History
-- ------------------------------------------------------------
-- Track database size over time
SELECT 
    date_trunc('day', check_time) as check_date,
    sum(size_bytes) as total_size_bytes,
    pg_size_pretty(sum(size_bytes)) as total_size
FROM (
    SELECT 
        table_name,
        pg_total_relation_size(table_name) as size_bytes,
        now() as check_time
    FROM information_schema.tables
    WHERE table_schema = 'public'
) t
GROUP BY check_date
ORDER BY check_date;

-- 1.7 Table Access Patterns
-- ------------------------------------------------------------
-- Analyze how often tables are being accessed
SELECT 
    schemaname,
    relname as table_name,
    seq_scan as sequential_scans,
    seq_tup_read as tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as tuples_fetched,
    n_tup_ins as rows_inserted,
    n_tup_upd as rows_updated,
    n_tup_del as rows_deleted
FROM pg_stat_user_tables
ORDER BY (seq_scan + idx_scan) DESC;

-- 1.8 Index Size Analysis
-- ------------------------------------------------------------
-- Analyze index sizes and their contribution to total table size
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) as index_size,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_table_size,
    (pg_relation_size(schemaname || '.' || indexname)::float / 
     pg_total_relation_size(schemaname || '.' || tablename) * 100) as index_size_percent
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(schemaname || '.' || indexname) DESC;

-- 1.9 Table Fragmentation
-- ------------------------------------------------------------
-- Check for table fragmentation
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
    CASE 
        WHEN pg_relation_size(schemaname || '.' || tablename) = 0 THEN 0
        ELSE (pg_total_relation_size(schemaname || '.' || tablename) - 
              pg_relation_size(schemaname || '.' || tablename))::float / 
             pg_relation_size(schemaname || '.' || tablename) * 100
    END as fragmentation_percent
FROM pg_stat_user_tables
WHERE pg_relation_size(schemaname || '.' || tablename) > 0
ORDER BY fragmentation_percent DESC;

-- 1.10 Database Maintenance Recommendations
-- ------------------------------------------------------------
-- Generate maintenance recommendations based on various metrics
WITH table_stats AS (
    SELECT 
        schemaname,
        tablename,
        n_dead_tup as dead_tuples,
        n_live_tup as live_tuples,
        last_vacuum,
        last_autovacuum,
        last_analyze,
        last_autoanalyze
    FROM pg_stat_user_tables
)
SELECT 
    tablename,
    CASE 
        WHEN dead_tuples > live_tuples * 0.2 THEN 'VACUUM recommended - high dead tuple ratio'
        WHEN last_vacuum IS NULL AND last_autovacuum IS NULL THEN 'VACUUM recommended - never vacuumed'
        WHEN last_vacuum < now() - interval '7 days' THEN 'VACUUM recommended - not vacuumed in 7 days'
        ELSE 'No VACUUM needed'
    END as vacuum_recommendation,
    CASE 
        WHEN last_analyze IS NULL AND last_autoanalyze IS NULL THEN 'ANALYZE recommended - never analyzed'
        WHEN last_analyze < now() - interval '7 days' THEN 'ANALYZE recommended - not analyzed in 7 days'
        ELSE 'No ANALYZE needed'
    END as analyze_recommendation
FROM table_stats
WHERE dead_tuples > live_tuples * 0.2
   OR last_vacuum IS NULL 
   OR last_vacuum < now() - interval '7 days'
   OR last_analyze IS NULL
   OR last_analyze < now() - interval '7 days'
ORDER BY dead_tuples DESC; 