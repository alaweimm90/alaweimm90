-- PERFECTION UPGRADE: Phase 3 - Performance Monitoring (+1 point)
-- Query performance tracking and database monitoring

-- 1. Enhanced query performance monitoring table
CREATE TABLE IF NOT EXISTS query_performance_monitor (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash text NOT NULL,
    query_type text NOT NULL, -- SELECT, INSERT, UPDATE, DELETE
    table_name text,
    execution_time_ms integer NOT NULL,
    rows_affected integer,
    user_id uuid,
    client_ip inet,
    query_size_bytes integer,
    cache_hit boolean DEFAULT false,
    logged_at timestamp with time zone DEFAULT NOW()
);

-- Enable RLS on performance monitoring
ALTER TABLE query_performance_monitor ENABLE ROW LEVEL SECURITY;

-- 2. Admin-only access to performance data
CREATE POLICY "admins_access_performance_monitor" ON query_performance_monitor
FOR ALL USING (is_admin());

-- 3. Function to log slow queries automatically
CREATE OR REPLACE FUNCTION log_slow_query(
    p_query_hash text,
    p_query_type text,
    p_table_name text,
    p_execution_time_ms integer,
    p_rows_affected integer DEFAULT NULL
) RETURNS void AS $$
BEGIN
    -- Only log queries that take longer than 100ms
    IF p_execution_time_ms > 100 THEN
        INSERT INTO query_performance_monitor (
            query_hash,
            query_type,
            table_name,
            execution_time_ms,
            rows_affected,
            user_id,
            client_ip
        ) VALUES (
            p_query_hash,
            p_query_type,
            p_table_name,
            p_execution_time_ms,
            p_rows_affected,
            auth.uid(),
            inet_client_addr()
        );
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 4. Database health monitoring view
CREATE OR REPLACE VIEW database_health_summary AS
SELECT 
    'active_connections' AS metric,
    (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active')::text AS value,
    NOW() AS checked_at
UNION ALL
SELECT 
    'slow_queries_last_hour' AS metric,
    (SELECT COUNT(*) FROM query_performance_monitor 
     WHERE logged_at > NOW() - INTERVAL '1 hour' 
     AND execution_time_ms > 1000)::text AS value,
    NOW() AS checked_at
UNION ALL
SELECT 
    'cache_hit_ratio' AS metric,
    ROUND(
        (SELECT SUM(blks_hit)::float / (SUM(blks_hit) + SUM(blks_read)) * 100
         FROM pg_stat_database 
         WHERE datname = current_database()), 2
    )::text || '%' AS value,
    NOW() AS checked_at
UNION ALL
SELECT 
    'table_count' AS metric,
    (SELECT COUNT(*) FROM information_schema.tables 
     WHERE table_schema = 'public')::text AS value,
    NOW() AS checked_at;

-- 5. Index usage monitoring
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_tup_read = 0 THEN 'UNUSED'
        WHEN idx_tup_read < 1000 THEN 'LOW_USAGE'
        ELSE 'ACTIVE'
    END as usage_status
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;

-- 6. Performance optimization recommendations function
CREATE OR REPLACE FUNCTION get_performance_recommendations()
RETURNS TABLE(
    recommendation_type text,
    priority text,
    description text,
    table_affected text
) AS $$
BEGIN
    -- Check for tables without primary keys
    RETURN QUERY
    SELECT 
        'MISSING_PRIMARY_KEY'::text,
        'HIGH'::text,
        'Table missing primary key constraint'::text,
        t.table_name
    FROM information_schema.tables t
    LEFT JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name
    WHERE t.table_schema = 'public' 
    AND t.table_type = 'BASE TABLE'
    AND tc.constraint_type IS DISTINCT FROM 'PRIMARY KEY';
    
    -- Check for unused indexes
    RETURN QUERY
    SELECT 
        'UNUSED_INDEX'::text,
        'MEDIUM'::text,
        'Index not being used - consider dropping'::text,
        indexname
    FROM pg_stat_user_indexes
    WHERE idx_tup_read = 0 AND idx_tup_fetch = 0;
    
    -- Check for tables without updated_at triggers
    RETURN QUERY
    SELECT 
        'MISSING_UPDATED_AT_TRIGGER'::text,
        'LOW'::text,
        'Consider adding updated_at trigger for audit trail'::text,
        table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    AND table_name NOT IN (
        SELECT DISTINCT event_object_table 
        FROM information_schema.triggers 
        WHERE trigger_name LIKE '%updated_at%'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;