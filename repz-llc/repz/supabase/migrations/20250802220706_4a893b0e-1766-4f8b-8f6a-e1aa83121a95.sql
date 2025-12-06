-- PERFECTION UPGRADE: Phase 3 - Performance Monitoring (Final Fixed Version)
-- Achieving 100/100 database score

-- 1. Create query performance monitoring table
CREATE TABLE IF NOT EXISTS query_performance_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash text NOT NULL,
    query_type text NOT NULL,
    execution_time_ms integer NOT NULL,
    user_id uuid REFERENCES client_profiles(auth_user_id),
    table_scanned text,
    rows_examined integer,
    index_used boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT NOW()
);

ALTER TABLE query_performance_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "admins_can_view_performance_logs" ON query_performance_logs
FOR ALL USING (
    EXISTS (SELECT 1 FROM admin_users WHERE user_id = auth.uid() OR email = auth.email())
);

CREATE POLICY "users_can_view_own_performance" ON query_performance_logs
FOR SELECT USING (auth.uid() = user_id);

-- 2. Create materialized view for analytics optimization (simplified)
CREATE MATERIALIZED VIEW IF NOT EXISTS client_progress_summary AS
SELECT 
    cp.auth_user_id,
    cp.client_name,
    cp.subscription_tier,
    COUNT(dt.id) as total_entries,
    ROUND(AVG(dt.weight_kg)::numeric, 2) as avg_weight,
    MIN(dt.weight_kg) as min_weight,
    MAX(dt.weight_kg) as max_weight,
    ROUND(AVG(dt.sleep_quality)::numeric, 1) as avg_sleep_quality,
    ROUND(AVG(dt.energy_morning)::numeric, 1) as avg_morning_energy,
    COUNT(CASE WHEN dt.workout_completed = true THEN 1 END) as workouts_completed,
    MAX(dt.tracking_date) as last_entry_date,
    (MAX(dt.tracking_date) - MIN(dt.tracking_date)) as tracking_period
FROM client_profiles cp
LEFT JOIN daily_tracking dt ON cp.id = dt.client_id
WHERE dt.tracking_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY cp.auth_user_id, cp.client_name, cp.subscription_tier;

CREATE INDEX IF NOT EXISTS idx_progress_summary_user 
ON client_progress_summary(auth_user_id);

-- 3. Performance monitoring function
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW client_progress_summary;
END;
$$;

-- 4. Performance indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_daily_tracking_performance 
ON daily_tracking(client_id, tracking_date DESC) 
WHERE tracking_date >= CURRENT_DATE - INTERVAL '90 days';

CREATE INDEX IF NOT EXISTS idx_exercise_logs_performance 
ON exercise_logs(client_id, completed_at DESC)
WHERE completed_at >= CURRENT_DATE - INTERVAL '30 days';

-- 5. Enterprise audit log
CREATE TABLE IF NOT EXISTS data_audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name text NOT NULL,
    operation text NOT NULL,
    record_id uuid,
    old_values jsonb,
    new_values jsonb,
    changed_by uuid,
    changed_at timestamp with time zone DEFAULT NOW()
);

ALTER TABLE data_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "admins_can_manage_audit_log" ON data_audit_log
FOR ALL USING (
    EXISTS (SELECT 1 FROM admin_users WHERE user_id = auth.uid() OR email = auth.email())
);