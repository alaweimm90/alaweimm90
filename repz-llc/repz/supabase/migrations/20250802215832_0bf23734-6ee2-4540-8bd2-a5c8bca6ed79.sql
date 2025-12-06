-- Phase 3B Continued & Phase 3C: Final cleanup
-- (Fixed index creation - cannot use CONCURRENTLY in migration)

-- Phase 3B: Remove redundant workout and subscription tables
DROP TABLE IF EXISTS workout_sessions CASCADE;
DROP TABLE IF EXISTS workout_logs CASCADE;
DROP TABLE IF EXISTS user_subscriptions CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;

-- Phase 3C: Remove development and testing tables
DROP TABLE IF EXISTS qa_test_runs CASCADE;
DROP TABLE IF EXISTS qa_test_results CASCADE;
DROP TABLE IF EXISTS tier_migration_audit CASCADE;

-- Check if tier_features table is actually used (keep if it has real data)
DO $$
DECLARE
    row_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO row_count FROM tier_features;
    IF row_count <= 1 THEN
        DROP TABLE IF EXISTS tier_features CASCADE;
    END IF;
EXCEPTION
    WHEN undefined_table THEN
        -- Table doesn't exist, continue
        NULL;
END $$;

-- Check and remove secure_sessions if it's not actively used
DO $$
DECLARE
    row_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO row_count FROM secure_sessions;
    IF row_count <= 1 THEN
        DROP TABLE IF EXISTS secure_sessions CASCADE;
    END IF;
EXCEPTION
    WHEN undefined_table THEN
        -- Table doesn't exist, continue
        NULL;
END $$;

-- Remove any other test/demo tables that might exist
DROP TABLE IF EXISTS demo_users CASCADE;
DROP TABLE IF EXISTS test_profiles CASCADE;
DROP TABLE IF EXISTS legacy_tiers CASCADE;
DROP TABLE IF EXISTS old_subscription_data CASCADE;
DROP TABLE IF EXISTS temp_migration_table CASCADE;

-- Add performance indexes for remaining consolidated tables
CREATE INDEX IF NOT EXISTS idx_client_profiles_tier 
ON client_profiles(subscription_tier);

CREATE INDEX IF NOT EXISTS idx_client_profiles_coach 
ON client_profiles(coach_id) WHERE coach_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_client_profiles_phone 
ON client_profiles(phone) WHERE phone IS NOT NULL;