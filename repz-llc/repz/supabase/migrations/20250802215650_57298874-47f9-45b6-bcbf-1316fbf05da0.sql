-- Phase 3B: Table Consolidation (Medium Priority)
-- Consolidating redundant and overlapping tables

-- 1. Migrate data from user_profiles to client_profiles (if any exists)
-- First check if user_profiles has data not in client_profiles
INSERT INTO client_profiles (auth_user_id, client_name, created_at, updated_at)
SELECT 
    user_id,
    COALESCE(display_name, email, 'User') as client_name,
    COALESCE(created_at, NOW()) as created_at,
    COALESCE(updated_at, NOW()) as updated_at
FROM user_profiles up
WHERE NOT EXISTS (
    SELECT 1 FROM client_profiles cp 
    WHERE cp.auth_user_id = up.user_id
)
AND user_id IS NOT NULL;

-- 2. Migrate any additional data from profiles table to client_profiles
INSERT INTO client_profiles (auth_user_id, client_name, created_at, updated_at)
SELECT 
    user_id,
    COALESCE(display_name, 'User') as client_name,
    NOW() as created_at,
    NOW() as updated_at
FROM profiles p
WHERE NOT EXISTS (
    SELECT 1 FROM client_profiles cp 
    WHERE cp.auth_user_id = p.user_id
)
AND user_id IS NOT NULL;

-- 3. Drop redundant user tables after data migration
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS profiles CASCADE;

-- 4. Consolidate workout tracking tables
-- Migrate any essential data from workout_sessions to live_workout_sessions if needed
-- (keeping live_workout_sessions as the primary workout table)
DROP TABLE IF EXISTS workout_sessions CASCADE;
DROP TABLE IF EXISTS workout_logs CASCADE;

-- 5. Clean up redundant subscription tables
-- Keep main tables: subscriptions, payment_events, orders
-- Remove potential duplicates
DROP TABLE IF EXISTS user_subscriptions CASCADE;

-- 6. Remove session tracking duplicates  
-- Keep secure_sessions only if it's actively used
DROP TABLE IF EXISTS sessions CASCADE;