-- REPZ FITNESS PLATFORM: Surgical Tier System Fix
-- This version works with existing functions and only updates what's needed

BEGIN;

-- Step 1: First check what tier values currently exist
DO $$
DECLARE
    current_values text[];
BEGIN
    SELECT array_agg(enumlabel ORDER BY enumsortorder) 
    INTO current_values
    FROM pg_enum 
    WHERE enumtypid = 'tier_enum'::regtype;
    
    RAISE NOTICE 'Current tier_enum values: %', current_values;
END $$;

-- Step 2: Create temporary mapping function
CREATE OR REPLACE FUNCTION temp_fix_tier_value(old_tier text)
RETURNS text
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN CASE old_tier
    WHEN 'foundational' THEN 'core'
    WHEN 'essentials' THEN 'adaptive'
    WHEN 'deluxe' THEN 'performance'
    WHEN 'baseline' THEN 'core'
    WHEN 'prime' THEN 'adaptive'
    WHEN 'precision' THEN 'performance'
    WHEN 'longevity' THEN 'longevity'
    ELSE 'core'
  END;
END;
$$;

-- Step 3: Add temporary columns for data migration
ALTER TABLE client_profiles ADD COLUMN IF NOT EXISTS subscription_tier_temp text;
ALTER TABLE demo_profiles ADD COLUMN IF NOT EXISTS subscription_tier_temp text;

-- Add to other tables if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    ALTER TABLE subscription_tiers ADD COLUMN IF NOT EXISTS tier_name_temp text;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    ALTER TABLE leaderboards ADD COLUMN IF NOT EXISTS tier_filter_temp text;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    ALTER TABLE leaderboard_entries ADD COLUMN IF NOT EXISTS tier_temp text;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    ALTER TABLE achievements ADD COLUMN IF NOT EXISTS tier_requirement_temp text;
  END IF;
END $$;

-- Step 4: Migrate data to temporary columns
UPDATE client_profiles 
SET subscription_tier_temp = temp_fix_tier_value(subscription_tier::text);

UPDATE demo_profiles 
SET subscription_tier_temp = temp_fix_tier_value(subscription_tier::text);

-- Migrate other tables if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    UPDATE subscription_tiers 
    SET tier_name_temp = temp_fix_tier_value(tier_name::text)
    WHERE tier_name IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    UPDATE leaderboards 
    SET tier_filter_temp = temp_fix_tier_value(tier_filter::text)
    WHERE tier_filter IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    UPDATE leaderboard_entries 
    SET tier_temp = temp_fix_tier_value(tier::text)
    WHERE tier IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    UPDATE achievements 
    SET tier_requirement_temp = temp_fix_tier_value(tier_requirement::text)
    WHERE tier_requirement IS NOT NULL;
  END IF;
END $$;

-- Step 5: Check what needs to be done with the enum
DO $$
DECLARE
    needs_recreation boolean := false;
    current_values text[];
    expected_values text[] := ARRAY['core', 'adaptive', 'performance', 'longevity'];
BEGIN
    SELECT array_agg(enumlabel ORDER BY enumsortorder) 
    INTO current_values
    FROM pg_enum 
    WHERE enumtypid = 'tier_enum'::regtype;
    
    -- Check if current values match expected values
    IF current_values IS DISTINCT FROM expected_values THEN
        needs_recreation := true;
        RAISE NOTICE 'Current values: %, Expected: %, Needs recreation: %', 
                     current_values, expected_values, needs_recreation;
    END IF;
    
    -- Store result for next step
    INSERT INTO pg_temp.migration_state (key, value) 
    VALUES ('needs_enum_recreation', needs_recreation::text)
    ON CONFLICT (key) DO UPDATE SET value = needs_recreation::text;
EXCEPTION
    WHEN undefined_table THEN
        -- Create temp table if it doesn't exist
        CREATE TEMP TABLE migration_state (key text PRIMARY KEY, value text);
        INSERT INTO migration_state (key, value) 
        VALUES ('needs_enum_recreation', needs_recreation::text);
END $$;

-- Step 6: Recreate enum only if needed
DO $$
DECLARE
    needs_recreation boolean;
BEGIN
    SELECT value::boolean INTO needs_recreation 
    FROM pg_temp.migration_state 
    WHERE key = 'needs_enum_recreation';
    
    IF needs_recreation THEN
        RAISE NOTICE 'Recreating tier_enum...';
        
        -- Drop all dependent functions temporarily (they'll be recreated)
        DROP FUNCTION IF EXISTS has_tier_access(tier_enum) CASCADE;
        DROP FUNCTION IF EXISTS get_tier_features(tier_enum) CASCADE;
        DROP FUNCTION IF EXISTS get_new_tier_features(tier_enum) CASCADE;
        DROP FUNCTION IF EXISTS check_tier_feature_access(tier_enum, text) CASCADE;
        DROP FUNCTION IF EXISTS sync_tier_features() CASCADE;
        DROP FUNCTION IF EXISTS sync_new_tier_features() CASCADE;
        DROP FUNCTION IF EXISTS get_user_tier() CASCADE;
        DROP FUNCTION IF EXISTS update_client_tier_features() CASCADE;
        DROP FUNCTION IF EXISTS validate_tier_system() CASCADE;
        
        -- Drop policies that reference the enum
        DROP POLICY IF EXISTS daily_tracking_tier_access ON daily_tracking;
        DROP POLICY IF EXISTS biomarker_precision_access ON biomarker_tests;
        DROP POLICY IF EXISTS progress_photos_tier_access ON progress_photos;
        
        -- Drop and recreate the enum
        DROP TYPE tier_enum CASCADE;
        CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');
        
        RAISE NOTICE 'tier_enum recreated successfully';
    ELSE
        RAISE NOTICE 'tier_enum already has correct values, skipping recreation';
    END IF;
END $$;

-- Step 7: Recreate columns with new enum type
DO $$
DECLARE
    needs_recreation boolean;
BEGIN
    SELECT value::boolean INTO needs_recreation 
    FROM pg_temp.migration_state 
    WHERE key = 'needs_enum_recreation';
    
    IF needs_recreation THEN
        -- Recreate columns with new enum type
        ALTER TABLE client_profiles DROP COLUMN IF EXISTS subscription_tier;
        ALTER TABLE client_profiles ADD COLUMN subscription_tier tier_enum;
        
        ALTER TABLE demo_profiles DROP COLUMN IF EXISTS subscription_tier;
        ALTER TABLE demo_profiles ADD COLUMN subscription_tier tier_enum;
        
        -- Handle other tables
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
            ALTER TABLE subscription_tiers DROP COLUMN IF EXISTS tier_name;
            ALTER TABLE subscription_tiers ADD COLUMN tier_name tier_enum;
        END IF;
        
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
            ALTER TABLE leaderboards DROP COLUMN IF EXISTS tier_filter;
            ALTER TABLE leaderboards ADD COLUMN tier_filter tier_enum;
        END IF;
        
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
            ALTER TABLE leaderboard_entries DROP COLUMN IF EXISTS tier;
            ALTER TABLE leaderboard_entries ADD COLUMN tier tier_enum;
        END IF;
        
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
            ALTER TABLE achievements DROP COLUMN IF EXISTS tier_requirement;
            ALTER TABLE achievements ADD COLUMN tier_requirement tier_enum;
        END IF;
        
        RAISE NOTICE 'Columns recreated with new enum type';
    END IF;
END $$;

-- Step 8: Copy data back from temporary columns
UPDATE client_profiles 
SET subscription_tier = subscription_tier_temp::tier_enum
WHERE subscription_tier_temp IS NOT NULL;

UPDATE demo_profiles 
SET subscription_tier = subscription_tier_temp::tier_enum  
WHERE subscription_tier_temp IS NOT NULL;

-- Copy other table data
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    UPDATE subscription_tiers 
    SET tier_name = tier_name_temp::tier_enum
    WHERE tier_name_temp IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    UPDATE leaderboards 
    SET tier_filter = tier_filter_temp::tier_enum
    WHERE tier_filter_temp IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    UPDATE leaderboard_entries 
    SET tier = tier_temp::tier_enum
    WHERE tier_temp IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    UPDATE achievements 
    SET tier_requirement = tier_requirement_temp::tier_enum
    WHERE tier_requirement_temp IS NOT NULL;
  END IF;
END $$;

-- Step 9: Set constraints and defaults
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET NOT NULL;

-- Step 10: Recreate essential functions
CREATE OR REPLACE FUNCTION public.get_tier_features(tier_name tier_enum)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $function$
BEGIN
  RETURN CASE tier_name::text
    WHEN 'core' THEN '{
      "training_program": true,
      "nutrition_plan": true,
      "dashboard_type": "static",
      "qa_access": true,
      "response_time_hours": 72,
      "weekly_checkins": false,
      "form_reviews": false,
      "wearable_integration": false,
      "ai_assistant": false,
      "biomarker_integration": false,
      "supplements": false,
      "telegram_community": false,
      "in_person_training": false
    }'::jsonb
    WHEN 'adaptive' THEN '{
      "training_program": true,
      "nutrition_plan": true,
      "dashboard_type": "interactive",
      "qa_access": true,
      "response_time_hours": 48,
      "weekly_checkins": true,
      "form_reviews": true,
      "wearable_integration": true,
      "ai_assistant": false,
      "biomarker_integration": false,
      "supplements": true,
      "telegram_community": false,
      "in_person_training": false
    }'::jsonb
    WHEN 'performance' THEN '{
      "training_program": true,
      "nutrition_plan": true,
      "dashboard_type": "advanced",
      "qa_access": true,
      "response_time_hours": 24,
      "weekly_checkins": true,
      "form_reviews": true,
      "wearable_integration": true,
      "ai_assistant": true,
      "biomarker_integration": true,
      "supplements": true,
      "peptides": true,
      "telegram_community": true,
      "in_person_training": false
    }'::jsonb
    WHEN 'longevity' THEN '{
      "training_program": true,
      "nutrition_plan": true,
      "dashboard_type": "premium",
      "qa_access": true,
      "response_time_hours": 12,
      "weekly_checkins": true,
      "form_reviews": true,
      "wearable_integration": true,
      "ai_assistant": true,
      "biomarker_integration": true,
      "supplements": true,
      "peptides": true,
      "bioregulators": true,
      "telegram_community": true,
      "in_person_training": true
    }'::jsonb
    ELSE '{}'::jsonb
  END;
END;
$function$;

CREATE OR REPLACE FUNCTION public.has_tier_access(required_tier tier_enum)
RETURNS boolean
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = public
AS $function$
  SELECT EXISTS (
    SELECT 1 FROM public.client_profiles
    WHERE auth_user_id = auth.uid()
    AND CASE required_tier::text
      WHEN 'core' THEN true
      WHEN 'adaptive' THEN subscription_tier::text IN ('adaptive', 'performance', 'longevity')
      WHEN 'performance' THEN subscription_tier::text IN ('performance', 'longevity')
      WHEN 'longevity' THEN subscription_tier::text = 'longevity'
      ELSE false
    END
  );
$function$;

-- Step 11: Update tier_features for all users
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier)
WHERE subscription_tier IS NOT NULL;

-- Step 12: Clean up temporary columns and functions
ALTER TABLE client_profiles DROP COLUMN IF EXISTS subscription_tier_temp;
ALTER TABLE demo_profiles DROP COLUMN IF EXISTS subscription_tier_temp;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'subscription_tiers' AND column_name = 'tier_name_temp') THEN
    ALTER TABLE subscription_tiers DROP COLUMN tier_name_temp;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'leaderboards' AND column_name = 'tier_filter_temp') THEN
    ALTER TABLE leaderboards DROP COLUMN tier_filter_temp;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'leaderboard_entries' AND column_name = 'tier_temp') THEN
    ALTER TABLE leaderboard_entries DROP COLUMN tier_temp;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'achievements' AND column_name = 'tier_requirement_temp') THEN
    ALTER TABLE achievements DROP COLUMN tier_requirement_temp;
  END IF;
END $$;

DROP FUNCTION IF EXISTS temp_fix_tier_value(text);

-- Step 13: Create indexes
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);

-- Step 14: Final verification
DO $$
DECLARE
    final_values text[];
BEGIN
    SELECT array_agg(enumlabel ORDER BY enumsortorder) 
    INTO final_values
    FROM pg_enum 
    WHERE enumtypid = 'tier_enum'::regtype;
    
    RAISE NOTICE 'Final tier_enum values: %', final_values;
    
    -- Count migrated records
    RAISE NOTICE 'Client profiles with tiers: %', 
        (SELECT count(*) FROM client_profiles WHERE subscription_tier IS NOT NULL);
    RAISE NOTICE 'Demo profiles with tiers: %', 
        (SELECT count(*) FROM demo_profiles WHERE subscription_tier IS NOT NULL);
END $$;

COMMIT;