-- REPZ FITNESS PLATFORM: Fix Tier System to Correct Structure (DEBUGGED VERSION)
-- This version includes better error handling and checks for existing objects

BEGIN;

-- Step 1: Create mapping function for final migration
CREATE OR REPLACE FUNCTION fix_tier_value(old_tier text)
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

-- Step 2: Add temporary columns to all affected tables (with existence checks)
DO $$
BEGIN
  -- Check and add columns only if they don't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'client_profiles' AND column_name = 'subscription_tier_fixed') THEN
    ALTER TABLE client_profiles ADD COLUMN subscription_tier_fixed text;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'demo_profiles' AND column_name = 'subscription_tier_fixed') THEN
    ALTER TABLE demo_profiles ADD COLUMN subscription_tier_fixed text;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'subscription_tiers' AND column_name = 'tier_name_fixed') THEN
      ALTER TABLE subscription_tiers ADD COLUMN tier_name_fixed text;
    END IF;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'leaderboards' AND column_name = 'tier_filter_fixed') THEN
      ALTER TABLE leaderboards ADD COLUMN tier_filter_fixed text;
    END IF;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'leaderboard_entries' AND column_name = 'tier_fixed') THEN
      ALTER TABLE leaderboard_entries ADD COLUMN tier_fixed text;
    END IF;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'achievements' AND column_name = 'tier_requirement_fixed') THEN
      ALTER TABLE achievements ADD COLUMN tier_requirement_fixed text;
    END IF;
  END IF;
END $$;

-- Step 3: Migrate data using the mapping function
UPDATE client_profiles SET subscription_tier_fixed = fix_tier_value(subscription_tier::text);
UPDATE demo_profiles SET subscription_tier_fixed = fix_tier_value(subscription_tier::text);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    UPDATE subscription_tiers SET tier_name_fixed = fix_tier_value(tier_name::text);
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    UPDATE leaderboards SET tier_filter_fixed = fix_tier_value(tier_filter::text) WHERE tier_filter IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    UPDATE leaderboard_entries SET tier_fixed = fix_tier_value(tier::text) WHERE tier IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    UPDATE achievements SET tier_requirement_fixed = fix_tier_value(tier_requirement::text) WHERE tier_requirement IS NOT NULL;
  END IF;
END $$;

-- Step 4: Drop ALL existing triggers on client_profiles (to avoid conflicts)
DO $$
DECLARE
    trigger_rec RECORD;
BEGIN
    FOR trigger_rec IN 
        SELECT tgname 
        FROM pg_trigger 
        WHERE tgrelid = 'client_profiles'::regclass 
        AND tgname LIKE '%tier%'
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON client_profiles CASCADE', trigger_rec.tgname);
    END LOOP;
END $$;

-- Step 5: Drop dependent functions with CASCADE
DROP FUNCTION IF EXISTS has_tier_access(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS get_tier_features(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS get_new_tier_features(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS check_tier_feature_access(tier_enum, text) CASCADE;
DROP FUNCTION IF EXISTS sync_tier_features() CASCADE;
DROP FUNCTION IF EXISTS sync_new_tier_features() CASCADE;

-- Step 6: Drop policies if tables exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'daily_tracking') THEN
    DROP POLICY IF EXISTS daily_tracking_tier_access ON daily_tracking;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'biomarker_tests') THEN
    DROP POLICY IF EXISTS biomarker_precision_access ON biomarker_tests;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'progress_photos') THEN
    DROP POLICY IF EXISTS progress_photos_tier_access ON progress_photos;
  END IF;
END $$;

-- Step 7: Drop old columns
ALTER TABLE client_profiles DROP COLUMN IF EXISTS subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN IF EXISTS subscription_tier;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    ALTER TABLE subscription_tiers DROP COLUMN IF EXISTS tier_name;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    ALTER TABLE leaderboards DROP COLUMN IF EXISTS tier_filter;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    ALTER TABLE leaderboard_entries DROP COLUMN IF EXISTS tier;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    ALTER TABLE achievements DROP COLUMN IF EXISTS tier_requirement;
  END IF;
END $$;

-- Step 8: Drop old enum and create correct one
DROP TYPE IF EXISTS tier_enum CASCADE;
CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Step 9: Add new properly typed columns
ALTER TABLE client_profiles ADD COLUMN subscription_tier tier_enum;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier tier_enum;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    ALTER TABLE subscription_tiers ADD COLUMN tier_name tier_enum;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    ALTER TABLE leaderboards ADD COLUMN tier_filter tier_enum;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    ALTER TABLE leaderboard_entries ADD COLUMN tier tier_enum;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    ALTER TABLE achievements ADD COLUMN tier_requirement tier_enum;
  END IF;
END $$;

-- Step 10: Copy data from temporary columns
UPDATE client_profiles SET subscription_tier = subscription_tier_fixed::tier_enum;
UPDATE demo_profiles SET subscription_tier = subscription_tier_fixed::tier_enum;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    UPDATE subscription_tiers SET tier_name = tier_name_fixed::tier_enum;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    UPDATE leaderboards SET tier_filter = tier_filter_fixed::tier_enum WHERE tier_filter_fixed IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    UPDATE leaderboard_entries SET tier = tier_fixed::tier_enum WHERE tier_fixed IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    UPDATE achievements SET tier_requirement = tier_requirement_fixed::tier_enum WHERE tier_requirement_fixed IS NOT NULL;
  END IF;
END $$;

-- Step 11: Set defaults and constraints
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET NOT NULL;

-- Step 12: Drop temporary columns
ALTER TABLE client_profiles DROP COLUMN IF EXISTS subscription_tier_fixed;
ALTER TABLE demo_profiles DROP COLUMN IF EXISTS subscription_tier_fixed;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'subscription_tiers') THEN
    ALTER TABLE subscription_tiers DROP COLUMN IF EXISTS tier_name_fixed;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboards') THEN
    ALTER TABLE leaderboards DROP COLUMN IF EXISTS tier_filter_fixed;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'leaderboard_entries') THEN
    ALTER TABLE leaderboard_entries DROP COLUMN IF EXISTS tier_fixed;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'achievements') THEN
    ALTER TABLE achievements DROP COLUMN IF EXISTS tier_requirement_fixed;
  END IF;
END $$;

-- Step 13: Recreate functions with correct tier enum
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

-- Step 14: Create trigger function
CREATE OR REPLACE FUNCTION public.sync_tier_features()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $function$
BEGIN
  NEW.tier_features = public.get_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$function$;

-- Step 15: Create trigger (only if it doesn't exist)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger 
    WHERE tgname = 'sync_tier_features_trigger' 
    AND tgrelid = 'client_profiles'::regclass
  ) THEN
    CREATE TRIGGER sync_tier_features_trigger
      BEFORE INSERT OR UPDATE OF subscription_tier ON client_profiles
      FOR EACH ROW
      EXECUTE FUNCTION sync_tier_features();
  END IF;
END $$;

-- Step 16: Recreate RLS policies (only if tables exist)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'daily_tracking') THEN
    CREATE POLICY "daily_tracking_tier_access" ON daily_tracking
    FOR ALL
    USING (
      EXISTS (
        SELECT 1 FROM client_profiles cp
        WHERE cp.id = daily_tracking.client_id
        AND cp.auth_user_id = auth.uid()
        AND cp.subscription_tier = ANY (ARRAY['adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum])
      )
    );
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'biomarker_tests') THEN
    CREATE POLICY "biomarker_precision_access" ON biomarker_tests
    FOR ALL
    USING (
      EXISTS (
        SELECT 1 FROM client_profiles cp
        WHERE cp.id = biomarker_tests.client_id
        AND cp.auth_user_id = auth.uid()
        AND cp.subscription_tier = ANY (ARRAY['performance'::tier_enum, 'longevity'::tier_enum])
      )
    );
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'progress_photos') THEN
    CREATE POLICY "progress_photos_tier_access" ON progress_photos
    FOR ALL
    USING (
      EXISTS (
        SELECT 1 FROM client_profiles cp
        WHERE cp.id = progress_photos.client_id
        AND cp.auth_user_id = auth.uid()
        AND cp.subscription_tier = ANY (ARRAY['adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum])
      )
    );
  END IF;
END $$;

-- Step 17: Update tier_features for all users
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier)
WHERE subscription_tier IS NOT NULL;

-- Step 18: Clean up
DROP FUNCTION IF EXISTS fix_tier_value(text);

-- Step 19: Create indexes
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);

COMMIT;