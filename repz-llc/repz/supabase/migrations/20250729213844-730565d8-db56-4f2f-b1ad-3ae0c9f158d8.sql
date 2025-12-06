-- REPZ FITNESS PLATFORM: Comprehensive Tier System Migration
-- Handle ALL dependencies on tier_enum type

BEGIN;

-- Step 1: Create mapping function for migration
CREATE OR REPLACE FUNCTION migrate_tier_value(old_tier text)
RETURNS text
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN CASE old_tier
    WHEN 'baseline' THEN 'foundational'
    WHEN 'prime' THEN 'essentials'  
    WHEN 'precision' THEN 'deluxe'
    WHEN 'core' THEN 'foundational'
    WHEN 'adaptive' THEN 'essentials'
    WHEN 'performance' THEN 'deluxe'
    WHEN 'longevity' THEN 'longevity'
    ELSE 'foundational'
  END;
END;
$$;

-- Step 2: Add temporary columns to all affected tables
ALTER TABLE client_profiles ADD COLUMN subscription_tier_new text;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier_new text;
ALTER TABLE subscription_tiers ADD COLUMN tier_name_new text;
ALTER TABLE leaderboards ADD COLUMN tier_filter_new text;
ALTER TABLE leaderboard_entries ADD COLUMN tier_new text;
ALTER TABLE achievements ADD COLUMN tier_requirement_new text;

-- Check if supplement tables exist before altering
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_protocols') THEN
    ALTER TABLE supplement_protocols ADD COLUMN tier_requirement_new text;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_library') THEN
    ALTER TABLE supplement_library ADD COLUMN tier_recommendation_new text;
  END IF;
END $$;

-- Step 3: Migrate data using the mapping function
UPDATE client_profiles SET subscription_tier_new = migrate_tier_value(subscription_tier::text);
UPDATE demo_profiles SET subscription_tier_new = migrate_tier_value(subscription_tier::text);
UPDATE subscription_tiers SET tier_name_new = migrate_tier_value(tier_name::text);
UPDATE leaderboards SET tier_filter_new = migrate_tier_value(tier_filter::text) WHERE tier_filter IS NOT NULL;
UPDATE leaderboard_entries SET tier_new = migrate_tier_value(tier::text) WHERE tier IS NOT NULL;
UPDATE achievements SET tier_requirement_new = migrate_tier_value(tier_requirement::text) WHERE tier_requirement IS NOT NULL;

-- Migrate supplement tables if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_protocols') THEN
    UPDATE supplement_protocols SET tier_requirement_new = migrate_tier_value(tier_requirement::text) WHERE tier_requirement IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_library') THEN
    UPDATE supplement_library SET tier_recommendation_new = migrate_tier_value(tier_recommendation::text) WHERE tier_recommendation IS NOT NULL;
  END IF;
END $$;

-- Step 4: Drop dependent functions first
DROP FUNCTION IF EXISTS has_tier_access(tier_enum);
DROP FUNCTION IF EXISTS get_tier_features(tier_enum);
DROP FUNCTION IF EXISTS check_tier_feature_access(tier_enum, text);
DROP FUNCTION IF EXISTS get_new_tier_features(tier_enum);

-- Step 5: Drop triggers and policies
DROP TRIGGER IF EXISTS update_tier_features_trigger ON client_profiles CASCADE;
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON client_profiles CASCADE;
DROP TRIGGER IF EXISTS sync_new_tier_features_trigger ON client_profiles CASCADE;

DROP POLICY IF EXISTS daily_tracking_tier_access ON daily_tracking;
DROP POLICY IF EXISTS biomarker_precision_access ON biomarker_tests;
DROP POLICY IF EXISTS progress_photos_tier_access ON progress_photos;

-- Step 6: Drop old columns
ALTER TABLE client_profiles DROP COLUMN subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier;
ALTER TABLE subscription_tiers DROP COLUMN tier_name;
ALTER TABLE leaderboards DROP COLUMN tier_filter;
ALTER TABLE leaderboard_entries DROP COLUMN tier;
ALTER TABLE achievements DROP COLUMN tier_requirement;

-- Drop supplement columns if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'supplement_protocols' AND column_name = 'tier_requirement') THEN
    ALTER TABLE supplement_protocols DROP COLUMN tier_requirement;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'supplement_library' AND column_name = 'tier_recommendation') THEN
    ALTER TABLE supplement_library DROP COLUMN tier_recommendation;
  END IF;
END $$;

-- Step 7: Create new enum type
CREATE TYPE tier_enum AS ENUM ('foundational', 'essentials', 'deluxe', 'longevity');

-- Step 8: Add new properly typed columns
ALTER TABLE client_profiles ADD COLUMN subscription_tier tier_enum;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier tier_enum;
ALTER TABLE subscription_tiers ADD COLUMN tier_name tier_enum;
ALTER TABLE leaderboards ADD COLUMN tier_filter tier_enum;
ALTER TABLE leaderboard_entries ADD COLUMN tier tier_enum;
ALTER TABLE achievements ADD COLUMN tier_requirement tier_enum;

-- Add supplement columns back if tables exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_protocols') THEN
    ALTER TABLE supplement_protocols ADD COLUMN tier_requirement tier_enum;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_library') THEN
    ALTER TABLE supplement_library ADD COLUMN tier_recommendation tier_enum;
  END IF;
END $$;

-- Step 9: Copy data from temporary columns
UPDATE client_profiles SET subscription_tier = subscription_tier_new::tier_enum;
UPDATE demo_profiles SET subscription_tier = subscription_tier_new::tier_enum;
UPDATE subscription_tiers SET tier_name = tier_name_new::tier_enum;
UPDATE leaderboards SET tier_filter = tier_filter_new::tier_enum WHERE tier_filter_new IS NOT NULL;
UPDATE leaderboard_entries SET tier = tier_new::tier_enum WHERE tier_new IS NOT NULL;
UPDATE achievements SET tier_requirement = tier_requirement_new::tier_enum WHERE tier_requirement_new IS NOT NULL;

-- Update supplement tables if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_protocols') THEN
    UPDATE supplement_protocols SET tier_requirement = tier_requirement_new::tier_enum WHERE tier_requirement_new IS NOT NULL;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'supplement_library') THEN
    UPDATE supplement_library SET tier_recommendation = tier_recommendation_new::tier_enum WHERE tier_recommendation_new IS NOT NULL;
  END IF;
END $$;

-- Step 10: Set defaults and constraints
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET NOT NULL;

-- Step 11: Drop temporary columns
ALTER TABLE client_profiles DROP COLUMN subscription_tier_new;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier_new;
ALTER TABLE subscription_tiers DROP COLUMN tier_name_new;
ALTER TABLE leaderboards DROP COLUMN tier_filter_new;
ALTER TABLE leaderboard_entries DROP COLUMN tier_new;
ALTER TABLE achievements DROP COLUMN tier_requirement_new;

-- Drop supplement temp columns if they exist
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'supplement_protocols' AND column_name = 'tier_requirement_new') THEN
    ALTER TABLE supplement_protocols DROP COLUMN tier_requirement_new;
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'supplement_library' AND column_name = 'tier_recommendation_new') THEN
    ALTER TABLE supplement_library DROP COLUMN tier_recommendation_new;
  END IF;
END $$;

-- Step 12: Recreate functions with new tier enum
CREATE OR REPLACE FUNCTION public.get_new_tier_features(tier_name tier_enum)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $function$
BEGIN
  RETURN CASE tier_name::text
    WHEN 'foundational' THEN '{
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
    WHEN 'essentials' THEN '{
      "training_program": true,
      "nutrition_plan": true,
      "dashboard_type": "interactive",
      "qa_access": true,
      "response_time_hours": 48,
      "weekly_checkins": true,
      "form_reviews": true,
      "wearable_integration": true,
      "ai_assistant": false,
      "biomarker_integration": true,
      "supplements": true,
      "telegram_community": false,
      "in_person_training": false
    }'::jsonb
    WHEN 'deluxe' THEN '{
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
SET search_path = ''
AS $function$
  SELECT EXISTS (
    SELECT 1 FROM public.client_profiles 
    WHERE auth_user_id = auth.uid()
    AND CASE required_tier::text
      WHEN 'foundational' THEN true
      WHEN 'essentials' THEN subscription_tier::text IN ('essentials', 'deluxe', 'longevity')
      WHEN 'deluxe' THEN subscription_tier::text IN ('deluxe', 'longevity')
      WHEN 'longevity' THEN subscription_tier::text = 'longevity'
      ELSE false
    END
  );
$function$;

-- Step 13: Recreate triggers and policies
CREATE OR REPLACE FUNCTION public.sync_new_tier_features()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $function$
BEGIN
  NEW.tier_features = public.get_new_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$function$;

CREATE TRIGGER sync_new_tier_features_trigger
  BEFORE INSERT OR UPDATE OF subscription_tier ON client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION sync_new_tier_features();

-- Recreate RLS policies
CREATE POLICY "daily_tracking_tier_access" ON daily_tracking
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = daily_tracking.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['essentials'::tier_enum, 'deluxe'::tier_enum, 'longevity'::tier_enum])
  )
);

CREATE POLICY "biomarker_precision_access" ON biomarker_tests
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = biomarker_tests.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['deluxe'::tier_enum, 'longevity'::tier_enum])
  )
);

CREATE POLICY "progress_photos_tier_access" ON progress_photos
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = progress_photos.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['essentials'::tier_enum, 'deluxe'::tier_enum, 'longevity'::tier_enum])
  )
);

-- Step 14: Update tier_features for all users
UPDATE client_profiles 
SET tier_features = public.get_new_tier_features(subscription_tier);

-- Step 15: Clean up
DROP FUNCTION migrate_tier_value(text);

-- Step 16: Create indexes
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);

COMMIT;