-- REPZ FITNESS PLATFORM: Safe Tier System Migration
-- Step 1: Drop dependent objects first, then migrate safely

BEGIN;

-- Step 1: Drop all dependent triggers and policies
DROP TRIGGER IF EXISTS update_tier_features_trigger ON client_profiles CASCADE;
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON client_profiles CASCADE;
DROP TRIGGER IF EXISTS sync_new_tier_features_trigger ON client_profiles CASCADE;

DROP POLICY IF EXISTS daily_tracking_tier_access ON daily_tracking;
DROP POLICY IF EXISTS biomarker_precision_access ON biomarker_tests;
DROP POLICY IF EXISTS progress_photos_tier_access ON progress_photos;

-- Step 2: Create new enum type
CREATE TYPE tier_enum_new AS ENUM ('foundational', 'essentials', 'deluxe', 'longevity');

-- Step 3: Add new column with migration data
ALTER TABLE client_profiles ADD COLUMN subscription_tier_new tier_enum_new;
UPDATE client_profiles SET subscription_tier_new = 
  CASE subscription_tier::text
    WHEN 'baseline' THEN 'foundational'::tier_enum_new
    WHEN 'prime' THEN 'essentials'::tier_enum_new  
    WHEN 'precision' THEN 'deluxe'::tier_enum_new
    WHEN 'core' THEN 'foundational'::tier_enum_new
    WHEN 'adaptive' THEN 'essentials'::tier_enum_new
    WHEN 'performance' THEN 'deluxe'::tier_enum_new
    WHEN 'longevity' THEN 'longevity'::tier_enum_new
    ELSE 'foundational'::tier_enum_new
  END;

-- Step 4: Update demo_profiles
ALTER TABLE demo_profiles ADD COLUMN subscription_tier_new tier_enum_new;
UPDATE demo_profiles SET subscription_tier_new = 
  CASE subscription_tier::text
    WHEN 'baseline' THEN 'foundational'::tier_enum_new
    WHEN 'prime' THEN 'essentials'::tier_enum_new
    WHEN 'precision' THEN 'deluxe'::tier_enum_new
    WHEN 'core' THEN 'foundational'::tier_enum_new
    WHEN 'adaptive' THEN 'essentials'::tier_enum_new
    WHEN 'performance' THEN 'deluxe'::tier_enum_new
    WHEN 'longevity' THEN 'longevity'::tier_enum_new
    ELSE 'foundational'::tier_enum_new
  END;

-- Step 5: Drop old columns
ALTER TABLE client_profiles DROP COLUMN subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier;

-- Step 6: Rename new columns
ALTER TABLE client_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE demo_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;

-- Step 7: Set defaults
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum_new;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum_new;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET NOT NULL;

-- Step 8: Replace old enum
DROP TYPE tier_enum;
ALTER TYPE tier_enum_new RENAME TO tier_enum;

-- Step 9: Update tier_features table if exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tier_features') THEN
    ALTER TABLE tier_features RENAME COLUMN core TO foundational;
    ALTER TABLE tier_features RENAME COLUMN adaptive TO essentials;
    ALTER TABLE tier_features RENAME COLUMN performance TO deluxe;
  END IF;
END $$;

-- Step 10: Recreate essential RLS policies with new tier names
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

-- Step 11: Update tier features function
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

-- Step 12: Recreate tier sync trigger
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

-- Step 13: Update existing tier_features for all users
UPDATE client_profiles 
SET tier_features = public.get_new_tier_features(subscription_tier);

-- Step 14: Create indexes
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);

COMMIT;