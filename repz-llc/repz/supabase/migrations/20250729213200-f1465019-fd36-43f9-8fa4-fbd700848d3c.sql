-- REPZ FITNESS PLATFORM: Tier System Migration
-- Rename tiers: Core→Foundational, Adaptive→Essentials, Performance→Deluxe, Longevity stays same
-- Update pricing and maintain backward compatibility

BEGIN;

-- Step 1: Create new tier enum type with updated names
CREATE TYPE tier_enum_new AS ENUM ('foundational', 'essentials', 'deluxe', 'longevity');

-- Step 2: Add temporary columns for migration
ALTER TABLE client_profiles ADD COLUMN subscription_tier_new tier_enum_new;

-- Step 3: Migrate existing data with tier name mapping
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

-- Step 4: Update demo_profiles table similarly
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

-- Step 5: Drop old columns and constraints
ALTER TABLE client_profiles DROP COLUMN subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier;

-- Step 6: Rename new columns
ALTER TABLE client_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE demo_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;

-- Step 7: Set default values
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum_new;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'foundational'::tier_enum_new;

-- Step 8: Drop old enum type and rename new one
DROP TYPE tier_enum;
ALTER TYPE tier_enum_new RENAME TO tier_enum;

-- Step 9: Update tier_features table if it exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tier_features') THEN
    -- Update tier column names in tier_features table
    ALTER TABLE tier_features RENAME COLUMN core TO foundational;
    ALTER TABLE tier_features RENAME COLUMN adaptive TO essentials;
    ALTER TABLE tier_features RENAME COLUMN performance TO deluxe;
    -- longevity stays the same
  END IF;
END $$;

-- Step 10: Update all tier_features JSON in client_profiles to use new tier names
UPDATE client_profiles 
SET tier_features = public.get_new_tier_features(subscription_tier)
WHERE tier_features IS NOT NULL;

-- Step 11: Recreate indexes and constraints as needed
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);

-- Step 12: Update RLS policies function to use new tier names
CREATE OR REPLACE FUNCTION public.get_new_tier_features(tier_name tier_enum)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $function$
DECLARE
  feature_map jsonb := '{}';
  tier_column text;
BEGIN
  -- Map tier names to column names
  tier_column := CASE tier_name::text
    WHEN 'foundational' THEN 'foundational'
    WHEN 'essentials' THEN 'essentials'
    WHEN 'deluxe' THEN 'deluxe'
    WHEN 'longevity' THEN 'longevity'
    ELSE 'foundational'
  END;
  
  -- Build feature map from tier_features table if it exists
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tier_features') THEN
    EXECUTE format('
      SELECT jsonb_object_agg(
        feature_key,
        CASE WHEN %I IS NOT NULL THEN %I ELSE false END
      ) FROM public.tier_features
    ', tier_column, tier_column) INTO feature_map;
  ELSE
    -- Fallback feature map
    feature_map := CASE tier_name::text
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
  END IF;
  
  RETURN feature_map;
END;
$function$;

-- Verify migration completed successfully
DO $$
DECLARE
  foundational_count integer;
  essentials_count integer;
  deluxe_count integer;
  longevity_count integer;
BEGIN
  SELECT COUNT(*) INTO foundational_count FROM client_profiles WHERE subscription_tier = 'foundational';
  SELECT COUNT(*) INTO essentials_count FROM client_profiles WHERE subscription_tier = 'essentials';
  SELECT COUNT(*) INTO deluxe_count FROM client_profiles WHERE subscription_tier = 'deluxe';
  SELECT COUNT(*) INTO longevity_count FROM client_profiles WHERE subscription_tier = 'longevity';
  
  RAISE NOTICE 'Migration completed successfully:';
  RAISE NOTICE 'Foundational users: %', foundational_count;
  RAISE NOTICE 'Essentials users: %', essentials_count;
  RAISE NOTICE 'Deluxe users: %', deluxe_count;
  RAISE NOTICE 'Longevity users: %', longevity_count;
END $$;

COMMIT;