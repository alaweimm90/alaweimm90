-- REPZ FITNESS PLATFORM: Final Tier Enum Cleanup
-- Clean up tier_enum to contain ONLY the 4 correct values: core, adaptive, performance, longevity
-- Handles all 10 tables that use tier_enum

BEGIN;

-- Step 1: Create comprehensive mapping function for all tier variations
CREATE OR REPLACE FUNCTION final_tier_mapping(old_tier text)
RETURNS text
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN CASE old_tier
    -- Old deprecated tiers
    WHEN 'baseline' THEN 'core'
    WHEN 'prime' THEN 'adaptive'
    WHEN 'precision' THEN 'performance'
    
    -- Intermediate deprecated tiers
    WHEN 'foundational' THEN 'core'
    WHEN 'essentials' THEN 'adaptive'
    WHEN 'deluxe' THEN 'performance'
    
    -- Extra deprecated tiers
    WHEN 'foundation_starter' THEN 'core'
    WHEN 'growth_accelerator' THEN 'adaptive'
    WHEN 'performance_pro' THEN 'performance'
    WHEN 'enterprise_elite' THEN 'longevity'
    
    -- Correct tiers (keep as-is)
    WHEN 'core' THEN 'core'
    WHEN 'adaptive' THEN 'adaptive'
    WHEN 'performance' THEN 'performance'
    WHEN 'longevity' THEN 'longevity'
    
    -- Default fallback
    ELSE 'core'
  END;
END;
$$;

-- Step 2: Add temporary migration columns to ALL affected tables
ALTER TABLE client_profiles ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE demo_profiles ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE subscription_tiers ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE leaderboards ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE leaderboard_entries ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE achievements ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE supplement_protocols ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE supplement_library ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE pricing_plans ADD COLUMN IF NOT EXISTS tier_final text;
ALTER TABLE user_tier_access ADD COLUMN IF NOT EXISTS tier_final text;

-- Step 3: Migrate ALL data to correct tier values
UPDATE client_profiles 
SET tier_final = final_tier_mapping(subscription_tier::text)
WHERE subscription_tier IS NOT NULL;

UPDATE demo_profiles 
SET tier_final = final_tier_mapping(subscription_tier::text)
WHERE subscription_tier IS NOT NULL;

UPDATE subscription_tiers 
SET tier_final = final_tier_mapping(tier_name::text)
WHERE tier_name IS NOT NULL;

UPDATE leaderboards 
SET tier_final = final_tier_mapping(tier_filter::text)
WHERE tier_filter IS NOT NULL;

UPDATE leaderboard_entries 
SET tier_final = final_tier_mapping(tier::text)
WHERE tier IS NOT NULL;

UPDATE achievements 
SET tier_final = final_tier_mapping(tier_requirement::text)
WHERE tier_requirement IS NOT NULL;

UPDATE supplement_protocols 
SET tier_final = final_tier_mapping(tier_requirement::text)
WHERE tier_requirement IS NOT NULL;

UPDATE supplement_library 
SET tier_final = final_tier_mapping(tier_recommendation::text)
WHERE tier_recommendation IS NOT NULL;

UPDATE pricing_plans 
SET tier_final = final_tier_mapping(tier_level::text)
WHERE tier_level IS NOT NULL;

UPDATE user_tier_access 
SET tier_final = final_tier_mapping(current_tier::text)
WHERE current_tier IS NOT NULL;

-- Step 4: Show migration summary
DO $$
DECLARE
    migration_summary text;
BEGIN
    migration_summary := format('
Migration Summary:
- client_profiles: %s records migrated
- demo_profiles: %s records migrated  
- subscription_tiers: %s records migrated
- leaderboards: %s records migrated
- leaderboard_entries: %s records migrated
- achievements: %s records migrated
- supplement_protocols: %s records migrated
- supplement_library: %s records migrated
- pricing_plans: %s records migrated
- user_tier_access: %s records migrated',
        (SELECT count(*) FROM client_profiles WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM demo_profiles WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM subscription_tiers WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM leaderboards WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM leaderboard_entries WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM achievements WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM supplement_protocols WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM supplement_library WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM pricing_plans WHERE tier_final IS NOT NULL),
        (SELECT count(*) FROM user_tier_access WHERE tier_final IS NOT NULL)
    );
    
    RAISE NOTICE '%', migration_summary;
END $$;

-- Step 5: Drop ALL functions and policies that depend on tier_enum
DROP FUNCTION IF EXISTS has_tier_access(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS get_tier_features(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS get_new_tier_features(tier_enum) CASCADE;
DROP FUNCTION IF EXISTS check_tier_feature_access(tier_enum, text) CASCADE;
DROP FUNCTION IF EXISTS sync_tier_features() CASCADE;
DROP FUNCTION IF EXISTS sync_new_tier_features() CASCADE;
DROP FUNCTION IF EXISTS get_user_tier() CASCADE;
DROP FUNCTION IF EXISTS update_client_tier_features() CASCADE;
DROP FUNCTION IF EXISTS validate_tier_system() CASCADE;

-- Drop all tier-related policies
DROP POLICY IF EXISTS daily_tracking_tier_access ON daily_tracking;
DROP POLICY IF EXISTS biomarker_precision_access ON biomarker_tests;
DROP POLICY IF EXISTS progress_photos_tier_access ON progress_photos;

-- Drop all tier-related triggers
DO $$
DECLARE
    trigger_rec RECORD;
BEGIN
    FOR trigger_rec IN 
        SELECT schemaname, tablename, triggername
        FROM pg_triggers 
        WHERE triggername ILIKE '%tier%'
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I.%I CASCADE', 
                      trigger_rec.triggername, trigger_rec.schemaname, trigger_rec.tablename);
        RAISE NOTICE 'Dropped trigger: %', trigger_rec.triggername;
    END LOOP;
END $$;

-- Step 6: Drop all tier_enum columns from all tables
ALTER TABLE client_profiles DROP COLUMN IF EXISTS subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN IF EXISTS subscription_tier;
ALTER TABLE subscription_tiers DROP COLUMN IF EXISTS tier_name;
ALTER TABLE leaderboards DROP COLUMN IF EXISTS tier_filter;
ALTER TABLE leaderboard_entries DROP COLUMN IF EXISTS tier;
ALTER TABLE achievements DROP COLUMN IF EXISTS tier_requirement;
ALTER TABLE supplement_protocols DROP COLUMN IF EXISTS tier_requirement;
ALTER TABLE supplement_library DROP COLUMN IF EXISTS tier_recommendation;
ALTER TABLE pricing_plans DROP COLUMN IF EXISTS tier_level;
ALTER TABLE user_tier_access DROP COLUMN IF EXISTS current_tier;

-- Step 7: Drop and recreate the enum with ONLY correct values
DROP TYPE tier_enum CASCADE;
CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');

RAISE NOTICE 'tier_enum recreated with correct values: core, adaptive, performance, longevity';

-- Step 8: Add back tier_enum columns to all tables
ALTER TABLE client_profiles ADD COLUMN subscription_tier tier_enum;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier tier_enum;
ALTER TABLE subscription_tiers ADD COLUMN tier_name tier_enum;
ALTER TABLE leaderboards ADD COLUMN tier_filter tier_enum;
ALTER TABLE leaderboard_entries ADD COLUMN tier tier_enum;
ALTER TABLE achievements ADD COLUMN tier_requirement tier_enum;
ALTER TABLE supplement_protocols ADD COLUMN tier_requirement tier_enum;
ALTER TABLE supplement_library ADD COLUMN tier_recommendation tier_enum;
ALTER TABLE pricing_plans ADD COLUMN tier_level tier_enum;
ALTER TABLE user_tier_access ADD COLUMN current_tier tier_enum;

-- Step 9: Restore data from temporary columns
UPDATE client_profiles 
SET subscription_tier = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE demo_profiles 
SET subscription_tier = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE subscription_tiers 
SET tier_name = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE leaderboards 
SET tier_filter = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE leaderboard_entries 
SET tier = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE achievements 
SET tier_requirement = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE supplement_protocols 
SET tier_requirement = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE supplement_library 
SET tier_recommendation = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE pricing_plans 
SET tier_level = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

UPDATE user_tier_access 
SET current_tier = tier_final::tier_enum
WHERE tier_final IS NOT NULL;

-- Step 10: Set proper defaults and constraints for main tables
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
ALTER TABLE user_tier_access ALTER COLUMN current_tier SET DEFAULT 'core'::tier_enum;

-- Set NOT NULL where appropriate
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier SET NOT NULL;

-- Step 11: Recreate essential functions
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

CREATE OR REPLACE FUNCTION public.check_tier_feature_access(user_tier tier_enum, feature_name text)
RETURNS boolean
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = public
AS $function$
  SELECT CASE feature_name
    WHEN 'training_program' THEN true
    WHEN 'nutrition_plan' THEN true
    WHEN 'qa_access' THEN true
    WHEN 'weekly_checkins' THEN user_tier::text IN ('adaptive', 'performance', 'longevity')
    WHEN 'form_reviews' THEN user_tier::text IN ('adaptive', 'performance', 'longevity')
    WHEN 'wearable_integration' THEN user_tier::text IN ('adaptive', 'performance', 'longevity')
    WHEN 'ai_assistant' THEN user_tier::text IN ('performance', 'longevity')
    WHEN 'biomarker_integration' THEN user_tier::text IN ('performance', 'longevity')
    WHEN 'supplements' THEN user_tier::text IN ('adaptive', 'performance', 'longevity')
    WHEN 'peptides' THEN user_tier::text IN ('performance', 'longevity')
    WHEN 'bioregulators' THEN user_tier::text = 'longevity'
    WHEN 'telegram_community' THEN user_tier::text IN ('performance', 'longevity')
    WHEN 'in_person_training' THEN user_tier::text = 'longevity'
    ELSE false
  END;
$function$;

-- Step 12: Recreate the main trigger
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

CREATE TRIGGER sync_tier_features_trigger
  BEFORE INSERT OR UPDATE OF subscription_tier ON client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION sync_tier_features();

-- Step 13: Update tier_features for all existing users
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier)
WHERE subscription_tier IS NOT NULL;

-- Step 14: Clean up temporary columns
ALTER TABLE client_profiles DROP COLUMN IF EXISTS tier_final;
ALTER TABLE demo_profiles DROP COLUMN IF EXISTS tier_final;
ALTER TABLE subscription_tiers DROP COLUMN IF EXISTS tier_final;
ALTER TABLE leaderboards DROP COLUMN IF EXISTS tier_final;
ALTER TABLE leaderboard_entries DROP COLUMN IF EXISTS tier_final;
ALTER TABLE achievements DROP COLUMN IF EXISTS tier_final;
ALTER TABLE supplement_protocols DROP COLUMN IF EXISTS tier_final;
ALTER TABLE supplement_library DROP COLUMN IF EXISTS tier_final;
ALTER TABLE pricing_plans DROP COLUMN IF EXISTS tier_final;
ALTER TABLE user_tier_access DROP COLUMN IF EXISTS tier_final;

-- Step 15: Drop the mapping function
DROP FUNCTION final_tier_mapping(text);

-- Step 16: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_user_tier_access_current_tier ON user_tier_access(current_tier);

-- Step 17: Final verification and summary
DO $$
DECLARE
    final_enum_values text[];
    table_summary text;
BEGIN
    -- Get final enum values
    SELECT array_agg(enumlabel ORDER BY enumsortorder) 
    INTO final_enum_values
    FROM pg_enum 
    WHERE enumtypid = 'tier_enum'::regtype;
    
    RAISE NOTICE 'MIGRATION COMPLETE!';
    RAISE NOTICE 'Final tier_enum values: %', final_enum_values;
    
    -- Generate summary of all tables
    table_summary := format('
Final Table Summary:
- client_profiles: %s records with tiers
- demo_profiles: %s records with tiers  
- subscription_tiers: %s records with tiers
- leaderboards: %s records with tiers
- leaderboard_entries: %s records with tiers
- achievements: %s records with tiers
- supplement_protocols: %s records with tiers
- supplement_library: %s records with tiers
- pricing_plans: %s records with tiers
- user_tier_access: %s records with tiers',
        (SELECT count(*) FROM client_profiles WHERE subscription_tier IS NOT NULL),
        (SELECT count(*) FROM demo_profiles WHERE subscription_tier IS NOT NULL),
        (SELECT count(*) FROM subscription_tiers WHERE tier_name IS NOT NULL),
        (SELECT count(*) FROM leaderboards WHERE tier_filter IS NOT NULL),
        (SELECT count(*) FROM leaderboard_entries WHERE tier IS NOT NULL),
        (SELECT count(*) FROM achievements WHERE tier_requirement IS NOT NULL),
        (SELECT count(*) FROM supplement_protocols WHERE tier_requirement IS NOT NULL),
        (SELECT count(*) FROM supplement_library WHERE tier_recommendation IS NOT NULL),
        (SELECT count(*) FROM pricing_plans WHERE tier_level IS NOT NULL),
        (SELECT count(*) FROM user_tier_access WHERE current_tier IS NOT NULL)
    );
    
    RAISE NOTICE '%', table_summary;
    
    -- Test the functions work
    PERFORM get_tier_features('core'::tier_enum);
    PERFORM has_tier_access('adaptive'::tier_enum);
    
    RAISE NOTICE 'All functions tested successfully!';
END $$;

COMMIT;