-- CRITICAL: Clean up tier_enum to contain only canonical values (Final Version)
-- This migration addresses the mixed tier enum values identified in the audit
-- Handling all dependent objects properly

-- Step 1: Drop all dependent triggers and policies first
DROP TRIGGER IF EXISTS update_tier_features_trigger ON client_profiles;
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON client_profiles;
DROP TRIGGER IF EXISTS sync_new_tier_features_trigger ON client_profiles;

-- Drop dependent policies temporarily
DROP POLICY IF EXISTS "daily_tracking_tier_access" ON daily_tracking;
DROP POLICY IF EXISTS "biomarker_precision_access" ON biomarker_tests;
DROP POLICY IF EXISTS "progress_photos_tier_access" ON progress_photos;

-- Step 2: Create new clean tier enum with only canonical values
DROP TYPE IF EXISTS tier_enum_new CASCADE;
CREATE TYPE tier_enum_new AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Step 3: Add temporary columns with new enum type
ALTER TABLE client_profiles ADD COLUMN subscription_tier_new tier_enum_new;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier_new tier_enum_new;
ALTER TABLE pricing_plans ADD COLUMN tier_level_new tier_enum_new;

-- Step 4: Convert existing data to canonical tier names
UPDATE client_profiles 
SET subscription_tier_new = CASE subscription_tier::text
  WHEN 'baseline' THEN 'core'::tier_enum_new
  WHEN 'prime' THEN 'adaptive'::tier_enum_new  
  WHEN 'precision' THEN 'performance'::tier_enum_new
  WHEN 'longevity' THEN 'longevity'::tier_enum_new
  WHEN 'foundation_starter' THEN 'core'::tier_enum_new
  WHEN 'growth_accelerator' THEN 'adaptive'::tier_enum_new
  WHEN 'performance_pro' THEN 'performance'::tier_enum_new
  WHEN 'enterprise_elite' THEN 'longevity'::tier_enum_new
  -- Handle canonical names that are already correct
  WHEN 'core' THEN 'core'::tier_enum_new
  WHEN 'adaptive' THEN 'adaptive'::tier_enum_new
  WHEN 'performance' THEN 'performance'::tier_enum_new
  ELSE 'core'::tier_enum_new -- Default fallback
END;

UPDATE demo_profiles 
SET subscription_tier_new = CASE subscription_tier::text
  WHEN 'baseline' THEN 'core'::tier_enum_new
  WHEN 'prime' THEN 'adaptive'::tier_enum_new
  WHEN 'precision' THEN 'performance'::tier_enum_new
  WHEN 'longevity' THEN 'longevity'::tier_enum_new
  WHEN 'core' THEN 'core'::tier_enum_new
  WHEN 'adaptive' THEN 'adaptive'::tier_enum_new
  WHEN 'performance' THEN 'performance'::tier_enum_new
  ELSE 'core'::tier_enum_new
END;

UPDATE pricing_plans 
SET tier_level_new = CASE tier_level::text
  WHEN 'baseline' THEN 'core'::tier_enum_new
  WHEN 'prime' THEN 'adaptive'::tier_enum_new
  WHEN 'precision' THEN 'performance'::tier_enum_new
  WHEN 'longevity' THEN 'longevity'::tier_enum_new
  WHEN 'core' THEN 'core'::tier_enum_new
  WHEN 'adaptive' THEN 'adaptive'::tier_enum_new
  WHEN 'performance' THEN 'performance'::tier_enum_new
  ELSE 'core'::tier_enum_new
END
WHERE tier_level IS NOT NULL;

-- Step 5: Drop old columns with CASCADE to remove dependencies
ALTER TABLE client_profiles DROP COLUMN subscription_tier CASCADE;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier CASCADE;
ALTER TABLE pricing_plans DROP COLUMN tier_level CASCADE;

-- Drop old enum
DROP TYPE tier_enum CASCADE;

-- Step 6: Rename new enum and columns
ALTER TYPE tier_enum_new RENAME TO tier_enum;
ALTER TABLE client_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE demo_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE pricing_plans RENAME COLUMN tier_level_new TO tier_level;

-- Step 7: Set NOT NULL constraints and defaults
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core';

-- Step 8: Recreate updated database functions
CREATE OR REPLACE FUNCTION public.get_tier_features(tier_name tier_enum)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
BEGIN
  RETURN CASE tier_name
    WHEN 'core' THEN jsonb_build_object(
      'progressPhotos', true,
      'nutritionTracking', true, 
      'customWorkouts', true,
      'coachAccess', true,
      'weeklyCheckins', false,
      'formAnalysis', false,
      'bodyCompositionTracking', false,
      'injuryPrevention', false,
      'performanceTracking', false,
      'hrv_optimization', false,
      'biomarker_integration', false,
      'enhanced_analytics', false,
      'ai_coaching', false,
      'live_coaching', false,
      'response_time_hours', 72,
      'dashboard_type', 'static_fixed'
    )
    WHEN 'adaptive' THEN jsonb_build_object(
      'progressPhotos', true,
      'nutritionTracking', true,
      'customWorkouts', true,
      'coachAccess', true,
      'weeklyCheckins', true,
      'formAnalysis', true,
      'bodyCompositionTracking', true,
      'injuryPrevention', true,
      'performanceTracking', true,
      'hrv_optimization', true,
      'biomarker_integration', true,
      'enhanced_analytics', true,
      'ai_coaching', false,
      'live_coaching', false,
      'response_time_hours', 48,
      'dashboard_type', 'interactive_adjustable'
    )
    WHEN 'performance' THEN jsonb_build_object(
      'progressPhotos', true,
      'nutritionTracking', true,
      'customWorkouts', true,
      'coachAccess', true,
      'weeklyCheckins', true,
      'formAnalysis', true,
      'bodyCompositionTracking', true,
      'injuryPrevention', true,
      'performanceTracking', true,
      'hrv_optimization', true,
      'biomarker_integration', true,
      'enhanced_analytics', true,
      'ai_coaching', true,
      'live_coaching', true,
      'response_time_hours', 24,
      'dashboard_type', 'interactive_adjustable'
    )
    WHEN 'longevity' THEN jsonb_build_object(
      'progressPhotos', true,
      'nutritionTracking', true,
      'customWorkouts', true,
      'coachAccess', true,
      'weeklyCheckins', true,
      'formAnalysis', true,
      'bodyCompositionTracking', true,
      'injuryPrevention', true,
      'performanceTracking', true,
      'hrv_optimization', true,
      'biomarker_integration', true,
      'enhanced_analytics', true,
      'ai_coaching', true,
      'live_coaching', true,
      'response_time_hours', 12,
      'dashboard_type', 'interactive_adjustable'
    )
    ELSE jsonb_build_object('error', 'invalid_tier')
  END;
END;
$function$;

-- Step 9: Recreate tier features sync trigger
CREATE OR REPLACE FUNCTION public.sync_tier_features()
RETURNS TRIGGER AS $$
BEGIN
  NEW.tier_features = public.get_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER sync_tier_features_trigger
  BEFORE INSERT OR UPDATE ON client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION sync_tier_features();

-- Step 10: Recreate RLS policies with updated tier names
CREATE POLICY "daily_tracking_tier_access" ON daily_tracking
FOR ALL USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = daily_tracking.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum])
  )
);

CREATE POLICY "biomarker_precision_access" ON biomarker_tests
FOR ALL USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = biomarker_tests.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['performance'::tier_enum, 'longevity'::tier_enum])
  )
);

-- Step 11: Update user tier function
CREATE OR REPLACE FUNCTION public.get_user_tier(user_uuid uuid)
RETURNS text
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO ''
AS $function$
  SELECT COALESCE(cp.subscription_tier::text, 'core')
  FROM public.client_profiles cp 
  WHERE cp.auth_user_id = user_uuid
  UNION ALL
  SELECT 'longevity' 
  FROM public.coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  LIMIT 1;
$function$;

-- Step 12: Refresh tier features for all existing client profiles
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier),
    updated_at = NOW();

-- Step 13: Final validation
DO $$
DECLARE
  invalid_count INTEGER;
  total_count INTEGER;
  core_count INTEGER;
  adaptive_count INTEGER;
  performance_count INTEGER;
  longevity_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO total_count FROM client_profiles;
  SELECT COUNT(*) INTO invalid_count
  FROM client_profiles 
  WHERE subscription_tier::text NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  SELECT COUNT(*) INTO core_count FROM client_profiles WHERE subscription_tier = 'core';
  SELECT COUNT(*) INTO adaptive_count FROM client_profiles WHERE subscription_tier = 'adaptive';
  SELECT COUNT(*) INTO performance_count FROM client_profiles WHERE subscription_tier = 'performance';
  SELECT COUNT(*) INTO longevity_count FROM client_profiles WHERE subscription_tier = 'longevity';
  
  IF invalid_count > 0 THEN
    RAISE EXCEPTION 'Migration validation failed: % out of % profiles have invalid tier values', invalid_count, total_count;
  END IF;
  
  RAISE NOTICE 'TIER ENUM MIGRATION COMPLETED SUCCESSFULLY!';
  RAISE NOTICE 'Total profiles: %, Core: %, Adaptive: %, Performance: %, Longevity: %', total_count, core_count, adaptive_count, performance_count, longevity_count;
  RAISE NOTICE 'Database now contains only canonical tier values: core, adaptive, performance, longevity';
END $$;