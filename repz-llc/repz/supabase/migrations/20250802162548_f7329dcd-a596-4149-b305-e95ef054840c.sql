-- CRITICAL: Clean up tier_enum to contain only canonical values (Fixed Version)
-- This migration addresses the mixed tier enum values identified in the audit

-- Step 1: Create new clean tier enum with only canonical values
DROP TYPE IF EXISTS tier_enum_new CASCADE;
CREATE TYPE tier_enum_new AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Step 2: Add temporary column with new enum type
ALTER TABLE client_profiles ADD COLUMN subscription_tier_new tier_enum_new;
ALTER TABLE demo_profiles ADD COLUMN subscription_tier_new tier_enum_new;
ALTER TABLE pricing_plans ADD COLUMN tier_level_new tier_enum_new;

-- Step 3: Update all existing data to use canonical tier names
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
  -- Also handle canonical names that are already correct
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

-- Step 4: Drop old columns and enum
ALTER TABLE client_profiles DROP COLUMN subscription_tier;
ALTER TABLE demo_profiles DROP COLUMN subscription_tier;
ALTER TABLE pricing_plans DROP COLUMN tier_level;
DROP TYPE tier_enum CASCADE;

-- Step 5: Rename new enum and columns
ALTER TYPE tier_enum_new RENAME TO tier_enum;
ALTER TABLE client_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE demo_profiles RENAME COLUMN subscription_tier_new TO subscription_tier;
ALTER TABLE pricing_plans RENAME COLUMN tier_level_new TO tier_level;

-- Step 6: Set NOT NULL constraints where appropriate
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET NOT NULL;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core';

-- Step 7: Recreate dependent database functions with updated logic
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

-- Step 8: Update user_tier function to handle canonical names
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

-- Step 9: Refresh tier features for all existing client profiles
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier),
    updated_at = NOW();

-- Step 10: Validate migration success
DO $$
DECLARE
  invalid_count INTEGER;
  total_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO total_count FROM client_profiles;
  SELECT COUNT(*) INTO invalid_count
  FROM client_profiles 
  WHERE subscription_tier::text NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  IF invalid_count > 0 THEN
    RAISE EXCEPTION 'Migration validation failed: % out of % profiles have invalid tier values', invalid_count, total_count;
  END IF;
  
  RAISE NOTICE 'Migration completed successfully. All % client profiles now use canonical tier values.', total_count;
END $$;