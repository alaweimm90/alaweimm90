-- Migration: Fix tier enum to correct structure and migrate data
-- This migration updates the tier system from old names to new structure

-- Step 1: Create new tier enum with correct values
DO $$ 
BEGIN
  -- Drop the old enum if it exists and create new one
  DROP TYPE IF EXISTS tier_enum CASCADE;
  CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');
END $$;

-- Step 2: Update client_profiles table to use new tier enum
ALTER TABLE public.client_profiles 
ALTER COLUMN subscription_tier TYPE tier_enum 
USING CASE 
  WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum
  WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum  
  WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum
  WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum
  ELSE 'core'::tier_enum
END;

-- Step 3: Update demo_profiles table to use new tier enum
ALTER TABLE public.demo_profiles 
ALTER COLUMN subscription_tier TYPE tier_enum 
USING CASE 
  WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum
  WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum
  WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum
  WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum
  ELSE 'core'::tier_enum
END;

-- Step 4: Update primary_goal enum to match expected values
DO $$
BEGIN
  DROP TYPE IF EXISTS goal_enum CASCADE;
  CREATE TYPE goal_enum AS ENUM ('weight_loss', 'muscle_gain', 'strength', 'endurance', 'general_fitness', 'longevity');
END $$;

-- Step 5: Update client_profiles primary_goal column
ALTER TABLE public.client_profiles 
ALTER COLUMN primary_goal TYPE goal_enum 
USING CASE 
  WHEN primary_goal::text = 'weight_loss' THEN 'weight_loss'::goal_enum
  WHEN primary_goal::text = 'muscle_gain' THEN 'muscle_gain'::goal_enum
  WHEN primary_goal::text = 'strength' THEN 'strength'::goal_enum
  WHEN primary_goal::text = 'endurance' THEN 'endurance'::goal_enum
  WHEN primary_goal::text = 'general_fitness' THEN 'general_fitness'::goal_enum
  WHEN primary_goal::text = 'longevity' THEN 'longevity'::goal_enum
  ELSE 'general_fitness'::goal_enum
END;

-- Step 6: Update demo_profiles primary_goal column
ALTER TABLE public.demo_profiles 
ALTER COLUMN primary_goal TYPE goal_enum 
USING CASE 
  WHEN primary_goal::text = 'weight_loss' THEN 'weight_loss'::goal_enum
  WHEN primary_goal::text = 'muscle_gain' THEN 'muscle_gain'::goal_enum
  WHEN primary_goal::text = 'strength' THEN 'strength'::goal_enum
  WHEN primary_goal::text = 'endurance' THEN 'endurance'::goal_enum
  WHEN primary_goal::text = 'general_fitness' THEN 'general_fitness'::goal_enum
  WHEN primary_goal::text = 'longevity' THEN 'longevity'::goal_enum
  ELSE 'general_fitness'::goal_enum
END;

-- Step 7: Update get_tier_features function to use new tier names
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
      'dashboard_type', 'premium_advanced'
    )
    ELSE jsonb_build_object('error', 'invalid_tier')
  END;
END;
$function$;

-- Step 8: Update get_user_tier function
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

-- Step 9: Trigger to sync tier features when tier changes
CREATE OR REPLACE FUNCTION public.sync_tier_features()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $function$
BEGIN
  NEW.tier_features = public.get_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$function$;

-- Step 10: Create trigger for client_profiles
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON public.client_profiles;
CREATE TRIGGER trigger_sync_tier_features
  BEFORE UPDATE OF subscription_tier ON public.client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.sync_tier_features();

-- Step 11: Update all existing client profiles with correct tier features
UPDATE public.client_profiles 
SET tier_features = public.get_tier_features(subscription_tier),
    updated_at = NOW();

-- Step 12: Set default tier for client_profiles
ALTER TABLE public.client_profiles 
ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;

-- Step 13: Update RLS policies to use new tier names
DROP POLICY IF EXISTS "biomarker_precision_access" ON public.biomarker_tests;
CREATE POLICY "biomarker_precision_access" ON public.biomarker_tests
FOR ALL USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = biomarker_tests.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['performance'::tier_enum, 'longevity'::tier_enum])
  )
);

DROP POLICY IF EXISTS "daily_tracking_tier_access" ON public.daily_tracking;
CREATE POLICY "daily_tracking_tier_access" ON public.daily_tracking
FOR ALL USING (
  EXISTS (
    SELECT 1 FROM client_profiles cp
    WHERE cp.id = daily_tracking.client_id 
    AND cp.auth_user_id = auth.uid() 
    AND cp.subscription_tier = ANY (ARRAY['adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum])
  )
);

-- Step 14: Add data validation function
CREATE OR REPLACE FUNCTION public.validate_tier_migration()
RETURNS TABLE(check_name text, status text, count integer, details text)
LANGUAGE plpgsql
SECURITY DEFINER
AS $function$
BEGIN
  -- Check client profiles have valid tiers
  RETURN QUERY
  SELECT 
    'Client Profiles Tier Validation'::text,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::text,
    COUNT(*)::integer,
    CASE WHEN COUNT(*) = 0 THEN 'All tiers valid' ELSE 'Invalid tiers found' END::text
  FROM public.client_profiles
  WHERE subscription_tier NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  -- Check tier features are properly synced
  RETURN QUERY
  SELECT 
    'Tier Features Sync'::text,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::text,
    COUNT(*)::integer,
    CASE WHEN COUNT(*) = 0 THEN 'All features synced' ELSE 'Features need sync' END::text
  FROM public.client_profiles
  WHERE tier_features != public.get_tier_features(subscription_tier);
  
END;
$function$;