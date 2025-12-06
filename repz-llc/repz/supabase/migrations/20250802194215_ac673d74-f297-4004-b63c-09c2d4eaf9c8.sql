-- Migration: Ensure tier system consistency and update functions
-- The enum structure is already correct, now updating functions and policies

-- Step 1: Update get_tier_features function to use correct tier names and modern features
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

-- Step 2: Ensure trigger exists for tier feature sync
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

-- Step 3: Create/update trigger for client_profiles
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON public.client_profiles;
CREATE TRIGGER trigger_sync_tier_features
  BEFORE UPDATE OF subscription_tier ON public.client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.sync_tier_features();

-- Step 4: Update all existing client profiles with correct tier features
UPDATE public.client_profiles 
SET tier_features = public.get_tier_features(subscription_tier),
    updated_at = NOW();

-- Step 5: Ensure RLS policies use correct tier names
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

-- Step 6: Add validation function to check tier system health
CREATE OR REPLACE FUNCTION public.validate_tier_system()
RETURNS TABLE(check_name text, status text, issues_count integer, details text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
BEGIN
  -- Check 1: Users with valid tiers
  RETURN QUERY
  SELECT 
    'Tier Validation'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All tiers valid' ELSE 'Invalid tiers found: ' || COUNT(*)::TEXT END::TEXT
  FROM public.client_profiles 
  WHERE subscription_tier NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  -- Check 2: Tier-feature consistency
  RETURN QUERY
  SELECT 
    'Tier Feature Consistency'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All features consistent' ELSE 'Features need sync: ' || COUNT(*)::TEXT END::TEXT
  FROM public.client_profiles cp
  WHERE cp.tier_features != public.get_tier_features(cp.subscription_tier);
  
  -- Check 3: Required tier features present
  RETURN QUERY
  SELECT 
    'Essential Features Check'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All essential features present' ELSE 'Missing essential features' END::TEXT
  FROM public.client_profiles 
  WHERE NOT (tier_features ? 'progressPhotos' AND tier_features ? 'coachAccess' AND tier_features ? 'response_time_hours');
  
END;
$function$;

-- Step 7: Create production readiness validation
CREATE OR REPLACE FUNCTION public.validate_production_readiness()
RETURNS TABLE(component text, status text, priority text, message text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
BEGIN
  -- Database tier system
  RETURN QUERY
  SELECT 
    'Tier System'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'READY' ELSE 'NEEDS_ATTENTION' END::TEXT,
    'HIGH'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'Tier system validated' ELSE 'Tier system has issues' END::TEXT
  FROM public.client_profiles 
  WHERE subscription_tier NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  -- User profiles validation
  RETURN QUERY
  SELECT 
    'User Profiles'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'READY' ELSE 'NEEDS_ATTENTION' END::TEXT,
    'HIGH'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'All users have valid profiles' ELSE 'Some users missing profiles' END::TEXT
  FROM auth.users au
  LEFT JOIN public.client_profiles cp ON au.id = cp.auth_user_id
  LEFT JOIN public.coach_profiles coach ON au.id = coach.auth_user_id
  WHERE au.email NOT LIKE '%@supabase.io'
    AND cp.auth_user_id IS NULL 
    AND coach.auth_user_id IS NULL;
    
  -- Coach contact info
  RETURN QUERY
  SELECT 
    'Coach Contact'::TEXT,
    CASE WHEN COUNT(*) > 0 THEN 'READY' ELSE 'NEEDS_ATTENTION' END::TEXT,
    'MEDIUM'::TEXT,
    CASE WHEN COUNT(*) > 0 THEN 'Primary coach configured' ELSE 'No primary coach set' END::TEXT
  FROM public.coach_contact_info 
  WHERE is_primary = true;
  
END;
$function$;