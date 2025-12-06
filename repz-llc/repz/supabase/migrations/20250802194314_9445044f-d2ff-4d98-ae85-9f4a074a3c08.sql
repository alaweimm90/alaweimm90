-- Fix security warnings by setting search_path for functions

-- Fix validate_tier_system function
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

-- Fix validate_production_readiness function
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