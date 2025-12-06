-- Fix validation function with correct table reference
CREATE OR REPLACE FUNCTION public.validate_tier_system()
RETURNS TABLE(check_name TEXT, status TEXT, issues_count INTEGER, details TEXT) AS $$
BEGIN
  -- Check 1: Users with outdated tier_features format
  RETURN QUERY
  SELECT 
    'Outdated Tier Features Format'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All tier features up to date' ELSE 'Users with old format: ' || COUNT(*)::TEXT END::TEXT
  FROM public.client_profiles 
  WHERE tier_features IS NULL 
     OR tier_features = '{}'::jsonb
     OR NOT (tier_features ? 'progressPhotos');
  
  -- Check 2: Tier-feature consistency
  RETURN QUERY
  SELECT 
    'Tier Feature Consistency'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All features consistent' ELSE 'Inconsistent features found' END::TEXT
  FROM public.client_profiles cp
  WHERE cp.tier_features != public.get_tier_features(cp.subscription_tier);
  
  -- Check 3: Missing required tier features
  RETURN QUERY
  SELECT 
    'Missing Required Features'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All required features present' ELSE 'Missing features detected' END::TEXT
  FROM public.client_profiles 
  WHERE NOT (tier_features ? 'progressPhotos' AND tier_features ? 'coachAccess' AND tier_features ? 'response_time_hours');
  
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';