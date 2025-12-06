-- PHASE 2 CRITICAL FIXES: Standardize Tier System & Feature Management

-- 1. Create standardized tier feature mapping function
CREATE OR REPLACE FUNCTION public.get_tier_features(tier_name tier_enum)
RETURNS jsonb AS $$
BEGIN
  RETURN CASE tier_name
    WHEN 'baseline' THEN jsonb_build_object(
      -- Basic features only
      'progressPhotos', false,
      'nutritionTracking', false, 
      'customWorkouts', false,
      'coachAccess', false,
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
      'dashboard_type', 'static'
    )
    WHEN 'prime' THEN jsonb_build_object(
      -- Prime features
      'progressPhotos', true,
      'nutritionTracking', true,
      'customWorkouts', true,
      'coachAccess', true,
      'weeklyCheckins', true,
      'formAnalysis', false,
      'bodyCompositionTracking', true,
      'injuryPrevention', true,
      'performanceTracking', true,
      'hrv_optimization', false,
      'biomarker_integration', false,
      'enhanced_analytics', true,
      'ai_coaching', false,
      'live_coaching', false,
      'response_time_hours', 48,
      'dashboard_type', 'interactive'
    )
    WHEN 'precision' THEN jsonb_build_object(
      -- Precision features  
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
      'live_coaching', false,
      'response_time_hours', 24,
      'dashboard_type', 'advanced'
    )
    WHEN 'longevity' THEN jsonb_build_object(
      -- Longevity features (all features)
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
      'dashboard_type', 'premium'
    )
    ELSE jsonb_build_object('error', 'invalid_tier')
  END;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- 2. Create trigger to automatically update tier features when tier changes
CREATE OR REPLACE FUNCTION public.sync_tier_features()
RETURNS TRIGGER AS $$
BEGIN
  -- Update tier_features based on subscription_tier change
  NEW.tier_features = public.get_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- Create or replace the trigger
DROP TRIGGER IF EXISTS trigger_sync_tier_features ON client_profiles;
CREATE TRIGGER trigger_sync_tier_features
  BEFORE INSERT OR UPDATE OF subscription_tier ON client_profiles
  FOR EACH ROW 
  EXECUTE FUNCTION public.sync_tier_features();

-- 3. Update existing users with correct tier features
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier),
    updated_at = NOW()
WHERE tier_features IS NULL 
   OR tier_features = '{}'::jsonb
   OR NOT (tier_features ? 'progressPhotos'); -- Check for new format

-- 4. Create function to validate tier access for features
CREATE OR REPLACE FUNCTION public.check_tier_feature_access(
  user_tier tier_enum,
  feature_name text
)
RETURNS boolean AS $$
DECLARE
  tier_features jsonb;
  has_access boolean := false;
BEGIN
  -- Get tier features
  tier_features := public.get_tier_features(user_tier);
  
  -- Check if feature exists and is enabled
  IF tier_features ? feature_name THEN
    has_access := (tier_features ->> feature_name)::boolean;
  END IF;
  
  RETURN COALESCE(has_access, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- 5. Create comprehensive tier validation function
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
  FROM client_profiles 
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
  FROM client_profiles cp
  WHERE cp.tier_features != public.get_tier_features(cp.subscription_tier);
  
  -- Check 3: Missing required tier features
  RETURN QUERY
  SELECT 
    'Missing Required Features'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All required features present' ELSE 'Missing features detected' END::TEXT
  FROM client_profiles 
  WHERE NOT (tier_features ? 'progressPhotos' AND tier_features ? 'coachAccess' AND tier_features ? 'response_time_hours');
  
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';