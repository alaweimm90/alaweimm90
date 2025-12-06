-- Fix security warnings: Set search_path for functions to prevent injection attacks

-- Fix function 1: handle_new_user_profile_creation
CREATE OR REPLACE FUNCTION public.handle_new_user_profile_creation()
RETURNS TRIGGER AS $$
BEGIN
  -- Check if user already has a profile
  IF NOT EXISTS (
    SELECT 1 FROM public.client_profiles WHERE auth_user_id = NEW.id
    UNION
    SELECT 1 FROM public.coach_profiles WHERE auth_user_id = NEW.id
  ) THEN
    -- Create default client profile for new users
    INSERT INTO public.client_profiles (
      auth_user_id,
      client_name,
      subscription_tier,
      created_at,
      updated_at
    ) VALUES (
      NEW.id,
      COALESCE(NEW.raw_user_meta_data->>'name', split_part(NEW.email, '@', 1)),
      'baseline',
      NOW(),
      NOW()
    );
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- Fix function 2: validate_platform_consistency
CREATE OR REPLACE FUNCTION public.validate_platform_consistency()
RETURNS TABLE(check_name TEXT, status TEXT, issues_count INTEGER, details TEXT) AS $$
BEGIN
  -- Check 1: Users without profiles
  RETURN QUERY
  SELECT 
    'Users Without Profiles'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All users have profiles' ELSE 'Users missing profiles: ' || string_agg(email, ', ') END::TEXT
  FROM auth.users au
  LEFT JOIN public.client_profiles cp ON au.id = cp.auth_user_id
  LEFT JOIN public.coach_profiles coach ON au.id = coach.auth_user_id
  WHERE au.email NOT LIKE '%@supabase.io'
    AND cp.auth_user_id IS NULL 
    AND coach.auth_user_id IS NULL;
  
  -- Check 2: Invalid subscription tiers
  RETURN QUERY
  SELECT 
    'Invalid Subscription Tiers'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All tiers valid' ELSE 'Invalid tiers found' END::TEXT
  FROM public.client_profiles 
  WHERE subscription_tier NOT IN ('baseline', 'prime', 'precision', 'longevity');
  
  -- Check 3: Empty profile names
  RETURN QUERY
  SELECT 
    'Empty Profile Names'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All profiles have names' ELSE 'Profiles missing names' END::TEXT
  FROM (
    SELECT client_name as name FROM public.client_profiles WHERE client_name IS NULL OR length(trim(client_name)) = 0
    UNION ALL
    SELECT coach_name as name FROM public.coach_profiles WHERE coach_name IS NULL OR length(trim(coach_name)) = 0
  ) empty_names;
  
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';