-- PHASE 1 CRITICAL FIXES: Database Integrity & Profile Management

-- 1. Add missing foreign key constraints for data integrity
ALTER TABLE client_profiles 
ADD CONSTRAINT fk_client_profiles_auth_user_id 
FOREIGN KEY (auth_user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

ALTER TABLE coach_profiles 
ADD CONSTRAINT fk_coach_profiles_auth_user_id 
FOREIGN KEY (auth_user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

-- 2. Create automatic profile creation trigger
CREATE OR REPLACE FUNCTION public.handle_new_user_profile_creation()
RETURNS TRIGGER AS $$
BEGIN
  -- Check if user already has a profile
  IF NOT EXISTS (
    SELECT 1 FROM client_profiles WHERE auth_user_id = NEW.id
    UNION
    SELECT 1 FROM coach_profiles WHERE auth_user_id = NEW.id
  ) THEN
    -- Create default client profile for new users
    INSERT INTO client_profiles (
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
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger for automatic profile creation
DROP TRIGGER IF EXISTS trigger_handle_new_user_profile ON auth.users;
CREATE TRIGGER trigger_handle_new_user_profile
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user_profile_creation();

-- 3. Fix existing users without profiles
INSERT INTO client_profiles (auth_user_id, client_name, subscription_tier, created_at, updated_at)
SELECT 
  au.id,
  COALESCE(au.raw_user_meta_data->>'name', split_part(au.email, '@', 1)),
  'baseline',
  NOW(),
  NOW()
FROM auth.users au
LEFT JOIN client_profiles cp ON au.id = cp.auth_user_id
LEFT JOIN coach_profiles coach ON au.id = coach.auth_user_id
WHERE au.email NOT LIKE '%@supabase.io'
  AND cp.auth_user_id IS NULL 
  AND coach.auth_user_id IS NULL;

-- 4. Add data validation constraints
ALTER TABLE client_profiles 
ADD CONSTRAINT check_subscription_tier 
CHECK (subscription_tier IN ('baseline', 'prime', 'precision', 'longevity'));

-- 5. Add constraint to ensure client_name is not empty
ALTER TABLE client_profiles 
ADD CONSTRAINT check_client_name_not_empty 
CHECK (client_name IS NOT NULL AND length(trim(client_name)) > 0);

ALTER TABLE coach_profiles 
ADD CONSTRAINT check_coach_name_not_empty 
CHECK (coach_name IS NOT NULL AND length(trim(coach_name)) > 0);

-- 6. Create comprehensive data validation function
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
  LEFT JOIN client_profiles cp ON au.id = cp.auth_user_id
  LEFT JOIN coach_profiles coach ON au.id = coach.auth_user_id
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
  FROM client_profiles 
  WHERE subscription_tier NOT IN ('baseline', 'prime', 'precision', 'longevity');
  
  -- Check 3: Empty profile names
  RETURN QUERY
  SELECT 
    'Empty Profile Names'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All profiles have names' ELSE 'Profiles missing names' END::TEXT
  FROM (
    SELECT client_name as name FROM client_profiles WHERE client_name IS NULL OR length(trim(client_name)) = 0
    UNION ALL
    SELECT coach_name as name FROM coach_profiles WHERE coach_name IS NULL OR length(trim(coach_name)) = 0
  ) empty_names;
  
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;