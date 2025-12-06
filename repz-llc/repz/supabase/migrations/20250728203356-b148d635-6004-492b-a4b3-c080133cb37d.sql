-- Fix security warnings: Set search_path for functions

-- Drop and recreate functions with proper search_path
DROP FUNCTION IF EXISTS get_user_tier(uuid);
DROP FUNCTION IF EXISTS get_user_role_text(uuid);

-- Create secure helper functions with search_path set
CREATE OR REPLACE FUNCTION get_user_tier(user_uuid uuid)
RETURNS text
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT COALESCE(cp.subscription_tier::text, 'baseline')
  FROM public.client_profiles cp 
  WHERE cp.auth_user_id = user_uuid
  UNION ALL
  SELECT 'longevity' 
  FROM public.coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION get_user_role_text(user_uuid uuid)
RETURNS text  
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT 'client'
  FROM public.client_profiles cp 
  WHERE cp.auth_user_id = user_uuid
  UNION ALL
  SELECT 'coach'
  FROM public.coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  UNION ALL
  SELECT 'admin'
  FROM public.admin_users au
  WHERE au.user_id = user_uuid OR au.email = (SELECT email FROM auth.users WHERE id = user_uuid)
  LIMIT 1;
$$;