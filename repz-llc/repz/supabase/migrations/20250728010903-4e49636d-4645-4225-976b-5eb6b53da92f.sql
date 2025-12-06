-- Fix security warnings by adding search_path to functions
CREATE OR REPLACE FUNCTION update_client_tier_features()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Update tier_features based on subscription_tier
  SELECT features INTO NEW.tier_features
  FROM public.subscription_tiers 
  WHERE tier_name = NEW.subscription_tier;
  
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION has_tier_access(required_tier tier_enum)
RETURNS boolean
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.client_profiles 
    WHERE auth_user_id = auth.uid()
    AND subscription_tier::text >= required_tier::text
  );
$$;

CREATE OR REPLACE FUNCTION get_user_role()
RETURNS text
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT 
    CASE 
      WHEN EXISTS (SELECT 1 FROM public.coach_profiles WHERE auth_user_id = auth.uid()) THEN 'coach'
      WHEN EXISTS (SELECT 1 FROM public.client_profiles WHERE auth_user_id = auth.uid()) THEN 'client'
      ELSE null
    END;
$$;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;