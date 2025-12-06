-- Fix remaining security warning: Set search_path for can_access_tier function

CREATE OR REPLACE FUNCTION public.can_access_tier(_user_tier subscription_tier, _required_tier subscription_tier)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
SET search_path = public
AS $$
  SELECT CASE _user_tier
    WHEN 'longevity' THEN true
    WHEN 'performance' THEN _required_tier IN ('core', 'adaptive', 'performance')
    WHEN 'adaptive' THEN _required_tier IN ('core', 'adaptive')
    WHEN 'core' THEN _required_tier = 'core'
    ELSE false
  END
$$;