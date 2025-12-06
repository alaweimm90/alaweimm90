-- Drop the check constraint that's preventing the update
ALTER TABLE public.client_profiles DROP CONSTRAINT IF EXISTS check_subscription_tier;

-- Phase 3: Update existing client_profiles to use new tier names
UPDATE public.client_profiles SET
  subscription_tier = CASE 
    WHEN subscription_tier = 'baseline' THEN 'core'::tier_enum
    WHEN subscription_tier = 'prime' THEN 'adaptive'::tier_enum
    WHEN subscription_tier = 'precision' THEN 'performance'::tier_enum
    WHEN subscription_tier = 'longevity' THEN 'longevity'::tier_enum
    ELSE subscription_tier
  END;

-- Create function to get tier features dynamically
CREATE OR REPLACE FUNCTION public.get_new_tier_features(tier_name tier_enum)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
DECLARE
  feature_map jsonb := '{}';
  tier_column text;
BEGIN
  -- Map tier names to column names
  tier_column := CASE tier_name::text
    WHEN 'core' THEN 'core'
    WHEN 'adaptive' THEN 'adaptive'
    WHEN 'performance' THEN 'performance'
    WHEN 'longevity' THEN 'longevity'
    ELSE 'core'
  END;
  
  -- Build feature map from tier_features table
  SELECT jsonb_object_agg(
    feature_key,
    CASE tier_column
      WHEN 'core' THEN core
      WHEN 'adaptive' THEN adaptive
      WHEN 'performance' THEN performance
      WHEN 'longevity' THEN longevity
    END
  ) INTO feature_map
  FROM public.tier_features;
  
  -- Add tier-specific metadata
  feature_map := feature_map || jsonb_build_object(
    'dashboard_type', CASE tier_column
      WHEN 'core' THEN 'static'
      WHEN 'adaptive' THEN 'interactive'
      WHEN 'performance' THEN 'advanced'
      WHEN 'longevity' THEN 'premium'
    END,
    'response_time_hours', CASE tier_column
      WHEN 'core' THEN 72
      WHEN 'adaptive' THEN 48
      WHEN 'performance' THEN 24
      WHEN 'longevity' THEN 12
    END
  );
  
  RETURN feature_map;
END;
$function$;

-- Update tier_features for all existing client profiles
UPDATE public.client_profiles 
SET tier_features = public.get_new_tier_features(subscription_tier);

-- Create trigger to automatically update tier_features when subscription_tier changes
CREATE OR REPLACE FUNCTION public.sync_new_tier_features()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO ''
AS $function$
BEGIN
  NEW.tier_features = public.get_new_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$function$;

-- Create trigger
DROP TRIGGER IF EXISTS sync_new_tier_features_trigger ON public.client_profiles;
CREATE TRIGGER sync_new_tier_features_trigger
  BEFORE UPDATE OF subscription_tier ON public.client_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.sync_new_tier_features();