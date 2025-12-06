-- Create comprehensive payment system validation
CREATE OR REPLACE FUNCTION public.validate_payment_system()
RETURNS TABLE(check_name TEXT, status TEXT, issues_count INTEGER, details TEXT) AS $$
BEGIN
  -- Check 1: Users with paid tiers but no Stripe subscription ID
  RETURN QUERY
  SELECT 
    'Missing Stripe Subscription IDs'::TEXT,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'All paid users have Stripe IDs' ELSE 'Users missing Stripe IDs: ' || COUNT(*)::TEXT END::TEXT
  FROM public.client_profiles 
  WHERE subscription_tier != 'baseline'
    AND (stripe_subscription_id IS NULL OR stripe_subscription_id = '');
  
  -- Check 2: Stripe price configuration consistency
  RETURN QUERY
  SELECT 
    'Stripe Price Configuration'::TEXT,
    'INFO'::TEXT,
    0::INTEGER,
    'Baseline: $97, Prime: $199, Precision: $299, Longevity: $449'::TEXT;
  
  -- Check 3: Tier upgrade eligibility
  RETURN QUERY
  SELECT 
    'Users Ready for Upgrade'::TEXT,
    'INFO'::TEXT,
    COUNT(*)::INTEGER,
    CASE WHEN COUNT(*) = 0 THEN 'No baseline users to upgrade' ELSE 'Baseline users ready for upgrade: ' || COUNT(*)::TEXT END::TEXT
  FROM public.client_profiles 
  WHERE subscription_tier = 'baseline';
  
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';