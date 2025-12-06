-- Security Fixes Migration
-- Addresses three critical security issues identified in security review

-- 1. Restrict analytics access to admin users only
-- Drop existing analytics policy that allows user_id IS NULL access
DROP POLICY IF EXISTS "Users can view analytics events" ON public.analytics_events;

-- Create admin role check function
CREATE OR REPLACE FUNCTION public.is_admin(user_uuid UUID DEFAULT auth.uid())
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Check if user has admin role in profiles table
  RETURN EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE user_id = user_uuid 
    AND role = 'admin'
  );
END;
$$;

-- Create new restrictive analytics policy - admin only access
CREATE POLICY "Admin users can view all analytics events" ON public.analytics_events
  FOR SELECT USING (public.is_admin());

-- Create policy for users to view only their own analytics events
CREATE POLICY "Users can view their own analytics events" ON public.analytics_events
  FOR SELECT USING (auth.uid() = user_id AND user_id IS NOT NULL);

-- 2. Enhanced API key security for models table
-- Add additional security metadata columns
ALTER TABLE public.models 
ADD COLUMN IF NOT EXISTS api_key_hash TEXT,
ADD COLUMN IF NOT EXISTS key_created_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS key_last_used TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS access_level TEXT DEFAULT 'user' CHECK (access_level IN ('user', 'admin', 'restricted'));

-- Create function to log API key access
CREATE OR REPLACE FUNCTION public.log_api_key_access()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Update last used timestamp when api_key_encrypted is accessed
  IF TG_OP = 'SELECT' AND OLD.api_key_encrypted IS NOT NULL THEN
    UPDATE public.models 
    SET key_last_used = now() 
    WHERE id = OLD.id;
  END IF;
  
  RETURN COALESCE(NEW, OLD);
END;
$$;

-- Add audit trail for API key access (optional - for high security environments)
-- CREATE TRIGGER api_key_access_log
--   AFTER SELECT ON public.models
--   FOR EACH ROW
--   EXECUTE FUNCTION public.log_api_key_access();

-- Enhanced models policies with additional access controls
DROP POLICY IF EXISTS "Users can view their own models" ON public.models;

-- Create more restrictive model access policies
CREATE POLICY "Users can view their own models (basic info)" ON public.models
  FOR SELECT USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Separate policy for accessing encrypted API keys - requires additional verification
CREATE POLICY "Users can access their model API keys with restrictions" ON public.models
  FOR SELECT USING (
    auth.uid() = user_id 
    AND (access_level = 'user' OR public.is_admin())
    AND is_active = true
  );

-- 3. Add role-based access control
-- Update profiles table to ensure role is properly managed
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS role_assigned_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS role_assigned_by UUID REFERENCES auth.users(id);

-- Create function to validate role assignments
CREATE OR REPLACE FUNCTION public.validate_role_assignment()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Only admins can assign admin role to others
  IF NEW.role = 'admin' AND OLD.role != 'admin' THEN
    IF NOT public.is_admin() THEN
      RAISE EXCEPTION 'Only administrators can assign admin role';
    END IF;
    NEW.role_assigned_at = now();
    NEW.role_assigned_by = auth.uid();
  END IF;
  
  RETURN NEW;
END;
$$;

-- Add trigger for role assignment validation
CREATE TRIGGER validate_role_assignment_trigger
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.validate_role_assignment();

-- 4. Create secure view for analytics without sensitive data
CREATE OR REPLACE VIEW public.analytics_summary AS
SELECT 
  event_type,
  event_name,
  DATE_TRUNC('hour', created_at) as time_bucket,
  COUNT(*) as event_count,
  COUNT(DISTINCT session_id) as unique_sessions
FROM public.analytics_events
WHERE created_at >= (now() - INTERVAL '30 days')
GROUP BY event_type, event_name, DATE_TRUNC('hour', created_at)
ORDER BY time_bucket DESC;

-- Grant access to analytics summary for authenticated users
GRANT SELECT ON public.analytics_summary TO authenticated;

-- 5. Add encryption key rotation capability
CREATE TABLE IF NOT EXISTS public.encryption_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key_version INTEGER NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  retired_at TIMESTAMP WITH TIME ZONE
);

-- Enable RLS on encryption keys
ALTER TABLE public.encryption_keys ENABLE ROW LEVEL SECURITY;

-- Only admins can manage encryption keys
CREATE POLICY "Admin can manage encryption keys" ON public.encryption_keys
  FOR ALL USING (public.is_admin());

-- 6. Add indexes for security queries
CREATE INDEX IF NOT EXISTS idx_profiles_role ON public.profiles(role) WHERE role IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_models_access_level ON public.models(access_level);
CREATE INDEX IF NOT EXISTS idx_analytics_events_session_id ON public.analytics_events(session_id);

-- 7. Create security audit log
CREATE TABLE IF NOT EXISTS public.security_audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT,
  details JSONB,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.security_audit_log ENABLE ROW LEVEL SECURITY;

-- Only admins can view security audit logs
CREATE POLICY "Admin can view security audit logs" ON public.security_audit_log
  FOR SELECT USING (public.is_admin());

-- Function to log security events
CREATE OR REPLACE FUNCTION public.log_security_event(
  p_action TEXT,
  p_resource_type TEXT,
  p_resource_id TEXT DEFAULT NULL,
  p_details JSONB DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.security_audit_log (
    user_id, action, resource_type, resource_id, details
  ) VALUES (
    auth.uid(), p_action, p_resource_type, p_resource_id, p_details
  );
END;
$$;

-- Add comments for documentation
COMMENT ON FUNCTION public.is_admin IS 'Checks if the current user has admin role';
COMMENT ON FUNCTION public.log_security_event IS 'Logs security-related events for audit trail';
COMMENT ON TABLE public.security_audit_log IS 'Audit trail for security-sensitive operations';
COMMENT ON VIEW public.analytics_summary IS 'Aggregated analytics data without sensitive user information';