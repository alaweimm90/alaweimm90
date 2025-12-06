-- Create security audit logs table
CREATE TABLE public.security_audit_logs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  event_category TEXT NOT NULL,
  event_details JSONB NOT NULL DEFAULT '{}',
  ip_address INET,
  user_agent TEXT,
  session_id TEXT,
  risk_score INTEGER DEFAULT 0,
  requires_action BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create compliance monitoring table
CREATE TABLE public.compliance_events (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  compliance_type TEXT NOT NULL, -- GDPR, HIPAA, etc.
  event_type TEXT NOT NULL,
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  data_subject TEXT,
  legal_basis TEXT,
  purpose TEXT,
  retention_period INTERVAL,
  event_details JSONB NOT NULL DEFAULT '{}',
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE
);

-- Create data encryption keys table
CREATE TABLE public.data_encryption_keys (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  key_name TEXT NOT NULL UNIQUE,
  key_type TEXT NOT NULL,
  key_status TEXT DEFAULT 'active',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE,
  rotation_schedule INTERVAL
);

-- Create session security table
CREATE TABLE public.secure_sessions (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  session_token TEXT NOT NULL UNIQUE,
  device_fingerprint TEXT,
  ip_address INET,
  location_country TEXT,
  location_city TEXT,
  risk_score INTEGER DEFAULT 0,
  is_suspicious BOOLEAN DEFAULT false,
  last_activity TIMESTAMP WITH TIME ZONE DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on all security tables
ALTER TABLE public.security_audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.compliance_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.data_encryption_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.secure_sessions ENABLE ROW LEVEL SECURITY;

-- Create policies for security audit logs
CREATE POLICY "Admins can view all security logs" 
ON public.security_audit_logs 
FOR SELECT 
USING (is_admin());

CREATE POLICY "System can insert security logs" 
ON public.security_audit_logs 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Users can view their own security logs" 
ON public.security_audit_logs 
FOR SELECT 
USING (auth.uid() = user_id);

-- Create policies for compliance events
CREATE POLICY "Admins can manage compliance events" 
ON public.compliance_events 
FOR ALL 
USING (is_admin());

CREATE POLICY "System can insert compliance events" 
ON public.compliance_events 
FOR INSERT 
WITH CHECK (true);

-- Create policies for encryption keys
CREATE POLICY "Admins can manage encryption keys" 
ON public.data_encryption_keys 
FOR ALL 
USING (is_admin());

-- Create policies for secure sessions
CREATE POLICY "Users can view their own sessions" 
ON public.secure_sessions 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own sessions" 
ON public.secure_sessions 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "System can manage secure sessions" 
ON public.secure_sessions 
FOR ALL 
WITH CHECK (true);

CREATE POLICY "Admins can view all sessions" 
ON public.secure_sessions 
FOR SELECT 
USING (is_admin());

-- Create function for security event logging
CREATE OR REPLACE FUNCTION public.log_security_event(
  p_user_id UUID,
  p_event_type TEXT,
  p_event_category TEXT,
  p_event_details JSONB DEFAULT '{}',
  p_ip_address INET DEFAULT NULL,
  p_user_agent TEXT DEFAULT NULL,
  p_session_id TEXT DEFAULT NULL,
  p_risk_score INTEGER DEFAULT 0
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  log_id UUID;
BEGIN
  INSERT INTO public.security_audit_logs (
    user_id,
    event_type,
    event_category,
    event_details,
    ip_address,
    user_agent,
    session_id,
    risk_score,
    requires_action
  ) VALUES (
    p_user_id,
    p_event_type,
    p_event_category,
    p_event_details,
    p_ip_address,
    p_user_agent,
    p_session_id,
    p_risk_score,
    p_risk_score > 70
  ) RETURNING id INTO log_id;
  
  RETURN log_id;
END;
$$;

-- Create function for compliance event logging
CREATE OR REPLACE FUNCTION public.log_compliance_event(
  p_compliance_type TEXT,
  p_event_type TEXT,
  p_user_id UUID DEFAULT NULL,
  p_data_subject TEXT DEFAULT NULL,
  p_legal_basis TEXT DEFAULT NULL,
  p_purpose TEXT DEFAULT NULL,
  p_retention_period INTERVAL DEFAULT NULL,
  p_event_details JSONB DEFAULT '{}'
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  event_id UUID;
  expires_at TIMESTAMP WITH TIME ZONE;
BEGIN
  -- Calculate expiration date if retention period is provided
  IF p_retention_period IS NOT NULL THEN
    expires_at := NOW() + p_retention_period;
  END IF;
  
  INSERT INTO public.compliance_events (
    compliance_type,
    event_type,
    user_id,
    data_subject,
    legal_basis,
    purpose,
    retention_period,
    event_details,
    expires_at
  ) VALUES (
    p_compliance_type,
    p_event_type,
    p_user_id,
    p_data_subject,
    p_legal_basis,
    p_purpose,
    p_retention_period,
    p_event_details,
    expires_at
  ) RETURNING id INTO event_id;
  
  RETURN event_id;
END;
$$;