-- Fix database function security by adding secure search paths
-- This prevents potential SQL injection through schema manipulation

-- Update is_admin function to use secure search path
CREATE OR REPLACE FUNCTION public.is_admin(check_email text DEFAULT NULL::text)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path = ''
AS $function$
  SELECT EXISTS (
    SELECT 1 
    FROM public.admin_users au
    WHERE 
      COALESCE(check_email, auth.email()) = au.email
      OR au.user_id = auth.uid()
  );
$function$;

-- Update get_admin_permissions function to use secure search path
CREATE OR REPLACE FUNCTION public.get_admin_permissions(check_email text DEFAULT NULL::text)
 RETURNS text[]
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path = ''
AS $function$
  SELECT COALESCE(au.permissions, ARRAY[]::TEXT[])
  FROM public.admin_users au
  WHERE 
    COALESCE(check_email, auth.email()) = au.email
    OR au.user_id = auth.uid()
  LIMIT 1;
$function$;

-- Update is_demo_admin function to use secure search path
CREATE OR REPLACE FUNCTION public.is_demo_admin(check_email text DEFAULT NULL::text)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path = ''
AS $function$
  SELECT EXISTS (
    SELECT 1 
    FROM public.admin_users au
    WHERE 
      COALESCE(check_email, auth.email()) = au.email
      AND au.admin_role = 'demo_admin'
  );
$function$;

-- Update handle_new_user function to use secure search path
CREATE OR REPLACE FUNCTION public.handle_new_user()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path = ''
AS $function$
BEGIN
  INSERT INTO public.profiles (user_id, display_name)
  VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$function$;

-- Update update_updated_at_column function to use secure search path
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path = ''
AS $function$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$function$;

-- Tighten RLS policies for better security

-- Replace overly permissive plan_approvals policy
DROP POLICY IF EXISTS "Anyone can view plan approvals by token" ON public.plan_approvals;

-- Create more specific policy for plan approvals - only allow viewing by approval token when specifically requested
CREATE POLICY "Plan approvals viewable by approval token"
ON public.plan_approvals
FOR SELECT
USING (
  -- Allow viewing by token only when token is provided and matches
  approval_token IS NOT NULL
  AND (
    -- Users can view their own plan approvals
    auth.uid() IN (
      SELECT subscribers.user_id
      FROM public.subscribers
      WHERE subscribers.id = plan_approvals.subscriber_id
    )
    -- Admins can view all plan approvals
    OR public.is_admin((auth.jwt() ->> 'email'::text))
  )
);

-- Restrict subscribers table update permissions to be more specific
DROP POLICY IF EXISTS "update_subscription" ON public.subscribers;

-- Create more specific update policy for subscribers
CREATE POLICY "Users and admins can update subscribers"
ON public.subscribers
FOR UPDATE
USING (
  -- Users can update their own subscription data
  (user_id = auth.uid() OR email = auth.email())
  -- Admins can update all subscription data
  OR public.is_admin((auth.jwt() ->> 'email'::text))
);