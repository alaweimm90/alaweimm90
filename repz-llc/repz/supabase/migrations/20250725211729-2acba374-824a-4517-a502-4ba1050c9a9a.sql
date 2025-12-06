-- Fix security warnings by setting search_path for admin functions
CREATE OR REPLACE FUNCTION public.is_admin(check_email TEXT DEFAULT NULL)
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
STABLE
SET search_path = ''
AS $$
  SELECT EXISTS (
    SELECT 1 
    FROM public.admin_users au
    WHERE 
      COALESCE(check_email, auth.email()) = au.email
      OR au.user_id = auth.uid()
  );
$$;

CREATE OR REPLACE FUNCTION public.get_admin_permissions(check_email TEXT DEFAULT NULL)
RETURNS TEXT[]
LANGUAGE SQL
SECURITY DEFINER
STABLE
SET search_path = ''
AS $$
  SELECT COALESCE(au.permissions, ARRAY[]::TEXT[])
  FROM public.admin_users au
  WHERE 
    COALESCE(check_email, auth.email()) = au.email
    OR au.user_id = auth.uid()
  LIMIT 1;
$$;