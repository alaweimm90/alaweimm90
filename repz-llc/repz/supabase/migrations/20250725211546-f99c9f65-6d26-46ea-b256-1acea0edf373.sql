-- Create admin_users table for admin access control
CREATE TABLE public.admin_users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE,
  admin_role TEXT NOT NULL DEFAULT 'admin',
  permissions TEXT[] DEFAULT ARRAY['reservation_admin', 'client_data', 'stripe_access', 'email_settings'],
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  created_by UUID REFERENCES auth.users(id)
);

-- Enable Row Level Security
ALTER TABLE public.admin_users ENABLE ROW LEVEL SECURITY;

-- Create policies for admin_users table
CREATE POLICY "Admins can view admin_users" 
ON public.admin_users 
FOR SELECT 
USING (
  email IN ('meshal@berkeley.edu', 'contact@repzcoach.com') OR
  EXISTS (
    SELECT 1 FROM public.admin_users au 
    WHERE au.user_id = auth.uid() OR au.email = auth.email()
  )
);

CREATE POLICY "Admins can insert admin_users" 
ON public.admin_users 
FOR INSERT 
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.admin_users au 
    WHERE au.user_id = auth.uid() OR au.email = auth.email()
  )
);

CREATE POLICY "Admins can update admin_users" 
ON public.admin_users 
FOR UPDATE 
USING (
  EXISTS (
    SELECT 1 FROM public.admin_users au 
    WHERE au.user_id = auth.uid() OR au.email = auth.email()
  )
);

-- Insert initial admin users
INSERT INTO public.admin_users (email, admin_role, permissions) VALUES 
('meshal@berkeley.edu', 'super_admin', ARRAY['reservation_admin', 'client_data', 'stripe_access', 'email_settings', 'user_management']),
('contact@repzcoach.com', 'super_admin', ARRAY['reservation_admin', 'client_data', 'stripe_access', 'email_settings', 'user_management'])
ON CONFLICT (email) DO UPDATE SET
  admin_role = EXCLUDED.admin_role,
  permissions = EXCLUDED.permissions,
  updated_at = now();

-- Create function to check if user is admin
CREATE OR REPLACE FUNCTION public.is_admin(check_email TEXT DEFAULT NULL)
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
STABLE
AS $$
  SELECT EXISTS (
    SELECT 1 
    FROM public.admin_users au
    WHERE 
      COALESCE(check_email, auth.email()) = au.email
      OR au.user_id = auth.uid()
  );
$$;

-- Create function to get admin permissions
CREATE OR REPLACE FUNCTION public.get_admin_permissions(check_email TEXT DEFAULT NULL)
RETURNS TEXT[]
LANGUAGE SQL
SECURITY DEFINER
STABLE
AS $$
  SELECT COALESCE(au.permissions, ARRAY[]::TEXT[])
  FROM public.admin_users au
  WHERE 
    COALESCE(check_email, auth.email()) = au.email
    OR au.user_id = auth.uid()
  LIMIT 1;
$$;

-- Update subscribers table policies to allow admin access
DROP POLICY IF EXISTS "select_own_subscription" ON public.subscribers;
CREATE POLICY "select_own_subscription" ON public.subscribers
FOR SELECT 
USING (
  user_id = auth.uid() 
  OR email = auth.email() 
  OR public.is_admin()
);

-- Create trigger for updated_at on admin_users
CREATE TRIGGER update_admin_users_updated_at
  BEFORE UPDATE ON public.admin_users
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();