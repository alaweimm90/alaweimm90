-- Create proper roles system following best practices

-- 1. Create enum for app roles
CREATE TYPE public.app_role AS ENUM ('admin', 'coach', 'client');

-- 2. Create user_roles table
CREATE TABLE public.user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    role app_role NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_by UUID REFERENCES auth.users(id),
    UNIQUE(user_id, role)
);

-- 3. Enable RLS on user_roles
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- 4. Create security definer function to check roles
CREATE OR REPLACE FUNCTION public.has_role(_user_id UUID, _role app_role)
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.user_roles
    WHERE user_id = _user_id
      AND role = _role
  );
$$;

-- 5. Create function to get user's primary role
CREATE OR REPLACE FUNCTION public.get_user_role(_user_id UUID DEFAULT auth.uid())
RETURNS app_role
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT role
  FROM public.user_roles
  WHERE user_id = COALESCE(_user_id, auth.uid())
  ORDER BY 
    CASE role
      WHEN 'admin' THEN 1
      WHEN 'coach' THEN 2  
      WHEN 'client' THEN 3
    END
  LIMIT 1;
$$;

-- 6. Create RLS policies for user_roles
CREATE POLICY "Users can view their own roles"
ON public.user_roles
FOR SELECT
TO authenticated
USING (user_id = auth.uid());

CREATE POLICY "Admins can manage all roles"
ON public.user_roles
FOR ALL
TO authenticated
USING (public.has_role(auth.uid(), 'admin'))
WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- 7. Create trigger to automatically assign client role for new users
CREATE OR REPLACE FUNCTION public.assign_default_role()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  -- Only assign default role if no role exists yet
  IF NOT EXISTS (SELECT 1 FROM public.user_roles WHERE user_id = NEW.id) THEN
    INSERT INTO public.user_roles (user_id, role)
    VALUES (NEW.id, 'client');
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER assign_default_role_trigger
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.assign_default_role();