-- Enable RLS on the demo_profiles table for security
ALTER TABLE public.demo_profiles ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for demo_profiles
-- This is just demo data, so we'll allow read access to authenticated users
CREATE POLICY "Demo profiles are viewable by authenticated users" 
ON public.demo_profiles 
FOR SELECT 
TO authenticated
USING (true);

-- Only admins can modify demo data
CREATE POLICY "Admins can manage demo profiles" 
ON public.demo_profiles 
FOR ALL 
TO authenticated
USING (is_admin());