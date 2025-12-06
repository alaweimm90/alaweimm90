-- Create enhanced users tracking table for auth attempts and methods
CREATE TABLE IF NOT EXISTS public.auth_attempts (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT,
  phone TEXT,
  method TEXT NOT NULL CHECK (method IN ('email', 'phone', 'google', 'oauth')),
  success BOOLEAN NOT NULL DEFAULT false,
  error_type TEXT,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on auth_attempts
ALTER TABLE public.auth_attempts ENABLE ROW LEVEL SECURITY;

-- Allow admins to view all auth attempts
CREATE POLICY "Admins can view all auth attempts" 
ON public.auth_attempts 
FOR SELECT 
USING (is_admin((auth.jwt() ->> 'email'::text)));

-- Allow insertion for logging auth attempts
CREATE POLICY "Auth attempts can be logged" 
ON public.auth_attempts 
FOR INSERT 
WITH CHECK (true);

-- Create user profiles table to track agreements and verification status
CREATE TABLE IF NOT EXISTS public.user_profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE,
  terms_accepted BOOLEAN NOT NULL DEFAULT false,
  liability_waiver_signed BOOLEAN NOT NULL DEFAULT false,
  medical_clearance BOOLEAN NOT NULL DEFAULT false,
  fitness_level TEXT,
  email_verified BOOLEAN NOT NULL DEFAULT false,
  phone_verified BOOLEAN NOT NULL DEFAULT false,
  google_id TEXT UNIQUE,
  phone TEXT UNIQUE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on user_profiles
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- Users can view and update their own profile
CREATE POLICY "Users can view their own profile" 
ON public.user_profiles 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile" 
ON public.user_profiles 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" 
ON public.user_profiles 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- Admins can view all profiles
CREATE POLICY "Admins can view all profiles" 
ON public.user_profiles 
FOR ALL 
USING (is_admin((auth.jwt() ->> 'email'::text)));

-- Create trigger for automatic profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user_profile()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.user_profiles (user_id, email_verified)
  VALUES (NEW.id, COALESCE(NEW.email_confirmed_at IS NOT NULL, false));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger
DROP TRIGGER IF EXISTS on_auth_user_created_profile ON auth.users;
CREATE TRIGGER on_auth_user_created_profile
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user_profile();

-- Create function to check if user exists with any auth method
CREATE OR REPLACE FUNCTION public.check_user_exists(check_email TEXT DEFAULT NULL, check_phone TEXT DEFAULT NULL, check_google_id TEXT DEFAULT NULL)
RETURNS TABLE(
  exists_with_email BOOLEAN,
  exists_with_phone BOOLEAN,
  exists_with_google BOOLEAN,
  suggested_methods TEXT[]
) AS $$
DECLARE
  email_exists BOOLEAN := false;
  phone_exists BOOLEAN := false;
  google_exists BOOLEAN := false;
  methods TEXT[] := ARRAY[]::TEXT[];
BEGIN
  -- Check email in auth.users
  IF check_email IS NOT NULL THEN
    SELECT EXISTS(SELECT 1 FROM auth.users WHERE email = check_email) INTO email_exists;
    IF email_exists THEN
      methods := array_append(methods, 'email');
    END IF;
  END IF;

  -- Check phone in user_profiles
  IF check_phone IS NOT NULL THEN
    SELECT EXISTS(SELECT 1 FROM public.user_profiles WHERE phone = check_phone) INTO phone_exists;
    IF phone_exists THEN
      methods := array_append(methods, 'phone');
    END IF;
  END IF;

  -- Check Google ID in user_profiles
  IF check_google_id IS NOT NULL THEN
    SELECT EXISTS(SELECT 1 FROM public.user_profiles WHERE google_id = check_google_id) INTO google_exists;
    IF google_exists THEN
      methods := array_append(methods, 'google');
    END IF;
  END IF;

  RETURN QUERY SELECT email_exists, phone_exists, google_exists, methods;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create updated_at trigger
CREATE TRIGGER update_user_profiles_updated_at
  BEFORE UPDATE ON public.user_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_auth_attempts_email ON public.auth_attempts(email);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_method ON public.auth_attempts(method);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_created_at ON public.auth_attempts(created_at);
CREATE INDEX IF NOT EXISTS idx_user_profiles_phone ON public.user_profiles(phone);
CREATE INDEX IF NOT EXISTS idx_user_profiles_google_id ON public.user_profiles(google_id);