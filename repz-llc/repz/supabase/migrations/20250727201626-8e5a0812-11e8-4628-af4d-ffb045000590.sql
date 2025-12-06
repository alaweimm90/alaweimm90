-- Fix security warnings for function search paths
CREATE OR REPLACE FUNCTION public.handle_new_user_profile()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.user_profiles (user_id, email_verified)
  VALUES (NEW.id, COALESCE(NEW.email_confirmed_at IS NOT NULL, false));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

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
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- Create edge function for authentication rate limiting
CREATE OR REPLACE FUNCTION public.log_auth_attempt(
  attempt_email TEXT DEFAULT NULL,
  attempt_phone TEXT DEFAULT NULL,
  auth_method TEXT DEFAULT 'email',
  is_success BOOLEAN DEFAULT false,
  error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
  INSERT INTO public.auth_attempts (email, phone, method, success, error_type, ip_address)
  VALUES (
    attempt_email,
    attempt_phone,
    auth_method,
    is_success,
    error_message,
    inet_client_addr()
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

-- Create function to check rate limiting
CREATE OR REPLACE FUNCTION public.check_rate_limit(
  check_email TEXT DEFAULT NULL,
  check_phone TEXT DEFAULT NULL,
  time_window_minutes INTEGER DEFAULT 15,
  max_attempts INTEGER DEFAULT 5
) RETURNS BOOLEAN AS $$
DECLARE
  attempt_count INTEGER := 0;
BEGIN
  SELECT COUNT(*)
  INTO attempt_count
  FROM public.auth_attempts
  WHERE (
    (check_email IS NOT NULL AND email = check_email) OR
    (check_phone IS NOT NULL AND phone = check_phone)
  )
  AND success = false
  AND created_at > (now() - interval '1 minute' * time_window_minutes);

  RETURN attempt_count < max_attempts;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';