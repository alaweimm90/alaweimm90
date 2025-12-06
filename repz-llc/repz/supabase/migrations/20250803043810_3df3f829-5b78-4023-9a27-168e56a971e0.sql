-- Drop the trigger temporarily to avoid the profiles table issue
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Create demo auth users for the enterprise demo
INSERT INTO auth.users (
  instance_id,
  id,
  aud,
  role,
  email,
  encrypted_password,
  email_confirmed_at,
  recovery_sent_at,
  last_sign_in_at,
  raw_app_meta_data,
  raw_user_meta_data,
  created_at,
  updated_at,
  confirmation_token,
  email_change,
  email_change_token_new,
  recovery_token
) VALUES 
(
  '00000000-0000-0000-0000-000000000000',
  gen_random_uuid(),
  'authenticated',
  'authenticated',
  'demo.core@repzcoach.com',
  crypt('CoreDemo2024!', gen_salt('bf')),
  now(),
  NULL,
  NULL,
  '{"provider":"email","providers":["email"]}',
  '{}',
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  '00000000-0000-0000-0000-000000000000',
  gen_random_uuid(),
  'authenticated',
  'authenticated',
  'demo.adaptive@repzcoach.com',
  crypt('AdaptiveDemo2024!', gen_salt('bf')),
  now(),
  NULL,
  NULL,
  '{"provider":"email","providers":["email"]}',
  '{}',
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  '00000000-0000-0000-0000-000000000000',
  gen_random_uuid(),
  'authenticated',
  'authenticated',
  'demo.performance@repzcoach.com',
  crypt('PerformanceDemo2024!', gen_salt('bf')),
  now(),
  NULL,
  NULL,
  '{"provider":"email","providers":["email"]}',
  '{}',
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  '00000000-0000-0000-0000-000000000000',
  gen_random_uuid(),
  'authenticated',
  'authenticated',
  'demo.longevity@repzcoach.com',
  crypt('LongevityDemo2024!', gen_salt('bf')),
  now(),
  NULL,
  NULL,
  '{"provider":"email","providers":["email"]}',
  '{}',
  now(),
  now(),
  '',
  '',
  '',
  ''
);

-- Now create client profiles for the demo users
DO $$
DECLARE
    core_user_id uuid;
    adaptive_user_id uuid;
    performance_user_id uuid;
    longevity_user_id uuid;
BEGIN
    -- Get the user IDs
    SELECT id INTO core_user_id FROM auth.users WHERE email = 'demo.core@repzcoach.com';
    SELECT id INTO adaptive_user_id FROM auth.users WHERE email = 'demo.adaptive@repzcoach.com';
    SELECT id INTO performance_user_id FROM auth.users WHERE email = 'demo.performance@repzcoach.com';
    SELECT id INTO longevity_user_id FROM auth.users WHERE email = 'demo.longevity@repzcoach.com';

    -- Insert client profiles for demo users
    INSERT INTO public.client_profiles (
        auth_user_id,
        client_name,
        subscription_tier,
        tier_features,
        created_at,
        updated_at
    ) VALUES 
    (
        core_user_id,
        'Demo Core User',
        'core',
        public.get_tier_features('core'),
        now(),
        now()
    ),
    (
        adaptive_user_id,
        'Demo Adaptive User',
        'adaptive',
        public.get_tier_features('adaptive'),
        now(),
        now()
    ),
    (
        performance_user_id,
        'Demo Performance User',
        'performance',
        public.get_tier_features('performance'),
        now(),
        now()
    ),
    (
        longevity_user_id,
        'Demo Longevity User',
        'longevity',
        public.get_tier_features('longevity'),
        now(),
        now()
    );
END $$;

-- Recreate the trigger for future users
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user_profile_creation();