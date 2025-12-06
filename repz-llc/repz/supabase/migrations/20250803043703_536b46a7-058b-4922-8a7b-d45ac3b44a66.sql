-- Create demo auth users for the enterprise demo
-- These will be actual users that can log in

-- Insert demo users into auth.users table
INSERT INTO auth.users (
  id,
  email,
  encrypted_password,
  email_confirmed_at,
  created_at,
  updated_at,
  confirmation_token,
  recovery_token,
  email_change_token_new,
  email_change
) VALUES 
(
  gen_random_uuid(),
  'demo.core@repzcoach.com',
  crypt('CoreDemo2024!', gen_salt('bf')),
  now(),
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  gen_random_uuid(),
  'demo.adaptive@repzcoach.com',
  crypt('AdaptiveDemo2024!', gen_salt('bf')),
  now(),
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  gen_random_uuid(),
  'demo.performance@repzcoach.com',
  crypt('PerformanceDemo2024!', gen_salt('bf')),
  now(),
  now(),
  now(),
  '',
  '',
  '',
  ''
),
(
  gen_random_uuid(),
  'demo.longevity@repzcoach.com',
  crypt('LongevityDemo2024!', gen_salt('bf')),
  now(),
  now(),
  now(),
  '',
  '',
  '',
  ''
);

-- Create client profiles for demo users with appropriate tiers
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
        jsonb_build_object(
            'personalized_training', true,
            'nutrition_plan', true,
            'dashboard_type', 'static_fixed',
            'qa_access', 'limited',
            'response_time_hours', 72,
            'weekly_checkins', false,
            'form_review', false,
            'ai_fitness_assistant', false
        ),
        now(),
        now()
    ),
    (
        adaptive_user_id,
        'Demo Adaptive User',
        'adaptive',
        jsonb_build_object(
            'personalized_training', true,
            'nutrition_plan', true,
            'dashboard_type', 'interactive_adjustable',
            'qa_access', 'enhanced',
            'response_time_hours', 48,
            'weekly_checkins', true,
            'form_review', true,
            'ai_fitness_assistant', false
        ),
        now(),
        now()
    ),
    (
        performance_user_id,
        'Demo Performance User',
        'performance',
        jsonb_build_object(
            'personalized_training', true,
            'nutrition_plan', true,
            'dashboard_type', 'interactive_adjustable',
            'qa_access', 'premium',
            'response_time_hours', 24,
            'weekly_checkins', true,
            'form_review', true,
            'ai_fitness_assistant', true
        ),
        now(),
        now()
    ),
    (
        longevity_user_id,
        'Demo Longevity User',
        'longevity',
        jsonb_build_object(
            'personalized_training', true,
            'nutrition_plan', true,
            'dashboard_type', 'premium_advanced',
            'qa_access', 'unlimited',
            'response_time_hours', 12,
            'weekly_checkins', true,
            'form_review', true,
            'ai_fitness_assistant', true,
            'biomarker_integration', true,
            'in_person_training', true
        ),
        now(),
        now()
    );
END $$;