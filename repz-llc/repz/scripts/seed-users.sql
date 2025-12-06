-- ============================================================================
-- REPZ Platform - User Seed Script
-- ============================================================================
-- This script creates the admin and initial client accounts
-- Run this after the database migrations are complete
--
-- IMPORTANT: Replace the placeholder UUIDs with actual auth.users IDs
-- after creating the accounts through the signup flow or Supabase dashboard
-- ============================================================================

-- ============================================================================
-- STEP 1: Create users via Supabase Auth Dashboard or API
-- ============================================================================
-- Go to Supabase Dashboard > Authentication > Users > Add User
--
-- Admin Account:
--   Email: admin@repzcoach.com (or your email)
--   Password: [secure password]
--
-- Emilio Account:
--   Email: emilio@[email].com
--   Password: [secure password]
--
-- After creating, copy the user IDs below

-- ============================================================================
-- STEP 2: Set the User IDs (replace with actual IDs from Supabase)
-- ============================================================================

-- Replace these with actual UUIDs from auth.users after signup
DO $$
DECLARE
  admin_user_id UUID := '00000000-0000-0000-0000-000000000001'; -- Replace with actual admin user ID
  emilio_user_id UUID := '00000000-0000-0000-0000-000000000002'; -- Replace with actual Emilio user ID
BEGIN

  -- ============================================================================
  -- STEP 3: Create/Update Profiles
  -- ============================================================================

  -- Admin Profile
  INSERT INTO public.profiles (id, email, subscription_tier, subscription_status, created_at, updated_at)
  VALUES (
    admin_user_id,
    'admin@repzcoach.com',
    'longevity', -- Admin gets highest tier
    'active',
    NOW(),
    NOW()
  )
  ON CONFLICT (id) DO UPDATE SET
    subscription_tier = 'longevity',
    subscription_status = 'active',
    updated_at = NOW();

  -- Emilio Profile (Performance tier = 3rd tier)
  INSERT INTO public.profiles (id, email, subscription_tier, subscription_status, created_at, updated_at)
  VALUES (
    emilio_user_id,
    'emilio@example.com', -- Replace with actual email
    'performance', -- 3rd tier as requested
    'active',
    NOW(),
    NOW()
  )
  ON CONFLICT (id) DO UPDATE SET
    subscription_tier = 'performance',
    subscription_status = 'active',
    updated_at = NOW();

  -- ============================================================================
  -- STEP 4: Assign Roles
  -- ============================================================================

  -- Admin Role
  INSERT INTO public.user_roles (user_id, role, created_at)
  VALUES (admin_user_id, 'admin', NOW())
  ON CONFLICT (user_id, role) DO NOTHING;

  -- Emilio as Client
  INSERT INTO public.user_roles (user_id, role, created_at)
  VALUES (emilio_user_id, 'client', NOW())
  ON CONFLICT (user_id, role) DO NOTHING;

  -- ============================================================================
  -- STEP 5: Create Client Profiles (for coaching features)
  -- ============================================================================

  -- Emilio Client Profile
  INSERT INTO public.client_profiles (
    auth_user_id,
    client_name,
    subscription_tier,
    onboarding_completed,
    created_at,
    updated_at
  )
  VALUES (
    emilio_user_id,
    'Emilio',
    'performance',
    false, -- Will be set to true after intake form
    NOW(),
    NOW()
  )
  ON CONFLICT (auth_user_id) DO UPDATE SET
    subscription_tier = 'performance',
    updated_at = NOW();

  RAISE NOTICE 'User seed completed successfully!';
  RAISE NOTICE 'Admin ID: %', admin_user_id;
  RAISE NOTICE 'Emilio ID: %', emilio_user_id;

END $$;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check profiles
SELECT id, email, subscription_tier, subscription_status FROM public.profiles;

-- Check roles
SELECT ur.user_id, ur.role, p.email
FROM public.user_roles ur
JOIN public.profiles p ON ur.user_id = p.id;

-- Check client profiles
SELECT auth_user_id, client_name, subscription_tier, onboarding_completed
FROM public.client_profiles;
