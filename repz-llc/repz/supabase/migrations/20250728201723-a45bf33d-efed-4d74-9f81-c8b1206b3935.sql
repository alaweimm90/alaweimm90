-- Create test client profiles with different subscription tiers
-- Note: These are for testing purposes only

-- Test User 1: No subscription / Free tier
INSERT INTO public.client_profiles (
  auth_user_id,
  client_name,
  subscription_tier,
  tier_features,
  age_years,
  sex,
  primary_goal
) VALUES (
  gen_random_uuid(), -- Fake auth_user_id for testing
  'Test User - Free',
  'baseline',
  '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'::jsonb,
  25,
  'male',
  'muscle_gain'
);

-- Test User 2: Prime tier
INSERT INTO public.client_profiles (
  auth_user_id,
  client_name,
  subscription_tier,
  tier_features,
  age_years,
  sex,
  primary_goal
) VALUES (
  gen_random_uuid(), -- Fake auth_user_id for testing
  'Test User - Prime',
  'prime',
  '{"science_tips": true, "peds_protocol": false, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 24, "supplements_protocol": true, "biomarker_integration": false, "educational_materials": true, "bloodwork_interpretation": false}'::jsonb,
  30,
  'female',
  'fat_loss'
);

-- Test User 3: Precision tier
INSERT INTO public.client_profiles (
  auth_user_id,
  client_name,
  subscription_tier,
  tier_features,
  age_years,
  sex,
  primary_goal
) VALUES (
  gen_random_uuid(), -- Fake auth_user_id for testing
  'Test User - Precision',
  'precision',
  '{"science_tips": true, "peds_protocol": true, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": false, "response_time_hours": 12, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}'::jsonb,
  35,
  'male',
  'performance'
);

-- Test User 4: Longevity tier (highest)
INSERT INTO public.client_profiles (
  auth_user_id,
  client_name,
  subscription_tier,
  tier_features,
  age_years,
  sex,
  primary_goal
) VALUES (
  gen_random_uuid(), -- Fake auth_user_id for testing
  'Test User - Longevity',
  'longevity',
  '{"science_tips": true, "peds_protocol": true, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": true, "response_time_hours": 6, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}'::jsonb,
  40,
  'female',
  'longevity'
);

-- Test Coach Profile
INSERT INTO public.coach_profiles (
  auth_user_id,
  coach_name,
  specializations,
  credentials
) VALUES (
  gen_random_uuid(), -- Fake auth_user_id for testing
  'Test Coach - Fitness Expert',
  ARRAY['strength_training', 'nutrition', 'longevity'],
  ARRAY['NASM-CPT', 'CSCS', 'PhD Exercise Science']
);