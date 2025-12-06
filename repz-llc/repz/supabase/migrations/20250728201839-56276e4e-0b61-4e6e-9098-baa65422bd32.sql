-- First, let's create a demo table for testing different subscription tiers
-- This allows us to test without needing real auth users

CREATE TABLE IF NOT EXISTS public.demo_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  demo_name text NOT NULL,
  subscription_tier tier_enum NOT NULL DEFAULT 'baseline',
  tier_features jsonb DEFAULT '{}',
  age_years integer,
  sex sex_enum,
  primary_goal goal_enum,
  created_at timestamp with time zone DEFAULT now()
);

-- Insert demo profiles for testing
INSERT INTO public.demo_profiles (demo_name, subscription_tier, tier_features, age_years, sex, primary_goal) VALUES
('Demo User - Baseline', 'baseline', '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}', 25, 'male', 'muscle_gain'),

('Demo User - Prime', 'prime', '{"science_tips": true, "peds_protocol": false, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 24, "supplements_protocol": true, "biomarker_integration": false, "educational_materials": true, "bloodwork_interpretation": false}', 30, 'female', 'fat_loss'),

('Demo User - Precision', 'precision', '{"science_tips": true, "peds_protocol": true, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": false, "response_time_hours": 12, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}', 35, 'male', 'performance'),

('Demo User - Longevity', 'longevity', '{"science_tips": true, "peds_protocol": true, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": true, "response_time_hours": 6, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}', 40, 'female', 'longevity');

-- Also update your existing profile with more complete information
UPDATE public.client_profiles 
SET 
  age_years = 28,
  sex = 'male',
  primary_goal = 'muscle_gain',
  tier_features = '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'::jsonb
WHERE auth_user_id = '3dc5ab03-4f8f-467f-a5f5-f2f00d403c09';