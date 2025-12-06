-- Phase 1: Database Cleanup and Standardization
-- This enhances the existing schema without breaking functionality

-- 1. Create indexes for better performance on commonly queried fields
CREATE INDEX IF NOT EXISTS idx_client_profiles_auth_user_id ON client_profiles(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_coach_profiles_auth_user_id ON coach_profiles(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_daily_tracking_client_id ON daily_tracking(client_id);
CREATE INDEX IF NOT EXISTS idx_daily_tracking_date ON daily_tracking(tracking_date);
CREATE INDEX IF NOT EXISTS idx_exercise_logs_client_id ON exercise_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender_recipient ON messages(sender_id, recipient_id);
CREATE INDEX IF NOT EXISTS idx_subscribers_email ON subscribers(email);
CREATE INDEX IF NOT EXISTS idx_subscribers_user_id ON subscribers(user_id);

-- 2. Create a unified view for all users (combines existing tables)
CREATE OR REPLACE VIEW unified_users AS
SELECT 
  cp.auth_user_id as user_id,
  cp.client_name as full_name,
  au.email,
  up.phone,
  'client' as role,
  cp.subscription_tier,
  cp.tier_features,
  cp.age_years,
  cp.sex,
  cp.primary_goal,
  cp.created_at,
  cp.updated_at
FROM client_profiles cp
LEFT JOIN auth.users au ON au.id = cp.auth_user_id  
LEFT JOIN user_profiles up ON up.user_id = cp.auth_user_id

UNION ALL

SELECT 
  coach.auth_user_id as user_id,
  coach.coach_name as full_name,
  au.email,
  up.phone,
  'coach' as role,
  'premium' as subscription_tier, -- Coaches get premium access
  '{"all_features": true}' as tier_features,
  null as age_years,
  null as sex,
  null as primary_goal,
  coach.created_at,
  coach.updated_at
FROM coach_profiles coach
LEFT JOIN auth.users au ON au.id = coach.auth_user_id
LEFT JOIN user_profiles up ON up.user_id = coach.auth_user_id;

-- 3. Create helper functions for common queries
CREATE OR REPLACE FUNCTION get_user_tier(user_uuid uuid)
RETURNS text
LANGUAGE sql
STABLE SECURITY DEFINER
AS $$
  SELECT COALESCE(cp.subscription_tier::text, 'baseline')
  FROM client_profiles cp 
  WHERE cp.auth_user_id = user_uuid
  UNION ALL
  SELECT 'premium' 
  FROM coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION get_user_role(user_uuid uuid)
RETURNS text  
LANGUAGE sql
STABLE SECURITY DEFINER
AS $$
  SELECT 'client'
  FROM client_profiles cp 
  WHERE cp.auth_user_id = user_uuid
  UNION ALL
  SELECT 'coach'
  FROM coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  UNION ALL
  SELECT 'admin'
  FROM admin_users au
  WHERE au.user_id = user_uuid OR au.email = (SELECT email FROM auth.users WHERE id = user_uuid)
  LIMIT 1;
$$;

-- 4. Add missing constraints and validations
ALTER TABLE client_profiles 
ADD CONSTRAINT client_profiles_age_check CHECK (age_years IS NULL OR (age_years >= 13 AND age_years <= 120));

ALTER TABLE client_profiles 
ADD CONSTRAINT client_profiles_training_days_check CHECK (training_days_per_week IS NULL OR (training_days_per_week >= 1 AND training_days_per_week <= 7));

-- 5. Ensure data consistency
UPDATE client_profiles 
SET tier_features = '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'::jsonb
WHERE tier_features IS NULL;