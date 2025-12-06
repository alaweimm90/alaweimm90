-- Phase 1: Database Cleanup and Standardization (Final)

-- 1. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_client_profiles_auth_user_id ON client_profiles(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_coach_profiles_auth_user_id ON coach_profiles(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_daily_tracking_client_id ON daily_tracking(client_id);
CREATE INDEX IF NOT EXISTS idx_daily_tracking_date ON daily_tracking(tracking_date);
CREATE INDEX IF NOT EXISTS idx_exercise_logs_client_id ON exercise_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender_recipient ON messages(sender_id, recipient_id);
CREATE INDEX IF NOT EXISTS idx_subscribers_email ON subscribers(email);
CREATE INDEX IF NOT EXISTS idx_subscribers_user_id ON subscribers(user_id);

-- 2. Drop existing functions that conflict
DROP FUNCTION IF EXISTS get_user_role(uuid);
DROP FUNCTION IF EXISTS get_user_tier(uuid);

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
  SELECT 'longevity' 
  FROM coach_profiles coach 
  WHERE coach.auth_user_id = user_uuid
  LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION get_user_role_text(user_uuid uuid)
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

-- 4. Ensure data consistency
UPDATE client_profiles 
SET tier_features = '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'::jsonb
WHERE tier_features IS NULL;