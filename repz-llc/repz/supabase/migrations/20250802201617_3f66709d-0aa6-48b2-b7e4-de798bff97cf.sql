-- ==========================================
-- PHASE 1: Create Unified Profiles View
-- ==========================================

-- Create a unified profiles view that combines client and coach profiles
CREATE OR REPLACE VIEW profiles AS
  SELECT 
    auth_user_id as user_id,
    client_name as display_name,
    'client' as role,
    created_at,
    updated_at
  FROM client_profiles
  WHERE auth_user_id IS NOT NULL
  UNION ALL
  SELECT 
    auth_user_id as user_id,
    coach_name as display_name,
    'coach' as role,
    created_at,
    updated_at
  FROM coach_profiles  
  WHERE auth_user_id IS NOT NULL;

-- Enable RLS on the view
ALTER VIEW profiles ENABLE ROW LEVEL SECURITY;

-- Create RLS policy for the profiles view
CREATE POLICY "Users can view their own profile and coaches can view their clients"
ON profiles FOR SELECT
USING (
  auth.uid() = user_id OR
  EXISTS (
    SELECT 1 FROM coach_profiles cp
    JOIN client_profiles cl ON cp.id = cl.coach_id
    WHERE cp.auth_user_id = auth.uid() AND cl.auth_user_id = user_id
  ) OR
  is_admin()
);

-- ==========================================
-- PHASE 2: Add Missing Updated_At Fields
-- ==========================================

-- Add updated_at to critical tables that are missing it
ALTER TABLE achievements ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE ai_analysis_results ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE auth_attempts ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE biomarker_tests ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE cardio_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE challenge_participants ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE coaching_messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE community_challenges ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE daily_tracking ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE dashboard_performance_logs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE data_encryption_keys ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE demo_profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE error_logs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE exercise_library ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE exercise_logs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE exercise_sets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE food_database ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE form_checks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE meal_plans ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Create triggers for automatic updated_at management
CREATE TRIGGER update_achievements_updated_at
  BEFORE UPDATE ON achievements
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_biomarker_tests_updated_at
  BEFORE UPDATE ON biomarker_tests
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cardio_sessions_updated_at
  BEFORE UPDATE ON cardio_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_tracking_updated_at
  BEFORE UPDATE ON daily_tracking
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_exercise_library_updated_at
  BEFORE UPDATE ON exercise_library
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_food_database_updated_at
  BEFORE UPDATE ON food_database
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_meal_plans_updated_at
  BEFORE UPDATE ON meal_plans
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================
-- PHASE 3: Fix Critical Nullable Foreign Keys
-- ==========================================

-- Make critical foreign keys NOT NULL where they should be required
ALTER TABLE biomarker_tests ALTER COLUMN client_id SET NOT NULL;
ALTER TABLE cardio_sessions ALTER COLUMN client_id SET NOT NULL;
ALTER TABLE meal_plans ALTER COLUMN client_id SET NOT NULL;
ALTER TABLE daily_tracking ALTER COLUMN client_id SET NOT NULL;

-- ==========================================
-- PHASE 4: Performance Indexes
-- ==========================================

-- Add critical performance indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_daily_tracking_client_date ON daily_tracking(client_id, tracking_date);
CREATE INDEX IF NOT EXISTS idx_biomarker_tests_client_date ON biomarker_tests(client_id, test_date);
CREATE INDEX IF NOT EXISTS idx_live_workout_sessions_client ON live_workout_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_exercise_logs_session ON exercise_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_cardio_sessions_client ON cardio_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_meal_plans_client_date ON meal_plans(client_id, plan_date);

-- Performance index for auth lookups
CREATE INDEX IF NOT EXISTS idx_client_profiles_auth_user ON client_profiles(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_coach_profiles_auth_user ON coach_profiles(auth_user_id);

-- ==========================================
-- PHASE 5: Setup Realtime Properly
-- ==========================================

-- Enable REPLICA IDENTITY FULL for realtime tables
ALTER TABLE messages REPLICA IDENTITY FULL;
ALTER TABLE coach_notifications REPLICA IDENTITY FULL;
ALTER TABLE live_workout_sessions REPLICA IDENTITY FULL;
ALTER TABLE daily_tracking REPLICA IDENTITY FULL;

-- Add tables to realtime publication
ALTER PUBLICATION supabase_realtime ADD TABLE messages;
ALTER PUBLICATION supabase_realtime ADD TABLE coach_notifications;
ALTER PUBLICATION supabase_realtime ADD TABLE live_workout_sessions;
ALTER PUBLICATION supabase_realtime ADD TABLE daily_tracking;