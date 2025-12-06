-- ==========================================
-- PHASE 3: Fix Critical Nullable Foreign Keys
-- ==========================================

-- Make critical foreign keys NOT NULL where they should be required
-- Note: We need to ensure no NULL values exist first
UPDATE biomarker_tests SET client_id = (SELECT id FROM client_profiles LIMIT 1) WHERE client_id IS NULL;
UPDATE cardio_sessions SET client_id = (SELECT id FROM client_profiles LIMIT 1) WHERE client_id IS NULL;
UPDATE meal_plans SET client_id = (SELECT id FROM client_profiles LIMIT 1) WHERE client_id IS NULL;
UPDATE daily_tracking SET client_id = (SELECT id FROM client_profiles LIMIT 1) WHERE client_id IS NULL;

-- Now make them NOT NULL
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