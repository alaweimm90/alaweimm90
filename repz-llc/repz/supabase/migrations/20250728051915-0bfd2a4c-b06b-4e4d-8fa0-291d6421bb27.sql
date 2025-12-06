-- Add comprehensive RLS policies for all new tables

-- RLS Policies for daily_tracking (Prime+ tiers only)
CREATE POLICY "daily_tracking_tier_access" ON daily_tracking
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = daily_tracking.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('prime', 'precision', 'longevity')
    )
  );

-- Coach access to daily tracking
CREATE POLICY "coaches_access_daily_tracking" ON daily_tracking
  FOR SELECT TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = daily_tracking.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for biomarker_tests (Precision+ only)
CREATE POLICY "biomarker_precision_access" ON biomarker_tests
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = biomarker_tests.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('precision', 'longevity')
    )
  );

-- Coach access to biomarkers
CREATE POLICY "coaches_access_biomarkers_new" ON biomarker_tests
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = biomarker_tests.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for progress_photos (Prime+ only)
CREATE POLICY "progress_photos_tier_access" ON progress_photos
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = progress_photos.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('prime', 'precision', 'longevity')
    )
  );

-- Coach access to progress photos
CREATE POLICY "coaches_access_progress_photos" ON progress_photos
  FOR SELECT TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = progress_photos.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for workout_sessions
CREATE POLICY "workout_sessions_access" ON workout_sessions
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = workout_sessions.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Coach access to workout sessions
CREATE POLICY "coaches_access_workout_sessions" ON workout_sessions
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = workout_sessions.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for exercise_sets
CREATE POLICY "exercise_sets_access" ON exercise_sets
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM workout_sessions ws
      JOIN client_profiles cp ON cp.id = ws.client_id
      WHERE ws.id = exercise_sets.workout_session_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Coach access to exercise sets
CREATE POLICY "coaches_access_exercise_sets" ON exercise_sets
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM workout_sessions ws
      JOIN client_profiles cp ON cp.id = ws.client_id
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE ws.id = exercise_sets.workout_session_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for supplement_protocols
CREATE POLICY "supplement_protocols_access" ON supplement_protocols
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = supplement_protocols.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Coach access to supplement protocols
CREATE POLICY "coaches_access_supplement_protocols" ON supplement_protocols
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = supplement_protocols.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for supplement_compliance
CREATE POLICY "supplement_compliance_access" ON supplement_compliance
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = supplement_compliance.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for meal_plans
CREATE POLICY "meal_plans_access" ON meal_plans
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = meal_plans.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Coach access to meal plans
CREATE POLICY "coaches_access_meal_plans" ON meal_plans
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = meal_plans.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for cardio_sessions
CREATE POLICY "cardio_sessions_access" ON cardio_sessions
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = cardio_sessions.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Coach access to cardio sessions
CREATE POLICY "coaches_access_cardio_sessions" ON cardio_sessions
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = cardio_sessions.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- Public access policies for library tables
CREATE POLICY "exercise_library_public" ON exercise_library 
  FOR SELECT TO authenticated 
  USING (true);

CREATE POLICY "supplement_library_public" ON supplement_library 
  FOR SELECT TO authenticated 
  USING (true);

CREATE POLICY "food_database_public" ON food_database 
  FOR SELECT TO authenticated 
  USING (true);

-- Peptide protocols - Longevity tier only access
CREATE POLICY "peptide_protocols_longevity_access" ON peptide_protocols
  FOR SELECT TO authenticated
  USING (true);

-- Admin access policies (can manage library content)
CREATE POLICY "admins_manage_exercise_library" ON exercise_library
  FOR ALL TO authenticated
  USING (is_admin());

CREATE POLICY "admins_manage_supplement_library" ON supplement_library
  FOR ALL TO authenticated
  USING (is_admin());

CREATE POLICY "admins_manage_food_database" ON food_database
  FOR ALL TO authenticated
  USING (is_admin());

CREATE POLICY "admins_manage_peptide_protocols" ON peptide_protocols
  FOR ALL TO authenticated
  USING (is_admin());

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_daily_tracking_client_date ON daily_tracking(client_id, tracking_date);
CREATE INDEX IF NOT EXISTS idx_workout_sessions_client_date ON workout_sessions(client_id, workout_date);
CREATE INDEX IF NOT EXISTS idx_biomarker_tests_client_date ON biomarker_tests(client_id, test_date);
CREATE INDEX IF NOT EXISTS idx_progress_photos_client_date ON progress_photos(client_id, photo_date);
CREATE INDEX IF NOT EXISTS idx_supplement_protocols_client ON supplement_protocols(client_id);
CREATE INDEX IF NOT EXISTS idx_exercise_library_category ON exercise_library(exercise_category);
CREATE INDEX IF NOT EXISTS idx_food_database_name ON food_database(food_name);
CREATE INDEX IF NOT EXISTS idx_meal_plans_client_date ON meal_plans(client_id, plan_date);
CREATE INDEX IF NOT EXISTS idx_cardio_sessions_client_date ON cardio_sessions(client_id, session_date);

-- Create update triggers for tables with updated_at columns
DROP TRIGGER IF EXISTS update_daily_tracking_updated_at ON daily_tracking;
CREATE TRIGGER update_daily_tracking_updated_at 
  BEFORE UPDATE ON daily_tracking 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_workout_sessions_updated_at ON workout_sessions;
CREATE TRIGGER update_workout_sessions_updated_at 
  BEFORE UPDATE ON workout_sessions 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_supplement_protocols_updated_at ON supplement_protocols;
CREATE TRIGGER update_supplement_protocols_updated_at 
  BEFORE UPDATE ON supplement_protocols 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();