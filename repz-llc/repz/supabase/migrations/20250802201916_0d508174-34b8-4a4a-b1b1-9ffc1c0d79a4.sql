-- ==========================================
-- PHASE 6: Fix Function Security Issues
-- ==========================================

-- Fix the functions with mutable search paths
ALTER FUNCTION update_user_integrations_updated_at() SECURITY DEFINER SET search_path = '';
ALTER FUNCTION sync_tier_features() SECURITY DEFINER SET search_path = '';

-- ==========================================
-- PHASE 7: Improve RLS Policies
-- ==========================================

-- Replace overly permissive policies with more secure ones
DROP POLICY IF EXISTS "System can insert AI analysis results" ON ai_analysis_results;
CREATE POLICY "Authenticated users can insert AI analysis results" 
ON ai_analysis_results FOR INSERT 
WITH CHECK (auth.uid() IS NOT NULL);

-- Improve achievements policy to require authentication
DROP POLICY IF EXISTS "Anyone can view achievements" ON achievements;
CREATE POLICY "Authenticated users can view achievements" 
ON achievements FOR SELECT 
USING (auth.uid() IS NOT NULL);

-- Fix hardcoded admin emails in admin_users policy
DROP POLICY IF EXISTS "Admins can view admin_users" ON admin_users;
CREATE POLICY "Verified admins can view admin_users" 
ON admin_users FOR SELECT 
USING (is_admin() OR email = auth.email());

-- Ensure coaching_messages have proper user validation
DROP POLICY IF EXISTS "System can insert coaching messages" ON coaching_messages;
CREATE POLICY "Authenticated users can insert coaching messages" 
ON coaching_messages FOR INSERT 
WITH CHECK (auth.uid() = client_id);

-- Ensure form_checks have proper user validation
DROP POLICY IF EXISTS "System can insert form checks" ON form_checks;
CREATE POLICY "Authenticated users can insert form checks" 
ON form_checks FOR INSERT 
WITH CHECK (auth.uid() = client_id);

-- Ensure nutrition_recommendations have proper user validation
DROP POLICY IF EXISTS "System can insert nutrition recommendations" ON nutrition_recommendations;
CREATE POLICY "Authenticated users can insert nutrition recommendations" 
ON nutrition_recommendations FOR INSERT 
WITH CHECK (auth.uid() = client_id);

-- ==========================================
-- PHASE 8: Add Data Validation Constraints
-- ==========================================

-- Add reasonable constraints for data validation
ALTER TABLE client_profiles ADD CONSTRAINT check_age_range CHECK (age_years IS NULL OR (age_years >= 13 AND age_years <= 120));
ALTER TABLE client_profiles ADD CONSTRAINT check_height_range CHECK (height_cm IS NULL OR (height_cm >= 100 AND height_cm <= 250));
ALTER TABLE client_profiles ADD CONSTRAINT check_weight_range CHECK (start_weight_kg IS NULL OR (start_weight_kg >= 30 AND start_weight_kg <= 300));
ALTER TABLE client_profiles ADD CONSTRAINT check_target_weight_range CHECK (target_weight_kg IS NULL OR (target_weight_kg >= 30 AND target_weight_kg <= 300));
ALTER TABLE client_profiles ADD CONSTRAINT check_body_fat_percentage CHECK (body_fat_percentage IS NULL OR (body_fat_percentage >= 1 AND body_fat_percentage <= 50));

-- Add constraints for daily tracking
ALTER TABLE daily_tracking ADD CONSTRAINT check_weight_kg_range CHECK (weight_kg IS NULL OR (weight_kg >= 30 AND weight_kg <= 300));
ALTER TABLE daily_tracking ADD CONSTRAINT check_sleep_duration CHECK (sleep_duration_hours IS NULL OR (sleep_duration_hours >= 0 AND sleep_duration_hours <= 24));
ALTER TABLE daily_tracking ADD CONSTRAINT check_energy_levels CHECK (
  (energy_morning IS NULL OR (energy_morning >= 1 AND energy_morning <= 10)) AND
  (energy_afternoon IS NULL OR (energy_afternoon >= 1 AND energy_afternoon <= 10)) AND
  (energy_evening IS NULL OR (energy_evening >= 1 AND energy_evening <= 10))
);

-- Add constraints for workout data
ALTER TABLE live_workout_sessions ADD CONSTRAINT check_duration_positive CHECK (total_duration IS NULL OR total_duration >= 0);
ALTER TABLE live_workout_sessions ADD CONSTRAINT check_exercises_positive CHECK (exercises_completed IS NULL OR exercises_completed >= 0);
ALTER TABLE live_workout_sessions ADD CONSTRAINT check_performance_score CHECK (performance_score IS NULL OR (performance_score >= 0 AND performance_score <= 100));