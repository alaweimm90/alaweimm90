-- Phase 3A: Critical Foreign Key Implementation
-- Adding missing foreign key constraints for data integrity

-- 1. Add client-coach relationship constraint
ALTER TABLE client_profiles 
ADD CONSTRAINT fk_client_coach 
FOREIGN KEY (coach_id) REFERENCES coach_profiles(id) ON DELETE SET NULL;

-- 2. Add exercise logs to client relationship
ALTER TABLE exercise_logs 
ADD CONSTRAINT fk_exercise_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(auth_user_id) ON DELETE CASCADE;

-- 3. Add daily tracking to client relationship
ALTER TABLE daily_tracking 
ADD CONSTRAINT fk_tracking_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(id) ON DELETE CASCADE;

-- 4. Add live workout sessions to client relationship
ALTER TABLE live_workout_sessions 
ADD CONSTRAINT fk_workout_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(auth_user_id) ON DELETE CASCADE;

-- 5. Add biomarker tests to client relationship
ALTER TABLE biomarker_tests 
ADD CONSTRAINT fk_biomarker_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(id) ON DELETE CASCADE;

-- 6. Add cardio sessions to client relationship
ALTER TABLE cardio_sessions 
ADD CONSTRAINT fk_cardio_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(id) ON DELETE CASCADE;

-- 7. Add meal plans to client relationship
ALTER TABLE meal_plans 
ADD CONSTRAINT fk_meal_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(id) ON DELETE CASCADE;

-- 8. Add messages sender/recipient relationships
ALTER TABLE messages 
ADD CONSTRAINT fk_message_sender 
FOREIGN KEY (sender_id) REFERENCES client_profiles(auth_user_id) ON DELETE CASCADE;

ALTER TABLE messages 
ADD CONSTRAINT fk_message_recipient 
FOREIGN KEY (recipient_id) REFERENCES client_profiles(auth_user_id) ON DELETE CASCADE;

-- 9. Add AI analysis results to client relationship
ALTER TABLE ai_analysis_results 
ADD CONSTRAINT fk_ai_analysis_client 
FOREIGN KEY (client_id) REFERENCES client_profiles(auth_user_id) ON DELETE CASCADE;