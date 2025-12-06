-- Fix missing RLS policy for exercise_sets table
-- This table had policies referencing workout_sessions which we just dropped

-- Add proper RLS policies for exercise_sets table
CREATE POLICY "users_access_own_exercise_sets" ON exercise_sets
FOR ALL USING (
    EXISTS (
        SELECT 1 FROM live_workout_sessions lws
        JOIN client_profiles cp ON cp.auth_user_id = lws.client_id
        WHERE lws.id = exercise_sets.workout_session_id 
        AND cp.auth_user_id = auth.uid()
    )
);

-- Add coach access policy for exercise_sets
CREATE POLICY "coaches_access_client_exercise_sets" ON exercise_sets
FOR ALL USING (
    EXISTS (
        SELECT 1 FROM live_workout_sessions lws
        JOIN client_profiles cp ON cp.auth_user_id = lws.client_id
        JOIN coach_profiles coach ON coach.id = cp.coach_id
        WHERE lws.id = exercise_sets.workout_session_id 
        AND coach.auth_user_id = auth.uid()
    )
);