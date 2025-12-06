-- Fix security warnings by setting search_path for functions

-- Update the main dashboard function with secure search_path
CREATE OR REPLACE FUNCTION get_client_dashboard_data(user_auth_id UUID)
RETURNS JSON 
LANGUAGE plpgsql 
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  result JSON;
  input_client_id UUID;
  tier_name public.tier_enum;
BEGIN
  -- Get client ID and tier in one query
  SELECT cp.id, cp.subscription_tier 
  INTO input_client_id, tier_name
  FROM public.client_profiles cp
  WHERE cp.auth_user_id = user_auth_id;
  
  IF input_client_id IS NULL THEN
    RETURN json_build_object('error', 'Client profile not found');
  END IF;

  -- Build comprehensive dashboard data in single query
  SELECT json_build_object(
    'clientProfile', (
      SELECT json_build_object(
        'id', cp.id,
        'auth_user_id', cp.auth_user_id,
        'client_name', cp.client_name,
        'subscription_tier', cp.subscription_tier,
        'current_week', cp.current_week,
        'training_days_per_week', cp.training_days_per_week,
        'primary_goal', cp.primary_goal,
        'tier_features', cp.tier_features,
        'coach_id', cp.coach_id,
        'age_years', cp.age_years,
        'sex', cp.sex,
        'height_cm', cp.height_cm,
        'start_weight_kg', cp.start_weight_kg,
        'target_weight_kg', cp.target_weight_kg,
        'body_fat_percentage', cp.body_fat_percentage,
        'activity_level', cp.activity_level,
        'created_at', cp.created_at,
        'updated_at', cp.updated_at
      )
      FROM public.client_profiles cp
      WHERE cp.id = input_client_id
    ),
    'coachProfile', (
      SELECT json_build_object(
        'id', coach.id,
        'auth_user_id', coach.auth_user_id,
        'coach_name', coach.coach_name,
        'credentials', coach.credentials,
        'specializations', coach.specializations,
        'max_longevity_clients', coach.max_longevity_clients,
        'current_longevity_clients', coach.current_longevity_clients,
        'created_at', coach.created_at,
        'updated_at', coach.updated_at
      )
      FROM public.client_profiles cp
      JOIN public.coach_profiles coach ON cp.coach_id = coach.id
      WHERE cp.id = input_client_id
    ),
    'recentProgress', (
      SELECT COALESCE(json_agg(
        json_build_object(
          'id', dt.id,
          'client_id', dt.client_id,
          'tracking_date', dt.tracking_date,
          'weight_kg', dt.weight_kg,
          'body_fat_percentage', dt.body_fat_percentage,
          'muscle_mass_kg', dt.muscle_mass_kg,
          'energy_morning', dt.energy_morning,
          'energy_afternoon', dt.energy_afternoon,
          'energy_evening', dt.energy_evening,
          'mood_overall', dt.mood_overall,
          'stress_level', dt.stress_level,
          'motivation_level', dt.motivation_level,
          'sleep_duration_hours', dt.sleep_duration_hours,
          'sleep_quality', dt.sleep_quality,
          'workout_completed', dt.workout_completed,
          'workout_intensity', dt.workout_intensity,
          'daily_notes', dt.daily_notes,
          'created_at', dt.created_at
        ) ORDER BY dt.tracking_date DESC
      ), '[]'::json)
      FROM public.daily_tracking dt
      WHERE dt.client_id = input_client_id
      AND dt.tracking_date >= CURRENT_DATE - INTERVAL '30 days'
      LIMIT 30
    ),
    'recentWorkouts', (
      SELECT COALESCE(json_agg(
        json_build_object(
          'id', ws.id,
          'client_id', ws.client_id,
          'completed_at', ws.completed_at,
          'total_duration', ws.total_duration,
          'exercises_completed', ws.exercises_completed,
          'performance_score', ws.performance_score,
          'workout_plan_id', ws.workout_plan_id,
          'status', ws.status,
          'created_at', ws.created_at
        ) ORDER BY ws.completed_at DESC
      ), '[]'::json)
      FROM public.live_workout_sessions ws
      WHERE ws.client_id = input_client_id
      AND ws.completed_at >= CURRENT_DATE - INTERVAL '14 days'
      AND ws.completed_at IS NOT NULL
      LIMIT 10
    ),
    'progressMetrics', (
      SELECT json_build_object(
        'currentWeight', (
          SELECT dt.weight_kg 
          FROM public.daily_tracking dt 
          WHERE dt.client_id = input_client_id 
          AND dt.weight_kg IS NOT NULL 
          ORDER BY dt.tracking_date DESC 
          LIMIT 1
        ),
        'weightTrend30Days', (
          WITH weight_data AS (
            SELECT weight_kg, tracking_date,
                   LAG(weight_kg) OVER (ORDER BY tracking_date) as prev_weight
            FROM public.daily_tracking 
            WHERE client_id = input_client_id 
            AND weight_kg IS NOT NULL 
            AND tracking_date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY tracking_date DESC
            LIMIT 2
          )
          SELECT CASE 
            WHEN COUNT(*) >= 2 THEN 
              ROUND(((MAX(weight_kg) - MIN(weight_kg)) / MIN(weight_kg) * 100)::numeric, 2)
            ELSE 0 
          END
          FROM weight_data
        ),
        'workoutsThisWeek', (
          SELECT COUNT(*)
          FROM public.live_workout_sessions ws
          WHERE ws.client_id = input_client_id
          AND ws.started_at >= date_trunc('week', CURRENT_DATE)
        ),
        'totalWorkouts', (
          SELECT COUNT(*)
          FROM public.live_workout_sessions ws
          WHERE ws.client_id = input_client_id
        ),
        'averageIntensity', (
          SELECT ROUND(AVG(performance_score)::numeric, 1)
          FROM public.live_workout_sessions ws
          WHERE ws.client_id = input_client_id
          AND ws.completed_at >= CURRENT_DATE - INTERVAL '30 days'
          AND performance_score IS NOT NULL
        ),
        'streakDays', public.get_client_streak_days(input_client_id)
      )
    ),
    'lastUpdated', NOW()
  ) INTO result;
  
  RETURN result;
END;
$$;

-- Update the streak calculation function with secure search_path
CREATE OR REPLACE FUNCTION get_client_streak_days(input_client_id UUID)
RETURNS INTEGER 
LANGUAGE plpgsql 
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  current_streak INTEGER := 0;
  check_date DATE := CURRENT_DATE;
  has_activity BOOLEAN;
BEGIN
  -- Limit to reasonable streak calculation (max 365 days)
  FOR i IN 0..364 LOOP
    -- Check if there's any activity on this date (optimized with EXISTS)
    SELECT EXISTS(
      SELECT 1 FROM public.live_workout_sessions ws
      WHERE ws.client_id = input_client_id
      AND DATE(ws.started_at) = check_date
      UNION ALL
      SELECT 1 FROM public.daily_tracking dt
      WHERE dt.client_id = input_client_id
      AND dt.tracking_date = check_date
      LIMIT 1
    ) INTO has_activity;
    
    IF has_activity THEN
      current_streak := current_streak + 1;
      check_date := check_date - INTERVAL '1 day';
    ELSE
      EXIT;
    END IF;
  END LOOP;
  
  RETURN current_streak;
END;
$$;

-- Update the performance logging function with secure search_path
CREATE OR REPLACE FUNCTION log_dashboard_query_performance(
  user_id UUID,
  query_duration_ms INTEGER,
  data_size_kb INTEGER
)
RETURNS VOID 
LANGUAGE plpgsql 
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.dashboard_performance_logs (
    user_id,
    query_duration_ms,
    data_size_kb,
    logged_at
  ) VALUES (
    user_id,
    query_duration_ms,
    data_size_kb,
    NOW()
  );
END;
$$;