-- Create weekly_checkins table
CREATE TABLE public.weekly_checkins (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id TEXT NOT NULL,
  week_number INTEGER NOT NULL,
  checkin_date DATE NOT NULL DEFAULT CURRENT_DATE,
  current_weight_kg NUMERIC,
  energy_level INTEGER NOT NULL CHECK (energy_level >= 1 AND energy_level <= 10),
  sleep_quality INTEGER NOT NULL CHECK (sleep_quality >= 1 AND sleep_quality <= 10),
  stress_level INTEGER NOT NULL CHECK (stress_level >= 1 AND stress_level <= 10),
  motivation INTEGER NOT NULL CHECK (motivation >= 1 AND motivation <= 10),
  nutrition_adherence INTEGER NOT NULL CHECK (nutrition_adherence >= 1 AND nutrition_adherence <= 10),
  hydration_adherence INTEGER NOT NULL CHECK (hydration_adherence >= 1 AND hydration_adherence <= 10),
  workouts_completed INTEGER NOT NULL DEFAULT 0,
  workout_intensity INTEGER NOT NULL CHECK (workout_intensity >= 1 AND workout_intensity <= 10),
  recovery_quality INTEGER NOT NULL CHECK (recovery_quality >= 1 AND recovery_quality <= 10),
  progress_satisfaction INTEGER NOT NULL CHECK (progress_satisfaction >= 1 AND progress_satisfaction <= 10),
  program_difficulty INTEGER NOT NULL CHECK (program_difficulty >= 1 AND program_difficulty <= 10),
  client_notes TEXT DEFAULT '',
  questions_for_coach TEXT DEFAULT '',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create workout_logs table
CREATE TABLE public.workout_logs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id TEXT NOT NULL,
  workout_date DATE NOT NULL,
  session_type TEXT NOT NULL,
  duration_minutes INTEGER,
  perceived_exertion INTEGER NOT NULL CHECK (perceived_exertion >= 1 AND perceived_exertion <= 10),
  energy_pre_workout INTEGER NOT NULL CHECK (energy_pre_workout >= 1 AND energy_pre_workout <= 10),
  energy_post_workout INTEGER NOT NULL CHECK (energy_post_workout >= 1 AND energy_post_workout <= 10),
  pump_quality INTEGER NOT NULL CHECK (pump_quality >= 1 AND pump_quality <= 10),
  focus_level INTEGER NOT NULL CHECK (focus_level >= 1 AND focus_level <= 10),
  weight_progression BOOLEAN NOT NULL DEFAULT false,
  rep_progression BOOLEAN NOT NULL DEFAULT false,
  technique_rating INTEGER NOT NULL CHECK (technique_rating >= 1 AND technique_rating <= 10),
  notes TEXT DEFAULT '',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.weekly_checkins ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workout_logs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for weekly_checkins
CREATE POLICY "Users can view their own checkins" 
ON public.weekly_checkins 
FOR SELECT 
USING (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Users can create their own checkins" 
ON public.weekly_checkins 
FOR INSERT 
WITH CHECK (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Users can update their own checkins" 
ON public.weekly_checkins 
FOR UPDATE 
USING (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Coaches can view their clients' checkins" 
ON public.weekly_checkins 
FOR SELECT 
USING (
  EXISTS (
    SELECT 1 FROM public.client_profiles cp
    JOIN public.coach_profiles coach ON coach.id = cp.coach_id
    WHERE cp.auth_user_id::text = weekly_checkins.client_id
    AND coach.auth_user_id = auth.uid()
  )
);

-- Create RLS policies for workout_logs
CREATE POLICY "Users can view their own workout logs" 
ON public.workout_logs 
FOR SELECT 
USING (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Users can create their own workout logs" 
ON public.workout_logs 
FOR INSERT 
WITH CHECK (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Users can update their own workout logs" 
ON public.workout_logs 
FOR UPDATE 
USING (
  client_id IN (
    SELECT auth_user_id::text FROM public.client_profiles WHERE auth_user_id = auth.uid()
  )
);

CREATE POLICY "Coaches can view their clients' workout logs" 
ON public.workout_logs 
FOR SELECT 
USING (
  EXISTS (
    SELECT 1 FROM public.client_profiles cp
    JOIN public.coach_profiles coach ON coach.id = cp.coach_id
    WHERE cp.auth_user_id::text = workout_logs.client_id
    AND coach.auth_user_id = auth.uid()
  )
);

-- Create triggers for updated_at
CREATE TRIGGER update_weekly_checkins_updated_at
  BEFORE UPDATE ON public.weekly_checkins
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_workout_logs_updated_at
  BEFORE UPDATE ON public.workout_logs
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();