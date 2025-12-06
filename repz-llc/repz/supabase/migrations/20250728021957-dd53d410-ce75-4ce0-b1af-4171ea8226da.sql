-- Create live workout sessions table
CREATE TABLE public.live_workout_sessions (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id UUID REFERENCES auth.users(id) NOT NULL,
  workout_plan_id TEXT,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  completed_at TIMESTAMP WITH TIME ZONE,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
  total_duration INTEGER, -- in minutes
  exercises_completed INTEGER DEFAULT 0,
  performance_score INTEGER,
  heart_rate_data JSONB,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create exercise logs table for real-time tracking
CREATE TABLE public.exercise_logs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id UUID REFERENCES auth.users(id) NOT NULL,
  session_id UUID REFERENCES public.live_workout_sessions(id),
  exercise_name TEXT NOT NULL,
  sets_completed INTEGER NOT NULL,
  reps TEXT NOT NULL,
  weight TEXT NOT NULL,
  rest_time INTEGER, -- in seconds
  form_rating INTEGER CHECK (form_rating >= 1 AND form_rating <= 10),
  intensity INTEGER CHECK (intensity >= 1 AND intensity <= 10),
  notes TEXT,
  completed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create form checks table for AI analysis
CREATE TABLE public.form_checks (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id UUID REFERENCES auth.users(id) NOT NULL,
  session_id UUID REFERENCES public.live_workout_sessions(id),
  exercise_name TEXT NOT NULL,
  form_score INTEGER CHECK (form_score >= 1 AND form_score <= 10),
  corrections_needed TEXT[],
  confidence_score NUMERIC(3,2), -- 0.00 to 1.00
  analysis_data JSONB,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create coaching messages table for AI interactions
CREATE TABLE public.coaching_messages (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  client_id UUID REFERENCES auth.users(id) NOT NULL,
  session_id UUID REFERENCES public.live_workout_sessions(id),
  message_type TEXT NOT NULL CHECK (message_type IN ('motivation', 'form', 'warning', 'achievement', 'rest')),
  message_content TEXT NOT NULL,
  severity TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'success', 'error')),
  delivered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  is_spoken BOOLEAN DEFAULT false
);

-- Enable RLS on all tables
ALTER TABLE public.live_workout_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.exercise_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.form_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.coaching_messages ENABLE ROW LEVEL SECURITY;

-- RLS Policies for live_workout_sessions
CREATE POLICY "Users can view their own workout sessions"
ON public.live_workout_sessions
FOR SELECT
USING (auth.uid() = client_id);

CREATE POLICY "Users can create their own workout sessions"
ON public.live_workout_sessions
FOR INSERT
WITH CHECK (auth.uid() = client_id);

CREATE POLICY "Users can update their own workout sessions"
ON public.live_workout_sessions
FOR UPDATE
USING (auth.uid() = client_id);

CREATE POLICY "Coaches can view their clients' workout sessions"
ON public.live_workout_sessions
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.client_profiles cp
    JOIN public.coach_profiles coach ON coach.id = cp.coach_id
    WHERE cp.auth_user_id = live_workout_sessions.client_id
    AND coach.auth_user_id = auth.uid()
  )
);

-- RLS Policies for exercise_logs
CREATE POLICY "Users can manage their own exercise logs"
ON public.exercise_logs
FOR ALL
USING (auth.uid() = client_id);

CREATE POLICY "Coaches can view their clients' exercise logs"
ON public.exercise_logs
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.client_profiles cp
    JOIN public.coach_profiles coach ON coach.id = cp.coach_id
    WHERE cp.auth_user_id = exercise_logs.client_id
    AND coach.auth_user_id = auth.uid()
  )
);

-- RLS Policies for form_checks
CREATE POLICY "Users can view their own form checks"
ON public.form_checks
FOR SELECT
USING (auth.uid() = client_id);

CREATE POLICY "System can insert form checks"
ON public.form_checks
FOR INSERT
WITH CHECK (true);

CREATE POLICY "Coaches can view their clients' form checks"
ON public.form_checks
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.client_profiles cp
    JOIN public.coach_profiles coach ON coach.id = cp.coach_id
    WHERE cp.auth_user_id = form_checks.client_id
    AND coach.auth_user_id = auth.uid()
  )
);

-- RLS Policies for coaching_messages
CREATE POLICY "Users can view their own coaching messages"
ON public.coaching_messages
FOR SELECT
USING (auth.uid() = client_id);

CREATE POLICY "System can insert coaching messages"
ON public.coaching_messages
FOR INSERT
WITH CHECK (true);

-- Create indexes for performance
CREATE INDEX idx_live_workout_sessions_client_id ON public.live_workout_sessions(client_id);
CREATE INDEX idx_live_workout_sessions_status ON public.live_workout_sessions(status);
CREATE INDEX idx_exercise_logs_client_id ON public.exercise_logs(client_id);
CREATE INDEX idx_exercise_logs_session_id ON public.exercise_logs(session_id);
CREATE INDEX idx_form_checks_client_id ON public.form_checks(client_id);
CREATE INDEX idx_coaching_messages_client_id ON public.coaching_messages(client_id);
CREATE INDEX idx_coaching_messages_session_id ON public.coaching_messages(session_id);

-- Create trigger for updating timestamps
CREATE TRIGGER update_live_workout_sessions_updated_at
BEFORE UPDATE ON public.live_workout_sessions
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();