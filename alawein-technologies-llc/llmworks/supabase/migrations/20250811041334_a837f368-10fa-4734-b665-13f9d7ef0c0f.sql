-- Create profiles table for user data
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  avatar_url TEXT,
  bio TEXT,
  organization TEXT,
  role TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create evaluations table for Arena/Bench runs
CREATE TABLE public.evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN ('arena', 'bench')),
  mode TEXT NOT NULL, -- debate, creative, explanation, mmlu, etc.
  title TEXT NOT NULL,
  description TEXT,
  config JSONB NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  results JSONB,
  metrics JSONB,
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create models table for AI model configurations
CREATE TABLE public.models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  provider TEXT NOT NULL, -- openai, anthropic, etc.
  model_id TEXT NOT NULL, -- gpt-4, claude-3, etc.
  api_key_encrypted TEXT,
  config JSONB NOT NULL DEFAULT '{}',
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create analytics table for tracking events
CREATE TABLE public.analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  session_id TEXT,
  event_type TEXT NOT NULL,
  event_name TEXT NOT NULL,
  payload JSONB,
  url TEXT,
  referrer TEXT,
  user_agent TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analytics_events ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view their own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own profile" ON public.profiles
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = user_id);

-- Evaluations policies  
CREATE POLICY "Users can view their own evaluations" ON public.evaluations
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own evaluations" ON public.evaluations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own evaluations" ON public.evaluations
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own evaluations" ON public.evaluations
  FOR DELETE USING (auth.uid() = user_id);

-- Models policies
CREATE POLICY "Users can view their own models" ON public.models
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own models" ON public.models
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own models" ON public.models
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own models" ON public.models
  FOR DELETE USING (auth.uid() = user_id);

-- Analytics policies (allow read for own events, insert for anyone)
CREATE POLICY "Users can view analytics events" ON public.analytics_events
  FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Anyone can create analytics events" ON public.analytics_events
  FOR INSERT WITH CHECK (true);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_evaluations_updated_at
  BEFORE UPDATE ON public.evaluations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_models_updated_at
  BEFORE UPDATE ON public.models
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Create indexes for performance
CREATE INDEX idx_evaluations_user_id ON public.evaluations(user_id);
CREATE INDEX idx_evaluations_type ON public.evaluations(type);
CREATE INDEX idx_evaluations_status ON public.evaluations(status);
CREATE INDEX idx_evaluations_created_at ON public.evaluations(created_at DESC);

CREATE INDEX idx_models_user_id ON public.models(user_id);
CREATE INDEX idx_models_is_active ON public.models(is_active);

CREATE INDEX idx_analytics_events_user_id ON public.analytics_events(user_id);
CREATE INDEX idx_analytics_events_created_at ON public.analytics_events(created_at DESC);
CREATE INDEX idx_analytics_events_event_type ON public.analytics_events(event_type);