-- Create subscribers table for Stripe subscription management
CREATE TABLE IF NOT EXISTS public.subscribers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE,
  stripe_customer_id TEXT,
  subscribed BOOLEAN NOT NULL DEFAULT false,
  subscription_tier TEXT CHECK (subscription_tier IN ('core', 'adaptive', 'performance', 'longevity')),
  subscription_end TIMESTAMPTZ,
  stripe_subscription_id TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.subscribers ENABLE ROW LEVEL SECURITY;

-- Create policy for users to view their own subscription info
CREATE POLICY "select_own_subscription" ON public.subscribers
FOR SELECT
USING (user_id = auth.uid() OR email = auth.email());

-- Create policy for edge functions to update subscription info
CREATE POLICY "update_own_subscription" ON public.subscribers
FOR UPDATE
USING (true);

-- Create policy for edge functions to insert subscription info
CREATE POLICY "insert_subscription" ON public.subscribers
FOR INSERT
WITH CHECK (true);

-- Create demo users table for enterprise showcase
CREATE TABLE IF NOT EXISTS public.demo_users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tier TEXT NOT NULL CHECK (tier IN ('core', 'adaptive', 'performance', 'longevity')),
  email TEXT NOT NULL UNIQUE,
  password TEXT NOT NULL DEFAULT 'demo123!',
  demo_features JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Insert demo users for each tier
INSERT INTO public.demo_users (tier, email, demo_features) VALUES
('core', 'demo.core@repzcoach.com', '{"access_level": "basic", "dashboard_type": "static", "ai_coaching": false}'),
('adaptive', 'demo.adaptive@repzcoach.com', '{"access_level": "intermediate", "dashboard_type": "interactive", "ai_coaching": false, "weekly_checkins": true}'),
('performance', 'demo.performance@repzcoach.com', '{"access_level": "advanced", "dashboard_type": "interactive", "ai_coaching": true, "form_analysis": true}'),
('longevity', 'demo.longevity@repzcoach.com', '{"access_level": "premium", "dashboard_type": "advanced", "ai_coaching": true, "concierge_service": true, "in_person_training": true}')
ON CONFLICT (email) DO NOTHING;

-- Create enterprise metrics table for demo
CREATE TABLE IF NOT EXISTS public.enterprise_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  metric_name TEXT NOT NULL,
  metric_value NUMERIC NOT NULL,
  metric_unit TEXT,
  category TEXT NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Insert sample enterprise metrics
INSERT INTO public.enterprise_metrics (metric_name, metric_value, metric_unit, category) VALUES
('concurrent_users', 1247, 'users', 'performance'),
('response_time', 89, 'ms', 'performance'),
('uptime', 99.9, 'percent', 'reliability'),
('conversion_rate', 12.4, 'percent', 'business'),
('customer_satisfaction', 4.8, 'rating', 'quality'),
('security_score', 98, 'score', 'security'),
('compliance_score', 100, 'percent', 'compliance'),
('api_calls_per_minute', 45600, 'calls', 'usage'),
('database_size', 2.4, 'GB', 'infrastructure'),
('cdn_cache_hit_rate', 97.2, 'percent', 'performance')
ON CONFLICT DO NOTHING;