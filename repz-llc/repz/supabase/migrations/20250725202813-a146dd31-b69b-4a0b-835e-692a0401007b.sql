-- Create subscribers table for $0 subscriptions
CREATE TABLE public.subscribers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  phone TEXT NOT NULL,
  stripe_customer_id TEXT,
  stripe_subscription_id TEXT,
  selected_tier TEXT NOT NULL,
  tier_price TEXT NOT NULL,
  subscription_status TEXT NOT NULL DEFAULT 'payment_secured',
  plan_status TEXT NOT NULL DEFAULT 'design_in_progress',
  goals TEXT NOT NULL,
  fitness_level TEXT,
  preferred_start_date DATE,
  special_requirements TEXT,
  admin_notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.subscribers ENABLE ROW LEVEL SECURITY;

-- Create policies for users to view their own subscription info
CREATE POLICY "select_own_subscription" ON public.subscribers
FOR SELECT
USING (user_id = auth.uid() OR email = auth.email());

-- Create policies for edge functions to manage subscription info
CREATE POLICY "insert_subscription" ON public.subscribers
FOR INSERT
WITH CHECK (true);

CREATE POLICY "update_subscription" ON public.subscribers
FOR UPDATE
USING (true);

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_subscribers_updated_at
BEFORE UPDATE ON public.subscribers
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();