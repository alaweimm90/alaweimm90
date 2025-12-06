-- Create subscriptions table to track user subscriptions
CREATE TABLE public.subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  pricing_plan_id UUID REFERENCES public.pricing_plans(id),
  stripe_customer_id TEXT,
  stripe_subscription_id TEXT UNIQUE,
  stripe_checkout_session_id TEXT,
  status TEXT CHECK (status IN ('active', 'canceled', 'past_due', 'trialing', 'incomplete', 'unpaid')),
  billing_period TEXT CHECK (billing_period IN ('monthly', 'quarterly', 'annual')),
  current_period_start TIMESTAMPTZ,
  current_period_end TIMESTAMPTZ,
  cancel_at_period_end BOOLEAN DEFAULT false,
  canceled_at TIMESTAMPTZ,
  trial_start TIMESTAMPTZ,
  trial_end TIMESTAMPTZ,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS for subscriptions
ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;

-- Users can view their own subscriptions
CREATE POLICY "Users can view own subscriptions" ON public.subscriptions
FOR SELECT USING (user_id = auth.uid());

-- Admins can view all subscriptions
CREATE POLICY "Admins can view all subscriptions" ON public.subscriptions
FOR ALL USING (is_admin());

-- Edge functions can manage subscriptions (using service role)
CREATE POLICY "Service role can manage subscriptions" ON public.subscriptions
FOR ALL USING (true);

-- Create orders table for one-time payments
CREATE TABLE public.orders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  pricing_plan_id UUID REFERENCES public.pricing_plans(id),
  stripe_customer_id TEXT,
  stripe_payment_intent_id TEXT UNIQUE,
  stripe_checkout_session_id TEXT,
  status TEXT CHECK (status IN ('pending', 'paid', 'failed', 'refunded', 'canceled')),
  amount_cents INTEGER NOT NULL,
  currency TEXT DEFAULT 'usd',
  customer_email TEXT,
  customer_name TEXT,
  billing_address JSONB,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS for orders
ALTER TABLE public.orders ENABLE ROW LEVEL SECURITY;

-- Users can view their own orders
CREATE POLICY "Users can view own orders" ON public.orders
FOR SELECT USING (user_id = auth.uid() OR customer_email = auth.email());

-- Admins can view all orders
CREATE POLICY "Admins can view all orders" ON public.orders
FOR ALL USING (is_admin());

-- Edge functions can manage orders
CREATE POLICY "Service role can manage orders" ON public.orders
FOR ALL USING (true);

-- Create user_tier_access table for feature access control
CREATE TABLE public.user_tier_access (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  subscription_id UUID REFERENCES public.subscriptions(id) ON DELETE SET NULL,
  current_tier tier_enum NOT NULL,
  tier_features JSONB DEFAULT '{}',
  access_starts_at TIMESTAMPTZ DEFAULT now(),
  access_ends_at TIMESTAMPTZ,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS for user tier access
ALTER TABLE public.user_tier_access ENABLE ROW LEVEL SECURITY;

-- Users can view their own tier access
CREATE POLICY "Users can view own tier access" ON public.user_tier_access
FOR SELECT USING (user_id = auth.uid());

-- Edge functions can manage tier access
CREATE POLICY "Service role can manage tier access" ON public.user_tier_access
FOR ALL USING (true);

-- Create updated_at triggers
CREATE TRIGGER update_pricing_plans_updated_at BEFORE UPDATE ON public.pricing_plans FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON public.subscriptions FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON public.orders FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_user_tier_access_updated_at BEFORE UPDATE ON public.user_tier_access FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();