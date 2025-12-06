-- Update pricing structure with new Foundation Starter, Growth Accelerator, Performance Pro, and Enterprise Elite tiers
-- First, add new tier enum values
ALTER TYPE tier_enum ADD VALUE 'foundation_starter';
ALTER TYPE tier_enum ADD VALUE 'growth_accelerator'; 
ALTER TYPE tier_enum ADD VALUE 'performance_pro';
ALTER TYPE tier_enum ADD VALUE 'enterprise_elite';

-- Create new pricing table for flexible pricing management
CREATE TABLE public.pricing_plans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  plan_name TEXT NOT NULL,
  plan_type TEXT NOT NULL CHECK (plan_type IN ('subscription', 'one_time', 'bundle')),
  tier_level tier_enum,
  display_name TEXT NOT NULL,
  description TEXT,
  stripe_product_id TEXT,
  stripe_price_id_monthly TEXT,
  stripe_price_id_quarterly TEXT,
  stripe_price_id_annual TEXT,
  price_monthly_cents INTEGER,
  price_quarterly_cents INTEGER,
  price_annual_cents INTEGER,
  savings_quarterly_percent DECIMAL(5,2) DEFAULT 12.00,
  savings_annual_percent DECIMAL(5,2) DEFAULT 25.00,
  features JSONB DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  is_featured BOOLEAN DEFAULT false,
  is_most_popular BOOLEAN DEFAULT false,
  is_limited BOOLEAN DEFAULT false,
  max_users INTEGER,
  api_calls_monthly INTEGER,
  sort_order INTEGER DEFAULT 0,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.pricing_plans ENABLE ROW LEVEL SECURITY;

-- Create policy for viewing pricing plans
CREATE POLICY "Anyone can view active pricing plans" ON public.pricing_plans
FOR SELECT USING (is_active = true);

-- Create policy for admins to manage pricing plans
CREATE POLICY "Admins can manage pricing plans" ON public.pricing_plans
FOR ALL USING (is_admin());

-- Insert the new subscription pricing plans
INSERT INTO public.pricing_plans (
  plan_name, plan_type, tier_level, display_name, description,
  price_monthly_cents, price_quarterly_cents, price_annual_cents,
  features, metadata, is_featured, is_most_popular, sort_order
) VALUES
-- Foundation Starter
(
  'foundation_starter', 'subscription', 'foundation_starter',
  'Foundation Starter', 'Essential analytics and reporting tools',
  8900, 23592, 80099,
  '{"analytics_dashboard": true, "basic_reporting": true, "user_accounts": 5, "email_support": true, "api_calls_monthly": 1000}',
  '{"level": "starter", "response_time_hours": 72}',
  false, false, 1
),
-- Growth Accelerator
(
  'growth_accelerator', 'subscription', 'growth_accelerator', 
  'Growth Accelerator', 'Advanced analytics with priority support',
  14900, 39384, 134099,
  '{"analytics_dashboard": true, "basic_reporting": true, "advanced_analytics": true, "priority_support": true, "user_accounts": 15, "custom_integrations": true, "api_calls_monthly": 5000}',
  '{"level": "growth", "response_time_hours": 48, "featured": "true"}',
  true, true, 2
),
-- Performance Pro
(
  'performance_pro', 'subscription', 'performance_pro',
  'Performance Pro', 'White-label options with API access', 
  22900, 60552, 206099,
  '{"analytics_dashboard": true, "basic_reporting": true, "advanced_analytics": true, "priority_support": true, "white_label": true, "api_access": true, "user_accounts": -1, "account_manager": true, "api_calls_monthly": 25000}',
  '{"level": "performance", "response_time_hours": 24}',
  false, false, 3
),
-- Enterprise Elite
(
  'enterprise_elite', 'subscription', 'enterprise_elite',
  'Enterprise Elite', 'Custom development with SLA guarantees',
  34900, 92304, 314099,
  '{"analytics_dashboard": true, "basic_reporting": true, "advanced_analytics": true, "priority_support": true, "white_label": true, "api_access": true, "custom_development": true, "on_premise": true, "sla_guarantees": true, "phone_support": true, "user_accounts": -1, "api_calls_monthly": -1}',
  '{"level": "enterprise", "response_time_hours": 12}',
  false, false, 4
);

-- Insert training services (one-time payments)
INSERT INTO public.pricing_plans (
  plan_name, plan_type, display_name, description,
  price_monthly_cents, features, metadata, sort_order
) VALUES
-- Individual Training Sessions
(
  'discovery_session', 'one_time', 'Discovery Session',
  '90-minute breakthrough consultation',
  29700,
  '{"session_duration": 90, "session_type": "consultation", "includes": ["assessment", "goal_setting", "action_plan"]}',
  '{"category": "training", "session_count": 1}',
  10
),
(
  'weekly_coaching', 'one_time', 'Weekly Coaching Session',
  'Individual coaching session',
  19700,
  '{"session_duration": 60, "session_type": "coaching", "includes": ["progress_review", "training_adjustment", "q_and_a"]}',
  '{"category": "training", "session_count": 1}',
  11
),
-- Training Packages
(
  'transformation_package', 'one_time', 'Transformation Package',
  '8 comprehensive sessions (Save $79)',
  149700,
  '{"session_duration": 60, "session_type": "coaching", "includes": ["progress_review", "training_adjustment", "q_and_a", "meal_planning"]}',
  '{"category": "training", "session_count": 8, "savings": 79, "most_popular": true}',
  12
),
(
  'vip_intensive', 'one_time', 'VIP Intensive',
  '16 sessions over 4 months (Save $158)',
  299700,
  '{"session_duration": 60, "session_type": "coaching", "includes": ["progress_review", "training_adjustment", "q_and_a", "meal_planning", "supplement_guidance"]}',
  '{"category": "training", "session_count": 16, "savings": 158}',
  13
);

-- Insert bundle products
INSERT INTO public.pricing_plans (
  plan_name, plan_type, display_name, description,
  price_monthly_cents, features, metadata, sort_order
) VALUES
(
  'starter_combo_bundle', 'bundle', 'Starter Combo Bundle',
  'Foundation Starter (Annual) + Discovery + 4 Training Sessions',
  169700,
  '{"includes": ["foundation_starter_annual", "discovery_session", "4_training_sessions"], "total_savings": 292}',
  '{"bundle_type": "starter", "individual_price": 198900, "savings": 29200, "savings_percentage": 15}',
  20
);

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
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_pricing_plans_updated_at BEFORE UPDATE ON public.pricing_plans FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON public.subscriptions FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON public.orders FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_user_tier_access_updated_at BEFORE UPDATE ON public.user_tier_access FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();