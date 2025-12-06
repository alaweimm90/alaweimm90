-- Add authentication columns to subscribers table if they don't exist
ALTER TABLE public.subscribers 
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT,
ADD COLUMN IF NOT EXISTS stripe_payment_method_id TEXT,
ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT,
ADD COLUMN IF NOT EXISTS plan_ready_date TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS billing_start_date TIMESTAMP WITH TIME ZONE;

-- Update plan_status values if needed
UPDATE public.subscribers 
SET plan_status = 'design_in_progress' 
WHERE plan_status = 'design_in_progress' OR plan_status IS NULL;