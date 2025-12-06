-- Create referral_codes table for tracking user referral codes
CREATE TABLE public.referral_codes (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  code TEXT NOT NULL UNIQUE,
  uses_remaining INTEGER DEFAULT NULL, -- NULL means unlimited
  expires_at TIMESTAMP WITH TIME ZONE,
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(user_id, code)
);

-- Create referrals table for tracking who referred whom
CREATE TABLE public.referrals (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  referrer_user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  referred_user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  referral_code TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'cancelled')),
  reward_type TEXT, -- e.g., 'discount', 'free_month', 'credit'
  reward_amount DECIMAL(10, 2),
  reward_claimed BOOLEAN NOT NULL DEFAULT false,
  reward_claimed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(referred_user_id) -- A user can only be referred once
);

-- Create referral_rewards table for tracking available rewards
CREATE TABLE public.referral_rewards (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  tier TEXT NOT NULL, -- which tier this reward applies to
  reward_type TEXT NOT NULL,
  reward_value DECIMAL(10, 2) NOT NULL,
  referrer_reward TEXT,
  referee_reward TEXT,
  min_referrals INTEGER DEFAULT 1,
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.referral_codes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.referral_rewards ENABLE ROW LEVEL SECURITY;

-- RLS Policies for referral_codes
CREATE POLICY "Users can view their own referral codes"
ON public.referral_codes
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own referral codes"
ON public.referral_codes
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own referral codes"
ON public.referral_codes
FOR UPDATE
USING (auth.uid() = user_id);

-- RLS Policies for referrals
CREATE POLICY "Users can view referrals where they are referrer"
ON public.referrals
FOR SELECT
USING (auth.uid() = referrer_user_id);

CREATE POLICY "Users can view referrals where they are referred"
ON public.referrals
FOR SELECT
USING (auth.uid() = referred_user_id);

CREATE POLICY "Anyone can create referrals"
ON public.referrals
FOR INSERT
WITH CHECK (true);

CREATE POLICY "Users can update referrals where they are referrer"
ON public.referrals
FOR UPDATE
USING (auth.uid() = referrer_user_id);

-- RLS Policies for referral_rewards (read-only for all authenticated users)
CREATE POLICY "Authenticated users can view referral rewards"
ON public.referral_rewards
FOR SELECT
USING (auth.uid() IS NOT NULL);

-- Create function to generate unique referral code
CREATE OR REPLACE FUNCTION public.generate_referral_code()
RETURNS TEXT AS $$
DECLARE
  code TEXT;
  exists BOOLEAN;
BEGIN
  LOOP
    -- Generate a random 8-character code
    code := upper(substring(md5(random()::text) from 1 for 8));

    -- Check if code already exists
    SELECT EXISTS(SELECT 1 FROM public.referral_codes WHERE referral_codes.code = code) INTO exists;

    EXIT WHEN NOT exists;
  END LOOP;

  RETURN code;
END;
$$ LANGUAGE plpgsql;

-- Create function to auto-create referral code for new users
CREATE OR REPLACE FUNCTION public.create_user_referral_code()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.referral_codes (user_id, code)
  VALUES (NEW.id, public.generate_referral_code());
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger for automatic referral code creation
DROP TRIGGER IF EXISTS on_user_created_referral_code ON auth.users;
CREATE TRIGGER on_user_created_referral_code
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.create_user_referral_code();

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_referral_timestamps()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_referral_codes_updated_at
  BEFORE UPDATE ON public.referral_codes
  FOR EACH ROW
  EXECUTE FUNCTION public.update_referral_timestamps();

CREATE TRIGGER update_referrals_updated_at
  BEFORE UPDATE ON public.referrals
  FOR EACH ROW
  EXECUTE FUNCTION public.update_referral_timestamps();

CREATE TRIGGER update_referral_rewards_updated_at
  BEFORE UPDATE ON public.referral_rewards
  FOR EACH ROW
  EXECUTE FUNCTION public.update_referral_timestamps();

-- Create indexes for performance
CREATE INDEX idx_referral_codes_user_id ON public.referral_codes(user_id);
CREATE INDEX idx_referral_codes_code ON public.referral_codes(code);
CREATE INDEX idx_referrals_referrer_user_id ON public.referrals(referrer_user_id);
CREATE INDEX idx_referrals_referred_user_id ON public.referrals(referred_user_id);
CREATE INDEX idx_referrals_status ON public.referrals(status);

-- Insert default referral rewards
INSERT INTO public.referral_rewards (tier, reward_type, reward_value, referrer_reward, referee_reward, min_referrals) VALUES
  ('core', 'discount', 10.00, '10% off next month', '10% off first month', 1),
  ('adaptive', 'discount', 15.00, '15% off next month', '15% off first month', 1),
  ('performance', 'discount', 20.00, '20% off next month', '20% off first month', 1),
  ('longevity', 'free_month', 349.00, '1 free month', '1 free month', 1);

-- Create function to apply referral rewards
CREATE OR REPLACE FUNCTION public.apply_referral_reward(
  p_referral_code TEXT,
  p_referred_user_id UUID
)
RETURNS JSONB AS $$
DECLARE
  v_referrer_user_id UUID;
  v_code_valid BOOLEAN;
  v_referral_id UUID;
  v_reward RECORD;
BEGIN
  -- Check if referral code is valid
  SELECT user_id, is_active
  INTO v_referrer_user_id, v_code_valid
  FROM public.referral_codes
  WHERE code = p_referral_code
    AND (uses_remaining IS NULL OR uses_remaining > 0)
    AND (expires_at IS NULL OR expires_at > now());

  IF NOT FOUND OR NOT v_code_valid THEN
    RETURN jsonb_build_object('success', false, 'error', 'Invalid or expired referral code');
  END IF;

  -- Check if user is trying to refer themselves
  IF v_referrer_user_id = p_referred_user_id THEN
    RETURN jsonb_build_object('success', false, 'error', 'Cannot use your own referral code');
  END IF;

  -- Get reward details (using 'core' tier as default)
  SELECT * INTO v_reward
  FROM public.referral_rewards
  WHERE tier = 'core' AND is_active = true
  LIMIT 1;

  -- Create referral record
  INSERT INTO public.referrals (
    referrer_user_id,
    referred_user_id,
    referral_code,
    reward_type,
    reward_amount
  ) VALUES (
    v_referrer_user_id,
    p_referred_user_id,
    p_referral_code,
    v_reward.reward_type,
    v_reward.reward_value
  )
  RETURNING id INTO v_referral_id;

  -- Decrement uses_remaining if not unlimited
  UPDATE public.referral_codes
  SET uses_remaining = uses_remaining - 1
  WHERE code = p_referral_code
    AND uses_remaining IS NOT NULL;

  RETURN jsonb_build_object(
    'success', true,
    'referral_id', v_referral_id,
    'reward', v_reward
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
