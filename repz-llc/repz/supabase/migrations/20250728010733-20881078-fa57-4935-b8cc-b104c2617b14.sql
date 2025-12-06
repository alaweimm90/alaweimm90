-- Create enums for the fitness coaching platform
CREATE TYPE tier_enum AS ENUM ('baseline', 'prime', 'precision', 'longevity');
CREATE TYPE gender_enum AS ENUM ('male', 'female');
CREATE TYPE goal_enum AS ENUM ('fat_loss', 'muscle_gain', 'maintenance', 'recomposition', 'performance');
CREATE TYPE session_type_enum AS ENUM ('upper', 'lower', 'push', 'pull', 'legs', 'cardio', 'rest');

-- Coach profiles table
CREATE TABLE coach_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_user_id UUID REFERENCES auth.users(id) UNIQUE NOT NULL,
  coach_name TEXT NOT NULL,
  credentials TEXT[] DEFAULT ARRAY[]::TEXT[],
  specializations TEXT[] DEFAULT ARRAY[]::TEXT[],
  max_longevity_clients INTEGER DEFAULT 5,
  current_longevity_clients INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Client profiles with tier management
CREATE TABLE client_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_user_id UUID REFERENCES auth.users(id) UNIQUE,
  coach_id UUID REFERENCES coach_profiles(id),
  subscription_tier tier_enum DEFAULT 'baseline',
  stripe_subscription_id TEXT,
  
  -- Demographics
  client_name TEXT NOT NULL,
  age_years INTEGER CHECK (age_years BETWEEN 16 AND 100),
  sex gender_enum,
  height_cm DECIMAL(5,2) CHECK (height_cm BETWEEN 120 AND 250),
  
  -- Program Configuration
  program_start_date DATE,
  program_end_date DATE,
  current_week INTEGER DEFAULT 1,
  training_days_per_week INTEGER DEFAULT 3,
  primary_goal goal_enum,
  
  -- Metabolic Data
  start_weight_kg DECIMAL(5,2),
  target_weight_kg DECIMAL(5,2),
  body_fat_percentage DECIMAL(4,2),
  activity_level DECIMAL(3,2) DEFAULT 1.55,
  
  -- Calculated Metrics
  bmi DECIMAL(4,2),
  rmr_kcal_day INTEGER,
  tdee_kcal_day INTEGER,
  lbm_kg DECIMAL(5,2),
  
  -- Tier Features (JSON for flexibility)
  tier_features JSONB DEFAULT '{
    "dashboard_type": "static",
    "weekly_checkin": false,
    "workout_reviews": false,
    "science_tips": false,
    "supplements_protocol": false,
    "peptides_protocol": false,
    "peds_protocol": false,
    "biomarker_integration": false,
    "bloodwork_interpretation": false,
    "hrv_optimization": false,
    "educational_materials": false,
    "response_time_hours": 72
  }'::jsonb,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Subscription tiers definition table
CREATE TABLE subscription_tiers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tier_name tier_enum NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  price_cents INTEGER NOT NULL,
  stripe_price_id TEXT,
  features JSONB NOT NULL,
  is_popular BOOLEAN DEFAULT false,
  is_limited BOOLEAN DEFAULT false,
  max_clients INTEGER,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert the tier definitions
INSERT INTO subscription_tiers (tier_name, display_name, price_cents, features, is_popular, is_limited, max_clients, description) VALUES
('baseline', 'Baseline Coaching', 9700, '{
  "dashboard_type": "static",
  "weekly_checkin": false,
  "workout_reviews": false,
  "science_tips": false,
  "supplements_protocol": false,
  "peptides_protocol": false,
  "peds_protocol": false,
  "biomarker_integration": false,
  "bloodwork_interpretation": false,
  "hrv_optimization": false,
  "educational_materials": false,
  "response_time_hours": 72
}', false, false, null, 'Fixed customized program with static dashboard'),

('prime', 'Prime Performance', 17900, '{
  "dashboard_type": "interactive",
  "weekly_checkin": true,
  "workout_reviews": true,
  "science_tips": true,
  "supplements_protocol": true,
  "peptides_protocol": false,
  "peds_protocol": false,
  "biomarker_integration": false,
  "bloodwork_interpretation": false,
  "hrv_optimization": false,
  "educational_materials": false,
  "response_time_hours": 48
}', true, false, null, 'Core coaching features with adaptive real-time updates'),

('precision', 'Precision Protocol', 29900, '{
  "dashboard_type": "interactive",
  "weekly_checkin": true,
  "workout_reviews": true,
  "science_tips": true,
  "supplements_protocol": true,
  "peptides_protocol": true,
  "peds_protocol": true,
  "biomarker_integration": true,
  "bloodwork_interpretation": false,
  "hrv_optimization": false,
  "educational_materials": false,
  "response_time_hours": 24
}', false, false, null, 'Data-integrated tracking with dynamic program control'),

('longevity', 'Longevity Concierge', 44900, '{
  "dashboard_type": "interactive",
  "weekly_checkin": true,
  "workout_reviews": true,
  "science_tips": true,
  "supplements_protocol": true,
  "peptides_protocol": true,
  "peds_protocol": true,
  "biomarker_integration": true,
  "bloodwork_interpretation": true,
  "hrv_optimization": true,
  "educational_materials": true,
  "response_time_hours": 12
}', false, true, 5, 'Continuous optimization with full-access coaching insights');

-- Enable Row Level Security
ALTER TABLE coach_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_tiers ENABLE ROW LEVEL SECURITY;

-- RLS Policies for coach_profiles
CREATE POLICY "coaches_can_view_own_profile" ON coach_profiles
  FOR ALL USING (auth.uid() = auth_user_id);

CREATE POLICY "admins_can_view_all_coaches" ON coach_profiles
  FOR ALL USING (is_admin());

-- RLS Policies for client_profiles  
CREATE POLICY "clients_can_view_own_profile" ON client_profiles
  FOR ALL USING (auth.uid() = auth_user_id);

CREATE POLICY "coaches_can_view_their_clients" ON client_profiles
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM coach_profiles 
      WHERE coach_profiles.auth_user_id = auth.uid() 
      AND coach_profiles.id = client_profiles.coach_id
    )
  );

CREATE POLICY "admins_can_view_all_clients" ON client_profiles
  FOR ALL USING (is_admin());

-- RLS Policies for subscription_tiers (public read access)
CREATE POLICY "anyone_can_view_tiers" ON subscription_tiers
  FOR SELECT USING (true);

CREATE POLICY "admins_can_manage_tiers" ON subscription_tiers
  FOR ALL USING (is_admin());

-- Function to update tier features when subscription changes
CREATE OR REPLACE FUNCTION update_client_tier_features()
RETURNS TRIGGER AS $$
BEGIN
  -- Update tier_features based on subscription_tier
  SELECT features INTO NEW.tier_features
  FROM subscription_tiers 
  WHERE tier_name = NEW.subscription_tier;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update tier features
CREATE TRIGGER update_tier_features_trigger
  BEFORE INSERT OR UPDATE OF subscription_tier ON client_profiles
  FOR EACH ROW EXECUTE FUNCTION update_client_tier_features();

-- Function to check if user has tier access
CREATE OR REPLACE FUNCTION has_tier_access(required_tier tier_enum)
RETURNS boolean
LANGUAGE sql
STABLE SECURITY DEFINER
AS $$
  SELECT EXISTS (
    SELECT 1 FROM client_profiles 
    WHERE auth_user_id = auth.uid()
    AND subscription_tier::text >= required_tier::text
  );
$$;

-- Function to get user role (coach or client)
CREATE OR REPLACE FUNCTION get_user_role()
RETURNS text
LANGUAGE sql
STABLE SECURITY DEFINER
AS $$
  SELECT 
    CASE 
      WHEN EXISTS (SELECT 1 FROM coach_profiles WHERE auth_user_id = auth.uid()) THEN 'coach'
      WHEN EXISTS (SELECT 1 FROM client_profiles WHERE auth_user_id = auth.uid()) THEN 'client'
      ELSE null
    END;
$$;

-- Update timestamps trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add update triggers
CREATE TRIGGER update_coach_profiles_updated_at
  BEFORE UPDATE ON coach_profiles
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_client_profiles_updated_at
  BEFORE UPDATE ON client_profiles
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();