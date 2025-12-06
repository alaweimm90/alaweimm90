-- supabase/migrations/20250131_update_tier_features.sql
-- Migration to add new features and update tier access levels

-- First, ensure we have the tier_features table with all columns
CREATE TABLE IF NOT EXISTS tier_features (
    id SERIAL PRIMARY KEY,
    tier_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Core Platform & Program
    training_program BOOLEAN DEFAULT FALSE,
    training_program_type TEXT DEFAULT 'macro_based',
    nutrition_plan BOOLEAN DEFAULT FALSE,
    nutrition_plan_level TEXT DEFAULT 'basic',
    dashboard_type TEXT DEFAULT 'static_fixed',
    
    -- Coach Access & Support
    qa_access BOOLEAN DEFAULT FALSE,
    qa_support_tier TEXT DEFAULT 'limited',
    response_time_hours INTEGER DEFAULT 72,
    message_limit INTEGER DEFAULT 0,
    
    -- Progress Tracking & Analysis
    weekly_checkins BOOLEAN DEFAULT FALSE,
    checkin_depth TEXT DEFAULT 'none',
    workout_form_reviews BOOLEAN DEFAULT FALSE,
    sleep_recovery_optimization BOOLEAN DEFAULT FALSE,
    
    -- Convenience & Tech Features (NEW SECTION)
    auto_grocery_lists BOOLEAN DEFAULT FALSE,
    travel_workout_generator BOOLEAN DEFAULT FALSE,
    wearable_integration BOOLEAN DEFAULT FALSE,
    ai_fitness_assistant BOOLEAN DEFAULT FALSE,
    ai_progress_predictors BOOLEAN DEFAULT FALSE,
    
    -- Education & Content
    science_based_tips BOOLEAN DEFAULT FALSE,
    supplements_guide BOOLEAN DEFAULT FALSE,
    blog_articles_research BOOLEAN DEFAULT FALSE,
    
    -- Optimization Protocols
    supplements BOOLEAN DEFAULT FALSE,
    peptides BOOLEAN DEFAULT FALSE,
    peds BOOLEAN DEFAULT FALSE,
    nootropics_productivity BOOLEAN DEFAULT FALSE,
    bioregulators BOOLEAN DEFAULT FALSE,
    custom_cycling_schemes BOOLEAN DEFAULT FALSE,
    
    -- Health Analytics
    biomarker_integration BOOLEAN DEFAULT FALSE,
    blood_work_interpretation BOOLEAN DEFAULT FALSE,
    hrv_optimization BOOLEAN DEFAULT FALSE,
    
    -- Community & Access
    private_telegram_group BOOLEAN DEFAULT FALSE,
    exclusive_biohacking_protocols BOOLEAN DEFAULT FALSE,
    early_access_longevity_tools BOOLEAN DEFAULT FALSE,
    
    -- Premium Services
    in_person_training TEXT DEFAULT NULL,
    unlimited_coach_access BOOLEAN DEFAULT FALSE
);

-- Add columns if they don't exist (for incremental migrations)
DO $$ 
BEGIN
    -- Convenience & Tech Features
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'auto_grocery_lists') THEN
        ALTER TABLE tier_features ADD COLUMN auto_grocery_lists BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'travel_workout_generator') THEN
        ALTER TABLE tier_features ADD COLUMN travel_workout_generator BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'ai_fitness_assistant') THEN
        ALTER TABLE tier_features ADD COLUMN ai_fitness_assistant BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'ai_progress_predictors') THEN
        ALTER TABLE tier_features ADD COLUMN ai_progress_predictors BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'nootropics_productivity') THEN
        ALTER TABLE tier_features ADD COLUMN nootropics_productivity BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'blog_articles_research') THEN
        ALTER TABLE tier_features ADD COLUMN blog_articles_research BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'exclusive_biohacking_protocols') THEN
        ALTER TABLE tier_features ADD COLUMN exclusive_biohacking_protocols BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'early_access_longevity_tools') THEN
        ALTER TABLE tier_features ADD COLUMN early_access_longevity_tools BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'unlimited_coach_access') THEN
        ALTER TABLE tier_features ADD COLUMN unlimited_coach_access BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tier_features' AND column_name = 'message_limit') THEN
        ALTER TABLE tier_features ADD COLUMN message_limit INTEGER DEFAULT 0;
    END IF;
END $$;

-- Insert or update Core tier features
INSERT INTO tier_features (tier_name) VALUES ('core') ON CONFLICT (tier_name) DO NOTHING;
UPDATE tier_features 
SET 
    -- Core features remain basic
    training_program = TRUE,
    training_program_type = 'macro_based',
    nutrition_plan = TRUE,
    nutrition_plan_level = 'basic',
    dashboard_type = 'static_fixed',
    qa_access = TRUE,
    qa_support_tier = 'limited',
    response_time_hours = 72,
    message_limit = 5,
    
    -- Progress tracking - limited
    weekly_checkins = FALSE,
    checkin_depth = 'none',
    workout_form_reviews = FALSE,
    sleep_recovery_optimization = FALSE,
    
    -- Everything else is locked
    auto_grocery_lists = FALSE,
    travel_workout_generator = FALSE,
    wearable_integration = FALSE,
    ai_fitness_assistant = FALSE,
    ai_progress_predictors = FALSE,
    
    -- Education
    science_based_tips = TRUE,
    supplements_guide = FALSE,
    blog_articles_research = FALSE,
    
    -- Protocols
    supplements = FALSE,
    peptides = FALSE,
    peds = FALSE,
    nootropics_productivity = FALSE,
    bioregulators = FALSE,
    custom_cycling_schemes = FALSE,
    
    -- Analytics
    biomarker_integration = FALSE,
    blood_work_interpretation = FALSE,  
    hrv_optimization = FALSE,
    
    -- Community
    private_telegram_group = FALSE,
    exclusive_biohacking_protocols = FALSE,
    early_access_longevity_tools = FALSE,
    
    -- Premium
    in_person_training = NULL,
    unlimited_coach_access = FALSE,
    updated_at = NOW()
WHERE tier_name = 'core';

-- Insert or update Adaptive tier features
INSERT INTO tier_features (tier_name) VALUES ('adaptive') ON CONFLICT (tier_name) DO NOTHING;
UPDATE tier_features 
SET 
    -- Enhanced features
    training_program = TRUE,
    training_program_type = 'personalized',
    nutrition_plan = TRUE,
    nutrition_plan_level = 'adaptive',
    dashboard_type = 'interactive_adjustable',
    qa_access = TRUE,
    qa_support_tier = 'standard',
    response_time_hours = 48,
    message_limit = 15,
    
    -- Progress tracking enhanced
    weekly_checkins = TRUE,
    checkin_depth = 'basic',
    workout_form_reviews = TRUE,
    sleep_recovery_optimization = TRUE,
    
    -- NEW VALUE ADDS (Major differentiators)
    auto_grocery_lists = TRUE, -- NEW: Major value add
    travel_workout_generator = TRUE, -- NEW: Practical feature
    wearable_integration = FALSE,
    ai_fitness_assistant = FALSE,
    ai_progress_predictors = FALSE,
    
    -- Education enhanced
    science_based_tips = TRUE,
    supplements_guide = TRUE,
    blog_articles_research = FALSE,
    
    -- Protocols basic
    supplements = TRUE,
    peptides = FALSE,
    peds = FALSE,
    nootropics_productivity = FALSE,
    bioregulators = FALSE,
    custom_cycling_schemes = FALSE,
    
    -- Analytics - MOVED biomarker integration here
    biomarker_integration = TRUE, -- MOVED: Was Performance-only
    blood_work_interpretation = FALSE,
    hrv_optimization = FALSE,
    
    -- Community access
    private_telegram_group = TRUE,
    exclusive_biohacking_protocols = FALSE,
    early_access_longevity_tools = FALSE,
    
    -- Premium locked
    in_person_training = NULL,
    unlimited_coach_access = FALSE,
    updated_at = NOW()
WHERE tier_name = 'adaptive';

-- Insert or update Performance tier features
INSERT INTO tier_features (tier_name) VALUES ('performance') ON CONFLICT (tier_name) DO NOTHING;
UPDATE tier_features 
SET 
    -- All Adaptive features plus more
    training_program = TRUE,
    training_program_type = 'ai_optimized',
    nutrition_plan = TRUE,
    nutrition_plan_level = 'comprehensive',
    dashboard_type = 'interactive_adjustable',
    qa_access = TRUE,
    qa_support_tier = 'priority',
    response_time_hours = 24,
    message_limit = -1, -- Unlimited
    
    -- Progress tracking comprehensive
    weekly_checkins = TRUE,
    checkin_depth = 'comprehensive',
    workout_form_reviews = TRUE,
    sleep_recovery_optimization = TRUE,
    
    -- Tech features full
    auto_grocery_lists = TRUE,
    travel_workout_generator = TRUE,
    wearable_integration = TRUE,
    ai_fitness_assistant = TRUE, -- NEW: Major differentiator
    ai_progress_predictors = TRUE, -- NEW: Advanced analytics
    
    -- Education full
    science_based_tips = TRUE,
    supplements_guide = TRUE,
    blog_articles_research = TRUE, -- NEW: Educational content
    
    -- Advanced protocols
    supplements = TRUE,
    peptides = TRUE,
    peds = TRUE, -- MOVED: Was Longevity-only
    nootropics_productivity = TRUE, -- NEW: Cognitive enhancement
    bioregulators = FALSE,
    custom_cycling_schemes = FALSE,
    
    -- Analytics advanced
    biomarker_integration = TRUE,
    blood_work_interpretation = TRUE,
    hrv_optimization = TRUE,
    
    -- Community access
    private_telegram_group = TRUE,
    exclusive_biohacking_protocols = FALSE,
    early_access_longevity_tools = FALSE,
    
    -- Premium partial
    in_person_training = '50_percent_off',
    unlimited_coach_access = FALSE,
    updated_at = NOW()
WHERE tier_name = 'performance';

-- Insert or update Longevity tier features (everything enabled)
INSERT INTO tier_features (tier_name) VALUES ('longevity') ON CONFLICT (tier_name) DO NOTHING;
UPDATE tier_features 
SET 
    -- All features enabled
    training_program = TRUE,
    training_program_type = 'ai_optimized',
    nutrition_plan = TRUE,
    nutrition_plan_level = 'comprehensive',
    dashboard_type = 'interactive_adjustable',
    qa_access = TRUE,
    qa_support_tier = 'elite',
    response_time_hours = 12, -- Elite response time
    message_limit = -1, -- Unlimited
    
    -- Progress tracking elite
    weekly_checkins = TRUE,
    checkin_depth = 'comprehensive',
    workout_form_reviews = TRUE,
    sleep_recovery_optimization = TRUE,
    
    -- All tech features
    auto_grocery_lists = TRUE,
    travel_workout_generator = TRUE,
    wearable_integration = TRUE,
    ai_fitness_assistant = TRUE,
    ai_progress_predictors = TRUE,
    
    -- All education
    science_based_tips = TRUE,
    supplements_guide = TRUE,
    blog_articles_research = TRUE,
    
    -- All protocols including exclusive
    supplements = TRUE,
    peptides = TRUE,
    peds = TRUE,
    nootropics_productivity = TRUE,
    bioregulators = TRUE, -- EXCLUSIVE
    custom_cycling_schemes = TRUE, -- EXCLUSIVE
    
    -- All analytics
    biomarker_integration = TRUE,
    blood_work_interpretation = TRUE,
    hrv_optimization = TRUE,
    
    -- All community features including exclusive
    private_telegram_group = TRUE,
    exclusive_biohacking_protocols = TRUE, -- EXCLUSIVE
    early_access_longevity_tools = TRUE, -- EXCLUSIVE
    
    -- All premium services
    in_person_training = '2_sessions_60min_per_week', -- EXCLUSIVE
    unlimited_coach_access = TRUE,
    updated_at = NOW()
WHERE tier_name = 'longevity';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tier_features_tier_name ON tier_features(tier_name);
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);

-- Create or update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_tier_features_updated_at ON tier_features;
CREATE TRIGGER update_tier_features_updated_at
    BEFORE UPDATE ON tier_features
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add tier_features JSON column to client_profiles if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'client_profiles' AND column_name = 'tier_features') THEN
        ALTER TABLE client_profiles ADD COLUMN tier_features JSONB DEFAULT NULL;
    END IF;
END $$;

-- Update client_profiles to sync features for existing users
UPDATE client_profiles cp
SET tier_features = (
    SELECT row_to_json(tf.*)
    FROM tier_features tf
    WHERE tf.tier_name = cp.subscription_tier
)
WHERE subscription_tier IS NOT NULL AND subscription_tier IN ('core', 'adaptive', 'performance', 'longevity');

-- Add trigger to auto-sync tier features when subscription changes
CREATE OR REPLACE FUNCTION sync_tier_features()
RETURNS TRIGGER AS $$
BEGIN
    -- Update tier_features JSON when subscription_tier changes
    IF NEW.subscription_tier IS DISTINCT FROM OLD.subscription_tier THEN
        NEW.tier_features = (
            SELECT row_to_json(tf.*)
            FROM tier_features tf
            WHERE tf.tier_name = NEW.subscription_tier
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger if it doesn't exist
DROP TRIGGER IF EXISTS sync_tier_features_trigger ON client_profiles;
CREATE TRIGGER sync_tier_features_trigger
    BEFORE UPDATE OF subscription_tier ON client_profiles
    FOR EACH ROW
    EXECUTE FUNCTION sync_tier_features();

-- Enable RLS on tier_features table
ALTER TABLE tier_features ENABLE ROW LEVEL SECURITY;

-- Create policy for reading tier features (public access for pricing display)
DROP POLICY IF EXISTS "Enable read access for all users" ON tier_features;
CREATE POLICY "Enable read access for all users" ON tier_features
    FOR SELECT USING (true);

-- Create policy for updating tier features (admin only)
DROP POLICY IF EXISTS "Enable update for admin users only" ON tier_features;
CREATE POLICY "Enable update for admin users only" ON tier_features
    FOR UPDATE USING (
        auth.role() = 'authenticated' AND 
        EXISTS (
            SELECT 1 FROM client_profiles 
            WHERE user_id = auth.uid() 
            AND role = 'admin'
        )
    );

-- Verify the migration by selecting the data
SELECT 
    tier_name,
    auto_grocery_lists,
    biomarker_integration,
    ai_fitness_assistant,
    peds,
    bioregulators,
    in_person_training,
    response_time_hours,
    message_limit
FROM tier_features
ORDER BY 
    CASE tier_name 
        WHEN 'core' THEN 1
        WHEN 'adaptive' THEN 2
        WHEN 'performance' THEN 3
        WHEN 'longevity' THEN 4
    END;

-- Create a view for easy feature checking
CREATE OR REPLACE VIEW tier_feature_comparison AS
SELECT 
    tier_name,
    
    -- Core features
    training_program,
    training_program_type,
    nutrition_plan,
    nutrition_plan_level,
    dashboard_type,
    
    -- Support
    qa_access,
    response_time_hours,
    message_limit,
    
    -- NEW convenience features
    auto_grocery_lists,
    travel_workout_generator,
    
    -- Tech features  
    wearable_integration,
    ai_fitness_assistant,
    ai_progress_predictors,
    
    -- Protocols
    supplements,
    peptides,
    peds,
    nootropics_productivity,
    bioregulators,
    
    -- Analytics
    biomarker_integration,
    blood_work_interpretation,
    hrv_optimization,
    
    -- Premium
    in_person_training,
    unlimited_coach_access,
    
    -- Metadata
    updated_at
FROM tier_features
ORDER BY 
    CASE tier_name 
        WHEN 'core' THEN 1
        WHEN 'adaptive' THEN 2
        WHEN 'performance' THEN 3
        WHEN 'longevity' THEN 4
    END;