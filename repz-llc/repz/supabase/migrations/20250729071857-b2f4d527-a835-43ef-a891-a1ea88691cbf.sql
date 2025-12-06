-- Phase 1: Add new tier values to the enum type
ALTER TYPE tier_enum ADD VALUE IF NOT EXISTS 'core';
ALTER TYPE tier_enum ADD VALUE IF NOT EXISTS 'adaptive';
ALTER TYPE tier_enum ADD VALUE IF NOT EXISTS 'performance';

-- Phase 2: Create new tier_features table with complete feature matrix
CREATE TABLE IF NOT EXISTS public.tier_features (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  category TEXT NOT NULL,
  feature_name TEXT NOT NULL,
  feature_key TEXT UNIQUE NOT NULL,
  core BOOLEAN DEFAULT FALSE,
  adaptive BOOLEAN DEFAULT FALSE,
  performance BOOLEAN DEFAULT FALSE,
  longevity BOOLEAN DEFAULT FALSE,
  display_order INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.tier_features ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access
CREATE POLICY "tier_features_public_read" ON public.tier_features
FOR SELECT
USING (true);

-- Insert complete feature matrix
INSERT INTO public.tier_features (category, feature_name, feature_key, core, adaptive, performance, longevity, display_order) VALUES
-- Core Platform & Program
('CORE_PLATFORM', 'Training program (personalized)', 'training_program', true, true, true, true, 1),
('CORE_PLATFORM', 'Nutrition plan (macro- or meal-based)', 'nutrition_plan', true, true, true, true, 2),
('CORE_PLATFORM', 'Dashboard type', 'dashboard_type', true, true, true, true, 3),

-- Coach Access & Support
('COACH_ACCESS', 'Q&A access', 'qa_access', true, true, true, true, 4),
('COACH_ACCESS', 'Response time', 'response_time', true, true, true, true, 5),

-- Progress Tracking & Analysis
('PROGRESS_TRACKING', 'Weekly check-ins & photo reviews', 'weekly_checkins', false, true, true, true, 6),
('PROGRESS_TRACKING', 'Workout form reviews', 'form_reviews', false, true, true, true, 7),
('PROGRESS_TRACKING', 'Wearable device integration', 'wearable_integration', false, true, true, true, 8),
('PROGRESS_TRACKING', 'Sleep & recovery optimization', 'sleep_recovery', false, true, true, true, 9),
('PROGRESS_TRACKING', 'AI fitness assistant', 'ai_assistant', false, false, true, true, 10),
('PROGRESS_TRACKING', 'AI progress predictions', 'ai_predictions', false, false, true, true, 11),

-- Convenience Features
('CONVENIENCE', 'Automated grocery lists', 'grocery_lists', false, true, true, true, 12),
('CONVENIENCE', 'Travel workout generator', 'travel_workouts', false, false, true, true, 13),

-- Educational Resources
('EDUCATION', 'Science-based daily/weekly tips', 'science_tips', false, true, true, true, 14),
('EDUCATION', 'Supplements/compounds sourcing & guide', 'supplement_guide', false, false, true, true, 15),
('EDUCATION', 'Scientific blog & research summaries', 'research_summaries', false, false, true, true, 16),

-- Biohacking & Advanced Supplementation
('BIOHACKING', 'Supplements protocols', 'supplements', false, true, true, true, 17),
('BIOHACKING', 'Peptides protocols', 'peptides', false, false, true, true, 18),
('BIOHACKING', 'PEDs protocols', 'peds', false, false, true, true, 19),
('BIOHACKING', 'Nootropics & cognitive enhancement', 'nootropics', false, false, true, true, 20),
('BIOHACKING', 'Bioregulators', 'bioregulators', false, false, false, true, 21),
('BIOHACKING', 'Advanced & customized cycling schemes', 'cycling_schemes', false, false, false, true, 22),

-- Health Analytics
('HEALTH_ANALYTICS', 'Biomarker integration', 'biomarker_integration', false, false, true, true, 23),
('HEALTH_ANALYTICS', 'Blood work interpretation', 'blood_work', false, false, true, true, 24),
('HEALTH_ANALYTICS', 'Sleep, stress, and recovery guidance', 'health_guidance', false, false, true, true, 25),
('HEALTH_ANALYTICS', 'HRV-based optimization', 'hrv_optimization', false, false, false, true, 26),

-- Community & Exclusive Access
('COMMUNITY', 'Private Telegram community', 'telegram_community', false, false, true, true, 27),
('COMMUNITY', 'Exclusive materials & biohacking protocols', 'exclusive_materials', false, false, false, true, 28),
('COMMUNITY', 'Latest health & longevity insights early access', 'early_access', false, false, false, true, 29),

-- Premium Services
('PREMIUM', 'In-person training (local clients)', 'in_person_training', false, false, false, true, 30),
('PREMIUM', 'Monthly group calls', 'group_calls', false, true, true, true, 31),
('PREMIUM', 'Basic peptide guidance', 'peptide_intro', false, true, true, true, 32),
('PREMIUM', 'Quarterly in-person intensives', 'quarterly_intensives', false, false, false, true, 33),
('PREMIUM', 'Dedicated account manager', 'account_manager', false, false, false, true, 34)
ON CONFLICT (feature_key) DO NOTHING;