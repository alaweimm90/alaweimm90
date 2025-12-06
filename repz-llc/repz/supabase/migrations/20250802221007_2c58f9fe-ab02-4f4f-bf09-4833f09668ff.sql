-- FINAL REVISION: Critical Integration & Compatibility Check
-- Fixing any mismatches between database and frontend expectations

-- 1. Update tier_features default to match frontend expectations
-- The current default has outdated keys that don't match featureMatrix.ts
ALTER TABLE client_profiles 
ALTER COLUMN tier_features 
SET DEFAULT jsonb_build_object(
    'personalized_training', true,
    'nutrition_plan', true,
    'dashboard_type', 'static_fixed',
    'qa_access', 'limited',
    'response_time_hours', 72,
    'weekly_checkins', false,
    'form_review', false,
    'wearable_sync', false,
    'sleep_optimization', false,
    'ai_fitness_assistant', false,
    'ai_progress_predictors', false,
    'voice_coaching', false,
    'video_analysis', false,
    'universal_search', false,
    'production_monitoring', false,
    'quality_assurance', false,
    'auto_grocery_lists', false,
    'travel_workouts', false,
    'science_tips', true,
    'supplement_guide', false,
    'research_blog_access', false,
    'supplement_protocols', false,
    'peptides', false,
    'peds', false,
    'nootropics', false,
    'bioregulators', false,
    'custom_cycling', false,
    'biomarker_integration', false,
    'blood_work_review', false,
    'recovery_guidance', false,
    'hrv_optimization', false,
    'telegram_group', false,
    'exclusive_protocols', false,
    'early_access_tools', false,
    'in_person_training', false
);

-- 2. Update existing client profiles to have consistent tier_features
-- This ensures all profiles have the latest feature structure
UPDATE client_profiles 
SET tier_features = public.get_tier_features(subscription_tier)
WHERE tier_features IS NULL 
   OR NOT tier_features ? 'personalized_training'
   OR NOT tier_features ? 'dashboard_type';

-- 3. Ensure auth_user_id is not nullable for data integrity
-- (This is critical for frontend authentication checks)
UPDATE client_profiles 
SET auth_user_id = id 
WHERE auth_user_id IS NULL;

-- Now make it NOT NULL to prevent future issues
ALTER TABLE client_profiles 
ALTER COLUMN auth_user_id SET NOT NULL;

-- 4. Add constraint to ensure tier consistency
ALTER TABLE client_profiles 
ADD CONSTRAINT tier_features_consistency 
CHECK (
    tier_features IS NOT NULL AND 
    tier_features ? 'response_time_hours' AND
    tier_features ? 'dashboard_type'
);

-- 5. Create validation function for tier feature integrity
CREATE OR REPLACE FUNCTION validate_tier_features_integrity()
RETURNS TABLE(
    client_id uuid,
    tier text,
    missing_features text[],
    outdated_structure boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cp.id,
        cp.subscription_tier::text,
        ARRAY(
            SELECT key FROM (
                VALUES ('personalized_training'), ('nutrition_plan'), ('dashboard_type'), 
                       ('response_time_hours'), ('weekly_checkins'), ('ai_fitness_assistant')
            ) AS required_keys(key)
            WHERE NOT cp.tier_features ? key
        ) as missing_features,
        (cp.tier_features ? 'science_tips' AND NOT cp.tier_features ? 'personalized_training') as outdated_structure
    FROM client_profiles cp
    WHERE cp.tier_features IS NOT NULL;
END;
$$;