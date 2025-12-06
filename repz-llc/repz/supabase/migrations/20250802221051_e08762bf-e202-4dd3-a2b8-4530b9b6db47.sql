-- Fix the validation function with proper schema reference
CREATE OR REPLACE FUNCTION validate_tier_features_integrity()
RETURNS TABLE(
    client_id uuid,
    tier text,
    missing_features text[],
    outdated_structure boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = 'public'
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
    FROM public.client_profiles cp
    WHERE cp.tier_features IS NOT NULL;
END;
$$;