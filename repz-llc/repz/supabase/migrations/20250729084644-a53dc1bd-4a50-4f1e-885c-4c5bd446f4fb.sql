-- REPZ Platform Tier Restructuring Migration - Phase 1 (Safe Update)
-- Check if we need to migrate the tier structure

-- 1. Clear existing tier_features if it exists to avoid duplicates
DELETE FROM public.tier_features WHERE feature_key IN ('training_program', 'nutrition_plan');

-- 2. Update the tier enum to use new tier names (only if not already updated)
DO $$
BEGIN
    -- Check if 'core' already exists in the enum
    IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'core' AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'tier_enum')) THEN
        -- Drop and recreate the enum with new values
        ALTER TYPE tier_enum RENAME TO tier_enum_old;
        CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');
        
        -- Update client_profiles subscription_tier column
        ALTER TABLE public.client_profiles ALTER COLUMN subscription_tier TYPE tier_enum USING (
          CASE subscription_tier::text
            WHEN 'baseline' THEN 'core'::tier_enum
            WHEN 'prime' THEN 'adaptive'::tier_enum
            WHEN 'precision' THEN 'performance'::tier_enum
            WHEN 'longevity' THEN 'longevity'::tier_enum
            ELSE 'core'::tier_enum
          END
        );
        
        -- Set default for new clients
        ALTER TABLE public.client_profiles ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;
        
        -- Drop the old enum type
        DROP TYPE tier_enum_old;
    END IF;
END $$;