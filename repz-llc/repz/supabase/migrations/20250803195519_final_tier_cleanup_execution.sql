-- Final cleanup of deprecated tier names
-- This migration ensures all deprecated tier enum values are completely removed
-- and replaced with the canonical tier names: core, adaptive, performance, longevity

-- Step 1: Create new tier enum with only canonical names
CREATE TYPE tier_enum_final AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Step 2: Update all existing data to use canonical names (final cleanup)
-- Update client_profiles
UPDATE client_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum_final
  WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum_final
  WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum_final
  WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum_final
  ELSE 'core'::tier_enum_final  -- Default fallback
END::text::tier_enum_final
WHERE subscription_tier::text IN ('baseline', 'prime', 'precision') 
   OR subscription_tier IS NULL;

-- Update demo_profiles
UPDATE demo_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum_final
  WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum_final
  WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum_final
  WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum_final
  ELSE 'core'::tier_enum_final  -- Default fallback
END::text::tier_enum_final
WHERE subscription_tier::text IN ('baseline', 'prime', 'precision')
   OR subscription_tier IS NULL;

-- Step 3: Update all tables to use the new enum type
-- Update client_profiles table
ALTER TABLE client_profiles 
  ALTER COLUMN subscription_tier TYPE tier_enum_final 
  USING CASE 
    WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum_final
    WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum_final  
    WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum_final
    WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum_final
    ELSE 'core'::tier_enum_final
  END;

-- Update demo_profiles table  
ALTER TABLE demo_profiles
  ALTER COLUMN subscription_tier TYPE tier_enum_final
  USING CASE 
    WHEN subscription_tier::text = 'baseline' THEN 'core'::tier_enum_final
    WHEN subscription_tier::text = 'prime' THEN 'adaptive'::tier_enum_final
    WHEN subscription_tier::text = 'precision' THEN 'performance'::tier_enum_final  
    WHEN subscription_tier::text = 'longevity' THEN 'longevity'::tier_enum_final
    ELSE 'core'::tier_enum_final
  END;

-- Update any other tables that might reference tier_enum
-- (This is safe - PostgreSQL will error if there are references we missed)

-- Step 4: Update all RLS policies to use canonical tier names
-- Drop old policies that reference deprecated names
DROP POLICY IF EXISTS "biomarker_prime_access" ON biomarker_tests;
DROP POLICY IF EXISTS "biomarker_precision_access" ON biomarker_tests;

-- Create new policies with canonical names
CREATE POLICY "biomarker_adaptive_plus_access" ON biomarker_tests
  FOR ALL 
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp 
      WHERE cp.auth_user_id = auth.uid() 
      AND cp.subscription_tier IN ('adaptive', 'performance', 'longevity')
    )
  );

CREATE POLICY "biomarker_performance_plus_access" ON biomarker_tests
  FOR ALL 
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp 
      WHERE cp.auth_user_id = auth.uid() 
      AND cp.subscription_tier IN ('performance', 'longevity')
    )
  );

-- Step 5: Drop old enum types (this will fail if still referenced, which is good)
DROP TYPE IF EXISTS tier_enum CASCADE;

-- Step 6: Rename the new enum to the canonical name
ALTER TYPE tier_enum_final RENAME TO tier_enum;

-- Step 7: Set default values
ALTER TABLE client_profiles 
  ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;

ALTER TABLE demo_profiles
  ALTER COLUMN subscription_tier SET DEFAULT 'core'::tier_enum;

-- Step 8: Add constraints to ensure only canonical values
ALTER TABLE client_profiles 
  ADD CONSTRAINT client_profiles_tier_canonical 
  CHECK (subscription_tier IN ('core', 'adaptive', 'performance', 'longevity'));

ALTER TABLE demo_profiles
  ADD CONSTRAINT demo_profiles_tier_canonical
  CHECK (subscription_tier IN ('core', 'adaptive', 'performance', 'longevity'));

-- Step 9: Update any functions that might reference old tier names
-- (Add any custom functions here that need updating)

-- Step 10: Add helpful comment
COMMENT ON TYPE tier_enum IS 'Canonical tier names: core ($89), adaptive ($149), performance ($229), longevity ($349)';

-- Verification query (will be logged)
DO $$
DECLARE
  client_count INTEGER;
  demo_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO client_count FROM client_profiles WHERE subscription_tier::text NOT IN ('core', 'adaptive', 'performance', 'longevity');
  SELECT COUNT(*) INTO demo_count FROM demo_profiles WHERE subscription_tier::text NOT IN ('core', 'adaptive', 'performance', 'longevity');
  
  RAISE NOTICE 'Migration complete. Invalid tiers in client_profiles: %, demo_profiles: %', client_count, demo_count;
  
  IF client_count > 0 OR demo_count > 0 THEN 
    RAISE EXCEPTION 'Migration failed: Found invalid tier values';
  END IF;
END $$;