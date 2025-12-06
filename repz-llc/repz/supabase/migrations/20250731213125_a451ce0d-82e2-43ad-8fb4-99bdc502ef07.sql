-- First, safely update the enum values by creating a new enum and migrating data
CREATE TYPE tier_enum_new AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Update demo_profiles using string casting
UPDATE demo_profiles SET subscription_tier = 
  CASE subscription_tier::text
    WHEN 'baseline' THEN 'core'::tier_enum_new
    WHEN 'prime' THEN 'adaptive'::tier_enum_new  
    WHEN 'precision' THEN 'performance'::tier_enum_new
    WHEN 'longevity' THEN 'longevity'::tier_enum_new
    ELSE 'core'::tier_enum_new
  END::tier_enum
WHERE subscription_tier::text IN ('baseline', 'prime', 'precision');

-- Update client_profiles using string casting  
UPDATE client_profiles SET subscription_tier =
  CASE subscription_tier::text
    WHEN 'baseline' THEN 'core'::tier_enum_new
    WHEN 'prime' THEN 'adaptive'::tier_enum_new
    WHEN 'precision' THEN 'performance'::tier_enum_new
    WHEN 'longevity' THEN 'longevity'::tier_enum_new
    ELSE 'core'::tier_enum_new
  END::tier_enum
WHERE subscription_tier::text IN ('baseline', 'prime', 'precision');

-- Update demo profile names
UPDATE demo_profiles SET demo_name = 
  CASE demo_name
    WHEN 'Alex - Baseline Client' THEN 'Alex - Core Client'
    WHEN 'Demo User - Baseline' THEN 'Demo User - Core'
    WHEN 'Sarah - Prime Client' THEN 'Sarah - Adaptive Client'
    WHEN 'Demo User - Prime' THEN 'Demo User - Adaptive'
    WHEN 'Marcus - Precision Client' THEN 'Marcus - Performance Client'
    WHEN 'Demo User - Precision' THEN 'Demo User - Performance'
    ELSE demo_name
  END;

-- Clean up
DROP TYPE tier_enum_new;