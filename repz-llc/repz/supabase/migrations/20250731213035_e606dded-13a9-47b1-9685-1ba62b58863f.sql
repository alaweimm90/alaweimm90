-- Update tier enum to match project standards
ALTER TYPE tier_enum RENAME TO tier_enum_old;

CREATE TYPE tier_enum AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Update demo_profiles table to use correct tier names
UPDATE demo_profiles SET subscription_tier = 'core'::tier_enum WHERE subscription_tier::text = 'baseline';
UPDATE demo_profiles SET subscription_tier = 'adaptive'::tier_enum WHERE subscription_tier::text = 'prime';
UPDATE demo_profiles SET subscription_tier = 'performance'::tier_enum WHERE subscription_tier::text = 'precision';
-- longevity stays the same

-- Update client_profiles table to use correct tier names  
UPDATE client_profiles SET subscription_tier = 'core'::tier_enum WHERE subscription_tier::text = 'baseline';
UPDATE client_profiles SET subscription_tier = 'adaptive'::tier_enum WHERE subscription_tier::text = 'prime';
UPDATE client_profiles SET subscription_tier = 'performance'::tier_enum WHERE subscription_tier::text = 'precision';
-- longevity stays the same

-- Update demo profile names to match tier standards
UPDATE demo_profiles SET demo_name = 'Alex - Core Client' WHERE demo_name = 'Alex - Baseline Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Core' WHERE demo_name = 'Demo User - Baseline';
UPDATE demo_profiles SET demo_name = 'Sarah - Adaptive Client' WHERE demo_name = 'Sarah - Prime Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Adaptive' WHERE demo_name = 'Demo User - Prime';
UPDATE demo_profiles SET demo_name = 'Marcus - Performance Client' WHERE demo_name = 'Marcus - Precision Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Performance' WHERE demo_name = 'Demo User - Precision';
UPDATE demo_profiles SET demo_name = 'Elena - Longevity Client' WHERE demo_name = 'Elena - Longevity Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Longevity' WHERE demo_name = 'Demo User - Longevity';

-- Update all column references to use new enum type
ALTER TABLE demo_profiles ALTER COLUMN subscription_tier TYPE tier_enum USING subscription_tier::text::tier_enum;
ALTER TABLE client_profiles ALTER COLUMN subscription_tier TYPE tier_enum USING subscription_tier::text::tier_enum;
ALTER TABLE biomarker_tests ALTER COLUMN test_type TYPE tier_enum USING test_type::text::tier_enum;
ALTER TABLE leaderboard_entries ALTER COLUMN tier TYPE tier_enum USING tier::text::tier_enum;
ALTER TABLE leaderboards ALTER COLUMN tier_filter TYPE tier_enum USING tier_filter::text::tier_enum;
ALTER TABLE pricing_plans ALTER COLUMN tier_level TYPE tier_enum USING tier_level::text::tier_enum;

-- Drop old enum type
DROP TYPE tier_enum_old;