-- Add new values to existing enum first
ALTER TYPE tier_enum ADD VALUE 'core';
ALTER TYPE tier_enum ADD VALUE 'adaptive'; 
ALTER TYPE tier_enum ADD VALUE 'performance';

-- Update demo_profiles to use new tier names
UPDATE demo_profiles SET subscription_tier = 'core' WHERE subscription_tier = 'baseline';
UPDATE demo_profiles SET subscription_tier = 'adaptive' WHERE subscription_tier = 'prime';
UPDATE demo_profiles SET subscription_tier = 'performance' WHERE subscription_tier = 'precision';

-- Update client_profiles to use new tier names
UPDATE client_profiles SET subscription_tier = 'core' WHERE subscription_tier = 'baseline';
UPDATE client_profiles SET subscription_tier = 'adaptive' WHERE subscription_tier = 'prime';
UPDATE client_profiles SET subscription_tier = 'performance' WHERE subscription_tier = 'precision';

-- Update demo profile names
UPDATE demo_profiles SET demo_name = 'Alex - Core Client' WHERE demo_name = 'Alex - Baseline Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Core' WHERE demo_name = 'Demo User - Baseline';
UPDATE demo_profiles SET demo_name = 'Sarah - Adaptive Client' WHERE demo_name = 'Sarah - Prime Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Adaptive' WHERE demo_name = 'Demo User - Prime';
UPDATE demo_profiles SET demo_name = 'Marcus - Performance Client' WHERE demo_name = 'Marcus - Precision Client';
UPDATE demo_profiles SET demo_name = 'Demo User - Performance' WHERE demo_name = 'Demo User - Precision';