-- Phase 1: Legacy Cleanup - Safe tier migration for RepzCoach Pro Platform
-- Update existing legacy tier references to new structure

-- Update client_profiles table - only existing records with legacy names
UPDATE client_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier = 'baseline'::tier_enum THEN 'core'::tier_enum
  WHEN subscription_tier = 'prime'::tier_enum THEN 'adaptive'::tier_enum
  WHEN subscription_tier = 'precision'::tier_enum THEN 'performance'::tier_enum
  ELSE subscription_tier
END
WHERE subscription_tier IN ('baseline'::tier_enum, 'prime'::tier_enum, 'precision'::tier_enum);

-- Update demo_profiles table - legacy references
UPDATE demo_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier = 'baseline'::tier_enum THEN 'core'::tier_enum
  WHEN subscription_tier = 'prime'::tier_enum THEN 'adaptive'::tier_enum  
  WHEN subscription_tier = 'precision'::tier_enum THEN 'performance'::tier_enum
  ELSE subscription_tier
END
WHERE subscription_tier IN ('baseline'::tier_enum, 'prime'::tier_enum, 'precision'::tier_enum);

-- Update subscribers table - selected_tier field
UPDATE subscribers 
SET selected_tier = CASE 
  WHEN selected_tier = 'baseline' THEN 'core'
  WHEN selected_tier = 'prime' THEN 'adaptive'
  WHEN selected_tier = 'precision' THEN 'performance'
  ELSE selected_tier
END
WHERE selected_tier IN ('baseline', 'prime', 'precision');

-- Update user_tier_access table - current_tier field
UPDATE user_tier_access 
SET current_tier = CASE 
  WHEN current_tier = 'baseline'::tier_enum THEN 'core'::tier_enum
  WHEN current_tier = 'prime'::tier_enum THEN 'adaptive'::tier_enum
  WHEN current_tier = 'precision'::tier_enum THEN 'performance'::tier_enum
  ELSE current_tier
END
WHERE current_tier IN ('baseline'::tier_enum, 'prime'::tier_enum, 'precision'::tier_enum);

-- Add performance indexes for tier-based queries
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_demo_profiles_subscription_tier ON demo_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_subscribers_selected_tier ON subscribers(selected_tier);
CREATE INDEX IF NOT EXISTS idx_user_tier_access_current_tier ON user_tier_access(current_tier);

-- Create migration audit log for transparency
CREATE TABLE IF NOT EXISTS tier_migration_audit (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  table_name TEXT NOT NULL,
  old_tier TEXT,
  new_tier TEXT,
  record_count INTEGER,
  migration_phase TEXT DEFAULT 'Phase 1: Legacy Cleanup',
  migrated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Log the migration completion
INSERT INTO tier_migration_audit (table_name, old_tier, new_tier, record_count)
VALUES 
  ('client_profiles', 'Legacy tiers', 'New tier structure', 
   (SELECT COUNT(*) FROM client_profiles WHERE subscription_tier IN ('core'::tier_enum, 'adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum))),
  ('demo_profiles', 'Legacy tiers', 'New tier structure', 
   (SELECT COUNT(*) FROM demo_profiles WHERE subscription_tier IN ('core'::tier_enum, 'adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum))),
  ('subscribers', 'Legacy tier names', 'New tier names', 
   (SELECT COUNT(*) FROM subscribers WHERE selected_tier IN ('core', 'adaptive', 'performance', 'longevity'))),
  ('user_tier_access', 'Legacy tiers', 'New tier structure', 
   (SELECT COUNT(*) FROM user_tier_access WHERE current_tier IN ('core'::tier_enum, 'adaptive'::tier_enum, 'performance'::tier_enum, 'longevity'::tier_enum)));