-- Phase 1: Legacy Cleanup - Update tier references in database
-- This ensures any remaining legacy tier names are migrated to new structure

-- Update client_profiles table - normalize any legacy tier references
UPDATE client_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier = 'baseline' THEN 'core'
  WHEN subscription_tier = 'prime' THEN 'adaptive'
  WHEN subscription_tier = 'precision' THEN 'performance'
  ELSE subscription_tier
END
WHERE subscription_tier IN ('baseline', 'prime', 'precision');

-- Update user_profiles table if it has tier references
UPDATE user_profiles 
SET subscription_tier = CASE 
  WHEN subscription_tier = 'baseline' THEN 'core'
  WHEN subscription_tier = 'prime' THEN 'adaptive'
  WHEN subscription_tier = 'precision' THEN 'performance'
  ELSE subscription_tier
END
WHERE subscription_tier IN ('baseline', 'prime', 'precision');

-- Update orders table to reflect new tier structure
UPDATE orders 
SET tier = CASE 
  WHEN tier = 'baseline' THEN 'core'
  WHEN tier = 'prime' THEN 'adaptive'
  WHEN tier = 'precision' THEN 'performance'
  ELSE tier
END
WHERE tier IN ('baseline', 'prime', 'precision');

-- Add indexes for better performance on tier-based queries
CREATE INDEX IF NOT EXISTS idx_client_profiles_subscription_tier ON client_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_user_profiles_subscription_tier ON user_profiles(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_orders_tier ON orders(tier);

-- Create a backup audit table to track the migration
CREATE TABLE IF NOT EXISTS tier_migration_audit (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  table_name TEXT NOT NULL,
  old_tier TEXT,
  new_tier TEXT,
  record_count INTEGER,
  migrated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert audit records for the migration
INSERT INTO tier_migration_audit (table_name, old_tier, new_tier, record_count)
SELECT 'client_profiles', 'baseline', 'core', 
  (SELECT COUNT(*) FROM client_profiles WHERE subscription_tier = 'core')
WHERE EXISTS (SELECT 1 FROM client_profiles WHERE subscription_tier = 'core');

INSERT INTO tier_migration_audit (table_name, old_tier, new_tier, record_count)
SELECT 'client_profiles', 'prime', 'adaptive',
  (SELECT COUNT(*) FROM client_profiles WHERE subscription_tier = 'adaptive')
WHERE EXISTS (SELECT 1 FROM client_profiles WHERE subscription_tier = 'adaptive');

INSERT INTO tier_migration_audit (table_name, old_tier, new_tier, record_count)
SELECT 'client_profiles', 'precision', 'performance',
  (SELECT COUNT(*) FROM client_profiles WHERE subscription_tier = 'performance')
WHERE EXISTS (SELECT 1 FROM client_profiles WHERE subscription_tier = 'performance');