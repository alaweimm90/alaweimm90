-- First, add new tier enum values
ALTER TYPE tier_enum ADD VALUE 'foundation_starter';
ALTER TYPE tier_enum ADD VALUE 'growth_accelerator'; 
ALTER TYPE tier_enum ADD VALUE 'performance_pro';
ALTER TYPE tier_enum ADD VALUE 'enterprise_elite';