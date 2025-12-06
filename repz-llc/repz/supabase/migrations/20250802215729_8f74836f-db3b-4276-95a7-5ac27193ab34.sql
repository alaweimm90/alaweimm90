-- Phase 3B: Table Consolidation (Medium Priority) - Corrected
-- Consolidating redundant and overlapping tables with correct column references

-- 1. Migrate data from user_profiles to client_profiles (preserve verification data)
-- Add columns to client_profiles for user verification data
ALTER TABLE client_profiles 
ADD COLUMN IF NOT EXISTS terms_accepted boolean DEFAULT false,
ADD COLUMN IF NOT EXISTS liability_waiver_signed boolean DEFAULT false,
ADD COLUMN IF NOT EXISTS medical_clearance boolean DEFAULT false,
ADD COLUMN IF NOT EXISTS email_verified boolean DEFAULT true,
ADD COLUMN IF NOT EXISTS phone_verified boolean DEFAULT false,
ADD COLUMN IF NOT EXISTS phone text;

-- Migrate user_profiles data to client_profiles
UPDATE client_profiles 
SET 
    terms_accepted = COALESCE(up.terms_accepted, false),
    liability_waiver_signed = COALESCE(up.liability_waiver_signed, false),
    medical_clearance = COALESCE(up.medical_clearance, false),
    email_verified = COALESCE(up.email_verified, true),
    phone_verified = COALESCE(up.phone_verified, false),
    phone = up.phone
FROM user_profiles up
WHERE client_profiles.auth_user_id = up.user_id;

-- 2. Migrate profiles data to client_profiles (display_name, avatar_url, bio)
ALTER TABLE client_profiles 
ADD COLUMN IF NOT EXISTS avatar_url text,
ADD COLUMN IF NOT EXISTS bio text;

UPDATE client_profiles 
SET 
    avatar_url = p.avatar_url,
    bio = p.bio
FROM profiles p
WHERE client_profiles.auth_user_id = p.user_id;

-- Update client_name if it's generic and we have display_name
UPDATE client_profiles 
SET client_name = p.display_name
FROM profiles p
WHERE client_profiles.auth_user_id = p.user_id 
AND p.display_name IS NOT NULL 
AND p.display_name != ''
AND (client_profiles.client_name IS NULL OR client_profiles.client_name = 'User');

-- 3. Drop redundant user tables after data migration
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS profiles CASCADE;