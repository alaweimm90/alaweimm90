-- Fix security issue: Enable RLS on tier_migration_audit table
ALTER TABLE tier_migration_audit ENABLE ROW LEVEL SECURITY;

-- Create RLS policy for audit table - only admins should access migration logs
CREATE POLICY "tier_migration_audit_admin_access" ON tier_migration_audit
FOR ALL 
TO authenticated
USING (is_admin((auth.jwt() ->> 'email'::text)));

-- Verify RLS is enabled on all critical tables
ALTER TABLE tier_migration_audit ENABLE ROW LEVEL SECURITY;