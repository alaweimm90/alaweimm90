-- Create admin account for testing (promote your account to admin)
INSERT INTO admin_users (user_id, email, admin_role, permissions)
VALUES (
  '3dc5ab03-4f8f-467f-a5f5-f2f00d403c09',
  'meshal.ow@live.com',
  'admin',
  ARRAY['reservation_admin', 'client_data', 'stripe_access', 'email_settings', 'testing_dashboard']
) 
ON CONFLICT (email) DO UPDATE SET
  admin_role = 'admin',
  permissions = ARRAY['reservation_admin', 'client_data', 'stripe_access', 'email_settings', 'testing_dashboard'];