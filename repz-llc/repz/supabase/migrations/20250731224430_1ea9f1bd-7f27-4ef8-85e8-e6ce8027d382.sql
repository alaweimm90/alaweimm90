-- Remove the admin user meshal.ow@live.com from admin_users table
DELETE FROM admin_users WHERE email = 'meshal.ow@live.com';

-- Remove the client profile for meshal.ow@live.com
DELETE FROM client_profiles WHERE auth_user_id = (
  SELECT id FROM auth.users WHERE email = 'meshal.ow@live.com'
);

-- Remove all demo profiles - we'll create new ones later
DELETE FROM demo_profiles;