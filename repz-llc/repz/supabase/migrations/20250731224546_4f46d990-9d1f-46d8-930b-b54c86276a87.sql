-- Remove client profiles for the coach accounts
DELETE FROM client_profiles WHERE auth_user_id IN (
  SELECT id FROM auth.users WHERE email IN ('meshal@berkeley.edu', 'contact@repzcoach.com')
);

-- Create coach profiles for both admin accounts
INSERT INTO coach_profiles (auth_user_id, coach_name, credentials, specializations, max_longevity_clients, current_longevity_clients)
SELECT 
  au.id,
  'Meshal (Coach)',
  ARRAY['NASM-CPT', 'Precision Nutrition', 'Biomarker Analysis'],
  ARRAY['Strength Training', 'Body Recomposition', 'Longevity Optimization', 'Peptide Protocols'],
  5,
  0
FROM auth.users au 
WHERE au.email IN ('meshal@berkeley.edu', 'contact@repzcoach.com');