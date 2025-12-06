-- Create demo user role and sample data for testing

-- Add demo admin user
INSERT INTO public.admin_users (email, admin_role, permissions) VALUES 
('demo@repzcoach.com', 'demo_admin', ARRAY['reservation_admin', 'client_data'])
ON CONFLICT (email) DO UPDATE SET
  admin_role = EXCLUDED.admin_role,
  permissions = EXCLUDED.permissions,
  updated_at = now();

-- Create sample subscriber data for demo purposes
INSERT INTO public.subscribers (
  email, 
  full_name, 
  phone, 
  selected_tier, 
  tier_price, 
  subscription_status, 
  plan_status, 
  goals, 
  fitness_level, 
  preferred_start_date, 
  special_requirements, 
  stripe_customer_id, 
  stripe_subscription_id,
  admin_notes,
  created_at
) VALUES 
(
  'client1.demo@example.com',
  'Sarah Johnson',
  '+1 (555) 123-4567',
  '🟠 Precision Protocol',
  '$299',
  'payment_secured',
  'design_in_progress',
  'I want to lose 20 pounds and build lean muscle. Currently working out 3x per week but not seeing results. Looking for a structured approach with nutrition guidance.',
  'intermediate',
  '2024-02-01',
  'Previous knee injury - need modifications for squats and lunges',
  'cus_demo_precision_001',
  'sub_demo_precision_001',
  'High priority client - competitive athlete background',
  now() - interval '2 days'
),
(
  'client2.demo@example.com',
  'Marcus Chen',
  '+1 (555) 234-5678',
  '⚫ Longevity Concierge',
  '$449',
  'payment_secured',
  'design_in_progress',
  'Optimize health span and performance. Executive lifestyle with limited time. Want comprehensive approach including biomarker optimization.',
  'advanced',
  '2024-01-28',
  'High stress job, travel frequently, prefer early morning workouts',
  'cus_demo_concierge_001',
  'sub_demo_concierge_001',
  'VIP client - CEO of tech company',
  now() - interval '1 day'
),
(
  'client3.demo@example.com',
  'Emma Rodriguez',
  '+1 (555) 345-6789',
  '🧑‍💻 Baseline Coaching',
  '$97',
  'payment_secured',
  'design_in_progress',
  'New to fitness, want to establish healthy habits. Work from home and need structure. Goals are weight loss and general health improvement.',
  'beginner',
  '2024-02-05',
  'No gym access - home workouts only',
  'cus_demo_baseline_001',
  'sub_demo_baseline_001',
  'First-time client - needs extra guidance',
  now() - interval '3 hours'
),
(
  'client4.demo@example.com',
  'David Thompson',
  '+1 (555) 456-7890',
  '🟠 Prime Performance',
  '$179',
  'payment_secured',
  'plan_presented',
  'Marathon training focused. Need periodized training plan and nutrition strategy. Currently running 40 miles per week.',
  'advanced',
  '2024-01-25',
  'Vegetarian diet, history of IT band issues',
  'cus_demo_prime_001',
  'sub_demo_prime_001',
  'Plan ready for approval - marathon in 16 weeks',
  now() - interval '5 days'
),
(
  'client5.demo@example.com',
  'Lisa Park',
  '+1 (555) 567-8901',
  '🟠 Precision Protocol',
  '$299',
  'payment_secured',
  'design_in_progress',
  'Post-pregnancy fitness return. Need safe progressive program to regain strength and energy. Breastfeeding so nutrition needs to be carefully planned.',
  'beginner',
  '2024-02-10',
  'Post-partum (6 months), breastfeeding, diastasis recti concerns',
  'cus_demo_precision_002',
  'sub_demo_precision_002',
  'Medical clearance received - can start program',
  now() - interval '6 hours'
)
ON CONFLICT (email) DO NOTHING;

-- Function to check if user is demo admin
CREATE OR REPLACE FUNCTION public.is_demo_admin(check_email TEXT DEFAULT NULL)
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
STABLE
SET search_path = ''
AS $$
  SELECT EXISTS (
    SELECT 1 
    FROM public.admin_users au
    WHERE 
      COALESCE(check_email, auth.email()) = au.email
      AND au.admin_role = 'demo_admin'
  );
$$;