-- Create a client profile for the existing user who doesn't have one
INSERT INTO public.client_profiles (
  auth_user_id,
  client_name,
  subscription_tier,
  tier_features
) VALUES (
  '3dc5ab03-4f8f-467f-a5f5-f2f00d403c09',
  'Meshal',
  'baseline',
  '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'::jsonb
);