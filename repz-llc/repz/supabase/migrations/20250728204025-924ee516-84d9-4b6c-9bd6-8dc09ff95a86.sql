-- Insert demo profiles for testing different subscription tiers
INSERT INTO demo_profiles (demo_name, subscription_tier, age_years, sex, primary_goal, tier_features) VALUES
(
  'Alex - Baseline Client',
  'baseline',
  28,
  'male',
  'muscle_gain',
  '{"science_tips": false, "peds_protocol": false, "dashboard_type": "static", "weekly_checkin": false, "workout_reviews": false, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 72, "supplements_protocol": false, "biomarker_integration": false, "educational_materials": false, "bloodwork_interpretation": false}'
),
(
  'Sarah - Prime Client',
  'prime',
  32,
  'female',
  'fat_loss',
  '{"science_tips": true, "peds_protocol": false, "dashboard_type": "interactive", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": false, "peptides_protocol": false, "response_time_hours": 48, "supplements_protocol": true, "biomarker_integration": false, "educational_materials": true, "bloodwork_interpretation": false}'
),
(
  'Marcus - Precision Client',
  'precision',
  35,
  'male',
  'performance',
  '{"science_tips": true, "peds_protocol": true, "dashboard_type": "advanced", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": false, "response_time_hours": 24, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}'
),
(
  'Elena - Longevity Client',
  'longevity',
  42,
  'female',
  'general_health',
  '{"science_tips": true, "peds_protocol": true, "dashboard_type": "premium", "weekly_checkin": true, "workout_reviews": true, "hrv_optimization": true, "peptides_protocol": true, "response_time_hours": 12, "supplements_protocol": true, "biomarker_integration": true, "educational_materials": true, "bloodwork_interpretation": true}'
);