-- Enhanced Database Schema for REPZ Coach Elite Platform

-- Create necessary enums
CREATE TYPE gender_enum AS ENUM ('male', 'female', 'other');
CREATE TYPE experience_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE time_of_day_enum AS ENUM ('early_morning', 'morning', 'afternoon', 'evening', 'night');
CREATE TYPE nutrition_skill_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE workout_type_enum AS ENUM ('strength', 'cardio', 'hiit', 'yoga', 'pilates', 'sports', 'flexibility', 'recovery');
CREATE TYPE exercise_category_enum AS ENUM ('compound', 'isolation', 'cardio', 'plyometric', 'flexibility', 'core', 'functional');
CREATE TYPE photo_type_enum AS ENUM ('intake', 'weekly', 'monthly', 'milestone', 'other');
CREATE TYPE photo_angle_enum AS ENUM ('front', 'back', 'left_side', 'right_side', 'close_up');
CREATE TYPE biomarker_test_type_enum AS ENUM ('comprehensive_metabolic', 'lipid_panel', 'hormonal_panel', 'nutritional_markers', 'inflammatory_markers', 'advanced_longevity');
CREATE TYPE protocol_type_enum AS ENUM ('supplement', 'peptide', 'hormone', 'lifestyle');
CREATE TYPE supplement_category_enum AS ENUM ('basic_health', 'performance', 'recovery', 'longevity', 'hormone_support', 'cognitive', 'digestive');
CREATE TYPE safety_rating_enum AS ENUM ('safe', 'caution', 'prescription_only', 'experimental');
CREATE TYPE meal_type_enum AS ENUM ('breakfast', 'lunch', 'dinner', 'snack', 'pre_workout', 'post_workout');
CREATE TYPE cardio_type_enum AS ENUM ('steady_state', 'hiit', 'liss', 'incline_walking', 'cycling', 'rowing', 'swimming', 'stairmaster', 'elliptical');

-- Enhanced Daily Tracking Table
CREATE TABLE daily_tracking (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  tracking_date DATE NOT NULL,
  
  -- Basic Metrics (Prime+ only)
  weight_kg DECIMAL(5,2),
  body_fat_percentage DECIMAL(4,2),
  muscle_mass_kg DECIMAL(5,2),
  water_percentage DECIMAL(4,2),
  
  -- Energy & Mood (1-10 scale)
  energy_morning INTEGER CHECK (energy_morning BETWEEN 1 AND 10),
  energy_afternoon INTEGER CHECK (energy_afternoon BETWEEN 1 AND 10),
  energy_evening INTEGER CHECK (energy_evening BETWEEN 1 AND 10),
  mood_overall INTEGER CHECK (mood_overall BETWEEN 1 AND 10),
  stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
  motivation_level INTEGER CHECK (motivation_level BETWEEN 1 AND 10),
  
  -- Sleep Metrics
  sleep_duration_hours DECIMAL(3,1),
  sleep_quality INTEGER CHECK (sleep_quality BETWEEN 1 AND 10),
  sleep_efficiency_percentage INTEGER,
  sleep_deep_percentage INTEGER,
  sleep_rem_percentage INTEGER,
  
  -- Recovery Metrics (Precision+ only)
  hrv_score INTEGER,
  resting_heart_rate INTEGER,
  heart_rate_variability DECIMAL(4,1),
  recovery_score INTEGER CHECK (recovery_score BETWEEN 1 AND 100),
  
  -- Nutrition Tracking
  calories_consumed INTEGER,
  protein_g DECIMAL(5,1),
  carbs_g DECIMAL(5,1),
  fat_g DECIMAL(5,1),
  fiber_g DECIMAL(4,1),
  water_liters DECIMAL(3,1),
  nutrition_adherence_percentage INTEGER,
  
  -- Training Metrics
  workout_completed BOOLEAN DEFAULT FALSE,
  workout_type workout_type_enum,
  workout_duration_minutes INTEGER,
  workout_intensity INTEGER CHECK (workout_intensity BETWEEN 1 AND 10),
  workout_enjoyment INTEGER CHECK (workout_enjoyment BETWEEN 1 AND 10),
  muscle_soreness INTEGER CHECK (muscle_soreness BETWEEN 1 AND 10),
  
  -- Biomarkers (Precision+ only)
  glucose_mg_dl INTEGER,
  ketones_mmol_l DECIMAL(3,1),
  blood_pressure_systolic INTEGER,
  blood_pressure_diastolic INTEGER,
  
  -- Notes
  daily_notes TEXT,
  coach_notes TEXT,
  
  UNIQUE(client_id, tracking_date),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Exercise Library
CREATE TABLE exercise_library (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  exercise_name TEXT NOT NULL UNIQUE,
  exercise_category exercise_category_enum,
  muscle_groups TEXT[],
  equipment_required TEXT[],
  difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
  instructions TEXT NOT NULL,
  video_url TEXT,
  demonstration_images TEXT[],
  contraindications TEXT[],
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workout Sessions
CREATE TABLE workout_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  workout_date DATE NOT NULL,
  workout_type workout_type_enum,
  duration_minutes INTEGER,
  notes TEXT,
  coach_feedback TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Exercise Sets
CREATE TABLE exercise_sets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workout_session_id UUID REFERENCES workout_sessions(id) ON DELETE CASCADE,
  exercise_id UUID REFERENCES exercise_library(id),
  set_number INTEGER,
  reps INTEGER,
  weight_kg DECIMAL(5,2),
  rest_seconds INTEGER,
  rpe INTEGER CHECK (rpe BETWEEN 1 AND 10),
  tempo TEXT, -- e.g., "3-1-2-1"
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Progress Photos
CREATE TABLE progress_photos (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  photo_date DATE NOT NULL,
  photo_type photo_type_enum,
  photo_angle photo_angle_enum,
  photo_url TEXT NOT NULL,
  week_number INTEGER,
  weight_at_photo DECIMAL(5,2),
  body_fat_at_photo DECIMAL(4,2),
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Biomarker Tests
CREATE TABLE biomarker_tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  test_date DATE NOT NULL,
  test_type biomarker_test_type_enum,
  lab_name TEXT,
  
  -- Metabolic Panel
  glucose_fasting INTEGER,
  insulin_fasting DECIMAL(4,1),
  hba1c DECIMAL(3,1),
  triglycerides INTEGER,
  total_cholesterol INTEGER,
  ldl_cholesterol INTEGER,
  hdl_cholesterol INTEGER,
  
  -- Hormonal Panel
  testosterone_total INTEGER,
  testosterone_free DECIMAL(4,1),
  estradiol INTEGER,
  cortisol_am DECIMAL(4,1),
  thyroid_tsh DECIMAL(4,2),
  thyroid_t3_free DECIMAL(3,1),
  thyroid_t4_free DECIMAL(3,1),
  
  -- Nutritional Markers
  vitamin_d INTEGER,
  vitamin_b12 INTEGER,
  folate INTEGER,
  iron_serum INTEGER,
  ferritin INTEGER,
  
  -- Inflammatory Markers
  crp_high_sensitivity DECIMAL(3,1),
  esr INTEGER,
  
  -- Advanced Markers (Longevity tier only)
  igf1 INTEGER,
  dhea_sulfate INTEGER,
  homocysteine DECIMAL(4,1),
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Supplement Protocols
CREATE TABLE supplement_protocols (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  protocol_name TEXT NOT NULL,
  protocol_type protocol_type_enum,
  start_date DATE,
  end_date DATE,
  tier_requirement tier_enum,
  created_by UUID REFERENCES coach_profiles(id),
  
  supplements JSONB,
  dosage_schedule JSONB,
  monitoring_markers TEXT[],
  expected_benefits TEXT[],
  potential_side_effects TEXT[],
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Supplement Compliance
CREATE TABLE supplement_compliance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  protocol_id UUID REFERENCES supplement_protocols(id) ON DELETE CASCADE,
  compliance_date DATE,
  supplements_taken JSONB,
  adherence_percentage INTEGER,
  side_effects TEXT,
  notes TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Food Database
CREATE TABLE food_database (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  food_name TEXT NOT NULL,
  brand TEXT,
  serving_size TEXT,
  calories_per_serving INTEGER,
  protein_g DECIMAL(4,1),
  carbs_g DECIMAL(4,1),
  fat_g DECIMAL(4,1),
  fiber_g DECIMAL(4,1),
  sugar_g DECIMAL(4,1),
  sodium_mg INTEGER,
  micronutrients JSONB,
  food_category TEXT,
  verified BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Meal Plans
CREATE TABLE meal_plans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  plan_date DATE,
  meal_type meal_type_enum,
  foods JSONB,
  total_calories INTEGER,
  total_protein DECIMAL(4,1),
  total_carbs DECIMAL(4,1),
  total_fat DECIMAL(4,1),
  coach_notes TEXT,
  client_feedback TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Supplement Library
CREATE TABLE supplement_library (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  supplement_name TEXT NOT NULL,
  category supplement_category_enum,
  primary_benefits TEXT[],
  dosage_range TEXT,
  timing_recommendations TEXT[],
  contraindications TEXT[],
  drug_interactions TEXT[],
  research_evidence TEXT[],
  tier_recommendation tier_enum,
  safety_profile safety_rating_enum,
  cost_effectiveness INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Peptide Protocols (Longevity Tier Only)
CREATE TABLE peptide_protocols (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  peptide_name TEXT NOT NULL,
  protocol_description TEXT,
  dosage_schedule TEXT,
  injection_sites TEXT[],
  cycle_length TEXT,
  monitoring_requirements TEXT[],
  potential_benefits TEXT[],
  risk_factors TEXT[],
  contraindications TEXT[],
  research_status TEXT,
  legal_status TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cardio Sessions
CREATE TABLE cardio_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  session_date DATE,
  cardio_type cardio_type_enum,
  duration_minutes INTEGER,
  distance_km DECIMAL(5,2),
  average_heart_rate INTEGER,
  max_heart_rate INTEGER,
  calories_burned INTEGER,
  perceived_exertion INTEGER,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on all new tables
ALTER TABLE daily_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_library ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE progress_photos ENABLE ROW LEVEL SECURITY;
ALTER TABLE biomarker_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE supplement_protocols ENABLE ROW LEVEL SECURITY;
ALTER TABLE supplement_compliance ENABLE ROW LEVEL SECURITY;
ALTER TABLE food_database ENABLE ROW LEVEL SECURITY;
ALTER TABLE meal_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE supplement_library ENABLE ROW LEVEL SECURITY;
ALTER TABLE peptide_protocols ENABLE ROW LEVEL SECURITY;
ALTER TABLE cardio_sessions ENABLE ROW LEVEL SECURITY;

-- RLS Policies for daily_tracking (Prime+ tiers only)
CREATE POLICY "daily_tracking_tier_access" ON daily_tracking
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = daily_tracking.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('prime', 'precision', 'longevity')
    )
  );

-- RLS Policies for biomarker_tests (Precision+ only)
CREATE POLICY "biomarker_precision_access" ON biomarker_tests
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = biomarker_tests.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('precision', 'longevity')
    )
  );

-- RLS Policies for peptide_protocols (Longevity only)
CREATE POLICY "peptide_longevity_access" ON peptide_protocols
  FOR SELECT TO authenticated
  USING (true);

-- RLS Policies for supplement_protocols
CREATE POLICY "supplement_protocols_access" ON supplement_protocols
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = supplement_protocols.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for progress_photos (Prime+ only)
CREATE POLICY "progress_photos_tier_access" ON progress_photos
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = progress_photos.client_id
      AND cp.auth_user_id = auth.uid()
      AND cp.subscription_tier IN ('prime', 'precision', 'longevity')
    )
  );

-- RLS Policies for workout_sessions
CREATE POLICY "workout_sessions_access" ON workout_sessions
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      WHERE cp.id = workout_sessions.client_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- RLS Policies for exercise_sets
CREATE POLICY "exercise_sets_access" ON exercise_sets
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM workout_sessions ws
      JOIN client_profiles cp ON cp.id = ws.client_id
      WHERE ws.id = exercise_sets.workout_session_id
      AND cp.auth_user_id = auth.uid()
    )
  );

-- Public access for exercise library and supplement library
CREATE POLICY "exercise_library_public" ON exercise_library FOR SELECT TO authenticated USING (true);
CREATE POLICY "supplement_library_public" ON supplement_library FOR SELECT TO authenticated USING (true);
CREATE POLICY "food_database_public" ON food_database FOR SELECT TO authenticated USING (true);

-- Coach access policies
CREATE POLICY "coaches_access_client_data" ON daily_tracking
  FOR SELECT TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = daily_tracking.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

CREATE POLICY "coaches_access_biomarkers" ON biomarker_tests
  FOR ALL TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM client_profiles cp
      JOIN coach_profiles coach ON coach.id = cp.coach_id
      WHERE cp.id = biomarker_tests.client_id
      AND coach.auth_user_id = auth.uid()
    )
  );

-- Create indexes for performance
CREATE INDEX idx_daily_tracking_client_date ON daily_tracking(client_id, tracking_date);
CREATE INDEX idx_workout_sessions_client_date ON workout_sessions(client_id, workout_date);
CREATE INDEX idx_biomarker_tests_client_date ON biomarker_tests(client_id, test_date);
CREATE INDEX idx_progress_photos_client_date ON progress_photos(client_id, photo_date);
CREATE INDEX idx_supplement_protocols_client ON supplement_protocols(client_id);
CREATE INDEX idx_exercise_library_category ON exercise_library(exercise_category);
CREATE INDEX idx_food_database_name ON food_database(food_name);

-- Add triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at column where missing and create triggers
ALTER TABLE daily_tracking ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
CREATE TRIGGER update_daily_tracking_updated_at BEFORE UPDATE ON daily_tracking FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE workout_sessions ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
CREATE TRIGGER update_workout_sessions_updated_at BEFORE UPDATE ON workout_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE supplement_protocols ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
CREATE TRIGGER update_supplement_protocols_updated_at BEFORE UPDATE ON supplement_protocols FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();