-- Enhanced Database Schema for REPZ Coach Elite Platform
-- Only create enums that don't already exist

DO $$ BEGIN
    CREATE TYPE experience_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE time_of_day_enum AS ENUM ('early_morning', 'morning', 'afternoon', 'evening', 'night');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE nutrition_skill_enum AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE workout_type_enum AS ENUM ('strength', 'cardio', 'hiit', 'yoga', 'pilates', 'sports', 'flexibility', 'recovery');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE exercise_category_enum AS ENUM ('compound', 'isolation', 'cardio', 'plyometric', 'flexibility', 'core', 'functional');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE photo_type_enum AS ENUM ('intake', 'weekly', 'monthly', 'milestone', 'other');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE photo_angle_enum AS ENUM ('front', 'back', 'left_side', 'right_side', 'close_up');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE biomarker_test_type_enum AS ENUM ('comprehensive_metabolic', 'lipid_panel', 'hormonal_panel', 'nutritional_markers', 'inflammatory_markers', 'advanced_longevity');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE protocol_type_enum AS ENUM ('supplement', 'peptide', 'hormone', 'lifestyle');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE supplement_category_enum AS ENUM ('basic_health', 'performance', 'recovery', 'longevity', 'hormone_support', 'cognitive', 'digestive');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE safety_rating_enum AS ENUM ('safe', 'caution', 'prescription_only', 'experimental');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE meal_type_enum AS ENUM ('breakfast', 'lunch', 'dinner', 'snack', 'pre_workout', 'post_workout');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE cardio_type_enum AS ENUM ('steady_state', 'hiit', 'liss', 'incline_walking', 'cycling', 'rowing', 'swimming', 'stairmaster', 'elliptical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Enhanced Daily Tracking Table
CREATE TABLE IF NOT EXISTS daily_tracking (
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
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(client_id, tracking_date)
);

-- Exercise Library
CREATE TABLE IF NOT EXISTS exercise_library (
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
CREATE TABLE IF NOT EXISTS workout_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
  workout_date DATE NOT NULL,
  workout_type workout_type_enum,
  duration_minutes INTEGER,
  notes TEXT,
  coach_feedback TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Exercise Sets
CREATE TABLE IF NOT EXISTS exercise_sets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workout_session_id UUID REFERENCES workout_sessions(id) ON DELETE CASCADE,
  exercise_id UUID REFERENCES exercise_library(id),
  set_number INTEGER,
  reps INTEGER,
  weight_kg DECIMAL(5,2),
  rest_seconds INTEGER,
  rpe INTEGER CHECK (rpe BETWEEN 1 AND 10),
  tempo TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Progress Photos
CREATE TABLE IF NOT EXISTS progress_photos (
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
CREATE TABLE IF NOT EXISTS biomarker_tests (
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
CREATE TABLE IF NOT EXISTS supplement_protocols (
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
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Supplement Compliance
CREATE TABLE IF NOT EXISTS supplement_compliance (
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
CREATE TABLE IF NOT EXISTS food_database (
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
CREATE TABLE IF NOT EXISTS meal_plans (
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
CREATE TABLE IF NOT EXISTS supplement_library (
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
CREATE TABLE IF NOT EXISTS peptide_protocols (
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
CREATE TABLE IF NOT EXISTS cardio_sessions (
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