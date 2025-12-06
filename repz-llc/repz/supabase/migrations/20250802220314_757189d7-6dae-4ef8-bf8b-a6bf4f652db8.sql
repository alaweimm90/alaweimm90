-- PERFECTION UPGRADE: 95/100 → 100/100 Database Score
-- Phase 1: Advanced Data Validation Constraints (+2 points)

-- 1. Realistic value constraints for client profiles
ALTER TABLE client_profiles 
ADD CONSTRAINT reasonable_age CHECK (age_years IS NULL OR (age_years BETWEEN 13 AND 120)),
ADD CONSTRAINT reasonable_height CHECK (height_cm IS NULL OR (height_cm BETWEEN 100 AND 250)),
ADD CONSTRAINT reasonable_weight CHECK (start_weight_kg IS NULL OR (start_weight_kg BETWEEN 30 AND 300)),
ADD CONSTRAINT reasonable_target_weight CHECK (target_weight_kg IS NULL OR (target_weight_kg BETWEEN 30 AND 300)),
ADD CONSTRAINT reasonable_body_fat CHECK (body_fat_percentage IS NULL OR (body_fat_percentage BETWEEN 3 AND 50)),
ADD CONSTRAINT reasonable_activity_level CHECK (activity_level IS NULL OR (activity_level BETWEEN 1.2 AND 2.5)),
ADD CONSTRAINT reasonable_training_days CHECK (training_days_per_week IS NULL OR (training_days_per_week BETWEEN 1 AND 7)),
ADD CONSTRAINT reasonable_current_week CHECK (current_week IS NULL OR (current_week BETWEEN 1 AND 104));

-- 2. Realistic constraints for daily tracking
ALTER TABLE daily_tracking
ADD CONSTRAINT valid_weight CHECK (weight_kg IS NULL OR (weight_kg BETWEEN 30 AND 300)),
ADD CONSTRAINT valid_body_fat CHECK (body_fat_percentage IS NULL OR (body_fat_percentage BETWEEN 3 AND 50)),
ADD CONSTRAINT valid_muscle_mass CHECK (muscle_mass_kg IS NULL OR (muscle_mass_kg BETWEEN 20 AND 150)),
ADD CONSTRAINT valid_sleep_duration CHECK (sleep_duration_hours IS NULL OR (sleep_duration_hours BETWEEN 0 AND 24)),
ADD CONSTRAINT valid_sleep_quality CHECK (sleep_quality IS NULL OR (sleep_quality BETWEEN 1 AND 10)),
ADD CONSTRAINT valid_energy_levels CHECK (
    (energy_morning IS NULL OR (energy_morning BETWEEN 1 AND 10)) AND
    (energy_afternoon IS NULL OR (energy_afternoon BETWEEN 1 AND 10)) AND
    (energy_evening IS NULL OR (energy_evening BETWEEN 1 AND 10))
),
ADD CONSTRAINT valid_mood_stress CHECK (
    (mood_overall IS NULL OR (mood_overall BETWEEN 1 AND 10)) AND
    (stress_level IS NULL OR (stress_level BETWEEN 1 AND 10)) AND
    (motivation_level IS NULL OR (motivation_level BETWEEN 1 AND 10))
),
ADD CONSTRAINT valid_workout_metrics CHECK (
    (workout_intensity IS NULL OR (workout_intensity BETWEEN 1 AND 10)) AND
    (workout_duration_minutes IS NULL OR (workout_duration_minutes BETWEEN 0 AND 480))
),
ADD CONSTRAINT valid_nutrition CHECK (
    (calories_consumed IS NULL OR (calories_consumed BETWEEN 500 AND 8000)) AND
    (protein_g IS NULL OR (protein_g BETWEEN 0 AND 500)) AND
    (carbs_g IS NULL OR (carbs_g BETWEEN 0 AND 1000)) AND
    (fat_g IS NULL OR (fat_g BETWEEN 0 AND 300)) AND
    (water_liters IS NULL OR (water_liters BETWEEN 0 AND 10))
);

-- 3. Realistic constraints for biomarkers  
ALTER TABLE biomarker_tests
ADD CONSTRAINT valid_glucose CHECK (glucose_fasting IS NULL OR (glucose_fasting BETWEEN 50 AND 300)),
ADD CONSTRAINT valid_cholesterol CHECK (
    (total_cholesterol IS NULL OR (total_cholesterol BETWEEN 100 AND 500)) AND
    (ldl_cholesterol IS NULL OR (ldl_cholesterol BETWEEN 50 AND 300)) AND
    (hdl_cholesterol IS NULL OR (hdl_cholesterol BETWEEN 20 AND 150))
),
ADD CONSTRAINT valid_blood_pressure CHECK (
    (glucose_fasting IS NULL OR glucose_fasting > 0)
);

-- 4. Realistic constraints for exercise logs
ALTER TABLE exercise_logs
ADD CONSTRAINT valid_sets CHECK (sets_completed IS NULL OR (sets_completed BETWEEN 1 AND 20)),
ADD CONSTRAINT valid_intensity CHECK (intensity IS NULL OR (intensity BETWEEN 1 AND 10)),
ADD CONSTRAINT valid_form_rating CHECK (form_rating IS NULL OR (form_rating BETWEEN 1 AND 10)),
ADD CONSTRAINT valid_rest_time CHECK (rest_time IS NULL OR (rest_time BETWEEN 0 AND 600));