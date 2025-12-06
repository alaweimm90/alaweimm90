-- PERFECTION UPGRADE: Phase 2 - Data Consistency Triggers (+2 points)
-- Auto-calculation and data consistency functions

-- 1. Function to auto-calculate BMI
CREATE OR REPLACE FUNCTION calculate_bmi()
RETURNS TRIGGER AS $$
BEGIN
  -- Calculate BMI if both height and weight are present
  IF NEW.height_cm IS NOT NULL AND NEW.start_weight_kg IS NOT NULL THEN
    NEW.bmi = ROUND((NEW.start_weight_kg / POWER(NEW.height_cm / 100.0, 2))::numeric, 2);
  END IF;
  
  -- Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
  IF NEW.height_cm IS NOT NULL AND NEW.start_weight_kg IS NOT NULL AND NEW.age_years IS NOT NULL AND NEW.sex IS NOT NULL THEN
    IF NEW.sex = 'male' THEN
      NEW.rmr_kcal_day = ROUND(10 * NEW.start_weight_kg + 6.25 * NEW.height_cm - 5 * NEW.age_years + 5);
    ELSIF NEW.sex = 'female' THEN  
      NEW.rmr_kcal_day = ROUND(10 * NEW.start_weight_kg + 6.25 * NEW.height_cm - 5 * NEW.age_years - 161);
    END IF;
  END IF;
  
  -- Calculate TDEE (Total Daily Energy Expenditure)
  IF NEW.rmr_kcal_day IS NOT NULL AND NEW.activity_level IS NOT NULL THEN
    NEW.tdee_kcal_day = ROUND(NEW.rmr_kcal_day * NEW.activity_level);
  END IF;
  
  -- Calculate Lean Body Mass (Boer formula)
  IF NEW.height_cm IS NOT NULL AND NEW.start_weight_kg IS NOT NULL AND NEW.sex IS NOT NULL THEN
    IF NEW.sex = 'male' THEN
      NEW.lbm_kg = ROUND((0.407 * NEW.start_weight_kg + 0.267 * NEW.height_cm - 19.2)::numeric, 2);
    ELSIF NEW.sex = 'female' THEN
      NEW.lbm_kg = ROUND((0.252 * NEW.start_weight_kg + 0.473 * NEW.height_cm - 48.3)::numeric, 2);
    END IF;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Trigger for auto-calculating health metrics
CREATE TRIGGER calculate_health_metrics_trigger
BEFORE INSERT OR UPDATE ON client_profiles
FOR EACH ROW EXECUTE FUNCTION calculate_bmi();

-- 3. Function to update tier features when subscription changes
CREATE OR REPLACE FUNCTION sync_tier_features_advanced()
RETURNS TRIGGER AS $$
BEGIN
  -- Auto-sync tier features when subscription tier changes
  IF NEW.subscription_tier IS DISTINCT FROM OLD.subscription_tier THEN
    NEW.tier_features = get_tier_features(NEW.subscription_tier);
    NEW.updated_at = NOW();
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 4. Enhanced tier sync trigger
CREATE TRIGGER sync_tier_features_advanced_trigger
BEFORE UPDATE ON client_profiles
FOR EACH ROW EXECUTE FUNCTION sync_tier_features_advanced();

-- 5. Function to validate daily tracking consistency
CREATE OR REPLACE FUNCTION validate_daily_tracking()
RETURNS TRIGGER AS $$
BEGIN
  -- Ensure tracking date is not in the future
  IF NEW.tracking_date > CURRENT_DATE THEN
    RAISE EXCEPTION 'Tracking date cannot be in the future';
  END IF;
  
  -- Auto-calculate nutrition adherence if all macros are provided
  IF NEW.calories_consumed IS NOT NULL AND NEW.protein_g IS NOT NULL 
     AND NEW.carbs_g IS NOT NULL AND NEW.fat_g IS NOT NULL THEN
    -- Simple adherence calculation (can be enhanced with user targets)
    NEW.nutrition_adherence_percentage = 
      CASE 
        WHEN NEW.calories_consumed BETWEEN 1200 AND 3000 AND
             NEW.protein_g > 0.8 * 70 AND -- minimum protein per kg body weight
             NEW.water_liters >= 2.0 THEN 90
        ELSE 70
      END;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 6. Daily tracking validation trigger
CREATE TRIGGER validate_daily_tracking_trigger
BEFORE INSERT OR UPDATE ON daily_tracking
FOR EACH ROW EXECUTE FUNCTION validate_daily_tracking();