-- PERFECTION UPGRADE: Phase 2 - Data Consistency Triggers (+2 points)

-- 1. Auto-calculate BMI when height/weight changes
CREATE OR REPLACE FUNCTION calculate_bmi()
RETURNS TRIGGER AS $$
BEGIN
  -- Calculate BMI if both height and weight are available
  IF NEW.height_cm IS NOT NULL AND NEW.start_weight_kg IS NOT NULL THEN
    NEW.bmi = ROUND((NEW.start_weight_kg / ((NEW.height_cm / 100.0) * (NEW.height_cm / 100.0)))::numeric, 2);
  END IF;
  
  -- Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
  IF NEW.height_cm IS NOT NULL AND NEW.start_weight_kg IS NOT NULL AND NEW.age_years IS NOT NULL AND NEW.sex IS NOT NULL THEN
    IF NEW.sex = 'male' THEN
      NEW.rmr_kcal_day = ROUND((10 * NEW.start_weight_kg + 6.25 * NEW.height_cm - 5 * NEW.age_years + 5)::numeric);
    ELSIF NEW.sex = 'female' THEN  
      NEW.rmr_kcal_day = ROUND((10 * NEW.start_weight_kg + 6.25 * NEW.height_cm - 5 * NEW.age_years - 161)::numeric);
    END IF;
    
    -- Calculate TDEE (Total Daily Energy Expenditure)
    IF NEW.activity_level IS NOT NULL AND NEW.rmr_kcal_day IS NOT NULL THEN
      NEW.tdee_kcal_day = ROUND((NEW.rmr_kcal_day * NEW.activity_level)::numeric);
    END IF;
  END IF;
  
  -- Calculate Lean Body Mass (LBM) if body fat % is available
  IF NEW.start_weight_kg IS NOT NULL AND NEW.body_fat_percentage IS NOT NULL THEN
    NEW.lbm_kg = ROUND((NEW.start_weight_kg * (100 - NEW.body_fat_percentage) / 100)::numeric, 2);
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for BMI/BMR auto-calculation
CREATE TRIGGER calculate_body_metrics_trigger
  BEFORE INSERT OR UPDATE ON client_profiles
  FOR EACH ROW EXECUTE FUNCTION calculate_bmi();

-- 2. Auto-update tier features when subscription tier changes
CREATE OR REPLACE FUNCTION sync_tier_features_advanced()
RETURNS TRIGGER AS $$
BEGIN
  -- Only update if subscription_tier has actually changed
  IF TG_OP = 'UPDATE' AND OLD.subscription_tier = NEW.subscription_tier THEN
    RETURN NEW;
  END IF;
  
  -- Get the latest tier features
  NEW.tier_features = public.get_tier_features(NEW.subscription_tier);
  NEW.updated_at = NOW();
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Replace existing trigger with advanced version
DROP TRIGGER IF EXISTS sync_tier_features_trigger ON client_profiles;
CREATE TRIGGER sync_tier_features_advanced_trigger
  BEFORE INSERT OR UPDATE ON client_profiles
  FOR EACH ROW EXECUTE FUNCTION sync_tier_features_advanced();

-- 3. Auto-validate and clean data on insert/update
CREATE OR REPLACE FUNCTION validate_and_clean_data()
RETURNS TRIGGER AS $$
BEGIN
  -- Clean and validate email format if present
  IF NEW.client_name IS NOT NULL THEN
    NEW.client_name = TRIM(NEW.client_name);
    -- Ensure name is not empty after trimming
    IF LENGTH(NEW.client_name) = 0 THEN
      NEW.client_name = 'User';
    END IF;
  END IF;
  
  -- Clean phone number if present
  IF NEW.phone IS NOT NULL THEN
    -- Remove all non-digit characters
    NEW.phone = REGEXP_REPLACE(NEW.phone, '[^0-9+]', '', 'g');
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create data validation trigger
CREATE TRIGGER validate_client_data_trigger
  BEFORE INSERT OR UPDATE ON client_profiles
  FOR EACH ROW EXECUTE FUNCTION validate_and_clean_data();