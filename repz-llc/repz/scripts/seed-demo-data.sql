-- ============================================================================
-- REPZ DEMO DATA SEED SCRIPT
-- Populates new systems with realistic demo data for testing and development
-- ============================================================================

-- ============================================================================
-- 1. DEMO MEDICAL PROFESSIONALS
-- ============================================================================

INSERT INTO medical_professionals (
    full_name, 
    license_number, 
    medical_specialty, 
    credentials, 
    email, 
    phone, 
    clinic_name, 
    areas_of_expertise, 
    accepts_consultations, 
    consultation_fee, 
    license_verified, 
    platform_approved
) VALUES 
(
    'Dr. Marcus Thompson',
    'MD123456789',
    'Sports Medicine & Hormone Therapy',
    ARRAY['MD', 'FACSM', 'ABAAHP'],
    'dr.thompson@example.com',
    '+1-555-0101',
    'Elite Performance Medical Center',
    ARRAY['hormone_therapy', 'sports_medicine', 'peptide_therapy', 'performance_enhancement'],
    true,
    250.00,
    true,
    true
),
(
    'Dr. Sarah Chen',
    'MD987654321',
    'Anti-Aging & Longevity Medicine',
    ARRAY['MD', 'A4M', 'ABAARM'],
    'dr.chen@longevitymed.com',
    '+1-555-0102',
    'Longevity & Wellness Institute',
    ARRAY['anti_aging', 'longevity', 'bioregulators', 'preventive_medicine'],
    true,
    300.00,
    true,
    true
),
(
    'Dr. James Rodriguez',
    'MD456789123',
    'Endocrinology & Metabolic Medicine',
    ARRAY['MD', 'FACE', 'BCE'],
    'dr.rodriguez@endocare.com',
    '+1-555-0103',
    'Advanced Endocrine Solutions',
    ARRAY['hormone_therapy', 'metabolic_optimization', 'diabetes_care', 'thyroid_disorders'],
    true,
    275.00,
    true,
    true
);

-- ============================================================================
-- 2. DEMO FOOD ITEMS (EXPANDED BODYBUILDING FOODS)
-- ============================================================================

-- Additional Protein Sources
INSERT INTO food_items (name, category_id, calories_per_100g, protein_per_100g, carbs_per_100g, fat_per_100g, fiber_per_100g, dietary_labels, serving_sizes) 
SELECT 
    'Salmon (Atlantic, farmed)', 
    c.id, 
    208, 25.4, 0, 12.4, 0,
    ARRAY['high_protein', 'omega_3'],
    '[{"name": "6 oz fillet", "grams": 170}, {"name": "1 cup flaked", "grams": 154}]'::jsonb
FROM food_categories c WHERE c.name = 'Proteins'
UNION ALL
SELECT 
    'Lean Ground Beef (93/7)', 
    c.id, 
    152, 22.6, 0, 6.2, 0,
    ARRAY['high_protein', 'iron_rich'],
    '[{"name": "4 oz patty", "grams": 113}, {"name": "1 cup cooked", "grams": 135}]'::jsonb
FROM food_categories c WHERE c.name = 'Proteins'
UNION ALL
SELECT 
    'Greek Yogurt (Plain, Non-fat)', 
    c.id, 
    59, 10.3, 3.6, 0.4, 0,
    ARRAY['high_protein', 'probiotics', 'low_fat'],
    '[{"name": "1 cup", "grams": 245}, {"name": "6 oz container", "grams": 170}]'::jsonb
FROM food_categories c WHERE c.name = 'Dairy'
UNION ALL
SELECT 
    'Whey Protein Isolate', 
    c.id, 
    370, 90.0, 0, 1.0, 0,
    ARRAY['high_protein', 'post_workout', 'fast_digesting'],
    '[{"name": "1 scoop", "grams": 30}, {"name": "2 scoops", "grams": 60}]'::jsonb
FROM food_categories c WHERE c.name = 'Supplements'
ON CONFLICT DO NOTHING;

-- Complex Carbohydrates
INSERT INTO food_items (name, category_id, calories_per_100g, protein_per_100g, carbs_per_100g, fat_per_100g, fiber_per_100g, dietary_labels, serving_sizes) 
SELECT 
    'Oatmeal (Steel Cut, Cooked)', 
    c.id, 
    124, 4.3, 22.0, 2.3, 4.0,
    ARRAY['complex_carbs', 'high_fiber', 'slow_digesting'],
    '[{"name": "1 cup cooked", "grams": 245}, {"name": "1/2 cup dry", "grams": 40}]'::jsonb
FROM food_categories c WHERE c.name = 'Grains'
UNION ALL
SELECT 
    'Sweet Potato (Baked)', 
    c.id, 
    86, 1.6, 20.1, 0.1, 3.0,
    ARRAY['complex_carbs', 'beta_carotene', 'pre_workout'],
    '[{"name": "1 medium", "grams": 128}, {"name": "1 cup cubed", "grams": 133}]'::jsonb
FROM food_categories c WHERE c.name = 'Vegetables'
UNION ALL
SELECT 
    'Quinoa (Cooked)', 
    c.id, 
    120, 4.4, 21.8, 1.9, 2.8,
    ARRAY['complete_protein', 'gluten_free', 'complex_carbs'],
    '[{"name": "1 cup cooked", "grams": 185}, {"name": "1/2 cup dry", "grams": 85}]'::jsonb
FROM food_categories c WHERE c.name = 'Grains'
ON CONFLICT DO NOTHING;

-- Healthy Fats
INSERT INTO food_items (name, category_id, calories_per_100g, protein_per_100g, carbs_per_100g, fat_per_100g, fiber_per_100g, dietary_labels, serving_sizes) 
SELECT 
    'Avocado', 
    c.id, 
    160, 2.0, 8.5, 14.7, 6.7,
    ARRAY['healthy_fats', 'monounsaturated', 'potassium'],
    '[{"name": "1 medium", "grams": 150}, {"name": "1/2 cup sliced", "grams": 75}]'::jsonb
FROM food_categories c WHERE c.name = 'Fats & Oils'
UNION ALL
SELECT 
    'Almonds (Raw)', 
    c.id, 
    579, 21.2, 21.6, 49.9, 12.5,
    ARRAY['healthy_fats', 'vitamin_e', 'magnesium'],
    '[{"name": "1 oz (23 almonds)", "grams": 28}, {"name": "1/4 cup", "grams": 35}]'::jsonb
FROM food_categories c WHERE c.name = 'Fats & Oils'
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 3. DEMO RECIPES
-- ============================================================================

-- Get client ID for demo data (using first client)
DO $$
DECLARE
    demo_client_id UUID;
    chicken_id UUID;
    rice_id UUID;
    broccoli_id UUID;
    salmon_id UUID;
    sweet_potato_id UUID;
    oatmeal_id UUID;
    protein_powder_id UUID;
BEGIN
    -- Get first client for demo
    SELECT id INTO demo_client_id FROM client_profiles LIMIT 1;
    
    -- Get food item IDs
    SELECT id INTO chicken_id FROM food_items WHERE name = 'Chicken Breast (skinless)';
    SELECT id INTO rice_id FROM food_items WHERE name = 'White Rice (cooked)';
    SELECT id INTO broccoli_id FROM food_items WHERE name = 'Broccoli (raw)';
    SELECT id INTO salmon_id FROM food_items WHERE name = 'Salmon (Atlantic, farmed)';
    SELECT id INTO sweet_potato_id FROM food_items WHERE name = 'Sweet Potato (Baked)';
    SELECT id INTO oatmeal_id FROM food_items WHERE name = 'Oatmeal (Steel Cut, Cooked)';
    SELECT id INTO protein_powder_id FROM food_items WHERE name = 'Whey Protein Isolate';
    
    -- Only insert if we have a client and food items
    IF demo_client_id IS NOT NULL AND chicken_id IS NOT NULL THEN
        -- Post-Workout Power Bowl
        INSERT INTO recipes (
            created_by,
            name,
            description,
            meal_type,
            prep_time_minutes,
            cook_time_minutes,
            servings,
            difficulty_level,
            ingredients,
            instructions,
            tags,
            dietary_labels
        ) VALUES (
            demo_client_id,
            'Post-Workout Power Bowl',
            'High-protein, balanced meal perfect for post-workout recovery',
            'post_workout',
            10,
            15,
            1,
            'beginner',
            jsonb_build_array(
                jsonb_build_object('food_item_id', chicken_id, 'amount_grams', 150, 'notes', 'grilled'),
                jsonb_build_object('food_item_id', rice_id, 'amount_grams', 100, 'notes', 'steamed'),
                jsonb_build_object('food_item_id', broccoli_id, 'amount_grams', 100, 'notes', 'steamed')
            ),
            ARRAY[
                'Grill chicken breast with minimal oil',
                'Steam rice according to package instructions',
                'Steam broccoli until tender-crisp',
                'Combine in bowl and season to taste'
            ],
            ARRAY['high_protein', 'post_workout', 'balanced', 'meal_prep'],
            ARRAY['gluten_free', 'dairy_free']
        );
        
        -- Salmon & Sweet Potato Dinner
        INSERT INTO recipes (
            created_by,
            name,
            description,
            meal_type,
            prep_time_minutes,
            cook_time_minutes,
            servings,
            difficulty_level,
            ingredients,
            instructions,
            tags,
            dietary_labels
        ) VALUES (
            demo_client_id,
            'Omega-3 Salmon Dinner',
            'Rich in healthy fats and complex carbs for evening meal',
            'dinner',
            5,
            25,
            1,
            'intermediate',
            jsonb_build_array(
                jsonb_build_object('food_item_id', salmon_id, 'amount_grams', 170, 'notes', '6oz fillet'),
                jsonb_build_object('food_item_id', sweet_potato_id, 'amount_grams', 200, 'notes', 'medium sized')
            ),
            ARRAY[
                'Preheat oven to 400°F',
                'Season salmon with herbs and lemon',
                'Bake sweet potato for 45 minutes',
                'Bake salmon for 12-15 minutes',
                'Serve immediately'
            ],
            ARRAY['omega_3', 'anti_inflammatory', 'nutrient_dense'],
            ARRAY['gluten_free', 'dairy_free', 'paleo']
        );
        
        -- Pre-Workout Oatmeal
        INSERT INTO recipes (
            created_by,
            name,
            description,
            meal_type,
            prep_time_minutes,
            cook_time_minutes,
            servings,
            difficulty_level,
            ingredients,
            instructions,
            tags,
            dietary_labels
        ) VALUES (
            demo_client_id,
            'Pre-Workout Power Oats',
            'Sustained energy breakfast with complete nutrition',
            'breakfast',
            5,
            10,
            1,
            'beginner',
            jsonb_build_array(
                jsonb_build_object('food_item_id', oatmeal_id, 'amount_grams', 245, 'notes', '1 cup cooked'),
                jsonb_build_object('food_item_id', protein_powder_id, 'amount_grams', 30, 'notes', '1 scoop vanilla')
            ),
            ARRAY[
                'Cook steel-cut oats according to package',
                'Let cool for 2 minutes',
                'Stir in protein powder until smooth',
                'Add berries or banana if desired'
            ],
            ARRAY['pre_workout', 'high_protein', 'sustained_energy'],
            ARRAY['vegetarian']
        );
    END IF;
END $$;

-- ============================================================================
-- 4. DEMO MEAL PLAN
-- ============================================================================

DO $$
DECLARE
    demo_client_id UUID;
    power_bowl_id UUID;
    salmon_dinner_id UUID;
    power_oats_id UUID;
BEGIN
    -- Get demo client and recipes
    SELECT id INTO demo_client_id FROM client_profiles LIMIT 1;
    SELECT id INTO power_bowl_id FROM recipes WHERE name = 'Post-Workout Power Bowl';
    SELECT id INTO salmon_dinner_id FROM recipes WHERE name = 'Omega-3 Salmon Dinner';
    SELECT id INTO power_oats_id FROM recipes WHERE name = 'Pre-Workout Power Oats';
    
    IF demo_client_id IS NOT NULL THEN
        INSERT INTO meal_plans (
            client_id,
            name,
            description,
            plan_type,
            start_date,
            target_calories,
            target_protein,
            target_carbs,
            target_fat,
            meals_per_day,
            meal_schedule,
            weekly_recipes,
            status
        ) VALUES (
            demo_client_id,
            'Performance Cutting Plan',
            'High-protein meal plan for lean muscle maintenance during cut',
            'cutting',
            CURRENT_DATE,
            2200,
            180,
            165,
            75,
            3,
            jsonb_build_object(
                'breakfast', jsonb_build_object('time', '07:00', 'calories', 600),
                'lunch', jsonb_build_object('time', '12:30', 'calories', 700),
                'dinner', jsonb_build_object('time', '18:00', 'calories', 900)
            ),
            jsonb_build_object(
                'monday', jsonb_build_object(
                    'breakfast', power_oats_id,
                    'lunch', power_bowl_id,
                    'dinner', salmon_dinner_id
                ),
                'tuesday', jsonb_build_object(
                    'breakfast', power_oats_id,
                    'lunch', power_bowl_id,
                    'dinner', salmon_dinner_id
                )
            ),
            'active'
        );
    END IF;
END $$;

-- ============================================================================
-- 5. DEMO PEDs PROTOCOL (PENDING MEDICAL APPROVAL)
-- ============================================================================

DO $$
DECLARE
    demo_client_id UUID;
    demo_doctor_id UUID;
BEGIN
    SELECT id INTO demo_client_id FROM client_profiles LIMIT 1;
    SELECT id INTO demo_doctor_id FROM medical_professionals WHERE full_name = 'Dr. Marcus Thompson';
    
    IF demo_client_id IS NOT NULL AND demo_doctor_id IS NOT NULL THEN
        INSERT INTO peds_protocols (
            client_id,
            medical_supervisor_id,
            protocol_name,
            protocol_type,
            cycle_phase,
            total_planned_weeks,
            compounds,
            cycle_schedule,
            required_bloodwork,
            potential_side_effects,
            status,
            medical_approval,
            client_consent
        ) VALUES (
            demo_client_id,
            demo_doctor_id,
            'Beginner Test Cycle',
            'peds_cycle',
            'prep',
            16,
            jsonb_build_array(
                jsonb_build_object(
                    'name', 'Testosterone Cypionate',
                    'dosage_mg', 300,
                    'frequency', 'weekly',
                    'injection_sites', jsonb_build_array('glute', 'quad'),
                    'start_week', 1,
                    'end_week', 12
                ),
                jsonb_build_object(
                    'name', 'Anastrozole',
                    'dosage_mg', 0.5,
                    'frequency', 'every_other_day',
                    'start_week', 3,
                    'end_week', 14
                )
            ),
            jsonb_build_object(
                'weeks_1_12', jsonb_build_object('test_cyp', '300mg/week'),
                'weeks_3_14', jsonb_build_object('anastrozole', '0.5mg EOD'),
                'weeks_13_16', jsonb_build_object('clomid', '50mg/day', 'nolva', '20mg/day')
            ),
            ARRAY['Complete Blood Count', 'Comprehensive Metabolic Panel', 'Lipid Panel', 'Hormone Panel'],
            ARRAY['Acne', 'Hair Loss', 'Gynecomastia', 'Mood Changes', 'Blood Pressure Changes'],
            'pending',
            false,
            true
        );
    END IF;
END $$;

-- ============================================================================
-- 6. DEMO BIOREGULATORS PROTOCOL
-- ============================================================================

DO $$
DECLARE
    demo_client_id UUID;
BEGIN
    SELECT id INTO demo_client_id FROM client_profiles LIMIT 1;
    
    IF demo_client_id IS NOT NULL THEN
        INSERT INTO bioregulators_protocols (
            client_id,
            protocol_name,
            bioregulator_type,
            peptide_category,
            regulatory_status,
            dosage_amount,
            dosage_unit,
            administration_route,
            frequency_per_day,
            cycle_length_weeks,
            start_date,
            primary_targets,
            expected_benefits,
            mechanism_of_action,
            evidence_level,
            research_citations,
            status,
            medical_approval
        ) VALUES (
            demo_client_id,
            'Longevity Optimization Protocol',
            'Epitalon',
            'geroprotective',
            'research',
            10.0,
            'mg',
            'injection',
            1,
            4,
            CURRENT_DATE,
            ARRAY['telomere_length', 'cellular_repair', 'sleep_quality'],
            ARRAY['Extended lifespan', 'Improved sleep patterns', 'Enhanced recovery', 'Cellular rejuvenation'],
            'Epitalon acts as a telomerase activator, potentially extending cellular lifespan by maintaining telomere length and promoting cellular repair mechanisms.',
            'moderate',
            ARRAY[
                'Khavinson VKh, et al. Peptides. 2003;24(11):1649-1652.',
                'Anisimov VN, et al. Ann N Y Acad Sci. 2003;994:133-139.',
                'Kossoy G, et al. Neuro Endocrinol Lett. 2006;27(1-2):190-194.'
            ],
            'active',
            true
        );
    END IF;
END $$;

-- ============================================================================
-- 7. DEMO MEDICAL CONSULTATION
-- ============================================================================

DO $$
DECLARE
    demo_client_id UUID;
    demo_doctor_id UUID;
    demo_protocol_id UUID;
BEGIN
    SELECT id INTO demo_client_id FROM client_profiles LIMIT 1;
    SELECT id INTO demo_doctor_id FROM medical_professionals WHERE full_name = 'Dr. Marcus Thompson';
    SELECT id INTO demo_protocol_id FROM peds_protocols WHERE protocol_name = 'Beginner Test Cycle';
    
    IF demo_client_id IS NOT NULL AND demo_doctor_id IS NOT NULL THEN
        INSERT INTO medical_consultations (
            client_id,
            medical_professional_id,
            consultation_type,
            consultation_date,
            duration_minutes,
            peds_protocol_id,
            chief_complaint,
            medical_recommendations,
            required_monitoring,
            follow_up_required,
            follow_up_weeks,
            status
        ) VALUES (
            demo_client_id,
            demo_doctor_id,
            'initial',
            CURRENT_TIMESTAMP + INTERVAL '3 days',
            60,
            demo_protocol_id,
            'Request for performance enhancement protocol evaluation',
            'Comprehensive evaluation required before PED protocol approval. Blood work analysis needed.',
            ARRAY['Monthly blood panels', 'Blood pressure monitoring', 'Liver function tests'],
            true,
            4,
            'scheduled'
        );
    END IF;
END $$;

-- ============================================================================
-- DEMO DATA COMPLETE
-- ============================================================================

-- Summary of demo data created:
-- ✅ 3 Medical Professionals (Sports Medicine, Anti-Aging, Endocrinology)
-- ✅ 8 Additional Food Items (Proteins, Carbs, Fats)
-- ✅ 3 Demo Recipes (Post-Workout Bowl, Salmon Dinner, Power Oats)
-- ✅ 1 Active Meal Plan (Performance Cutting Plan)
-- ✅ 1 Pending PEDs Protocol (awaiting medical approval)
-- ✅ 1 Active Bioregulators Protocol (Epitalon)
-- ✅ 1 Scheduled Medical Consultation

-- This provides realistic demo data for testing all four new systems:
-- 1. PEDs Protocol Dashboard - Shows pending protocol and scheduled consultation
-- 2. Food Database - Expanded food items and recipes
-- 3. Bioregulators Protocol - Active longevity protocol
-- 4. Medical Oversight - Doctor directory and scheduled consultation