-- ============================================================================
-- REPZ COMPLETE INTAKE SYSTEM MIGRATION
-- Date: 2025-12-05
-- Purpose: Complete intake form storage, client onboarding, and user management
-- ============================================================================

-- ============================================================================
-- 1. INTAKE FORM SUBMISSIONS TABLE
-- ============================================================================

-- Drop and recreate to ensure clean state
DROP TABLE IF EXISTS intake_form_submissions CASCADE;

CREATE TABLE intake_form_submissions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,

    -- Link to user (optional for non-portal clients)
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    client_profile_id UUID REFERENCES client_profiles(id) ON DELETE SET NULL,

    -- Submission metadata
    submission_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    form_version VARCHAR(20) DEFAULT '2.0',
    submission_source VARCHAR(50) DEFAULT 'web' CHECK (submission_source IN ('web', 'mobile', 'pdf', 'manual')),

    -- ========================================
    -- SECTION 1: PERSONAL INFORMATION
    -- ========================================
    personal_info JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "full_name": "string",
    --   "preferred_name": "string",
    --   "date_of_birth": "YYYY-MM-DD",
    --   "age": number,
    --   "gender": "male|female|non-binary|prefer_not_to_say",
    --   "email": "string",
    --   "phone": "string",
    --   "address": "string",
    --   "city": "string",
    --   "state": "string",
    --   "zip": "string",
    --   "emergency_contact": {
    --     "name": "string",
    --     "relationship": "string",
    --     "phone": "string",
    --     "alternate_phone": "string"
    --   }
    -- }

    -- ========================================
    -- SECTION 2: FITNESS ASSESSMENT
    -- ========================================
    fitness_assessment JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "current_weight_lbs": number,
    --   "goal_weight_lbs": number,
    --   "height_feet": number,
    --   "height_inches": number,
    --   "body_fat_percent": number,
    --   "goal_body_fat_percent": number,
    --   "bmi": number,
    --   "measurements": {
    --     "neck": number,
    --     "chest": number,
    --     "waist": number,
    --     "hips": number,
    --     "right_arm": number,
    --     "left_arm": number,
    --     "right_thigh": number,
    --     "left_thigh": number,
    --     "right_calf": number,
    --     "left_calf": number
    --   },
    --   "vitals": {
    --     "resting_heart_rate": number,
    --     "blood_pressure_systolic": number,
    --     "blood_pressure_diastolic": number
    --   }
    -- }

    -- ========================================
    -- SECTION 3: HEALTH HISTORY
    -- ========================================
    health_history JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "parq_responses": {
    --     "heart_condition": boolean,
    --     "chest_pain_activity": boolean,
    --     "chest_pain_rest": boolean,
    --     "dizziness": boolean,
    --     "bone_joint_problem": boolean,
    --     "blood_pressure_meds": boolean,
    --     "other_reason": boolean,
    --     "explanation": "string"
    --   },
    --   "medical_conditions": ["string"],
    --   "condition_details": "string",
    --   "current_medications": [
    --     {"name": "string", "dosage": "string", "frequency": "string", "purpose": "string"}
    --   ],
    --   "allergies": "string",
    --   "current_injuries": "string",
    --   "past_injuries": "string",
    --   "movements_to_avoid": "string",
    --   "physician": {
    --     "name": "string",
    --     "phone": "string",
    --     "last_physical_date": "YYYY-MM-DD"
    --   },
    --   "physician_clearance": boolean
    -- }

    -- ========================================
    -- SECTION 4: TRAINING EXPERIENCE
    -- ========================================
    training_experience JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "fitness_level": "beginner|intermediate|advanced|elite",
    --   "exercise_frequency": "none|1-2|3-4|5-6|daily",
    --   "training_duration": "less_than_6_months|6_months_to_1_year|1-3_years|3-5_years|5_plus_years",
    --   "training_background": "string",
    --   "strength_benchmarks": {
    --     "bench_press_current": number,
    --     "bench_press_goal": number,
    --     "squat_current": number,
    --     "squat_goal": number,
    --     "deadlift_current": number,
    --     "deadlift_goal": number,
    --     "overhead_press_current": number,
    --     "overhead_press_goal": number,
    --     "pullups_current": number,
    --     "pullups_goal": number,
    --     "pushups_current": number,
    --     "pushups_goal": number,
    --     "plank_seconds_current": number,
    --     "plank_seconds_goal": number,
    --     "mile_time_current": "string",
    --     "mile_time_goal": "string"
    --   },
    --   "training_location": "commercial_gym|home_gym|both|outdoors",
    --   "gym_name": "string",
    --   "home_equipment": ["string"]
    -- }

    -- ========================================
    -- SECTION 5: GOALS
    -- ========================================
    goals JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "primary_goal": "weight_loss|muscle_gain|strength|endurance|athletic_performance|body_recomp|general_fitness|sport_specific|rehabilitation|longevity",
    --   "secondary_goals": ["string"],
    --   "specific_targets": "string",
    --   "timeframe": "1-3_months|3-6_months|6-12_months|12_plus_months",
    --   "motivation": "string",
    --   "past_obstacles": "string",
    --   "accountability_level": "high|medium|low",
    --   "coaching_style": "strict|balanced|flexible"
    -- }

    -- ========================================
    -- SECTION 6: NUTRITION
    -- ========================================
    nutrition JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "meals_per_day": "1-2|3|4-5|6_plus",
    --   "current_diet": "none|balanced|high_protein|low_carb|vegetarian|vegan|paleo|mediterranean|other",
    --   "diet_other": "string",
    --   "dietary_restrictions": ["string"],
    --   "food_allergies": ["string"],
    --   "foods_disliked": "string",
    --   "typical_meals": {
    --     "breakfast": "string",
    --     "lunch": "string",
    --     "dinner": "string",
    --     "snacks": "string"
    --   },
    --   "hydration": {
    --     "daily_water_intake": "less_than_4|4-8|8-12|12_plus",
    --     "alcohol_consumption": "never|rarely|1-2_week|3-5_week|daily",
    --     "caffeine_intake": "none|1-2|3-4|5_plus",
    --     "eating_out_frequency": "rarely|1-2_week|3-5_week|daily"
    --   },
    --   "current_supplements": "string",
    --   "open_to_supplements": boolean,
    --   "willing_to_meal_prep": "yes|sometimes|no"
    -- }

    -- ========================================
    -- SECTION 7: LIFESTYLE
    -- ========================================
    lifestyle JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "occupation": "string",
    --   "work_schedule": "string",
    --   "occupational_activity": "sedentary|light|moderate|active|very_active",
    --   "daily_steps": number,
    --   "sleep": {
    --     "average_hours": "less_than_5|5-6|6-7|7-8|8-9|9_plus",
    --     "quality": "poor|fair|good|excellent",
    --     "bedtime": "string",
    --     "wake_time": "string",
    --     "issues": ["string"]
    --   },
    --   "stress_level": "low|moderate|high|very_high",
    --   "stress_sources": "string",
    --   "recovery_methods": ["string"],
    --   "smoking_status": "never|former|current",
    --   "recreational_drugs": boolean,
    --   "travel_frequency": "rarely|monthly|weekly"
    -- }

    -- ========================================
    -- SECTION 8: SCHEDULE
    -- ========================================
    schedule JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "preferred_training_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    --   "preferred_training_time": "early_morning|morning|midday|afternoon|evening|late_evening",
    --   "session_duration": "30-45|45-60|60-90|90_plus",
    --   "communication": {
    --     "preferred_method": "email|text|phone|app",
    --     "check_in_frequency": "daily|weekly|bi-weekly",
    --     "timezone": "string",
    --     "preferred_start_date": "YYYY-MM-DD"
    --   },
    --   "upcoming_events": "string"
    -- }

    -- ========================================
    -- SECTION 9: CONSENT
    -- ========================================
    consent JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "informed_consent": boolean,
    --   "liability_waiver": boolean,
    --   "privacy_policy": boolean,
    --   "photo_release_coaching": boolean,
    --   "photo_release_marketing": boolean,
    --   "medical_disclosure": boolean,
    --   "program_agreement": boolean,
    --   "signature": "string",
    --   "signature_date": "YYYY-MM-DD"
    -- }

    -- ========================================
    -- PROCESSING STATUS
    -- ========================================
    status VARCHAR(50) DEFAULT 'submitted' CHECK (status IN (
        'draft',
        'submitted',
        'under_review',
        'approved',
        'needs_info',
        'rejected',
        'onboarded'
    )),

    -- Coach assignment
    assigned_coach_id UUID REFERENCES coach_profiles(id) ON DELETE SET NULL,
    assigned_tier VARCHAR(50) CHECK (assigned_tier IN ('core', 'adaptive', 'performance', 'longevity')),

    -- Review notes
    coach_notes TEXT,
    admin_notes TEXT,

    -- Timestamps
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    onboarded_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 2. INTAKE FORM DRAFTS (Auto-save functionality)
-- ============================================================================

CREATE TABLE IF NOT EXISTS intake_form_drafts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,

    -- Session identifier (for non-logged-in users)
    session_id VARCHAR(255),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Draft data (all sections combined)
    form_data JSONB DEFAULT '{}'::jsonb,

    -- Current step
    current_step INTEGER DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'),

    -- Unique constraint
    UNIQUE(session_id),
    UNIQUE(user_id)
);

-- ============================================================================
-- 3. CLIENT ONBOARDING CHECKLIST
-- ============================================================================

CREATE TABLE IF NOT EXISTS client_onboarding (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE UNIQUE,
    intake_submission_id UUID REFERENCES intake_form_submissions(id) ON DELETE SET NULL,

    -- Onboarding steps
    intake_completed BOOLEAN DEFAULT FALSE,
    intake_completed_at TIMESTAMP WITH TIME ZONE,

    payment_completed BOOLEAN DEFAULT FALSE,
    payment_completed_at TIMESTAMP WITH TIME ZONE,
    stripe_subscription_id VARCHAR(255),

    coach_assigned BOOLEAN DEFAULT FALSE,
    coach_assigned_at TIMESTAMP WITH TIME ZONE,

    welcome_email_sent BOOLEAN DEFAULT FALSE,
    welcome_email_sent_at TIMESTAMP WITH TIME ZONE,

    initial_consultation_scheduled BOOLEAN DEFAULT FALSE,
    initial_consultation_date TIMESTAMP WITH TIME ZONE,

    program_created BOOLEAN DEFAULT FALSE,
    program_created_at TIMESTAMP WITH TIME ZONE,

    first_checkin_completed BOOLEAN DEFAULT FALSE,
    first_checkin_at TIMESTAMP WITH TIME ZONE,

    -- Overall status
    onboarding_status VARCHAR(50) DEFAULT 'pending' CHECK (onboarding_status IN (
        'pending',
        'intake_submitted',
        'payment_pending',
        'coach_assignment',
        'program_creation',
        'active',
        'paused',
        'cancelled'
    )),

    -- Progress percentage
    progress_percent INTEGER DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 4. SUBSCRIPTION MANAGEMENT
-- ============================================================================

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    client_profile_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,

    -- Stripe data
    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255) UNIQUE,
    stripe_price_id VARCHAR(255),

    -- Subscription details
    tier VARCHAR(50) NOT NULL CHECK (tier IN ('core', 'adaptive', 'performance', 'longevity')),
    billing_period VARCHAR(50) DEFAULT 'monthly' CHECK (billing_period IN ('monthly', 'quarterly', 'semiannual', 'annual')),

    -- Pricing
    amount_cents INTEGER,
    currency VARCHAR(3) DEFAULT 'usd',

    -- Status
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN (
        'trialing',
        'active',
        'past_due',
        'canceled',
        'unpaid',
        'paused'
    )),

    -- Dates
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    canceled_at TIMESTAMP WITH TIME ZONE,

    -- Trial
    trial_start TIMESTAMP WITH TIME ZONE,
    trial_end TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 5. COACH-CLIENT ASSIGNMENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS coach_client_assignments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    coach_id UUID REFERENCES coach_profiles(id) ON DELETE CASCADE,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,

    -- Assignment details
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assigned_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,

    -- Status
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'ended')),
    ended_at TIMESTAMP WITH TIME ZONE,
    end_reason TEXT,

    -- Primary coach flag
    is_primary BOOLEAN DEFAULT TRUE,

    -- Notes
    assignment_notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(coach_id, client_id)
);

-- ============================================================================
-- 6. ACTIVITY LOG (Audit Trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS activity_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,

    -- Actor
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    user_email VARCHAR(255),
    user_role VARCHAR(50),

    -- Action
    action_type VARCHAR(100) NOT NULL,
    action_description TEXT,

    -- Target
    target_type VARCHAR(100),
    target_id UUID,

    -- Context
    metadata JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 7. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_intake_submissions_user_id ON intake_form_submissions(user_id);
CREATE INDEX IF NOT EXISTS idx_intake_submissions_status ON intake_form_submissions(status);
CREATE INDEX IF NOT EXISTS idx_intake_submissions_date ON intake_form_submissions(submission_date);
CREATE INDEX IF NOT EXISTS idx_intake_drafts_session ON intake_form_drafts(session_id);
CREATE INDEX IF NOT EXISTS idx_intake_drafts_user ON intake_form_drafts(user_id);
CREATE INDEX IF NOT EXISTS idx_onboarding_client ON client_onboarding(client_id);
CREATE INDEX IF NOT EXISTS idx_onboarding_status ON client_onboarding(onboarding_status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe ON subscriptions(stripe_subscription_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_coach_assignments_coach ON coach_client_assignments(coach_id);
CREATE INDEX IF NOT EXISTS idx_coach_assignments_client ON coach_client_assignments(client_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_user ON activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_action ON activity_log(action_type);
CREATE INDEX IF NOT EXISTS idx_activity_log_date ON activity_log(created_at);

-- ============================================================================
-- 8. ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE intake_form_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE intake_form_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_onboarding ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE coach_client_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;

-- Intake submissions: Users can see their own, admins/coaches can see all
CREATE POLICY intake_submissions_user ON intake_form_submissions
    FOR ALL USING (
        auth.uid() = user_id
        OR EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role IN ('admin', 'coach'))
    );

-- Drafts: Users can manage their own
CREATE POLICY intake_drafts_user ON intake_form_drafts
    FOR ALL USING (auth.uid() = user_id);

-- Onboarding: Clients see their own, coaches/admins see all
CREATE POLICY onboarding_access ON client_onboarding
    FOR ALL USING (
        EXISTS (SELECT 1 FROM client_profiles WHERE id = client_id AND auth_user_id = auth.uid())
        OR EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role IN ('admin', 'coach'))
    );

-- Subscriptions: Users see their own, admins see all
CREATE POLICY subscriptions_user ON subscriptions
    FOR ALL USING (
        auth.uid() = user_id
        OR EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
    );

-- Coach assignments: Coaches see their assignments, admins see all
CREATE POLICY coach_assignments_access ON coach_client_assignments
    FOR ALL USING (
        EXISTS (SELECT 1 FROM coach_profiles WHERE id = coach_id AND auth_user_id = auth.uid())
        OR EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
    );

-- Activity log: Admins only
CREATE POLICY activity_log_admin ON activity_log
    FOR ALL USING (
        EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
    );

-- ============================================================================
-- 9. FUNCTIONS
-- ============================================================================

-- Function to update onboarding progress
CREATE OR REPLACE FUNCTION update_onboarding_progress()
RETURNS TRIGGER AS $$
DECLARE
    total_steps INTEGER := 7;
    completed_steps INTEGER := 0;
BEGIN
    IF NEW.intake_completed THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.payment_completed THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.coach_assigned THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.welcome_email_sent THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.initial_consultation_scheduled THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.program_created THEN completed_steps := completed_steps + 1; END IF;
    IF NEW.first_checkin_completed THEN completed_steps := completed_steps + 1; END IF;

    NEW.progress_percent := (completed_steps * 100) / total_steps;
    NEW.updated_at := NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER onboarding_progress_trigger
    BEFORE UPDATE ON client_onboarding
    FOR EACH ROW
    EXECUTE FUNCTION update_onboarding_progress();

-- Function to log activity
CREATE OR REPLACE FUNCTION log_activity(
    p_user_id UUID,
    p_action_type VARCHAR(100),
    p_action_description TEXT,
    p_target_type VARCHAR(100) DEFAULT NULL,
    p_target_id UUID DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    v_log_id UUID;
    v_user_email VARCHAR(255);
    v_user_role VARCHAR(50);
BEGIN
    -- Get user info
    SELECT email INTO v_user_email FROM auth.users WHERE id = p_user_id;
    SELECT role INTO v_user_role FROM user_roles WHERE user_id = p_user_id LIMIT 1;

    INSERT INTO activity_log (
        user_id, user_email, user_role,
        action_type, action_description,
        target_type, target_id, metadata
    ) VALUES (
        p_user_id, v_user_email, v_user_role,
        p_action_type, p_action_description,
        p_target_type, p_target_id, p_metadata
    ) RETURNING id INTO v_log_id;

    RETURN v_log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to create client from intake submission
CREATE OR REPLACE FUNCTION create_client_from_intake(
    p_submission_id UUID,
    p_tier VARCHAR(50) DEFAULT 'core'
)
RETURNS UUID AS $$
DECLARE
    v_submission intake_form_submissions%ROWTYPE;
    v_client_id UUID;
    v_personal_info JSONB;
BEGIN
    -- Get submission
    SELECT * INTO v_submission FROM intake_form_submissions WHERE id = p_submission_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Intake submission not found';
    END IF;

    v_personal_info := v_submission.personal_info;

    -- Create client profile
    INSERT INTO client_profiles (
        auth_user_id,
        client_name,
        subscription_tier,
        onboarding_completed
    ) VALUES (
        v_submission.user_id,
        v_personal_info->>'full_name',
        p_tier,
        FALSE
    ) RETURNING id INTO v_client_id;

    -- Update submission
    UPDATE intake_form_submissions
    SET client_profile_id = v_client_id,
        assigned_tier = p_tier,
        status = 'approved',
        updated_at = NOW()
    WHERE id = p_submission_id;

    -- Create onboarding record
    INSERT INTO client_onboarding (
        client_id,
        intake_submission_id,
        intake_completed,
        intake_completed_at,
        onboarding_status
    ) VALUES (
        v_client_id,
        p_submission_id,
        TRUE,
        NOW(),
        'intake_submitted'
    );

    RETURN v_client_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- 10. UPDATED_AT TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_intake_submissions_updated_at
    BEFORE UPDATE ON intake_form_submissions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_intake_drafts_updated_at
    BEFORE UPDATE ON intake_form_drafts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_coach_assignments_updated_at
    BEFORE UPDATE ON coach_client_assignments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

COMMENT ON TABLE intake_form_submissions IS 'Stores complete intake form submissions from clients';
COMMENT ON TABLE intake_form_drafts IS 'Auto-saved drafts of intake forms in progress';
COMMENT ON TABLE client_onboarding IS 'Tracks client onboarding progress and checklist';
COMMENT ON TABLE subscriptions IS 'Stripe subscription management';
COMMENT ON TABLE coach_client_assignments IS 'Maps coaches to their assigned clients';
COMMENT ON TABLE activity_log IS 'Audit trail for all system activities';
