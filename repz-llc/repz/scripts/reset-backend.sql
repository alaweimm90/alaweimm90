-- ============================================================================
-- REPZ BACKEND RESET - CLEAN SLATE
-- Preserves: subscription_tiers, pricing_plans, tier_features, tiers
-- Drops everything else and rebuilds from scratch
-- ============================================================================

-- ============================================================================
-- PHASE 1: BACKUP STRIPE/TIER DATA (into temp tables)
-- ============================================================================

-- Backup subscription_tiers
CREATE TEMP TABLE _backup_subscription_tiers AS SELECT * FROM subscription_tiers;

-- Backup pricing_plans if it has Stripe price IDs
CREATE TEMP TABLE _backup_pricing_plans AS SELECT * FROM pricing_plans;

-- Backup tiers
CREATE TEMP TABLE _backup_tiers AS SELECT * FROM tiers;

-- Backup tier_features
CREATE TEMP TABLE _backup_tier_features AS SELECT * FROM tier_features;

-- ============================================================================
-- PHASE 2: DROP ALL TABLES (except auth schema)
-- ============================================================================

-- Drop all tables in public schema
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename NOT IN ('schema_migrations')
    ) LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;

-- Drop all functions in public schema
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT proname, oidvectortypes(proargtypes) as args
        FROM pg_proc
        INNER JOIN pg_namespace ns ON (pg_proc.pronamespace = ns.oid)
        WHERE ns.nspname = 'public'
    ) LOOP
        BEGIN
            EXECUTE 'DROP FUNCTION IF EXISTS public.' || quote_ident(r.proname) || '(' || r.args || ') CASCADE';
        EXCEPTION WHEN OTHERS THEN
            NULL; -- Ignore errors
        END;
    END LOOP;
END $$;

-- ============================================================================
-- PHASE 3: CREATE FRESH SCHEMA
-- ============================================================================

-- 1. PROFILES (core user data)
CREATE TABLE profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE,
    full_name TEXT,
    avatar_url TEXT,
    phone TEXT,
    date_of_birth DATE,
    gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
    timezone TEXT DEFAULT 'America/Los_Angeles',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. USER ROLES
CREATE TABLE user_roles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'coach', 'client')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, role)
);

-- 3. SUBSCRIPTION TIERS (restore from backup)
CREATE TABLE subscription_tiers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT,
    price_cents INTEGER NOT NULL,
    billing_period TEXT DEFAULT 'monthly' CHECK (billing_period IN ('monthly', 'quarterly', 'annual')),
    stripe_price_id TEXT,
    features JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    is_popular BOOLEAN DEFAULT false,
    is_limited BOOLEAN DEFAULT false,
    max_clients INTEGER,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. COACH PROFILES
CREATE TABLE coach_profiles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    auth_user_id UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    coach_name TEXT NOT NULL,
    bio TEXT,
    specializations TEXT[],
    certifications TEXT[],
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    max_clients INTEGER DEFAULT 50,
    current_client_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. CLIENT PROFILES
CREATE TABLE client_profiles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    auth_user_id UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    client_name TEXT NOT NULL,
    subscription_tier_id UUID REFERENCES subscription_tiers(id),
    subscription_tier TEXT, -- Legacy field
    assigned_coach_id UUID REFERENCES coach_profiles(id),
    onboarding_completed BOOLEAN DEFAULT false,
    onboarding_completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'paused', 'cancelled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. SUBSCRIPTIONS (Stripe integration)
CREATE TABLE subscriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    client_profile_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    tier_id UUID REFERENCES subscription_tiers(id),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT UNIQUE,
    stripe_price_id TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('trialing', 'active', 'past_due', 'canceled', 'unpaid', 'paused')),
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    cancel_at_period_end BOOLEAN DEFAULT false,
    canceled_at TIMESTAMPTZ,
    trial_start TIMESTAMPTZ,
    trial_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. INTAKE FORM SUBMISSIONS
CREATE TABLE intake_form_submissions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    client_profile_id UUID REFERENCES client_profiles(id) ON DELETE SET NULL,

    -- Form sections (JSONB for flexibility)
    personal_info JSONB DEFAULT '{}'::jsonb,
    fitness_assessment JSONB DEFAULT '{}'::jsonb,
    health_history JSONB DEFAULT '{}'::jsonb,
    training_experience JSONB DEFAULT '{}'::jsonb,
    goals JSONB DEFAULT '{}'::jsonb,
    nutrition JSONB DEFAULT '{}'::jsonb,
    lifestyle JSONB DEFAULT '{}'::jsonb,
    schedule JSONB DEFAULT '{}'::jsonb,
    consent JSONB DEFAULT '{}'::jsonb,

    -- Status
    status TEXT DEFAULT 'submitted' CHECK (status IN ('draft', 'submitted', 'under_review', 'approved', 'needs_info', 'rejected')),
    assigned_coach_id UUID REFERENCES coach_profiles(id),
    assigned_tier_id UUID REFERENCES subscription_tiers(id),

    -- Notes
    coach_notes TEXT,
    admin_notes TEXT,

    -- Timestamps
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 8. INTAKE FORM DRAFTS (auto-save)
CREATE TABLE intake_form_drafts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id TEXT,
    form_data JSONB DEFAULT '{}'::jsonb,
    current_step INTEGER DEFAULT 1,
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days'),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 9. CLIENT ONBOARDING
CREATE TABLE client_onboarding (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id UUID UNIQUE REFERENCES client_profiles(id) ON DELETE CASCADE,
    intake_submission_id UUID REFERENCES intake_form_submissions(id),

    -- Steps
    intake_completed BOOLEAN DEFAULT false,
    intake_completed_at TIMESTAMPTZ,
    payment_completed BOOLEAN DEFAULT false,
    payment_completed_at TIMESTAMPTZ,
    coach_assigned BOOLEAN DEFAULT false,
    coach_assigned_at TIMESTAMPTZ,
    welcome_sent BOOLEAN DEFAULT false,
    welcome_sent_at TIMESTAMPTZ,
    consultation_scheduled BOOLEAN DEFAULT false,
    consultation_date TIMESTAMPTZ,
    program_created BOOLEAN DEFAULT false,
    program_created_at TIMESTAMPTZ,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'paused')),
    progress_percent INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 10. COACH-CLIENT ASSIGNMENTS
CREATE TABLE coach_client_assignments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    coach_id UUID REFERENCES coach_profiles(id) ON DELETE CASCADE,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    is_primary BOOLEAN DEFAULT true,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'paused', 'ended')),
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    assigned_by UUID REFERENCES auth.users(id),
    ended_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(coach_id, client_id)
);

-- 11. PROGRAMS
CREATE TABLE programs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    coach_id UUID REFERENCES coach_profiles(id),
    name TEXT NOT NULL,
    description TEXT,
    program_type TEXT CHECK (program_type IN ('strength', 'hypertrophy', 'endurance', 'weight_loss', 'general_fitness', 'sport_specific', 'rehabilitation')),
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'completed', 'archived')),
    start_date DATE,
    end_date DATE,
    program_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 12. WEEKLY CHECKINS
CREATE TABLE weekly_checkins (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    coach_id UUID REFERENCES coach_profiles(id),
    week_start DATE NOT NULL,

    -- Metrics
    weight_lbs DECIMAL(5,1),
    body_fat_percent DECIMAL(4,1),
    sleep_quality INTEGER CHECK (sleep_quality BETWEEN 1 AND 10),
    energy_level INTEGER CHECK (energy_level BETWEEN 1 AND 10),
    stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
    adherence_training INTEGER CHECK (adherence_training BETWEEN 0 AND 100),
    adherence_nutrition INTEGER CHECK (adherence_nutrition BETWEEN 0 AND 100),

    -- Notes
    wins TEXT,
    challenges TEXT,
    questions TEXT,
    coach_feedback TEXT,

    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'reviewed')),
    submitted_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 13. MESSAGES
CREATE TABLE messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sender_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    recipient_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    thread_id UUID,
    content TEXT NOT NULL,
    message_type TEXT DEFAULT 'text' CHECK (message_type IN ('text', 'image', 'file', 'system')),
    is_read BOOLEAN DEFAULT false,
    read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 14. ACTIVITY LOG (audit trail)
CREATE TABLE activity_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id UUID,
    metadata JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 15. PROGRESS PHOTOS
CREATE TABLE progress_photos (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    photo_url TEXT NOT NULL,
    photo_type TEXT CHECK (photo_type IN ('front', 'side', 'back', 'other')),
    taken_at DATE DEFAULT CURRENT_DATE,
    notes TEXT,
    is_private BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PHASE 4: RESTORE TIER DATA
-- ============================================================================

-- Restore subscription_tiers (map display_name to name as well)
INSERT INTO subscription_tiers (id, name, display_name, description, price_cents, stripe_price_id, features, is_popular, is_limited, max_clients, created_at)
SELECT
    id,
    LOWER(REGEXP_REPLACE(display_name, '[^a-zA-Z0-9]', '_', 'g')), -- Generate name from display_name
    display_name,
    description,
    price_cents,
    stripe_price_id,
    features,
    is_popular,
    is_limited,
    max_clients,
    created_at
FROM _backup_subscription_tiers
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- PHASE 5: INDEXES
-- ============================================================================

CREATE INDEX idx_profiles_email ON profiles(email);
CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role);
CREATE INDEX idx_client_profiles_user ON client_profiles(auth_user_id);
CREATE INDEX idx_client_profiles_coach ON client_profiles(assigned_coach_id);
CREATE INDEX idx_coach_profiles_user ON coach_profiles(auth_user_id);
CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_stripe ON subscriptions(stripe_subscription_id);
CREATE INDEX idx_intake_submissions_user ON intake_form_submissions(user_id);
CREATE INDEX idx_intake_submissions_status ON intake_form_submissions(status);
CREATE INDEX idx_coach_assignments_coach ON coach_client_assignments(coach_id);
CREATE INDEX idx_coach_assignments_client ON coach_client_assignments(client_id);
CREATE INDEX idx_programs_client ON programs(client_id);
CREATE INDEX idx_checkins_client ON weekly_checkins(client_id);
CREATE INDEX idx_messages_sender ON messages(sender_id);
CREATE INDEX idx_messages_recipient ON messages(recipient_id);
CREATE INDEX idx_activity_log_user ON activity_log(user_id);
CREATE INDEX idx_activity_log_created ON activity_log(created_at);

-- ============================================================================
-- PHASE 6: ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_tiers ENABLE ROW LEVEL SECURITY;
ALTER TABLE coach_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE intake_form_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE intake_form_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_onboarding ENABLE ROW LEVEL SECURITY;
ALTER TABLE coach_client_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_checkins ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE progress_photos ENABLE ROW LEVEL SECURITY;

-- RLS Policies

-- Profiles: users see their own
CREATE POLICY profiles_own ON profiles FOR ALL USING (auth.uid() = id);
CREATE POLICY profiles_admin ON profiles FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- User roles: admins manage, users see their own
CREATE POLICY roles_own ON user_roles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY roles_admin ON user_roles FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Subscription tiers: public read
CREATE POLICY tiers_public ON subscription_tiers FOR SELECT USING (true);
CREATE POLICY tiers_admin ON subscription_tiers FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Coach profiles: public read for active coaches
CREATE POLICY coaches_public ON coach_profiles FOR SELECT USING (is_active = true);
CREATE POLICY coaches_own ON coach_profiles FOR ALL USING (auth.uid() = auth_user_id);
CREATE POLICY coaches_admin ON coach_profiles FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Client profiles: own + assigned coach + admin
CREATE POLICY clients_own ON client_profiles FOR ALL USING (auth.uid() = auth_user_id);
CREATE POLICY clients_coach ON client_profiles FOR SELECT USING (
    EXISTS (SELECT 1 FROM coach_profiles WHERE auth_user_id = auth.uid() AND id = client_profiles.assigned_coach_id)
);
CREATE POLICY clients_admin ON client_profiles FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Subscriptions: own + admin
CREATE POLICY subs_own ON subscriptions FOR ALL USING (auth.uid() = user_id);
CREATE POLICY subs_admin ON subscriptions FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Intake forms: own + coach + admin
CREATE POLICY intake_own ON intake_form_submissions FOR ALL USING (auth.uid() = user_id);
CREATE POLICY intake_staff ON intake_form_submissions FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role IN ('admin', 'coach'))
);

-- Drafts: own only
CREATE POLICY drafts_own ON intake_form_drafts FOR ALL USING (auth.uid() = user_id);

-- Onboarding: client + coach + admin
CREATE POLICY onboarding_client ON client_onboarding FOR SELECT USING (
    EXISTS (SELECT 1 FROM client_profiles WHERE id = client_id AND auth_user_id = auth.uid())
);
CREATE POLICY onboarding_staff ON client_onboarding FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role IN ('admin', 'coach'))
);

-- Coach assignments: involved parties + admin
CREATE POLICY assignments_coach ON coach_client_assignments FOR ALL USING (
    EXISTS (SELECT 1 FROM coach_profiles WHERE id = coach_id AND auth_user_id = auth.uid())
);
CREATE POLICY assignments_admin ON coach_client_assignments FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Programs: client + coach + admin
CREATE POLICY programs_client ON programs FOR SELECT USING (
    EXISTS (SELECT 1 FROM client_profiles WHERE id = client_id AND auth_user_id = auth.uid())
);
CREATE POLICY programs_coach ON programs FOR ALL USING (
    EXISTS (SELECT 1 FROM coach_profiles WHERE id = coach_id AND auth_user_id = auth.uid())
);
CREATE POLICY programs_admin ON programs FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Checkins: client + coach + admin
CREATE POLICY checkins_client ON weekly_checkins FOR ALL USING (
    EXISTS (SELECT 1 FROM client_profiles WHERE id = client_id AND auth_user_id = auth.uid())
);
CREATE POLICY checkins_coach ON weekly_checkins FOR ALL USING (
    EXISTS (SELECT 1 FROM coach_profiles WHERE id = coach_id AND auth_user_id = auth.uid())
);
CREATE POLICY checkins_admin ON weekly_checkins FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Messages: sender or recipient
CREATE POLICY messages_own ON messages FOR ALL USING (
    auth.uid() = sender_id OR auth.uid() = recipient_id
);

-- Activity log: admin only
CREATE POLICY activity_admin ON activity_log FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- Progress photos: client + coach + admin
CREATE POLICY photos_client ON progress_photos FOR ALL USING (
    EXISTS (SELECT 1 FROM client_profiles WHERE id = client_id AND auth_user_id = auth.uid())
);
CREATE POLICY photos_coach ON progress_photos FOR SELECT USING (
    EXISTS (
        SELECT 1 FROM client_profiles cp
        JOIN coach_profiles coach ON cp.assigned_coach_id = coach.id
        WHERE cp.id = client_id AND coach.auth_user_id = auth.uid()
    )
);
CREATE POLICY photos_admin ON progress_photos FOR ALL USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'admin')
);

-- ============================================================================
-- PHASE 7: FUNCTIONS
-- ============================================================================

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1))
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger for new user
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER update_profiles_timestamp BEFORE UPDATE ON profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_subscription_tiers_timestamp BEFORE UPDATE ON subscription_tiers FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_coach_profiles_timestamp BEFORE UPDATE ON coach_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_client_profiles_timestamp BEFORE UPDATE ON client_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_subscriptions_timestamp BEFORE UPDATE ON subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_intake_submissions_timestamp BEFORE UPDATE ON intake_form_submissions FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_intake_drafts_timestamp BEFORE UPDATE ON intake_form_drafts FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_client_onboarding_timestamp BEFORE UPDATE ON client_onboarding FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_coach_assignments_timestamp BEFORE UPDATE ON coach_client_assignments FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_programs_timestamp BEFORE UPDATE ON programs FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_checkins_timestamp BEFORE UPDATE ON weekly_checkins FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Onboarding progress calculator
CREATE OR REPLACE FUNCTION update_onboarding_progress()
RETURNS TRIGGER AS $$
DECLARE
    steps_completed INTEGER := 0;
    total_steps INTEGER := 6;
BEGIN
    IF NEW.intake_completed THEN steps_completed := steps_completed + 1; END IF;
    IF NEW.payment_completed THEN steps_completed := steps_completed + 1; END IF;
    IF NEW.coach_assigned THEN steps_completed := steps_completed + 1; END IF;
    IF NEW.welcome_sent THEN steps_completed := steps_completed + 1; END IF;
    IF NEW.consultation_scheduled THEN steps_completed := steps_completed + 1; END IF;
    IF NEW.program_created THEN steps_completed := steps_completed + 1; END IF;

    NEW.progress_percent := (steps_completed * 100) / total_steps;

    IF steps_completed = total_steps THEN
        NEW.status := 'completed';
    ELSIF steps_completed > 0 THEN
        NEW.status := 'in_progress';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER onboarding_progress_trigger
    BEFORE UPDATE ON client_onboarding
    FOR EACH ROW EXECUTE FUNCTION update_onboarding_progress();

-- Helper: Check if user has role
CREATE OR REPLACE FUNCTION has_role(check_role TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM user_roles
        WHERE user_id = auth.uid() AND role = check_role
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Helper: Get user's primary role
CREATE OR REPLACE FUNCTION get_user_role()
RETURNS TEXT AS $$
DECLARE
    user_role TEXT;
BEGIN
    SELECT role INTO user_role FROM user_roles WHERE user_id = auth.uid() LIMIT 1;
    RETURN COALESCE(user_role, 'client');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- PHASE 8: SEED ADMIN
-- ============================================================================

-- Create admin role for existing user
INSERT INTO user_roles (user_id, role)
SELECT id, 'admin' FROM auth.users WHERE email = 'meshal@berkeley.edu'
ON CONFLICT (user_id, role) DO NOTHING;

-- ============================================================================
-- DONE!
-- ============================================================================

SELECT 'Backend reset complete!' as status,
       (SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public') as tables_created;
