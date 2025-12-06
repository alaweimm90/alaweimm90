-- REPZ Platform: Non-Portal Clients Table
-- Used for email-based intake form submissions

-- Create enum for payment status
CREATE TYPE payment_status AS ENUM ('pending', 'completed', 'failed', 'refunded');

-- Create enum for tier types
CREATE TYPE tier_type AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- Non-portal clients table (for intake form submissions)
CREATE TABLE IF NOT EXISTS non_portal_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    phone VARCHAR(50),

    -- Intake form data (JSON blob)
    intake_data JSONB NOT NULL DEFAULT '{}',

    -- Payment information
    payment_type tier_type DEFAULT 'core',
    payment_status payment_status DEFAULT 'pending',
    stripe_customer_id VARCHAR(255),
    stripe_session_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),

    -- Plan delivery
    plan_generated BOOLEAN DEFAULT FALSE,
    plan_sent_at TIMESTAMPTZ,
    plan_pdf_url TEXT,

    -- Coach notes
    coach_notes TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for email lookups
CREATE INDEX idx_non_portal_clients_email ON non_portal_clients(email);

-- Create index for payment status
CREATE INDEX idx_non_portal_clients_payment_status ON non_portal_clients(payment_status);

-- Create index for created_at (for sorting)
CREATE INDEX idx_non_portal_clients_created_at ON non_portal_clients(created_at DESC);

-- Enable Row Level Security
ALTER TABLE non_portal_clients ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access (for edge functions)
CREATE POLICY "Service role has full access to non_portal_clients"
    ON non_portal_clients
    FOR ALL
    USING (auth.role() = 'service_role');

-- Policy: Allow authenticated admins to view all clients
CREATE POLICY "Admins can view all non_portal_clients"
    ON non_portal_clients
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_roles.user_id = auth.uid()
            AND user_roles.role = 'admin'
        )
    );

-- Policy: Allow authenticated admins to update clients
CREATE POLICY "Admins can update non_portal_clients"
    ON non_portal_clients
    FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_roles.user_id = auth.uid()
            AND user_roles.role = 'admin'
        )
    );

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_non_portal_clients_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_non_portal_clients_updated_at
    BEFORE UPDATE ON non_portal_clients
    FOR EACH ROW
    EXECUTE FUNCTION update_non_portal_clients_updated_at();

-- User roles table (if not exists)
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, role)
);

-- Enable RLS on user_roles
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view their own roles
CREATE POLICY "Users can view own roles"
    ON user_roles
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid());

-- Create profiles table (if not exists)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255),
    full_name VARCHAR(255),
    avatar_url TEXT,
    phone VARCHAR(50),
    tier tier_type DEFAULT 'core',
    stripe_customer_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on profiles
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view their own profile
CREATE POLICY "Users can view own profile"
    ON profiles
    FOR SELECT
    TO authenticated
    USING (id = auth.uid());

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
    ON profiles
    FOR UPDATE
    TO authenticated
    USING (id = auth.uid());

-- Content table for tier-gated content
CREATE TABLE IF NOT EXISTS content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL CHECK (content_type IN ('article', 'video', 'guide', 'protocol')),
    required_tier tier_type DEFAULT 'core',
    is_published BOOLEAN DEFAULT FALSE,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on content
ALTER TABLE content ENABLE ROW LEVEL SECURITY;

-- Policy: Published content visible to users with required tier
CREATE POLICY "Users can view tier-appropriate published content"
    ON content
    FOR SELECT
    TO authenticated
    USING (
        is_published = TRUE
        AND required_tier <= (
            SELECT tier FROM profiles WHERE id = auth.uid()
        )
    );

-- Policy: Admins can manage all content
CREATE POLICY "Admins can manage all content"
    ON content
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM user_roles
            WHERE user_roles.user_id = auth.uid()
            AND user_roles.role = 'admin'
        )
    );

COMMENT ON TABLE non_portal_clients IS 'Stores intake form submissions from non-authenticated users';
COMMENT ON TABLE user_roles IS 'Stores user role assignments';
COMMENT ON TABLE profiles IS 'Stores user profile information';
COMMENT ON TABLE content IS 'Stores tier-gated content';
