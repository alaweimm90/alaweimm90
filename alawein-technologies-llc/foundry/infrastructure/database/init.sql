-- Database initialization script for CrazyIdeas platform
-- This script sets up the initial database schema and seed data

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS crazyideas;

-- Use the database
\c crazyideas;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS ghost_researcher;
CREATE SCHEMA IF NOT EXISTS scientific_tinder;
CREATE SCHEMA IF NOT EXISTS chaos_engine;
CREATE SCHEMA IF NOT EXISTS shared;

-- Set search path
SET search_path TO shared, public;

-- Create shared tables
-- Users table
CREATE TABLE IF NOT EXISTS shared.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url TEXT,
    bio TEXT,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    reset_token_expires TIMESTAMP,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS shared.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    token VARCHAR(500) NOT NULL UNIQUE,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API keys table
CREATE TABLE IF NOT EXISTS shared.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB DEFAULT '[]',
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS shared.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES shared.users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ghost Researcher schema
SET search_path TO ghost_researcher, public;

CREATE TABLE IF NOT EXISTS ghost_researcher.spooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    tags TEXT[],
    visibility VARCHAR(50) DEFAULT 'public',
    evidence_links TEXT[],
    credibility_score DECIMAL(3,2) DEFAULT 0.00,
    views_count INTEGER DEFAULT 0,
    likes_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    is_featured BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ghost_researcher.investigations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spook_id UUID NOT NULL REFERENCES ghost_researcher.spooks(id) ON DELETE CASCADE,
    investigator_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending',
    findings TEXT,
    evidence JSONB DEFAULT '[]',
    conclusion VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ghost_researcher.comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spook_id UUID NOT NULL REFERENCES ghost_researcher.spooks(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES ghost_researcher.comments(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    is_edited BOOLEAN DEFAULT false,
    likes_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scientific Tinder schema
SET search_path TO scientific_tinder, public;

CREATE TABLE IF NOT EXISTS scientific_tinder.ideas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    hypothesis TEXT NOT NULL,
    description TEXT,
    category VARCHAR(100),
    field_of_study VARCHAR(200),
    complexity_level INTEGER CHECK (complexity_level BETWEEN 1 AND 10),
    required_resources TEXT[],
    estimated_impact INTEGER CHECK (estimated_impact BETWEEN 1 AND 10),
    tags TEXT[],
    images TEXT[],
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scientific_tinder.swipes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    idea_id UUID NOT NULL REFERENCES scientific_tinder.ideas(id) ON DELETE CASCADE,
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('left', 'right', 'super')),
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, idea_id)
);

CREATE TABLE IF NOT EXISTS scientific_tinder.matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES scientific_tinder.ideas(id) ON DELETE CASCADE,
    user1_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    user2_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    compatibility_score DECIMAL(3,2),
    conversation_started BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(idea_id, user1_id, user2_id)
);

CREATE TABLE IF NOT EXISTS scientific_tinder.collaborations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID NOT NULL REFERENCES scientific_tinder.matches(id) ON DELETE CASCADE,
    project_name VARCHAR(200),
    description TEXT,
    status VARCHAR(50) DEFAULT 'proposed',
    progress_percentage INTEGER DEFAULT 0,
    milestones JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Chaos Engine schema
SET search_path TO chaos_engine, public;

CREATE TABLE IF NOT EXISTS chaos_engine.chaos_ideas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    chaos_level INTEGER CHECK (chaos_level BETWEEN 1 AND 100),
    category VARCHAR(100),
    ingredients TEXT[],
    expected_outcome TEXT,
    actual_outcome TEXT,
    side_effects TEXT[],
    danger_rating INTEGER CHECK (danger_rating BETWEEN 0 AND 10),
    fun_factor INTEGER CHECK (fun_factor BETWEEN 0 AND 10),
    tags TEXT[],
    status VARCHAR(50) DEFAULT 'concept',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chaos_engine.combinations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea1_id UUID NOT NULL REFERENCES chaos_engine.chaos_ideas(id) ON DELETE CASCADE,
    idea2_id UUID NOT NULL REFERENCES chaos_engine.chaos_ideas(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    result_title VARCHAR(500),
    result_description TEXT,
    chaos_multiplier DECIMAL(3,2) DEFAULT 1.00,
    stability_index DECIMAL(3,2),
    is_stable BOOLEAN DEFAULT false,
    warnings TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(idea1_id, idea2_id, user_id)
);

CREATE TABLE IF NOT EXISTS chaos_engine.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chaos_idea_id UUID NOT NULL REFERENCES chaos_engine.chaos_ideas(id) ON DELETE CASCADE,
    experimenter_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    hypothesis TEXT NOT NULL,
    procedure TEXT NOT NULL,
    observations TEXT,
    results TEXT,
    conclusion TEXT,
    safety_precautions TEXT[],
    equipment_used TEXT[],
    duration_minutes INTEGER,
    success_rate DECIMAL(3,2),
    replicable BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conducted_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chaos_engine.chaos_reactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chaos_idea_id UUID NOT NULL REFERENCES chaos_engine.chaos_ideas(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES shared.users(id) ON DELETE CASCADE,
    reaction_type VARCHAR(50) NOT NULL,
    intensity INTEGER CHECK (intensity BETWEEN 1 AND 10),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chaos_idea_id, user_id, reaction_type)
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON shared.users(email);
CREATE INDEX idx_users_username ON shared.users(username);
CREATE INDEX idx_sessions_token ON shared.sessions(token);
CREATE INDEX idx_sessions_user_id ON shared.sessions(user_id);

CREATE INDEX idx_spooks_user_id ON ghost_researcher.spooks(user_id);
CREATE INDEX idx_spooks_category ON ghost_researcher.spooks(category);
CREATE INDEX idx_spooks_created_at ON ghost_researcher.spooks(created_at DESC);
CREATE INDEX idx_spooks_tags ON ghost_researcher.spooks USING gin(tags);

CREATE INDEX idx_ideas_user_id ON scientific_tinder.ideas(user_id);
CREATE INDEX idx_ideas_category ON scientific_tinder.ideas(category);
CREATE INDEX idx_ideas_status ON scientific_tinder.ideas(status);
CREATE INDEX idx_swipes_user_idea ON scientific_tinder.swipes(user_id, idea_id);

CREATE INDEX idx_chaos_ideas_user_id ON chaos_engine.chaos_ideas(user_id);
CREATE INDEX idx_chaos_ideas_chaos_level ON chaos_engine.chaos_ideas(chaos_level);
CREATE INDEX idx_chaos_ideas_status ON chaos_engine.chaos_ideas(status);
CREATE INDEX idx_combinations_ideas ON chaos_engine.combinations(idea1_id, idea2_id);

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON shared.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_spooks_updated_at BEFORE UPDATE ON ghost_researcher.spooks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ideas_updated_at BEFORE UPDATE ON scientific_tinder.ideas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chaos_ideas_updated_at BEFORE UPDATE ON chaos_engine.chaos_ideas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW ghost_researcher.popular_spooks AS
SELECT s.*, u.username, u.avatar_url
FROM ghost_researcher.spooks s
JOIN shared.users u ON s.user_id = u.id
WHERE s.visibility = 'public'
ORDER BY s.views_count DESC, s.likes_count DESC;

CREATE VIEW scientific_tinder.active_ideas AS
SELECT i.*, u.username, u.avatar_url,
       COUNT(DISTINCT s.id) as swipe_count
FROM scientific_tinder.ideas i
JOIN shared.users u ON i.user_id = u.id
LEFT JOIN scientific_tinder.swipes s ON i.id = s.idea_id
WHERE i.status = 'active'
GROUP BY i.id, u.username, u.avatar_url;

CREATE VIEW chaos_engine.trending_chaos AS
SELECT c.*, u.username,
       COUNT(DISTINCT r.id) as reaction_count,
       AVG(r.intensity) as avg_intensity
FROM chaos_engine.chaos_ideas c
JOIN shared.users u ON c.user_id = u.id
LEFT JOIN chaos_engine.chaos_reactions r ON c.id = r.chaos_idea_id
GROUP BY c.id, u.username
ORDER BY reaction_count DESC, avg_intensity DESC;

-- Insert seed data for development
INSERT INTO shared.users (email, username, password_hash, first_name, last_name, is_active, is_verified)
VALUES
    ('admin@crazyideas.com', 'admin', '$2b$10$YourHashedPasswordHere', 'Admin', 'User', true, true),
    ('test@crazyideas.com', 'testuser', '$2b$10$YourHashedPasswordHere', 'Test', 'User', true, true),
    ('demo@crazyideas.com', 'demo', '$2b$10$YourHashedPasswordHere', 'Demo', 'User', true, true)
ON CONFLICT (email) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA shared TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ghost_researcher TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA scientific_tinder TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA chaos_engine TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA shared TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ghost_researcher TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA scientific_tinder TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA chaos_engine TO postgres;