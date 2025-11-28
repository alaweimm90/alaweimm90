-- Migration: 001_init
-- {{PROJECT_NAME}} Initial Schema

BEGIN;

-- Create migrations tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Apply initial schema
\i schema.sql

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001_init');

COMMIT;
