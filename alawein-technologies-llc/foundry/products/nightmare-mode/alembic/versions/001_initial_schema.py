"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2025-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial schema for Nightmare Mode"""

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('email_verified', sa.Boolean, default=False, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),  # bcrypt hash
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('institution', sa.String(255), nullable=True),
        sa.Column('role', sa.String(50), nullable=False, default='user'),  # user, admin
        sa.Column('subscription_status', sa.String(50), nullable=False, default='free'),  # free, pro, enterprise
        sa.Column('subscription_ends_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('stripe_customer_id', sa.String(255), nullable=True, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),  # Soft delete
    )

    # Create index on deleted_at for soft delete queries
    op.create_index('idx_users_deleted_at', 'users', ['deleted_at'])

    # Papers table (uploaded research papers)
    op.create_table(
        'papers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('title', sa.Text, nullable=True),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('file_name', sa.String(255), nullable=True),
        sa.Column('file_size', sa.Integer, nullable=True),  # bytes
        sa.Column('file_type', sa.String(50), nullable=True),  # pdf, txt, docx
        sa.Column('status', sa.String(50), nullable=False, default='uploaded'),  # uploaded, processing, analyzed, failed
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )

    # Analyses table (results of Nightmare Mode attacks)
    op.create_table(
        'analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('difficulty', sa.String(50), nullable=False, default='medium'),  # easy, medium, hard, nightmare
        sa.Column('survival_score', sa.Integer, nullable=False),  # 0-100
        sa.Column('status', sa.String(50), nullable=False, default='pending'),  # pending, running, completed, failed
        sa.Column('attacks_by_dimension', postgresql.JSONB, nullable=True),  # {"statistical": {...}, "methodological": {...}}
        sa.Column('debate_summary', postgresql.JSONB, nullable=True),
        sa.Column('consensus', postgresql.JSONB, nullable=True),
        sa.Column('critical_issues', postgresql.JSONB, nullable=True),  # Array of critical findings
        sa.Column('error_message', sa.Text, nullable=True),  # If analysis failed
        sa.Column('processing_time_ms', sa.Integer, nullable=True),  # How long it took
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Create index for querying by status
    op.create_index('idx_analyses_status', 'analyses', ['status'])
    op.create_index('idx_analyses_created_at', 'analyses', ['created_at'])

    # Attack findings table (individual attacks from each dimension)
    op.create_table(
        'attack_findings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('dimension', sa.String(100), nullable=False),  # statistical, methodological, etc.
        sa.Column('category', sa.String(100), nullable=False),  # sample_size, p_hacking, etc.
        sa.Column('severity', sa.Integer, nullable=False),  # 0-10
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('quote', sa.Text, nullable=True),  # Quote from paper
        sa.Column('suggestion', sa.Text, nullable=True),  # How to fix
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Create composite index for querying findings by analysis and severity
    op.create_index('idx_findings_analysis_severity', 'attack_findings', ['analysis_id', 'severity'])

    # Usage tracking (for rate limiting and billing)
    op.create_table(
        'usage_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),  # analyze_paper, export_report
        sa.Column('tokens_used', sa.Integer, nullable=True),  # For API cost tracking
        sa.Column('cost_usd', sa.Numeric(10, 4), nullable=True),  # Estimated cost
        sa.Column('ip_address', sa.String(45), nullable=True),  # IPv4 or IPv6
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Index for rate limiting queries (user + timestamp)
    op.create_index('idx_usage_user_created', 'usage_logs', ['user_id', 'created_at'])

    # Feedback table (user feedback on analyses)
    op.create_table(
        'feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('rating', sa.Integer, nullable=True),  # 1-5 stars
        sa.Column('comment', sa.Text, nullable=True),
        sa.Column('helpful', sa.Boolean, nullable=True),  # Was the analysis helpful?
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Subscriptions table (Stripe webhooks populate this)
    op.create_table(
        'subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('stripe_subscription_id', sa.String(255), unique=True, nullable=False),
        sa.Column('stripe_price_id', sa.String(255), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),  # active, canceled, past_due, trialing
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cancel_at_period_end', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )

    # API keys table (for programmatic access)
    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('key_hash', sa.String(255), unique=True, nullable=False, index=True),  # SHA-256 hash
        sa.Column('key_prefix', sa.String(20), nullable=False),  # First 8 chars for display
        sa.Column('name', sa.String(255), nullable=True),  # User-friendly name
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade():
    """Drop all tables"""
    op.drop_table('api_keys')
    op.drop_table('subscriptions')
    op.drop_table('feedback')
    op.drop_table('usage_logs')
    op.drop_table('attack_findings')
    op.drop_table('analyses')
    op.drop_table('papers')
    op.drop_table('users')
