"""
Authentication & Authorization Models
Enterprise-grade security models for TalAI
"""

import uuid
from enum import Enum
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text, JSON,
    ForeignKey, Table, UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from passlib.context import CryptContext

from database import Base, TimestampMixin, TenantMixin

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Association tables for many-to-many relationships
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('granted_at', DateTime(timezone=True), default=datetime.now(UTC)),
    Column('granted_by', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True),
    UniqueConstraint('user_id', 'role_id', name='uq_user_roles')
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE')),
    Column('granted_at', DateTime(timezone=True), default=datetime.now(UTC)),
    UniqueConstraint('role_id', 'permission_id', name='uq_role_permissions')
)

team_members = Table(
    'team_members',
    Base.metadata,
    Column('team_id', UUID(as_uuid=True), ForeignKey('teams.id', ondelete='CASCADE')),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role', String(50), default='member'),
    Column('joined_at', DateTime(timezone=True), default=datetime.now(UTC)),
    UniqueConstraint('team_id', 'user_id', name='uq_team_members')
)

# Enums
class UserStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class AuthProvider(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    SAML = "saml"
    LDAP = "ldap"

class PermissionType(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"

# Models
class Organization(Base, TimestampMixin):
    """Organization model for multi-tenant enterprise"""
    __tablename__ = 'organizations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    domain = Column(String(255), nullable=True)

    # Settings
    settings = Column(JSONB, default=dict)
    features = Column(ARRAY(String), default=list)

    # Subscription & limits
    subscription_tier = Column(String(50), default='free')
    user_limit = Column(Integer, default=5)
    storage_limit_gb = Column(Integer, default=10)
    api_rate_limit = Column(Integer, default=1000)

    # SSO Configuration
    sso_enabled = Column(Boolean, default=False)
    sso_config = Column(JSONB, nullable=True)

    # Security
    enforce_2fa = Column(Boolean, default=False)
    ip_whitelist = Column(ARRAY(String), nullable=True)
    password_policy = Column(JSONB, default={
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digits": True,
        "require_special": True,
        "password_history": 5,
        "max_age_days": 90
    })

    # Status
    is_active = Column(Boolean, default=True)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    suspended_at = Column(DateTime(timezone=True), nullable=True)
    suspension_reason = Column(Text, nullable=True)

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    teams = relationship("Team", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="organization", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_organizations_slug', 'slug'),
        CheckConstraint('user_limit > 0', name='check_user_limit_positive'),
    )

class User(Base, TimestampMixin):
    """User model with comprehensive security features"""
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)

    # Basic info
    email = Column(String(255), nullable=False, index=True)
    username = Column(String(100), nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    timezone = Column(String(50), default='UTC')
    locale = Column(String(10), default='en')

    # Authentication
    password_hash = Column(String(255), nullable=True)
    auth_provider = Column(String(50), default=AuthProvider.LOCAL)
    external_id = Column(String(255), nullable=True)  # ID from OAuth provider

    # Security
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255), nullable=True)
    backup_codes = Column(ARRAY(String), nullable=True)

    # Status
    status = Column(String(20), default=UserStatus.PENDING)
    email_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)

    # Access control
    is_org_admin = Column(Boolean, default=False)
    is_system_admin = Column(Boolean, default=False)

    # Session management
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)

    # Password management
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    password_reset_token = Column(String(255), nullable=True, unique=True)
    password_reset_expires = Column(DateTime(timezone=True), nullable=True)
    previous_passwords = Column(ARRAY(String), default=list)

    # Usage tracking
    api_calls_count = Column(Integer, default=0)
    storage_used_mb = Column(Integer, default=0)

    # Preferences
    preferences = Column(JSONB, default=dict)
    notification_settings = Column(JSONB, default={
        "email": True,
        "push": True,
        "sms": False,
        "digest_frequency": "daily"
    })

    # Relationships
    organization = relationship("Organization", back_populates="users")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    teams = relationship("Team", secondary=team_members, back_populates="members")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", foreign_keys="AuditLog.user_id", back_populates="user")

    __table_args__ = (
        UniqueConstraint('organization_id', 'email', name='uq_org_email'),
        UniqueConstraint('organization_id', 'username', name='uq_org_username'),
        Index('ix_users_organization_email', 'organization_id', 'email'),
        Index('ix_users_status', 'status'),
    )

    def set_password(self, password: str) -> None:
        """Hash and set user password"""
        self.password_hash = pwd_context.hash(password)
        self.password_changed_at = datetime.now(UTC)

        # Add to password history
        if not self.previous_passwords:
            self.previous_passwords = []
        self.previous_passwords.append(self.password_hash)
        # Keep only last 5 passwords
        self.previous_passwords = self.previous_passwords[-5:]

    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        if not self.password_hash:
            return False
        return pwd_context.verify(password, self.password_hash)

    def is_password_reused(self, password: str) -> bool:
        """Check if password was used before"""
        if not self.previous_passwords:
            return False
        return any(pwd_context.verify(password, old_hash) for old_hash in self.previous_passwords)

class Role(Base, TimestampMixin):
    """Role model for RBAC"""
    __tablename__ = 'roles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True)

    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_system_role = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)

    # Hierarchy
    parent_role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), nullable=True)

    # Constraints
    max_users = Column(Integer, nullable=True)
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    parent_role = relationship("Role", remote_side=[id])

    __table_args__ = (
        UniqueConstraint('organization_id', 'name', name='uq_org_role_name'),
        Index('ix_roles_organization', 'organization_id'),
    )

class Permission(Base, TimestampMixin):
    """Permission model for fine-grained access control"""
    __tablename__ = 'permissions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # Conditions (JSON for complex permission rules)
    conditions = Column(JSONB, nullable=True)

    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

    __table_args__ = (
        UniqueConstraint('resource', 'action', name='uq_resource_action'),
        Index('ix_permissions_resource', 'resource'),
    )

class Team(Base, TimestampMixin):
    """Team model for collaboration"""
    __tablename__ = 'teams'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)

    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    slug = Column(String(100), nullable=False)

    # Settings
    settings = Column(JSONB, default=dict)

    # Limits
    member_limit = Column(Integer, default=50)

    # Status
    is_active = Column(Boolean, default=True)

    # Relationships
    organization = relationship("Organization", back_populates="teams")
    members = relationship("User", secondary=team_members, back_populates="teams")

    __table_args__ = (
        UniqueConstraint('organization_id', 'slug', name='uq_org_team_slug'),
        Index('ix_teams_organization', 'organization_id'),
    )

class APIKey(Base, TimestampMixin):
    """API Key model for programmatic access"""
    __tablename__ = 'api_keys'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True)

    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(10), nullable=False)  # First few chars for identification

    # Permissions
    scopes = Column(ARRAY(String), default=list)

    # Rate limiting
    rate_limit = Column(Integer, nullable=True)
    rate_limit_window = Column(Integer, default=3600)  # seconds

    # Usage
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(String(45), nullable=True)
    usage_count = Column(Integer, default=0)

    # Security
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    allowed_ips = Column(ARRAY(String), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index('ix_api_keys_key_prefix', 'key_prefix'),
        Index('ix_api_keys_organization', 'organization_id'),
    )

class UserSession(Base, TimestampMixin):
    """User session tracking"""
    __tablename__ = 'user_sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)

    token_hash = Column(String(255), nullable=False, unique=True)
    refresh_token_hash = Column(String(255), nullable=True, unique=True)

    # Device info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    device_id = Column(String(100), nullable=True)

    # Session info
    expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_expires_at = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), default=datetime.now(UTC))

    # Security
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(String(255), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index('ix_user_sessions_user', 'user_id'),
        Index('ix_user_sessions_expires', 'expires_at'),
    )

class AuditLog(Base, TimestampMixin):
    """Comprehensive audit logging"""
    __tablename__ = 'audit_logs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='SET NULL'), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    # Event details
    event_type = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(100), nullable=True)
    action = Column(String(50), nullable=False)

    # Request info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    request_id = Column(String(100), nullable=True)

    # Change tracking
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="audit_logs")
    user = relationship("User", foreign_keys=[user_id], back_populates="audit_logs")

    __table_args__ = (
        Index('ix_audit_logs_organization', 'organization_id'),
        Index('ix_audit_logs_user', 'user_id'),
        Index('ix_audit_logs_created_at', 'created_at'),
        Index('ix_audit_logs_event_type', 'event_type'),
    )