"""
Multi-tenancy Architecture Service
Enterprise-grade tenant isolation and management
"""

import os
import uuid
import hashlib
import asyncio
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from contextlib import asynccontextmanager
import json

from fastapi import HTTPException, status, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import (
    select, and_, or_, func, text, create_engine,
    MetaData, Table, Column, String, event
)
from sqlalchemy.orm import selectinload
from sqlalchemy.schema import CreateSchema, DropSchema
from pydantic import BaseModel, EmailStr, Field, validator
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

from database import get_db, Base, DatabaseSettings
from auth_models import Organization, User, Role, Permission, Team
from billing_models import Subscription, SubscriptionPlan, SubscriptionTier

# Configuration
class TenancyConfig:
    # Tenant isolation strategies
    ISOLATION_STRATEGY = os.getenv("TALAI_TENANT_ISOLATION", "row_level")  # row_level, schema, database

    # Database pool settings per tenant
    TENANT_POOL_SIZE = 5
    TENANT_MAX_OVERFLOW = 5
    TENANT_POOL_TIMEOUT = 30

    # Resource limits by tier
    RESOURCE_LIMITS = {
        "free": {
            "max_connections": 5,
            "max_storage_gb": 5,
            "max_cpu_cores": 1,
            "max_memory_gb": 2,
            "max_api_rate": 100,  # per minute
            "max_concurrent_requests": 10
        },
        "starter": {
            "max_connections": 10,
            "max_storage_gb": 50,
            "max_cpu_cores": 2,
            "max_memory_gb": 4,
            "max_api_rate": 1000,
            "max_concurrent_requests": 50
        },
        "pro": {
            "max_connections": 25,
            "max_storage_gb": 500,
            "max_cpu_cores": 4,
            "max_memory_gb": 16,
            "max_api_rate": 5000,
            "max_concurrent_requests": 200
        },
        "team": {
            "max_connections": 50,
            "max_storage_gb": 2000,
            "max_cpu_cores": 8,
            "max_memory_gb": 32,
            "max_api_rate": 10000,
            "max_concurrent_requests": 500
        },
        "enterprise": {
            "max_connections": -1,  # unlimited
            "max_storage_gb": -1,
            "max_cpu_cores": -1,
            "max_memory_gb": -1,
            "max_api_rate": -1,
            "max_concurrent_requests": -1
        }
    }

    # Cache settings
    CACHE_TTL_SECONDS = 300
    TENANT_CACHE_PREFIX = "tenant:"

    # Data retention policies by tier
    RETENTION_POLICIES = {
        "free": {"audit_logs_days": 7, "analytics_days": 30},
        "starter": {"audit_logs_days": 30, "analytics_days": 90},
        "pro": {"audit_logs_days": 90, "analytics_days": 365},
        "team": {"audit_logs_days": 365, "analytics_days": 730},
        "enterprise": {"audit_logs_days": -1, "analytics_days": -1}  # unlimited
    }

# Tenant isolation strategies
class IsolationStrategy(str, Enum):
    ROW_LEVEL = "row_level"  # Shared tables with tenant_id column
    SCHEMA = "schema"  # Separate schema per tenant
    DATABASE = "database"  # Separate database per tenant

# Models
class TenantConfig(BaseModel):
    """Tenant configuration model"""
    organization_id: str
    isolation_strategy: IsolationStrategy
    database_url: Optional[str] = None
    schema_name: Optional[str] = None
    resource_limits: Dict[str, Any]
    features: List[str]
    settings: Dict[str, Any]

class TenantContext(BaseModel):
    """Runtime tenant context"""
    tenant_id: str
    organization_id: str
    organization_slug: str
    subscription_tier: str
    isolation_strategy: IsolationStrategy
    database_session: Optional[Any] = None
    schema_name: Optional[str] = None
    resource_quota: Dict[str, Any]
    cached_at: Optional[datetime] = None

class CreateTenantRequest(BaseModel):
    organization_name: str
    organization_slug: str
    admin_email: EmailStr
    admin_name: str
    subscription_plan_id: Optional[str] = None
    isolation_strategy: Optional[IsolationStrategy] = IsolationStrategy.ROW_LEVEL

class TenantResourceUsage(BaseModel):
    """Track tenant resource usage"""
    storage_used_gb: float
    storage_limit_gb: float
    api_calls_today: int
    api_calls_limit: int
    active_connections: int
    connection_limit: int
    active_users: int
    user_limit: int
    compute_hours_used: float
    compute_hours_limit: float

# Tenant Manager
class TenantManager:
    """Core tenant management and isolation"""

    def __init__(self):
        self._tenant_engines: Dict[str, Any] = {}
        self._tenant_sessions: Dict[str, Any] = {}
        self._redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize tenant manager"""
        # Initialize Redis for caching
        self._redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True
        )

    async def create_tenant(
        self,
        request: CreateTenantRequest,
        db: AsyncSession
    ) -> Organization:
        """Create a new tenant with full isolation"""

        # Create organization
        organization = Organization(
            name=request.organization_name,
            slug=request.organization_slug,
            settings={
                "isolation_strategy": request.isolation_strategy,
                "created_at": datetime.now(UTC).isoformat()
            },
            is_active=True,
            activated_at=datetime.now(UTC)
        )
        db.add(organization)
        await db.flush()

        # Setup tenant isolation based on strategy
        if request.isolation_strategy == IsolationStrategy.SCHEMA:
            await self._create_tenant_schema(organization, db)
        elif request.isolation_strategy == IsolationStrategy.DATABASE:
            await self._create_tenant_database(organization, db)
        else:
            # Row-level security setup
            await self._setup_row_level_security(organization, db)

        # Create admin user
        admin_user = User(
            organization_id=organization.id,
            email=request.admin_email,
            username=request.admin_email.split("@")[0],
            full_name=request.admin_name,
            status="active",
            email_verified=False,
            is_org_admin=True
        )
        admin_user.set_password(str(uuid.uuid4()))  # Temporary password
        db.add(admin_user)

        # Create default roles and permissions
        await self._create_default_roles(organization, db)

        # Setup subscription if plan provided
        if request.subscription_plan_id:
            plan_query = select(SubscriptionPlan).where(
                SubscriptionPlan.id == request.subscription_plan_id
            )
            plan_result = await db.execute(plan_query)
            plan = plan_result.scalar_one_or_none()

            if plan:
                subscription = Subscription(
                    organization_id=organization.id,
                    plan_id=plan.id,
                    billing_email=request.admin_email,
                    billing_name=organization.name,
                    start_date=datetime.now(UTC),
                    current_period_start=datetime.now(UTC),
                    current_period_end=datetime.now(UTC) + timedelta(days=30),
                    status='active',
                    is_trial=plan.trial_days > 0,
                    trial_end=datetime.now(UTC) + timedelta(days=plan.trial_days) if plan.trial_days > 0 else None
                )
                db.add(subscription)
                organization.subscription_tier = plan.tier

        await db.commit()

        # Initialize tenant resources
        await self._initialize_tenant_resources(organization)

        # Cache tenant configuration
        await self._cache_tenant_config(organization)

        return organization

    async def get_tenant_context(
        self,
        tenant_id: str,
        db: AsyncSession
    ) -> TenantContext:
        """Get tenant context with caching"""

        # Check cache first
        cache_key = f"{TenancyConfig.TENANT_CACHE_PREFIX}{tenant_id}"
        cached_data = await self._redis_client.get(cache_key) if self._redis_client else None

        if cached_data:
            context_data = json.loads(cached_data)
            context = TenantContext(**context_data)

            # Check if cache is still valid
            if context.cached_at and (
                datetime.now(UTC) - datetime.fromisoformat(context.cached_at)
            ).total_seconds() < TenancyConfig.CACHE_TTL_SECONDS:
                return context

        # Load from database
        org_query = select(Organization).options(
            selectinload(Organization.subscriptions)
        ).where(Organization.id == tenant_id)
        org_result = await db.execute(org_query)
        organization = org_result.scalar_one_or_none()

        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )

        # Get active subscription
        active_subscription = next(
            (s for s in organization.subscriptions if s.status in ['active', 'trialing']),
            None
        )

        # Build context
        context = TenantContext(
            tenant_id=str(organization.id),
            organization_id=str(organization.id),
            organization_slug=organization.slug,
            subscription_tier=active_subscription.plan.tier if active_subscription else "free",
            isolation_strategy=organization.settings.get("isolation_strategy", IsolationStrategy.ROW_LEVEL),
            schema_name=organization.settings.get("schema_name"),
            resource_quota=TenancyConfig.RESOURCE_LIMITS.get(
                active_subscription.plan.tier if active_subscription else "free",
                TenancyConfig.RESOURCE_LIMITS["free"]
            ),
            cached_at=datetime.now(UTC)
        )

        # Cache context
        if self._redis_client:
            await self._redis_client.setex(
                cache_key,
                TenancyConfig.CACHE_TTL_SECONDS,
                context.json()
            )

        return context

    async def get_tenant_session(
        self,
        tenant_context: TenantContext
    ) -> AsyncSession:
        """Get database session for tenant based on isolation strategy"""

        if tenant_context.isolation_strategy == IsolationStrategy.DATABASE:
            # Use tenant-specific database
            return await self._get_tenant_database_session(tenant_context)
        elif tenant_context.isolation_strategy == IsolationStrategy.SCHEMA:
            # Use tenant-specific schema
            return await self._get_tenant_schema_session(tenant_context)
        else:
            # Use shared database with row-level security
            session = await get_db()
            # Set tenant context for row-level security
            await self._set_row_level_context(session, tenant_context)
            return session

    async def check_resource_limits(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        requested_amount: int = 1
    ) -> bool:
        """Check if tenant has available resources"""

        if tenant_context.subscription_tier == "enterprise":
            return True  # Unlimited resources

        limits = tenant_context.resource_quota
        limit_key = f"max_{resource_type}"

        if limit_key not in limits:
            return True  # No limit defined

        limit = limits[limit_key]
        if limit == -1:
            return True  # Unlimited

        # Get current usage from cache or database
        usage_key = f"{TenancyConfig.TENANT_CACHE_PREFIX}usage:{tenant_context.tenant_id}:{resource_type}"
        current_usage = await self._redis_client.get(usage_key) if self._redis_client else 0
        current_usage = int(current_usage) if current_usage else 0

        return (current_usage + requested_amount) <= limit

    async def increment_resource_usage(
        self,
        tenant_context: TenantContext,
        resource_type: str,
        amount: int = 1
    ):
        """Increment resource usage counter"""

        if not self._redis_client:
            return

        usage_key = f"{TenancyConfig.TENANT_CACHE_PREFIX}usage:{tenant_context.tenant_id}:{resource_type}"

        # Increment with TTL reset
        await self._redis_client.incrby(usage_key, amount)
        await self._redis_client.expire(usage_key, 3600)  # 1 hour TTL

    async def get_tenant_resource_usage(
        self,
        organization: Organization,
        db: AsyncSession
    ) -> TenantResourceUsage:
        """Get current resource usage for tenant"""

        # Get storage usage
        storage_query = text("""
            SELECT
                pg_database_size(current_database()) / 1024 / 1024 / 1024 as size_gb
        """)
        storage_result = await db.execute(storage_query)
        storage_gb = storage_result.scalar() or 0

        # Get active users count
        users_query = select(func.count(User.id)).where(
            and_(
                User.organization_id == organization.id,
                User.status == "active"
            )
        )
        users_result = await db.execute(users_query)
        active_users = users_result.scalar() or 0

        # Get API calls from cache
        api_calls_key = f"{TenancyConfig.TENANT_CACHE_PREFIX}usage:{organization.id}:api_calls"
        api_calls = await self._redis_client.get(api_calls_key) if self._redis_client else 0
        api_calls = int(api_calls) if api_calls else 0

        # Get subscription tier limits
        subscription_tier = organization.subscription_tier or "free"
        limits = TenancyConfig.RESOURCE_LIMITS[subscription_tier]

        return TenantResourceUsage(
            storage_used_gb=float(storage_gb),
            storage_limit_gb=limits["max_storage_gb"] if limits["max_storage_gb"] != -1 else float('inf'),
            api_calls_today=api_calls,
            api_calls_limit=limits["max_api_rate"] * 60 * 24 if limits["max_api_rate"] != -1 else -1,
            active_connections=0,  # Would need connection pool monitoring
            connection_limit=limits["max_connections"],
            active_users=active_users,
            user_limit=organization.user_limit,
            compute_hours_used=0,  # Would need compute tracking
            compute_hours_limit=100  # Default
        )

    async def migrate_tenant_data(
        self,
        organization: Organization,
        from_strategy: IsolationStrategy,
        to_strategy: IsolationStrategy,
        db: AsyncSession
    ):
        """Migrate tenant data between isolation strategies"""

        if from_strategy == to_strategy:
            return

        # Export data
        exported_data = await self._export_tenant_data(organization, from_strategy, db)

        # Create new isolation
        if to_strategy == IsolationStrategy.SCHEMA:
            await self._create_tenant_schema(organization, db)
        elif to_strategy == IsolationStrategy.DATABASE:
            await self._create_tenant_database(organization, db)

        # Import data
        await self._import_tenant_data(organization, to_strategy, exported_data, db)

        # Update organization settings
        organization.settings["isolation_strategy"] = to_strategy
        await db.commit()

        # Clear cache
        await self._invalidate_tenant_cache(organization)

    async def delete_tenant(
        self,
        organization: Organization,
        hard_delete: bool,
        db: AsyncSession
    ):
        """Delete or archive tenant data"""

        if hard_delete:
            # Complete deletion
            isolation_strategy = organization.settings.get("isolation_strategy")

            if isolation_strategy == IsolationStrategy.SCHEMA:
                await self._drop_tenant_schema(organization, db)
            elif isolation_strategy == IsolationStrategy.DATABASE:
                await self._drop_tenant_database(organization)

            # Delete organization and cascade to related records
            await db.delete(organization)
        else:
            # Soft delete/archive
            organization.is_active = False
            organization.suspended_at = datetime.now(UTC)
            organization.suspension_reason = "Account deleted by user"

            # Deactivate all users
            users_query = update(User).where(
                User.organization_id == organization.id
            ).values(status="deleted")
            await db.execute(users_query)

        await db.commit()

        # Clear cache
        await self._invalidate_tenant_cache(organization)

    # Private methods for isolation strategies
    async def _create_tenant_schema(
        self,
        organization: Organization,
        db: AsyncSession
    ):
        """Create isolated schema for tenant"""

        schema_name = f"tenant_{organization.slug.replace('-', '_')}"

        # Create schema
        await db.execute(CreateSchema(schema_name, if_not_exists=True))

        # Create tables in schema
        async with db.begin():
            # This would create all tables in the tenant schema
            # For brevity, showing the concept
            await db.execute(text(f"SET search_path TO {schema_name}"))
            # Create tables...

        organization.settings["schema_name"] = schema_name

    async def _create_tenant_database(
        self,
        organization: Organization,
        db: AsyncSession
    ):
        """Create isolated database for tenant"""

        db_name = f"talai_tenant_{organization.slug.replace('-', '_')}"

        # Create database (requires superuser privileges)
        # This is simplified - in production, use proper database management
        await db.execute(text(f"CREATE DATABASE {db_name}"))

        # Store connection details
        organization.settings["database_name"] = db_name
        organization.settings["database_url"] = DatabaseSettings().database_url.replace(
            "talai_enterprise", db_name
        )

    async def _setup_row_level_security(
        self,
        organization: Organization,
        db: AsyncSession
    ):
        """Setup row-level security policies"""

        # Enable RLS on tables
        tables_to_secure = [
            "users", "roles", "permissions", "teams",
            "api_keys", "audit_logs", "subscriptions"
        ]

        for table in tables_to_secure:
            # Enable RLS
            await db.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))

            # Create policy for tenant isolation
            policy_name = f"{table}_tenant_isolation"
            await db.execute(text(f"""
                CREATE POLICY {policy_name} ON {table}
                FOR ALL
                USING (organization_id = current_setting('app.tenant_id')::uuid)
            """))

    async def _get_tenant_database_session(
        self,
        tenant_context: TenantContext
    ) -> AsyncSession:
        """Get session for tenant-specific database"""

        tenant_id = tenant_context.tenant_id

        # Check if engine exists
        if tenant_id not in self._tenant_engines:
            # Create engine for tenant database
            db_url = tenant_context.database_url or DatabaseSettings().database_url
            engine = create_async_engine(
                db_url,
                pool_size=TenancyConfig.TENANT_POOL_SIZE,
                max_overflow=TenancyConfig.TENANT_MAX_OVERFLOW,
                pool_timeout=TenancyConfig.TENANT_POOL_TIMEOUT
            )
            self._tenant_engines[tenant_id] = engine

            # Create session factory
            self._tenant_sessions[tenant_id] = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

        # Return new session
        return self._tenant_sessions[tenant_id]()

    async def _get_tenant_schema_session(
        self,
        tenant_context: TenantContext
    ) -> AsyncSession:
        """Get session with tenant schema set"""

        session = await get_db()

        # Set schema search path
        schema_name = tenant_context.schema_name
        if schema_name:
            await session.execute(text(f"SET search_path TO {schema_name}, public"))

        return session

    async def _set_row_level_context(
        self,
        session: AsyncSession,
        tenant_context: TenantContext
    ):
        """Set row-level security context"""

        # Set tenant ID for RLS policies
        await session.execute(
            text("SET app.tenant_id = :tenant_id"),
            {"tenant_id": tenant_context.tenant_id}
        )

    async def _create_default_roles(
        self,
        organization: Organization,
        db: AsyncSession
    ):
        """Create default roles for new tenant"""

        default_roles = [
            {
                "name": "Admin",
                "description": "Full administrative access",
                "is_default": False
            },
            {
                "name": "Manager",
                "description": "Team and project management",
                "is_default": False
            },
            {
                "name": "Developer",
                "description": "Development and deployment access",
                "is_default": False
            },
            {
                "name": "Viewer",
                "description": "Read-only access",
                "is_default": True
            }
        ]

        for role_data in default_roles:
            role = Role(
                organization_id=organization.id,
                **role_data
            )
            db.add(role)

    async def _initialize_tenant_resources(
        self,
        organization: Organization
    ):
        """Initialize tenant resources and quotas"""

        # This would set up:
        # - Storage quotas
        # - API rate limits
        # - Compute resources
        # - Network isolation
        pass

    async def _cache_tenant_config(
        self,
        organization: Organization
    ):
        """Cache tenant configuration"""

        if not self._redis_client:
            return

        cache_key = f"{TenancyConfig.TENANT_CACHE_PREFIX}{organization.id}"
        cache_data = {
            "organization_id": str(organization.id),
            "slug": organization.slug,
            "subscription_tier": organization.subscription_tier,
            "settings": organization.settings,
            "features": organization.features
        }

        await self._redis_client.setex(
            cache_key,
            TenancyConfig.CACHE_TTL_SECONDS,
            json.dumps(cache_data)
        )

    async def _invalidate_tenant_cache(
        self,
        organization: Organization
    ):
        """Clear tenant cache"""

        if not self._redis_client:
            return

        # Clear all cache keys for tenant
        pattern = f"{TenancyConfig.TENANT_CACHE_PREFIX}{organization.id}*"
        cursor = 0
        while True:
            cursor, keys = await self._redis_client.scan(
                cursor, match=pattern, count=100
            )
            if keys:
                await self._redis_client.delete(*keys)
            if cursor == 0:
                break

    async def _export_tenant_data(
        self,
        organization: Organization,
        strategy: IsolationStrategy,
        db: AsyncSession
    ) -> Dict:
        """Export all tenant data"""

        # This would export all tenant data for migration
        # Simplified for brevity
        return {}

    async def _import_tenant_data(
        self,
        organization: Organization,
        strategy: IsolationStrategy,
        data: Dict,
        db: AsyncSession
    ):
        """Import tenant data"""

        # This would import tenant data after migration
        # Simplified for brevity
        pass

    async def _drop_tenant_schema(
        self,
        organization: Organization,
        db: AsyncSession
    ):
        """Drop tenant schema"""

        schema_name = organization.settings.get("schema_name")
        if schema_name:
            await db.execute(DropSchema(schema_name, cascade=True, if_exists=True))

    async def _drop_tenant_database(
        self,
        organization: Organization
    ):
        """Drop tenant database"""

        db_name = organization.settings.get("database_name")
        if db_name:
            # This requires superuser privileges
            # In production, handle through proper database management
            pass

# Singleton instance
tenant_manager = TenantManager()

# FastAPI dependencies
async def get_tenant_context(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> TenantContext:
    """Get tenant context from request"""

    # Extract tenant ID from various sources
    tenant_id = None

    # 1. From subdomain
    host = request.headers.get("host", "")
    if "." in host:
        subdomain = host.split(".")[0]
        if subdomain and subdomain != "www":
            # Look up organization by slug
            org_query = select(Organization).where(
                Organization.slug == subdomain
            )
            org_result = await db.execute(org_query)
            org = org_result.scalar_one_or_none()
            if org:
                tenant_id = str(org.id)

    # 2. From header
    if not tenant_id:
        tenant_id = request.headers.get("X-Tenant-ID")

    # 3. From JWT token (if authenticated)
    # This would be extracted from the token

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant identification required"
        )

    return await tenant_manager.get_tenant_context(tenant_id, db)

# Middleware for tenant isolation
class TenantIsolationMiddleware:
    """Middleware to enforce tenant isolation"""

    async def __call__(self, request: Request, call_next):
        # Set tenant context for request
        # This would be integrated with FastAPI middleware
        response = await call_next(request)
        return response

# Export main components
__all__ = [
    "TenantManager",
    "tenant_manager",
    "TenantContext",
    "get_tenant_context",
    "CreateTenantRequest",
    "TenantResourceUsage",
    "TenantIsolationMiddleware",
    "IsolationStrategy"
]