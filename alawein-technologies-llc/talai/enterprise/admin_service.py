"""
Admin Dashboard Backend Service
System monitoring, user management, analytics, and configuration
"""

import os
import asyncio
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import json
import psutil
import platform

from fastapi import HTTPException, status, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text, case, extract
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, EmailStr, Field, validator
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import httpx

from database import get_db, check_database_health
from auth_models import User, Organization, Role, Team, APIKey, AuditLog, UserStatus
from auth_service import get_current_user, PermissionChecker
from billing_models import (
    Subscription, SubscriptionPlan, Invoice, Payment,
    UsageRecord, PaymentStatus, InvoiceStatus
)
from tenancy_service import TenantResourceUsage, tenant_manager

# Metrics
request_count = Counter('talai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('talai_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_users = Gauge('talai_active_users', 'Active users')
system_cpu = Gauge('talai_system_cpu_percent', 'System CPU usage')
system_memory = Gauge('talai_system_memory_percent', 'System memory usage')
system_disk = Gauge('talai_system_disk_percent', 'System disk usage')

# Configuration
class AdminConfig:
    # Feature flags
    FEATURE_FLAGS = {
        "new_dashboard": {"enabled": True, "rollout_percentage": 100},
        "advanced_analytics": {"enabled": True, "rollout_percentage": 50},
        "ai_insights": {"enabled": False, "rollout_percentage": 0},
        "bulk_operations": {"enabled": True, "rollout_percentage": 100},
        "export_import": {"enabled": True, "rollout_percentage": 100},
        "custom_branding": {"enabled": True, "rollout_percentage": 25},
        "sso_integration": {"enabled": True, "rollout_percentage": 100},
        "api_v2": {"enabled": False, "rollout_percentage": 10}
    }

    # System configuration
    SYSTEM_CONFIG = {
        "maintenance_mode": False,
        "rate_limiting_enabled": True,
        "signup_enabled": True,
        "trial_enabled": True,
        "default_trial_days": 14,
        "max_file_upload_mb": 100,
        "session_timeout_minutes": 60,
        "password_min_length": 8,
        "enforce_2fa": False,
        "audit_log_retention_days": 365
    }

    # Cost tracking
    COST_PER_UNIT = {
        "api_call": 0.00001,  # Per call
        "storage_gb": 0.10,    # Per GB per month
        "bandwidth_gb": 0.05,  # Per GB
        "compute_hour": 0.50   # Per hour
    }

# Models
class SystemHealth(BaseModel):
    """System health status"""
    status: str  # healthy, degraded, unhealthy
    database: Dict[str, Any]
    redis: Dict[str, Any]
    services: Dict[str, Any]
    metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]

class SystemStats(BaseModel):
    """System statistics"""
    total_users: int
    active_users_today: int
    active_users_week: int
    total_organizations: int
    active_subscriptions: int
    revenue_mtd: float
    revenue_ytd: float
    api_calls_today: int
    storage_used_gb: float
    top_features: List[Dict[str, Any]]

class UserManagementRequest(BaseModel):
    """User management operations"""
    action: str  # activate, deactivate, delete, reset_password, etc.
    user_ids: List[str]
    reason: Optional[str] = None
    notify: bool = True

class FeatureFlagUpdate(BaseModel):
    """Feature flag configuration"""
    flag_name: str
    enabled: bool
    rollout_percentage: int = Field(ge=0, le=100)
    metadata: Optional[Dict] = None

class SystemConfigUpdate(BaseModel):
    """System configuration update"""
    config_key: str
    config_value: Any
    effective_from: Optional[datetime] = None

class CostReport(BaseModel):
    """Cost tracking report"""
    organization_id: str
    period_start: datetime
    period_end: datetime
    total_cost: float
    breakdown: Dict[str, float]
    projections: Dict[str, float]

# Admin Dashboard Service
class AdminService:
    """Comprehensive admin dashboard functionality"""

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self._initialized = False

    async def initialize(self):
        """Initialize admin service"""
        if not self._initialized:
            self._redis_client = await redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                encoding="utf-8",
                decode_responses=True
            )
            self._initialized = True
            # Start background monitoring
            asyncio.create_task(self._monitor_system())

    async def get_system_health(self, db: AsyncSession) -> SystemHealth:
        """Get comprehensive system health status"""

        alerts = []

        # Database health
        db_health = await check_database_health()

        # Redis health
        redis_health = {"status": "unknown"}
        if self._redis_client:
            try:
                await self._redis_client.ping()
                redis_health = {
                    "status": "healthy",
                    "connected_clients": await self._redis_client.client_info()
                }
            except Exception as e:
                redis_health = {"status": "unhealthy", "error": str(e)}
                alerts.append({
                    "level": "critical",
                    "service": "redis",
                    "message": f"Redis connection failed: {str(e)}"
                })

        # Service health checks
        services = {}

        # Check Stripe
        try:
            import stripe
            stripe.Account.retrieve()
            services["stripe"] = {"status": "healthy"}
        except Exception as e:
            services["stripe"] = {"status": "unhealthy", "error": str(e)}
            alerts.append({
                "level": "warning",
                "service": "stripe",
                "message": "Stripe API unreachable"
            })

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "load_average": os.getloadavg()[0] if platform.system() != "Windows" else 0
        }

        # Update Prometheus metrics
        system_cpu.set(cpu_percent)
        system_memory.set(memory.percent)
        system_disk.set(disk.percent)

        # Generate alerts based on thresholds
        if cpu_percent > 80:
            alerts.append({
                "level": "warning",
                "service": "system",
                "message": f"High CPU usage: {cpu_percent}%"
            })
        if memory.percent > 85:
            alerts.append({
                "level": "warning",
                "service": "system",
                "message": f"High memory usage: {memory.percent}%"
            })
        if disk.percent > 90:
            alerts.append({
                "level": "critical",
                "service": "system",
                "message": f"Low disk space: {disk.free_gb:.1f}GB free"
            })

        # Determine overall status
        if any(a["level"] == "critical" for a in alerts):
            overall_status = "unhealthy"
        elif any(a["level"] == "warning" for a in alerts):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return SystemHealth(
            status=overall_status,
            database=db_health,
            redis=redis_health,
            services=services,
            metrics=metrics,
            alerts=alerts
        )

    async def get_system_stats(self, db: AsyncSession) -> SystemStats:
        """Get system-wide statistics"""

        # User statistics
        total_users_query = select(func.count(User.id))
        total_users = (await db.execute(total_users_query)).scalar() or 0

        active_today_query = select(func.count(User.id)).where(
            User.last_activity_at >= datetime.now(UTC) - timedelta(days=1)
        )
        active_today = (await db.execute(active_today_query)).scalar() or 0

        active_week_query = select(func.count(User.id)).where(
            User.last_activity_at >= datetime.now(UTC) - timedelta(days=7)
        )
        active_week = (await db.execute(active_week_query)).scalar() or 0

        # Organization statistics
        total_orgs_query = select(func.count(Organization.id))
        total_orgs = (await db.execute(total_orgs_query)).scalar() or 0

        active_subs_query = select(func.count(Subscription.id)).where(
            Subscription.status.in_(['active', 'trialing'])
        )
        active_subs = (await db.execute(active_subs_query)).scalar() or 0

        # Revenue statistics
        current_month = datetime.now(UTC).replace(day=1)
        current_year = datetime.now(UTC).replace(month=1, day=1)

        revenue_mtd_query = select(func.sum(Payment.amount)).where(
            and_(
                Payment.status == PaymentStatus.SUCCEEDED,
                Payment.processed_at >= current_month
            )
        )
        revenue_mtd = (await db.execute(revenue_mtd_query)).scalar() or 0
        revenue_mtd = revenue_mtd / 100  # Convert from cents

        revenue_ytd_query = select(func.sum(Payment.amount)).where(
            and_(
                Payment.status == PaymentStatus.SUCCEEDED,
                Payment.processed_at >= current_year
            )
        )
        revenue_ytd = (await db.execute(revenue_ytd_query)).scalar() or 0
        revenue_ytd = revenue_ytd / 100  # Convert from cents

        # API calls today (from cache)
        api_calls = 0
        if self._redis_client:
            api_pattern = "tenant:usage:*:api_calls"
            cursor = 0
            while True:
                cursor, keys = await self._redis_client.scan(
                    cursor, match=api_pattern, count=100
                )
                for key in keys:
                    count = await self._redis_client.get(key)
                    api_calls += int(count) if count else 0
                if cursor == 0:
                    break

        # Storage usage
        storage_query = text("""
            SELECT
                SUM(pg_database_size(datname)) / 1024 / 1024 / 1024 as total_gb
            FROM pg_database
            WHERE datname LIKE 'talai%'
        """)
        storage_result = await db.execute(storage_query)
        storage_gb = storage_result.scalar() or 0

        # Top features (from audit logs)
        top_features_query = select(
            AuditLog.event_type,
            func.count(AuditLog.id).label('count')
        ).where(
            AuditLog.created_at >= datetime.now(UTC) - timedelta(days=7)
        ).group_by(
            AuditLog.event_type
        ).order_by(
            func.count(AuditLog.id).desc()
        ).limit(10)

        top_features_result = await db.execute(top_features_query)
        top_features = [
            {"feature": row.event_type, "usage_count": row.count}
            for row in top_features_result
        ]

        # Update Prometheus metric
        active_users.set(active_today)

        return SystemStats(
            total_users=total_users,
            active_users_today=active_today,
            active_users_week=active_week,
            total_organizations=total_orgs,
            active_subscriptions=active_subs,
            revenue_mtd=revenue_mtd,
            revenue_ytd=revenue_ytd,
            api_calls_today=api_calls,
            storage_used_gb=float(storage_gb),
            top_features=top_features
        )

    async def manage_users(
        self,
        request: UserManagementRequest,
        admin_user: User,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Bulk user management operations"""

        if not admin_user.is_system_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System admin access required"
            )

        results = {"success": [], "failed": []}

        for user_id in request.user_ids:
            try:
                user_query = select(User).where(User.id == user_id)
                user_result = await db.execute(user_query)
                user = user_result.scalar_one_or_none()

                if not user:
                    results["failed"].append({
                        "user_id": user_id,
                        "error": "User not found"
                    })
                    continue

                if request.action == "activate":
                    user.status = UserStatus.ACTIVE
                    user.locked_until = None
                elif request.action == "deactivate":
                    user.status = UserStatus.INACTIVE
                elif request.action == "suspend":
                    user.status = UserStatus.SUSPENDED
                    user.locked_until = datetime.now(UTC) + timedelta(days=30)
                elif request.action == "delete":
                    user.status = UserStatus.DELETED
                elif request.action == "reset_password":
                    # Generate password reset token
                    import secrets
                    user.password_reset_token = secrets.token_urlsafe(32)
                    user.password_reset_expires = datetime.now(UTC) + timedelta(hours=1)
                elif request.action == "unlock":
                    user.locked_until = None
                    user.failed_login_attempts = 0
                else:
                    results["failed"].append({
                        "user_id": user_id,
                        "error": f"Unknown action: {request.action}"
                    })
                    continue

                # Log audit event
                audit_log = AuditLog(
                    user_id=admin_user.id,
                    organization_id=user.organization_id,
                    event_type="admin_user_management",
                    action=request.action,
                    resource_type="user",
                    resource_id=str(user.id),
                    metadata={
                        "reason": request.reason,
                        "admin_id": str(admin_user.id)
                    },
                    success=True
                )
                db.add(audit_log)

                results["success"].append({
                    "user_id": str(user.id),
                    "email": user.email,
                    "action": request.action
                })

                # Send notification if requested
                if request.notify:
                    # Queue notification task
                    pass

            except Exception as e:
                results["failed"].append({
                    "user_id": user_id,
                    "error": str(e)
                })

        await db.commit()
        return results

    async def get_feature_flags(
        self,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get feature flags configuration"""

        flags = AdminConfig.FEATURE_FLAGS.copy()

        # Check organization-specific overrides
        if organization_id and self._redis_client:
            org_flags_key = f"feature_flags:{organization_id}"
            org_flags = await self._redis_client.get(org_flags_key)
            if org_flags:
                org_flags = json.loads(org_flags)
                flags.update(org_flags)

        return flags

    async def update_feature_flag(
        self,
        update: FeatureFlagUpdate,
        admin_user: User,
        db: AsyncSession
    ):
        """Update feature flag configuration"""

        if not admin_user.is_system_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System admin access required"
            )

        # Update global configuration
        AdminConfig.FEATURE_FLAGS[update.flag_name] = {
            "enabled": update.enabled,
            "rollout_percentage": update.rollout_percentage,
            "metadata": update.metadata
        }

        # Persist to Redis
        if self._redis_client:
            await self._redis_client.set(
                f"feature_flag:{update.flag_name}",
                json.dumps(AdminConfig.FEATURE_FLAGS[update.flag_name]),
                ex=86400  # 24 hours
            )

        # Log audit event
        audit_log = AuditLog(
            user_id=admin_user.id,
            event_type="feature_flag_update",
            action="update",
            resource_type="feature_flag",
            resource_id=update.flag_name,
            old_values={"enabled": not update.enabled},
            new_values={"enabled": update.enabled},
            metadata=update.metadata,
            success=True
        )
        db.add(audit_log)
        await db.commit()

    async def get_system_configuration(self) -> Dict[str, Any]:
        """Get system configuration"""
        return AdminConfig.SYSTEM_CONFIG.copy()

    async def update_system_configuration(
        self,
        update: SystemConfigUpdate,
        admin_user: User,
        db: AsyncSession
    ):
        """Update system configuration"""

        if not admin_user.is_system_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System admin access required"
            )

        old_value = AdminConfig.SYSTEM_CONFIG.get(update.config_key)
        AdminConfig.SYSTEM_CONFIG[update.config_key] = update.config_value

        # Persist to Redis
        if self._redis_client:
            await self._redis_client.set(
                f"system_config:{update.config_key}",
                json.dumps(update.config_value),
                ex=86400  # 24 hours
            )

        # Log audit event
        audit_log = AuditLog(
            user_id=admin_user.id,
            event_type="system_config_update",
            action="update",
            resource_type="system_config",
            resource_id=update.config_key,
            old_values={"value": old_value},
            new_values={"value": update.config_value},
            success=True
        )
        db.add(audit_log)
        await db.commit()

    async def calculate_organization_costs(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        db: AsyncSession
    ) -> CostReport:
        """Calculate detailed costs for an organization"""

        # Get usage records
        usage_query = select(UsageRecord).where(
            and_(
                UsageRecord.organization_id == organization_id,
                UsageRecord.period_start >= start_date,
                UsageRecord.period_end <= end_date
            )
        )
        usage_result = await db.execute(usage_query)
        usage_records = usage_result.scalars().all()

        # Calculate costs by type
        breakdown = {
            "api_calls": 0,
            "storage": 0,
            "compute": 0,
            "bandwidth": 0,
            "subscription": 0
        }

        for record in usage_records:
            if record.usage_type == "api_calls":
                breakdown["api_calls"] += float(record.quantity) * AdminConfig.COST_PER_UNIT["api_call"]
            elif record.usage_type == "storage_gb":
                breakdown["storage"] += float(record.quantity) * AdminConfig.COST_PER_UNIT["storage_gb"]
            elif record.usage_type == "compute_hours":
                breakdown["compute"] += float(record.quantity) * AdminConfig.COST_PER_UNIT["compute_hour"]
            elif record.usage_type == "bandwidth_gb":
                breakdown["bandwidth"] += float(record.quantity) * AdminConfig.COST_PER_UNIT["bandwidth_gb"]

        # Get subscription costs
        sub_query = select(Subscription).where(
            and_(
                Subscription.organization_id == organization_id,
                Subscription.status.in_(['active', 'trialing'])
            )
        )
        sub_result = await db.execute(sub_query)
        subscription = sub_result.scalar_one_or_none()

        if subscription:
            # Calculate prorated subscription cost
            days_in_period = (end_date - start_date).days
            monthly_cost = float(subscription.plan.base_price)
            breakdown["subscription"] = (monthly_cost / 30) * days_in_period

        total_cost = sum(breakdown.values())

        # Calculate projections
        days_elapsed = (datetime.now(UTC) - start_date).days or 1
        daily_rate = total_cost / days_elapsed
        projections = {
            "monthly": daily_rate * 30,
            "quarterly": daily_rate * 90,
            "yearly": daily_rate * 365
        }

        return CostReport(
            organization_id=organization_id,
            period_start=start_date,
            period_end=end_date,
            total_cost=total_cost,
            breakdown=breakdown,
            projections=projections
        )

    async def get_analytics_dashboard(
        self,
        time_range: str,  # today, week, month, year
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get analytics dashboard data"""

        # Determine date range
        end_date = datetime.now(UTC)
        if time_range == "today":
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_range == "week":
            start_date = end_date - timedelta(days=7)
        elif time_range == "month":
            start_date = end_date - timedelta(days=30)
        elif time_range == "year":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)

        # User growth
        user_growth_query = select(
            func.date_trunc('day', User.created_at).label('date'),
            func.count(User.id).label('count')
        ).where(
            User.created_at >= start_date
        ).group_by(
            func.date_trunc('day', User.created_at)
        ).order_by('date')

        user_growth_result = await db.execute(user_growth_query)
        user_growth = [
            {"date": row.date.isoformat(), "count": row.count}
            for row in user_growth_result
        ]

        # Revenue trends
        revenue_query = select(
            func.date_trunc('day', Payment.processed_at).label('date'),
            func.sum(Payment.amount).label('amount')
        ).where(
            and_(
                Payment.status == PaymentStatus.SUCCEEDED,
                Payment.processed_at >= start_date
            )
        ).group_by(
            func.date_trunc('day', Payment.processed_at)
        ).order_by('date')

        revenue_result = await db.execute(revenue_query)
        revenue_trend = [
            {"date": row.date.isoformat(), "amount": float(row.amount) / 100}
            for row in revenue_result
        ]

        # API usage trends
        # This would need to aggregate from time-series data
        api_usage_trend = []

        # Top organizations by revenue
        top_orgs_query = select(
            Organization.name,
            func.sum(Payment.amount).label('total_revenue')
        ).join(
            Payment, Payment.organization_id == Organization.id
        ).where(
            and_(
                Payment.status == PaymentStatus.SUCCEEDED,
                Payment.processed_at >= start_date
            )
        ).group_by(
            Organization.id, Organization.name
        ).order_by(
            func.sum(Payment.amount).desc()
        ).limit(10)

        top_orgs_result = await db.execute(top_orgs_query)
        top_organizations = [
            {"name": row.name, "revenue": float(row.total_revenue) / 100}
            for row in top_orgs_result
        ]

        return {
            "time_range": time_range,
            "user_growth": user_growth,
            "revenue_trend": revenue_trend,
            "api_usage_trend": api_usage_trend,
            "top_organizations": top_organizations,
            "summary": {
                "new_users": len([u for u in user_growth]),
                "total_revenue": sum(r["amount"] for r in revenue_trend),
                "avg_daily_revenue": sum(r["amount"] for r in revenue_trend) / max(len(revenue_trend), 1)
            }
        }

    async def _monitor_system(self):
        """Background system monitoring task"""
        while True:
            try:
                # Update system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                system_cpu.set(cpu_percent)
                system_memory.set(memory.percent)
                system_disk.set(disk.percent)

                # Check for alerts
                if cpu_percent > 90:
                    # Send critical alert
                    pass
                if memory.percent > 95:
                    # Send critical alert
                    pass

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(60)

# Singleton instance
admin_service = AdminService()

# FastAPI dependencies
async def require_system_admin(
    user: User = Depends(get_current_user)
) -> User:
    """Require system admin access"""
    if not user.is_system_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System admin access required"
        )
    return user

# Export main components
__all__ = [
    "AdminService",
    "admin_service",
    "SystemHealth",
    "SystemStats",
    "UserManagementRequest",
    "FeatureFlagUpdate",
    "SystemConfigUpdate",
    "CostReport",
    "require_system_admin",
    "AdminConfig"
]