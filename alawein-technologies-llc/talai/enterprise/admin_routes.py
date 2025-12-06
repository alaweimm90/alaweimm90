"""
Admin Dashboard Routes
API endpoints for system administration
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from auth_service import get_current_user
from auth_models import User
from admin_service import (
    admin_service, require_system_admin,
    UserManagementRequest, FeatureFlagUpdate, SystemConfigUpdate
)

router = APIRouter()

@router.get("/health")
async def get_system_health(
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get system health status"""
    return await admin_service.get_system_health(db)

@router.get("/stats")
async def get_system_stats(
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get system statistics"""
    return await admin_service.get_system_stats(db)

@router.post("/users/manage")
async def manage_users(
    request: UserManagementRequest,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Bulk user management operations"""
    return await admin_service.manage_users(request, user, db)

@router.get("/feature-flags")
async def get_feature_flags(
    user: User = Depends(require_system_admin)
):
    """Get feature flags"""
    return await admin_service.get_feature_flags()

@router.put("/feature-flags")
async def update_feature_flag(
    update: FeatureFlagUpdate,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update feature flag"""
    await admin_service.update_feature_flag(update, user, db)
    return {"message": "Feature flag updated"}

@router.get("/config")
async def get_system_config(
    user: User = Depends(require_system_admin)
):
    """Get system configuration"""
    return await admin_service.get_system_configuration()

@router.put("/config")
async def update_system_config(
    update: SystemConfigUpdate,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update system configuration"""
    await admin_service.update_system_configuration(update, user, db)
    return {"message": "Configuration updated"}

@router.get("/analytics")
async def get_analytics(
    time_range: str = "month",
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get analytics dashboard data"""
    return await admin_service.get_analytics_dashboard(time_range, db)

@router.get("/costs/{organization_id}")
async def get_organization_costs(
    organization_id: str,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Calculate organization costs"""
    from datetime import datetime, timedelta, UTC

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)

    return await admin_service.calculate_organization_costs(
        organization_id, start_date, end_date, db
    )

# Export router
admin_router = router