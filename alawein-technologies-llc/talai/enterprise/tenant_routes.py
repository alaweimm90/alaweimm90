"""
Tenant Management Routes
API endpoints for multi-tenancy operations
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from auth_service import get_current_user, require_system_admin
from auth_models import User
from tenancy_service import (
    tenant_manager, get_tenant_context,
    CreateTenantRequest, TenantContext
)

router = APIRouter()

@router.post("/")
async def create_tenant(
    request: CreateTenantRequest,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new tenant"""
    organization = await tenant_manager.create_tenant(request, db)
    return {
        "id": str(organization.id),
        "name": organization.name,
        "slug": organization.slug,
        "message": "Tenant created successfully"
    }

@router.get("/context")
async def get_current_tenant_context(
    context: TenantContext = Depends(get_tenant_context)
):
    """Get current tenant context"""
    return context

@router.get("/{tenant_id}/usage")
async def get_tenant_usage(
    tenant_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get tenant resource usage"""

    # Verify access
    if not user.is_system_admin and str(user.organization_id) != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    from sqlalchemy import select
    from auth_models import Organization

    org_query = select(Organization).where(Organization.id == tenant_id)
    org_result = await db.execute(org_query)
    organization = org_result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )

    return await tenant_manager.get_tenant_resource_usage(organization, db)

@router.delete("/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    hard_delete: bool = False,
    user: User = Depends(require_system_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete or archive a tenant"""

    from sqlalchemy import select
    from auth_models import Organization

    org_query = select(Organization).where(Organization.id == tenant_id)
    org_result = await db.execute(org_query)
    organization = org_result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )

    await tenant_manager.delete_tenant(organization, hard_delete, db)

    return {
        "message": "Tenant deleted" if hard_delete else "Tenant archived"
    }

# Export router
tenant_router = router