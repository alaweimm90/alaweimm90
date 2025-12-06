"""
General API Routes
Core API endpoints for TalAI services
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, UTC

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from database import get_db
from auth_service import get_current_user, get_api_key_user
from auth_models import User
from tenancy_service import get_tenant_context, TenantContext
from billing_service import BillingService, RecordUsageRequest

router = APIRouter()

# Request/Response models
class TalAIRequest(BaseModel):
    """Base request for TalAI services"""
    module: str  # Which TalAI module to use
    action: str  # What action to perform
    parameters: Dict[str, Any]
    metadata: Optional[Dict] = None

class TalAIResponse(BaseModel):
    """Base response from TalAI services"""
    success: bool
    module: str
    action: str
    result: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict] = None

# TalAI Module Registry
TALAI_MODULES = {
    "abstract-writer": {
        "actions": ["generate", "improve", "analyze"],
        "description": "Abstract generation from research papers"
    },
    "adversarial-review": {
        "actions": ["critique", "review", "evaluate"],
        "description": "Adversarial critique generation"
    },
    "experiment-designer": {
        "actions": ["design", "optimize", "validate"],
        "description": "Experimental design automation"
    },
    "grant-writer": {
        "actions": ["draft", "review", "budget"],
        "description": "Grant proposal generation"
    },
    "lit-review-bot": {
        "actions": ["search", "summarize", "synthesize"],
        "description": "Literature review automation"
    },
    "prompt-marketplace": {
        "actions": ["search", "submit", "rate"],
        "description": "Prompt sharing and marketplace"
    }
}

@router.post("/execute", response_model=TalAIResponse)
async def execute_talai_module(
    request: TalAIRequest,
    req: Request,
    user: User = Depends(get_current_user),
    context: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Execute a TalAI module action"""

    # Validate module
    if request.module not in TALAI_MODULES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown module: {request.module}"
        )

    # Validate action
    if request.action not in TALAI_MODULES[request.module]["actions"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action '{request.action}' for module '{request.module}'"
        )

    # Check rate limits
    from tenancy_service import tenant_manager

    if not await tenant_manager.check_resource_limits(
        context, "api_calls", 1
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="API rate limit exceeded"
        )

    # Increment usage counter
    await tenant_manager.increment_resource_usage(
        context, "api_calls", 1
    )

    # Track usage for billing
    usage_request = RecordUsageRequest(
        usage_type="api_calls",
        quantity=1,
        unit="call",
        description=f"TalAI {request.module}:{request.action}",
        metadata={
            "module": request.module,
            "action": request.action,
            "user_id": str(user.id)
        }
    )
    await BillingService.record_usage(
        organization=user.organization,
        request=usage_request,
        user=user,
        db=db
    )

    try:
        # Here you would integrate with actual TalAI modules
        # For now, return a mock response
        result = {
            "output": f"Executed {request.action} on {request.module}",
            "processing_time_ms": 1234,
            "tokens_used": 567
        }

        return TalAIResponse(
            success=True,
            module=request.module,
            action=request.action,
            result=result,
            usage={
                "tokens": 567,
                "compute_ms": 1234
            }
        )

    except Exception as e:
        return TalAIResponse(
            success=False,
            module=request.module,
            action=request.action,
            error=str(e)
        )

@router.get("/modules")
async def list_modules(
    user: User = Depends(get_current_user)
):
    """List available TalAI modules"""
    return TALAI_MODULES

@router.get("/modules/{module_name}")
async def get_module_info(
    module_name: str,
    user: User = Depends(get_current_user)
):
    """Get detailed information about a module"""

    if module_name not in TALAI_MODULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module '{module_name}' not found"
        )

    return {
        "name": module_name,
        "description": TALAI_MODULES[module_name]["description"],
        "actions": TALAI_MODULES[module_name]["actions"],
        "pricing": {
            "per_request": 0.01,
            "per_1000_tokens": 0.02
        }
    }

@router.get("/quota")
async def get_api_quota(
    user: User = Depends(get_current_user),
    context: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Get current API quota and usage"""

    from tenancy_service import tenant_manager

    usage = await tenant_manager.get_tenant_resource_usage(
        user.organization, db
    )

    return {
        "api_calls": {
            "used_today": usage.api_calls_today,
            "limit_per_day": usage.api_calls_limit
        },
        "storage": {
            "used_gb": usage.storage_used_gb,
            "limit_gb": usage.storage_limit_gb
        },
        "users": {
            "active": usage.active_users,
            "limit": usage.user_limit
        },
        "subscription_tier": context.subscription_tier
    }

# API Key authenticated endpoints
@router.post("/v1/execute", response_model=TalAIResponse)
async def execute_with_api_key(
    request: TalAIRequest,
    user: Optional[User] = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute TalAI module using API key authentication"""

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Process request similar to above
    # but with API key specific handling

    return TalAIResponse(
        success=True,
        module=request.module,
        action=request.action,
        result={"message": "Executed via API key"}
    )

# Export router
api_router = router