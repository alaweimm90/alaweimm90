"""
Authentication Routes
API endpoints for authentication and authorization
"""

from datetime import datetime, timedelta, UTC
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from database import get_db
from auth_service import (
    AuthService, get_current_user, create_access_token, create_refresh_token,
    TokenData, TokenResponse, LoginRequest, OAuth2LoginRequest
)
from auth_models import User, Organization, Role, APIKey, UserSession

router = APIRouter()

# Request/Response models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    organization_name: Optional[str] = None
    organization_slug: Optional[str] = None

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class Enable2FAResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: List[str]

class Verify2FARequest(BaseModel):
    code: str

class CreateAPIKeyRequest(BaseModel):
    name: str
    scopes: List[str]
    expires_in_days: Optional[int] = None

# Authentication endpoints
@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user and organization"""

    # Check if email already exists
    from sqlalchemy import select
    existing_user = await db.execute(
        select(User).where(User.email == request.email)
    )
    if existing_user.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create organization if needed
    if request.organization_name:
        org = Organization(
            name=request.organization_name,
            slug=request.organization_slug or request.organization_name.lower().replace(" ", "-"),
            settings={},
            is_active=True
        )
        db.add(org)
        await db.flush()
    else:
        # Find or create default organization
        default_org = await db.execute(
            select(Organization).where(Organization.slug == "default")
        )
        org = default_org.scalar_one_or_none()
        if not org:
            org = Organization(
                name="Default Organization",
                slug="default",
                is_active=True
            )
            db.add(org)
            await db.flush()

    # Create user
    user = User(
        organization_id=org.id,
        email=request.email,
        username=request.email.split("@")[0],
        full_name=request.full_name,
        status="pending",
        email_verified=False,
        is_org_admin=True if request.organization_name else False
    )
    user.set_password(request.password)
    db.add(user)
    await db.commit()

    # Create session
    session = await AuthService.create_user_session(
        user=user,
        request=req,
        device_id=None,
        remember_me=False,
        db=db
    )

    # Generate tokens
    token_data = TokenData(
        user_id=str(user.id),
        org_id=str(user.organization_id),
        email=user.email,
        scopes=[],
        session_id=str(session.id)
    )

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "organization_id": str(user.organization_id)
        }
    )

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Login with email and password"""

    user = await AuthService.authenticate_user(
        email=request.email,
        password=request.password,
        organization_slug=request.organization_slug,
        db=db
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Create session
    session = await AuthService.create_user_session(
        user=user,
        request=req,
        device_id=request.device_id,
        remember_me=request.remember_me,
        db=db
    )

    # Generate tokens
    token_data = TokenData(
        user_id=str(user.id),
        org_id=str(user.organization_id),
        email=user.email,
        scopes=[],
        session_id=str(session.id)
    )

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Log audit event
    await AuthService.log_audit_event(
        db=db,
        user=user,
        event_type="user_login",
        action="login",
        request=req,
        success=True
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "organization_id": str(user.organization_id),
            "is_org_admin": user.is_org_admin,
            "is_system_admin": user.is_system_admin
        }
    )

@router.post("/oauth2/login", response_model=TokenResponse)
async def oauth2_login(
    request: OAuth2LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """Login via OAuth2 provider"""

    user = await AuthService.oauth2_authenticate(
        provider=request.provider,
        code=request.code,
        state=request.state,
        db=db
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OAuth2 authentication failed"
        )

    # Create session
    session = await AuthService.create_user_session(
        user=user,
        request=req,
        device_id=None,
        remember_me=True,
        db=db
    )

    # Generate tokens
    token_data = TokenData(
        user_id=str(user.id),
        org_id=str(user.organization_id),
        email=user.email,
        scopes=[],
        session_id=str(session.id)
    )

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "organization_id": str(user.organization_id)
        }
    )

@router.post("/logout")
async def logout(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Logout and invalidate session"""

    # Invalidate all user sessions
    from sqlalchemy import update
    await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user.id)
        .values(
            is_active=False,
            revoked_at=datetime.now(UTC),
            revoked_reason="User logout"
        )
    )
    await db.commit()

    return {"message": "Successfully logged out"}

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token"""

    from auth_service import decode_token

    token_data = await decode_token(refresh_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Verify session is still active
    from sqlalchemy import select
    session_query = select(UserSession).where(
        UserSession.id == token_data.session_id
    )
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()

    if not session or not session.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or revoked"
        )

    # Generate new access token
    new_access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=refresh_token,
        expires_in=1800,
        user={}
    )

@router.get("/me")
async def get_current_user_profile(
    user: User = Depends(get_current_user)
):
    """Get current user profile"""

    return {
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "avatar_url": user.avatar_url,
        "organization": {
            "id": str(user.organization.id),
            "name": user.organization.name,
            "slug": user.organization.slug,
            "subscription_tier": user.organization.subscription_tier
        },
        "roles": [
            {"id": str(r.id), "name": r.name}
            for r in user.roles
        ],
        "status": user.status,
        "email_verified": user.email_verified,
        "two_factor_enabled": user.two_factor_enabled,
        "is_org_admin": user.is_org_admin,
        "is_system_admin": user.is_system_admin,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
    }

@router.put("/me")
async def update_profile(
    updates: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user profile"""

    allowed_fields = ["full_name", "username", "avatar_url", "timezone", "locale", "preferences"]

    for field, value in updates.items():
        if field in allowed_fields:
            setattr(user, field, value)

    await db.commit()

    return {"message": "Profile updated successfully"}

@router.post("/password/reset")
async def request_password_reset(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset token"""

    from sqlalchemy import select
    import secrets

    user_query = select(User).where(User.email == request.email)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if user:
        # Generate reset token
        user.password_reset_token = secrets.token_urlsafe(32)
        user.password_reset_expires = datetime.now(UTC) + timedelta(hours=1)
        await db.commit()

        # Queue email sending
        # background_tasks.add_task(send_password_reset_email, user)

    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a reset link has been sent"}

@router.post("/password/reset/confirm")
async def confirm_password_reset(
    request: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """Reset password with token"""

    from sqlalchemy import select

    user_query = select(User).where(
        User.password_reset_token == request.token
    )
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user or user.password_reset_expires < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    # Check password reuse
    if user.is_password_reused(request.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password has been used before"
        )

    # Set new password
    user.set_password(request.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    await db.commit()

    return {"message": "Password reset successful"}

@router.post("/password/change")
async def change_password(
    request: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""

    # Verify current password
    if not user.verify_password(request.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    # Check password reuse
    if user.is_password_reused(request.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password has been used before"
        )

    # Set new password
    user.set_password(request.new_password)
    await db.commit()

    # Invalidate all sessions except current
    # This forces re-authentication on other devices

    return {"message": "Password changed successfully"}

@router.post("/2fa/enable", response_model=Enable2FAResponse)
async def enable_two_factor(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Enable two-factor authentication"""

    import pyotp
    import qrcode
    import io
    import base64

    # Generate secret
    secret = pyotp.random_base32()

    # Generate QR code
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user.email,
        issuer_name='TalAI'
    )

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    qr_code = base64.b64encode(buf.getvalue()).decode()

    # Generate backup codes
    import secrets
    backup_codes = [secrets.token_hex(4) for _ in range(10)]

    # Store temporarily (user must verify before enabling)
    user.two_factor_secret = secret
    user.backup_codes = backup_codes
    await db.commit()

    return Enable2FAResponse(
        secret=secret,
        qr_code=f"data:image/png;base64,{qr_code}",
        backup_codes=backup_codes
    )

@router.post("/2fa/verify")
async def verify_two_factor(
    request: Verify2FARequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Verify and activate two-factor authentication"""

    import pyotp

    if not user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA not initialized"
        )

    # Verify code
    totp = pyotp.TOTP(user.two_factor_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code"
        )

    # Enable 2FA
    user.two_factor_enabled = True
    await db.commit()

    return {"message": "Two-factor authentication enabled"}

@router.post("/2fa/disable")
async def disable_two_factor(
    password: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Disable two-factor authentication"""

    # Verify password
    if not user.verify_password(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid password"
        )

    # Disable 2FA
    user.two_factor_enabled = False
    user.two_factor_secret = None
    user.backup_codes = None
    await db.commit()

    return {"message": "Two-factor authentication disabled"}

# API Key management
@router.post("/api-keys", response_model=dict)
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key"""

    import secrets
    import hashlib

    # Generate API key
    key = f"sk_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    # Calculate expiry
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(UTC) + timedelta(days=request.expires_in_days)

    # Create API key record
    api_key = APIKey(
        organization_id=user.organization_id,
        user_id=user.id,
        name=request.name,
        key_hash=key_hash,
        key_prefix=key[:10],
        scopes=request.scopes,
        expires_at=expires_at,
        is_active=True
    )
    db.add(api_key)
    await db.commit()

    return {
        "id": str(api_key.id),
        "name": api_key.name,
        "key": key,  # Only shown once
        "expires_at": expires_at.isoformat() if expires_at else None,
        "message": "Store this key securely. It won't be shown again."
    }

@router.get("/api-keys")
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's API keys"""

    from sqlalchemy import select

    keys_query = select(APIKey).where(
        APIKey.user_id == user.id
    )
    keys_result = await db.execute(keys_query)
    keys = keys_result.scalars().all()

    return [
        {
            "id": str(k.id),
            "name": k.name,
            "key_prefix": k.key_prefix,
            "scopes": k.scopes,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            "expires_at": k.expires_at.isoformat() if k.expires_at else None,
            "is_active": k.is_active
        }
        for k in keys
    ]

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Revoke an API key"""

    from sqlalchemy import select

    key_query = select(APIKey).where(
        APIKey.id == key_id,
        APIKey.user_id == user.id
    )
    key_result = await db.execute(key_query)
    api_key = key_result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    api_key.is_active = False
    await db.commit()

    return {"message": "API key revoked"}

# Export router
auth_router = router