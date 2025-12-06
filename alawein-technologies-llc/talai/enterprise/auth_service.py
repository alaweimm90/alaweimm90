"""
Authentication & Authorization Service
Enterprise-grade authentication with JWT, OAuth2, and comprehensive security
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
import json
import asyncio
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import (
    Depends, HTTPException, status, Security,
    Request, Response, BackgroundTasks
)
from fastapi.security import (
    HTTPBearer, HTTPAuthorizationCredentials,
    OAuth2AuthorizationCodeBearer, OAuth2PasswordBearer
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, EmailStr, Field, validator
import httpx
from authlib.integrations.starlette_client import OAuth
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

from database import get_db, get_read_db
from auth_models import (
    User, Organization, Role, Permission, Team, APIKey,
    UserSession, AuditLog, UserStatus, AuthProvider
)

# Configuration
class AuthConfig:
    SECRET_KEY = os.getenv("TALAI_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 1
    EMAIL_VERIFICATION_TOKEN_EXPIRE_DAYS = 7

    # OAuth2 Providers
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
    MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
    MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")

    # Redis for session management
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Security
    MAX_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION_MINUTES = 30
    SESSION_TIMEOUT_MINUTES = 60
    REQUIRE_EMAIL_VERIFICATION = True

# Initialize components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
bearer_scheme = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

# Redis client for session and cache management
redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    redis_client = await redis.from_url(
        AuthConfig.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )

# OAuth2 clients
oauth = OAuth()

if AuthConfig.GOOGLE_CLIENT_ID:
    oauth.register(
        name='google',
        client_id=AuthConfig.GOOGLE_CLIENT_ID,
        client_secret=AuthConfig.GOOGLE_CLIENT_SECRET,
        authorize_url='https://accounts.google.com/o/oauth2/auth',
        access_token_url='https://accounts.google.com/o/oauth2/token',
        jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
        client_kwargs={'scope': 'openid email profile'},
    )

if AuthConfig.GITHUB_CLIENT_ID:
    oauth.register(
        name='github',
        client_id=AuthConfig.GITHUB_CLIENT_ID,
        client_secret=AuthConfig.GITHUB_CLIENT_SECRET,
        authorize_url='https://github.com/login/oauth/authorize',
        access_token_url='https://github.com/login/oauth/access_token',
        client_kwargs={'scope': 'user:email'},
    )

# Token models
class TokenData(BaseModel):
    user_id: str
    org_id: str
    email: str
    scopes: List[str] = []
    session_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    organization_slug: Optional[str] = None
    remember_me: bool = False
    device_id: Optional[str] = None

class OAuth2LoginRequest(BaseModel):
    provider: str
    code: str
    state: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

# JWT Token Management
def create_access_token(
    data: TokenData,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    to_encode = data.dict()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    return encoded_jwt

def create_refresh_token(
    data: TokenData,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token"""
    to_encode = data.dict()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
    return encoded_jwt

async def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])

        # Check if token is blacklisted in Redis
        if redis_client:
            is_blacklisted = await redis_client.get(f"blacklist:{token[:20]}")
            if is_blacklisted:
                return None

        return TokenData(
            user_id=payload["user_id"],
            org_id=payload["org_id"],
            email=payload["email"],
            scopes=payload.get("scopes", []),
            session_id=payload.get("session_id")
        )
    except JWTError:
        return None

# Authentication Service
class AuthService:
    """Main authentication service with comprehensive security features"""

    @staticmethod
    async def authenticate_user(
        email: str,
        password: str,
        organization_slug: Optional[str],
        db: AsyncSession
    ) -> Optional[User]:
        """Authenticate user with email and password"""

        # Build query
        query = select(User).options(
            selectinload(User.organization),
            selectinload(User.roles).selectinload(Role.permissions)
        )

        if organization_slug:
            query = query.join(Organization).where(
                and_(
                    func.lower(User.email) == func.lower(email),
                    Organization.slug == organization_slug
                )
            )
        else:
            query = query.where(func.lower(User.email) == func.lower(email))

        result = await db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            return None

        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(UTC):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked until {user.locked_until}"
            )

        # Verify password
        if not user.verify_password(password):
            # Increment failed attempts
            user.failed_login_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= AuthConfig.MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.now(UTC) + timedelta(
                    minutes=AuthConfig.ACCOUNT_LOCKOUT_DURATION_MINUTES
                )
                user.status = UserStatus.SUSPENDED

            await db.commit()
            return None

        # Check user status
        if user.status not in [UserStatus.ACTIVE, UserStatus.PENDING]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account status: {user.status}"
            )

        # Check organization status
        if not user.organization.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Organization is not active"
            )

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login_at = datetime.now(UTC)
        await db.commit()

        return user

    @staticmethod
    async def create_user_session(
        user: User,
        request: Request,
        device_id: Optional[str],
        remember_me: bool,
        db: AsyncSession
    ) -> UserSession:
        """Create a new user session"""

        # Calculate expiry
        if remember_me:
            expires_at = datetime.now(UTC) + timedelta(days=30)
            refresh_expires = datetime.now(UTC) + timedelta(days=60)
        else:
            expires_at = datetime.now(UTC) + timedelta(
                minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
            )
            refresh_expires = datetime.now(UTC) + timedelta(
                days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS
            )

        # Create session
        session = UserSession(
            user_id=user.id,
            token_hash=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
            refresh_token_hash=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            device_id=device_id,
            expires_at=expires_at,
            refresh_expires_at=refresh_expires,
            is_active=True
        )

        db.add(session)
        await db.commit()

        # Store session in Redis for quick validation
        if redis_client:
            await redis_client.setex(
                f"session:{session.id}",
                int((expires_at - datetime.now(UTC)).total_seconds()),
                json.dumps({
                    "user_id": str(user.id),
                    "org_id": str(user.organization_id),
                    "email": user.email
                })
            )

        return session

    @staticmethod
    async def oauth2_authenticate(
        provider: str,
        code: str,
        state: str,
        db: AsyncSession
    ) -> Optional[User]:
        """Authenticate user via OAuth2 provider"""

        if provider not in ["google", "github", "microsoft"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OAuth2 provider"
            )

        client = oauth.create_client(provider)
        if not client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{provider} OAuth2 not configured"
            )

        # Exchange code for token
        try:
            token = await client.authorize_access_token(code=code)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to exchange code: {str(e)}"
            )

        # Get user info from provider
        user_info = None
        if provider == "google":
            user_info = token.get("userinfo")
        elif provider == "github":
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token['access_token']}"}
                resp = await client.get("https://api.github.com/user", headers=headers)
                user_info = resp.json()
        elif provider == "microsoft":
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token['access_token']}"}
                resp = await client.get("https://graph.microsoft.com/v1.0/me", headers=headers)
                user_info = resp.json()

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve user info from provider"
            )

        # Find or create user
        email = user_info.get("email") or user_info.get("mail")
        external_id = str(user_info.get("id") or user_info.get("sub"))

        query = select(User).where(
            or_(
                User.email == email,
                and_(
                    User.auth_provider == provider,
                    User.external_id == external_id
                )
            )
        )
        result = await db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            # Create new user
            org_query = select(Organization).where(Organization.slug == "default")
            org_result = await db.execute(org_query)
            org = org_result.scalar_one_or_none()

            if not org:
                # Create default organization
                org = Organization(
                    name="Default Organization",
                    slug="default"
                )
                db.add(org)
                await db.flush()

            user = User(
                organization_id=org.id,
                email=email,
                username=user_info.get("login") or email.split("@")[0],
                full_name=user_info.get("name"),
                avatar_url=user_info.get("picture") or user_info.get("avatar_url"),
                auth_provider=provider,
                external_id=external_id,
                status=UserStatus.ACTIVE,
                email_verified=True,
                email_verified_at=datetime.now(UTC)
            )
            db.add(user)
            await db.commit()

        return user

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def log_audit_event(
        db: AsyncSession,
        user: Optional[User],
        event_type: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request: Optional[Request] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Log security audit event"""

        audit_log = AuditLog(
            organization_id=user.organization_id if user else None,
            user_id=user.id if user else None,
            event_type=event_type,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("User-Agent") if request else None,
            request_id=request.headers.get("X-Request-ID") if request else None,
            success=success,
            error_message=error_message,
            old_values=old_values,
            new_values=new_values,
            metadata=metadata
        )

        db.add(audit_log)
        await db.commit()

# Dependency injection for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    db: AsyncSession = Depends(get_read_db)
) -> User:
    """Get current authenticated user from JWT token"""

    token = credentials.credentials
    token_data = await decode_token(token)

    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    query = select(User).options(
        selectinload(User.organization),
        selectinload(User.roles).selectinload(Role.permissions)
    ).where(User.id == token_data.user_id)

    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User status: {user.status}"
        )

    # Validate session if provided
    if token_data.session_id and redis_client:
        session_data = await redis_client.get(f"session:{token_data.session_id}")
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )

    return user

# Permission checking decorator
class PermissionChecker:
    """Check if user has required permissions"""

    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action

    async def __call__(
        self,
        user: User = Depends(get_current_user)
    ) -> User:
        """Check user permissions"""

        # System admins bypass all checks
        if user.is_system_admin:
            return user

        # Org admins have full access to their org
        if user.is_org_admin and self.resource.startswith("org:"):
            return user

        # Check role-based permissions
        has_permission = False
        for role in user.roles:
            for permission in role.permissions:
                if (permission.resource == self.resource and
                    permission.action == self.action):
                    has_permission = True
                    break
            if has_permission:
                break

        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {self.resource}:{self.action}"
            )

        return user

# API Key authentication
async def get_api_key_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    db: AsyncSession = Depends(get_read_db)
) -> User:
    """Authenticate via API key"""

    api_key = credentials.credentials

    # API keys start with "sk_"
    if not api_key.startswith("sk_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )

    # Hash the API key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Find API key
    query = select(APIKey).options(
        selectinload(APIKey.user).selectinload(User.organization)
    ).where(
        and_(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True
        )
    )

    result = await db.execute(query)
    api_key_obj = result.scalar_one_or_none()

    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired"
        )

    # Update usage
    api_key_obj.last_used_at = datetime.now(UTC)
    api_key_obj.usage_count += 1
    await db.commit()

    return api_key_obj.user if api_key_obj.user else None

# Export commonly used functions
__all__ = [
    "AuthService",
    "AuthConfig",
    "get_current_user",
    "PermissionChecker",
    "get_api_key_user",
    "create_access_token",
    "create_refresh_token",
    "TokenResponse",
    "LoginRequest",
    "OAuth2LoginRequest",
    "init_redis"
]