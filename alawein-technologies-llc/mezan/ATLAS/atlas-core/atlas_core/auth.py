"""
Enterprise Authentication & Authorization System

Production-grade auth system with:
- JWT authentication
- API key management
- OAuth2 client credentials flow
- RBAC (Role-Based Access Control)
- Permission system
- Token refresh mechanism
- Audit logging
- MFA support

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple

import jwt
from passlib.context import CryptContext
from passlib.totp import TOTP

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthMethod(Enum):
    """Authentication methods"""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    SESSION = "session"


class TokenType(Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: Optional[str] = None
    password_hash: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    api_keys: List[str] = field(default_factory=list)
    active: bool = True
    verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if user has role"""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has permission"""
        return permission in self.permissions

    def verify_password(self, password: str) -> bool:
        """Verify password"""
        if not self.password_hash:
            return False
        return pwd_context.verify(password, self.password_hash)

    def set_password(self, password: str) -> None:
        """Set password"""
        self.password_hash = pwd_context.hash(password)
        self.updated_at = time.time()


@dataclass
class Role:
    """Role model"""
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # Role hierarchy
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        """Check if role has permission"""
        return permission in self.permissions


@dataclass
class Permission:
    """Permission model"""
    name: str
    resource: str
    action: str
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class APIKey:
    """API Key model"""
    id: str
    key: str
    user_id: str
    name: str = ""
    scopes: Set[str] = field(default_factory=set)
    expires_at: Optional[float] = None
    active: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    rate_limit: Optional[Dict[str, Any]] = None
    allowed_ips: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_valid(self, ip_address: Optional[str] = None) -> bool:
        """Check if key is valid"""
        if not self.active:
            return False
        if self.is_expired():
            return False
        if self.allowed_ips and ip_address not in self.allowed_ips:
            return False
        return True


@dataclass
class Session:
    """Session model"""
    id: str
    user_id: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float = 0
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return time.time() > self.expires_at


@dataclass
class AuditLog:
    """Audit log entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    action: str = ""
    resource: str = ""
    result: str = ""  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class AuthProvider(ABC):
    """Abstract authentication provider"""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with credentials"""
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate token and return user"""
        pass


class JWTProvider(AuthProvider):
    """JWT authentication provider"""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_ttl: int = 3600,  # 1 hour
        refresh_token_ttl: int = 86400 * 7,  # 7 days
        issuer: str = "MEZAN"
    ):
        """Initialize JWT provider"""
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl
        self.issuer = issuer

    def create_access_token(self, user: User, scopes: Optional[List[str]] = None) -> str:
        """Create access token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(seconds=self.access_token_ttl)

        payload = {
            "sub": user.id,
            "username": user.username,
            "roles": list(user.roles),
            "scopes": scopes or [],
            "type": TokenType.ACCESS.value,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "jti": str(uuid.uuid4()),  # JWT ID for revocation
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create refresh token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(seconds=self.refresh_token_ttl)

        payload = {
            "sub": user.id,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "jti": str(uuid.uuid4()),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate with username/password"""
        # This would typically query a database
        # For now, return None (implement in subclass)
        return None

    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        payload = self.validate_token(refresh_token)
        if not payload or payload.get("type") != TokenType.REFRESH.value:
            return None

        # Create new access token
        # (would need to fetch user from database)
        return None


class OAuth2Provider(AuthProvider):
    """OAuth2 authentication provider (client credentials flow)"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scopes: Optional[List[str]] = None
    ):
        """Initialize OAuth2 provider"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scopes = scopes or []
        self._token_cache: Dict[str, Tuple[str, float]] = {}

    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate using OAuth2"""
        # Implement OAuth2 client credentials flow
        # This would make HTTP request to token_url
        return None

    async def validate_token(self, token: str) -> Optional[User]:
        """Validate OAuth2 token"""
        # This would validate with OAuth2 server
        return None


class APIKeyProvider(AuthProvider):
    """API key authentication provider"""

    def __init__(self, storage: Optional[Any] = None):
        """Initialize API key provider"""
        self.storage = storage
        self.keys: Dict[str, APIKey] = {}  # In-memory storage

    def generate_api_key(
        self,
        user_id: str,
        name: str = "",
        scopes: Optional[Set[str]] = None,
        expires_in: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None
    ) -> APIKey:
        """Generate new API key"""
        api_key = APIKey(
            id=str(uuid.uuid4()),
            key=secrets.token_urlsafe(32),
            user_id=user_id,
            name=name,
            scopes=scopes or set(),
            expires_at=time.time() + expires_in if expires_in else None,
            allowed_ips=allowed_ips or []
        )

        self.keys[api_key.key] = api_key
        return api_key

    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate with API key"""
        api_key = credentials.get("api_key")
        if not api_key:
            return None

        key_obj = self.keys.get(api_key)
        if not key_obj or not key_obj.is_valid(credentials.get("ip_address")):
            return None

        # Update usage stats
        key_obj.last_used = time.time()
        key_obj.usage_count += 1

        # Return user (would fetch from database)
        return None

    async def validate_token(self, token: str) -> Optional[APIKey]:
        """Validate API key"""
        key_obj = self.keys.get(token)
        if key_obj and key_obj.is_valid():
            key_obj.last_used = time.time()
            key_obj.usage_count += 1
            return key_obj
        return None

    def revoke_key(self, key: str) -> bool:
        """Revoke API key"""
        if key in self.keys:
            self.keys[key].active = False
            return True
        return False


class RBACManager:
    """Role-Based Access Control manager"""

    def __init__(self, storage: Optional[Any] = None):
        """Initialize RBAC manager"""
        self.storage = storage
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}

        # Initialize default roles
        self._init_default_roles()

    def _init_default_roles(self):
        """Initialize default roles"""
        # Admin role
        self.roles["admin"] = Role(
            name="admin",
            description="Administrator with full access",
            permissions={
                "read:*", "write:*", "delete:*", "admin:*"
            }
        )

        # User role
        self.roles["user"] = Role(
            name="user",
            description="Regular user",
            permissions={
                "read:own", "write:own", "delete:own"
            }
        )

        # Guest role
        self.roles["guest"] = Role(
            name="guest",
            description="Guest with limited access",
            permissions={
                "read:public"
            }
        )

    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        roles: Optional[Set[str]] = None
    ) -> User:
        """Create new user"""
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            roles=roles or {"user"}
        )

        if password:
            user.set_password(password)

        # Compute permissions from roles
        self._compute_user_permissions(user)

        self.users[user.id] = user
        return user

    def _compute_user_permissions(self, user: User) -> None:
        """Compute user permissions from roles"""
        permissions = set()

        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)

                # Handle role hierarchy
                for parent_role_name in role.parent_roles:
                    parent_role = self.roles.get(parent_role_name)
                    if parent_role:
                        permissions.update(parent_role.permissions)

        user.permissions = permissions

    def create_role(
        self,
        name: str,
        description: str = "",
        permissions: Optional[Set[str]] = None,
        parent_roles: Optional[Set[str]] = None
    ) -> Role:
        """Create new role"""
        role = Role(
            name=name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set()
        )

        self.roles[name] = role

        # Update existing users with this role
        for user in self.users.values():
            if name in user.roles:
                self._compute_user_permissions(user)

        return role

    def create_permission(
        self,
        name: str,
        resource: str,
        action: str,
        description: str = "",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Permission:
        """Create new permission"""
        permission = Permission(
            name=name,
            resource=resource,
            action=action,
            description=description,
            constraints=constraints or {}
        )

        self.permissions[name] = permission
        return permission

    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user"""
        user = self.users.get(user_id)
        role = self.roles.get(role_name)

        if user and role:
            user.roles.add(role_name)
            self._compute_user_permissions(user)
            user.updated_at = time.time()
            return True

        return False

    def revoke_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user"""
        user = self.users.get(user_id)

        if user and role_name in user.roles:
            user.roles.discard(role_name)
            self._compute_user_permissions(user)
            user.updated_at = time.time()
            return True

        return False

    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if user has permission"""
        user = self.users.get(user_id)
        if not user or not user.active:
            return False

        # Check exact permission
        if permission in user.permissions:
            return True

        # Check wildcard permissions
        parts = permission.split(":")
        if len(parts) == 2:
            action, resource_name = parts

            # Check action:* permission
            if f"{action}:*" in user.permissions:
                return True

            # Check *:resource permission
            if f"*:{resource_name}" in user.permissions:
                return True

            # Check admin:* permission
            if "admin:*" in user.permissions:
                return True

        return False

    def load_config(self, config_path: str) -> None:
        """Load RBAC configuration from YAML file"""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load permissions
        for perm_config in config.get('permissions', []):
            self.create_permission(**perm_config)

        # Load roles
        for role_config in config.get('roles', []):
            self.create_role(**role_config)

        # Load users
        for user_config in config.get('users', []):
            self.create_user(**user_config)


class AuthManager:
    """Main authentication and authorization manager"""

    def __init__(
        self,
        jwt_secret: str,
        redis_client: Optional[Any] = None,
        enable_audit: bool = True
    ):
        """Initialize auth manager"""
        self.jwt_provider = JWTProvider(jwt_secret)
        self.api_key_provider = APIKeyProvider()
        self.rbac = RBACManager()
        self.redis_client = redis_client
        self.enable_audit = enable_audit

        # Session storage
        self.sessions: Dict[str, Session] = {}

        # Audit logs
        self.audit_logs: List[AuditLog] = []

        # Token blacklist (for revocation)
        self.token_blacklist: Set[str] = set()

        # Failed login attempts (for rate limiting)
        self.failed_attempts: Dict[str, List[float]] = {}

    async def authenticate(
        self,
        method: AuthMethod,
        credentials: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """Authenticate user and create session"""
        user = None

        # Check for too many failed attempts
        if await self._check_failed_attempts(credentials.get("username"), ip_address):
            self._audit("authentication", "login", "blocked", ip_address, user_agent, {
                "reason": "too_many_failed_attempts"
            })
            return None

        # Authenticate based on method
        if method == AuthMethod.JWT:
            token = credentials.get("token")
            if token and token not in self.token_blacklist:
                payload = await self.jwt_provider.validate_token(token)
                if payload:
                    user = self.rbac.users.get(payload.get("sub"))

        elif method == AuthMethod.API_KEY:
            key_obj = await self.api_key_provider.validate_token(
                credentials.get("api_key")
            )
            if key_obj:
                user = self.rbac.users.get(key_obj.user_id)

        elif method == AuthMethod.BASIC:
            username = credentials.get("username")
            password = credentials.get("password")

            # Find user
            for u in self.rbac.users.values():
                if u.username == username:
                    # Check MFA if enabled
                    if u.mfa_enabled:
                        totp_code = credentials.get("totp_code")
                        if not self._verify_totp(u, totp_code):
                            self._audit("authentication", "login", "failure", ip_address, user_agent, {
                                "reason": "invalid_mfa",
                                "username": username
                            })
                            await self._record_failed_attempt(username, ip_address)
                            return None

                    # Verify password
                    if u.verify_password(password):
                        user = u
                        break

        # Handle authentication result
        if user and user.active:
            # Create session
            session = Session(
                id=str(uuid.uuid4()),
                user_id=user.id,
                access_token=self.jwt_provider.create_access_token(user),
                refresh_token=self.jwt_provider.create_refresh_token(user),
                expires_at=time.time() + self.jwt_provider.access_token_ttl,
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.sessions[session.id] = session

            # Update user
            user.last_login = time.time()

            # Audit log
            self._audit("authentication", "login", "success", ip_address, user_agent, {
                "user_id": user.id,
                "method": method.value
            })

            return session

        else:
            # Record failed attempt
            username = credentials.get("username", credentials.get("api_key", "unknown"))
            await self._record_failed_attempt(username, ip_address)

            # Audit log
            self._audit("authentication", "login", "failure", ip_address, user_agent, {
                "method": method.value
            })

            return None

    async def validate_session(
        self,
        session_id: str,
        update_activity: bool = True
    ) -> Optional[Session]:
        """Validate session"""
        session = self.sessions.get(session_id)

        if session and not session.is_expired():
            if update_activity:
                session.last_activity = time.time()
            return session

        return None

    async def refresh_token(
        self,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """Refresh access token"""
        payload = await self.jwt_provider.validate_token(refresh_token)

        if payload and payload.get("type") == TokenType.REFRESH.value:
            user = self.rbac.users.get(payload.get("sub"))

            if user and user.active:
                # Create new session
                session = Session(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    access_token=self.jwt_provider.create_access_token(user),
                    refresh_token=self.jwt_provider.create_refresh_token(user),
                    expires_at=time.time() + self.jwt_provider.access_token_ttl,
                    ip_address=ip_address,
                    user_agent=user_agent
                )

                self.sessions[session.id] = session

                # Audit log
                self._audit("authentication", "token_refresh", "success", ip_address, user_agent, {
                    "user_id": user.id
                })

                return session

        return None

    def logout(self, session_id: str) -> bool:
        """Logout and invalidate session"""
        session = self.sessions.get(session_id)

        if session:
            # Add tokens to blacklist
            self.token_blacklist.add(session.access_token)
            if session.refresh_token:
                self.token_blacklist.add(session.refresh_token)

            # Remove session
            del self.sessions[session_id]

            # Audit log
            self._audit("authentication", "logout", "success", session.ip_address, session.user_agent, {
                "user_id": session.user_id
            })

            return True

        return False

    def revoke_token(self, token: str) -> None:
        """Revoke a token"""
        self.token_blacklist.add(token)

    def enable_mfa(self, user_id: str) -> Optional[str]:
        """Enable MFA for user"""
        user = self.rbac.users.get(user_id)

        if user:
            # Generate TOTP secret
            secret = secrets.token_urlsafe(32)
            user.mfa_enabled = True
            user.mfa_secret = secret
            user.updated_at = time.time()

            # Return provisioning URI for QR code
            totp = TOTP(secret)
            return totp.generate_provisioning_uri(
                user.email or user.username,
                "MEZAN"
            )

        return None

    def _verify_totp(self, user: User, code: str) -> bool:
        """Verify TOTP code"""
        if not user.mfa_enabled or not user.mfa_secret:
            return False

        totp = TOTP(user.mfa_secret)
        return totp.verify(code, window=1)

    async def _check_failed_attempts(
        self,
        username: Optional[str],
        ip_address: Optional[str]
    ) -> bool:
        """Check if too many failed attempts"""
        if not username and not ip_address:
            return False

        # Check by username
        if username:
            attempts = self.failed_attempts.get(f"user:{username}", [])
            recent_attempts = [t for t in attempts if time.time() - t < 300]  # Last 5 minutes

            if len(recent_attempts) >= 5:
                return True

        # Check by IP
        if ip_address:
            attempts = self.failed_attempts.get(f"ip:{ip_address}", [])
            recent_attempts = [t for t in attempts if time.time() - t < 300]

            if len(recent_attempts) >= 10:
                return True

        return False

    async def _record_failed_attempt(
        self,
        username: Optional[str],
        ip_address: Optional[str]
    ) -> None:
        """Record failed login attempt"""
        current_time = time.time()

        if username:
            key = f"user:{username}"
            if key not in self.failed_attempts:
                self.failed_attempts[key] = []
            self.failed_attempts[key].append(current_time)

        if ip_address:
            key = f"ip:{ip_address}"
            if key not in self.failed_attempts:
                self.failed_attempts[key] = []
            self.failed_attempts[key].append(current_time)

    def _audit(
        self,
        action: str,
        resource: str,
        result: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create audit log entry"""
        if not self.enable_audit:
            return

        log = AuditLog(
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )

        self.audit_logs.append(log)

        # Trim old logs (keep last 10000)
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs with filters"""
        logs = self.audit_logs

        # Apply filters
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]

        if action:
            logs = [log for log in logs if log.action == action]

        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]

        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.timestamp, reverse=True)

        return logs[:limit]