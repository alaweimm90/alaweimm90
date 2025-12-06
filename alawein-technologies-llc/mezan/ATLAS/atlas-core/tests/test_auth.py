"""
Comprehensive tests for Authentication & Authorization System

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import pytest
import time
import jwt
import secrets
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atlas_core.auth import (
    User,
    Role,
    Permission,
    APIKey,
    Session,
    JWTProvider,
    APIKeyProvider,
    RBACManager,
    AuthManager,
    AuthMethod,
    TokenType
)


class TestUser:
    """Test User model"""

    def test_user_creation(self):
        """Test user creation"""
        user = User(
            id="user-123",
            username="testuser",
            email="test@example.com"
        )

        assert user.id == "user-123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.active is True
        assert user.verified is False

    def test_password_management(self):
        """Test password hashing and verification"""
        user = User(id="user-123", username="testuser")

        # Set password
        user.set_password("SecurePass123!")
        assert user.password_hash is not None
        assert user.password_hash != "SecurePass123!"  # Should be hashed

        # Verify correct password
        assert user.verify_password("SecurePass123!") is True

        # Verify incorrect password
        assert user.verify_password("WrongPassword") is False

    def test_user_permissions(self):
        """Test user role and permission checks"""
        user = User(
            id="user-123",
            username="testuser",
            roles={"admin", "user"},
            permissions={"read:all", "write:all"}
        )

        assert user.has_role("admin") is True
        assert user.has_role("guest") is False

        assert user.has_permission("read:all") is True
        assert user.has_permission("delete:all") is False


class TestJWTProvider:
    """Test JWT authentication provider"""

    def test_create_access_token(self):
        """Test access token creation"""
        provider = JWTProvider(secret_key="test-secret-key")
        user = User(
            id="user-123",
            username="testuser",
            roles={"user", "premium"}
        )

        token = provider.create_access_token(user, scopes=["read", "write"])

        # Decode and verify token
        payload = jwt.decode(
            token,
            "test-secret-key",
            algorithms=["HS256"],
            issuer="MEZAN"
        )

        assert payload["sub"] == "user-123"
        assert payload["username"] == "testuser"
        assert set(payload["roles"]) == {"user", "premium"}
        assert payload["scopes"] == ["read", "write"]
        assert payload["type"] == TokenType.ACCESS.value

    def test_create_refresh_token(self):
        """Test refresh token creation"""
        provider = JWTProvider(secret_key="test-secret-key")
        user = User(id="user-123", username="testuser")

        token = provider.create_refresh_token(user)

        # Decode and verify token
        payload = jwt.decode(
            token,
            "test-secret-key",
            algorithms=["HS256"],
            issuer="MEZAN"
        )

        assert payload["sub"] == "user-123"
        assert payload["type"] == TokenType.REFRESH.value

    @pytest.mark.asyncio
    async def test_validate_token(self):
        """Test token validation"""
        provider = JWTProvider(secret_key="test-secret-key")
        user = User(id="user-123", username="testuser")

        token = provider.create_access_token(user)
        payload = await provider.validate_token(token)

        assert payload is not None
        assert payload["sub"] == "user-123"

    @pytest.mark.asyncio
    async def test_validate_expired_token(self):
        """Test expired token validation"""
        provider = JWTProvider(
            secret_key="test-secret-key",
            access_token_ttl=-1  # Expired immediately
        )
        user = User(id="user-123", username="testuser")

        token = provider.create_access_token(user)
        await asyncio.sleep(0.1)
        payload = await provider.validate_token(token)

        assert payload is None  # Expired token should be rejected

    @pytest.mark.asyncio
    async def test_validate_invalid_token(self):
        """Test invalid token validation"""
        provider = JWTProvider(secret_key="test-secret-key")

        payload = await provider.validate_token("invalid-token")
        assert payload is None


class TestAPIKeyProvider:
    """Test API Key authentication provider"""

    def test_generate_api_key(self):
        """Test API key generation"""
        provider = APIKeyProvider()

        api_key = provider.generate_api_key(
            user_id="user-123",
            name="Test API Key",
            scopes={"read", "write"},
            expires_in=3600,
            allowed_ips=["192.168.1.1"]
        )

        assert api_key.user_id == "user-123"
        assert api_key.name == "Test API Key"
        assert api_key.scopes == {"read", "write"}
        assert api_key.expires_at is not None
        assert api_key.allowed_ips == ["192.168.1.1"]
        assert len(api_key.key) > 0

    def test_api_key_validation(self):
        """Test API key validation"""
        provider = APIKeyProvider()

        api_key = provider.generate_api_key(
            user_id="user-123",
            expires_in=3600
        )

        assert api_key.is_valid() is True

        # Test expired key
        api_key.expires_at = time.time() - 1
        assert api_key.is_expired() is True
        assert api_key.is_valid() is False

        # Test inactive key
        api_key.expires_at = time.time() + 3600
        api_key.active = False
        assert api_key.is_valid() is False

        # Test IP restriction
        api_key.active = True
        api_key.allowed_ips = ["192.168.1.1"]
        assert api_key.is_valid(ip_address="192.168.1.1") is True
        assert api_key.is_valid(ip_address="192.168.1.2") is False

    @pytest.mark.asyncio
    async def test_validate_api_key(self):
        """Test API key validation through provider"""
        provider = APIKeyProvider()

        api_key = provider.generate_api_key(user_id="user-123")
        key_string = api_key.key

        # Validate valid key
        result = await provider.validate_token(key_string)
        assert result is not None
        assert result.user_id == "user-123"
        assert result.usage_count == 1

        # Validate invalid key
        result = await provider.validate_token("invalid-key")
        assert result is None

    def test_revoke_api_key(self):
        """Test API key revocation"""
        provider = APIKeyProvider()

        api_key = provider.generate_api_key(user_id="user-123")
        key_string = api_key.key

        # Revoke key
        success = provider.revoke_key(key_string)
        assert success is True
        assert api_key.active is False

        # Try to revoke non-existent key
        success = provider.revoke_key("non-existent-key")
        assert success is False


class TestRBACManager:
    """Test RBAC Manager"""

    def test_create_user_with_roles(self):
        """Test user creation with roles"""
        rbac = RBACManager()

        user = rbac.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            roles={"user", "premium"}
        )

        assert user.username == "testuser"
        assert user.roles == {"user", "premium"}
        assert len(user.permissions) > 0  # Should have computed permissions

    def test_create_role(self):
        """Test role creation"""
        rbac = RBACManager()

        role = rbac.create_role(
            name="custom_role",
            description="Custom test role",
            permissions={"read:custom", "write:custom"}
        )

        assert role.name == "custom_role"
        assert role.permissions == {"read:custom", "write:custom"}

    def test_role_hierarchy(self):
        """Test role hierarchy"""
        rbac = RBACManager()

        # Create parent role
        parent_role = rbac.create_role(
            name="parent",
            permissions={"read:parent", "write:parent"}
        )

        # Create child role with parent
        child_role = rbac.create_role(
            name="child",
            permissions={"read:child"},
            parent_roles={"parent"}
        )

        # Create user with child role
        user = rbac.create_user(
            username="testuser",
            roles={"child"}
        )

        # User should have permissions from both roles
        assert "read:child" in user.permissions
        assert "read:parent" in user.permissions
        assert "write:parent" in user.permissions

    def test_assign_revoke_role(self):
        """Test role assignment and revocation"""
        rbac = RBACManager()

        user = rbac.create_user(username="testuser", roles={"user"})
        user_id = user.id

        # Assign admin role
        success = rbac.assign_role_to_user(user_id, "admin")
        assert success is True
        assert "admin" in user.roles
        assert "admin:*" in user.permissions

        # Revoke admin role
        success = rbac.revoke_role_from_user(user_id, "admin")
        assert success is True
        assert "admin" not in user.roles
        assert "admin:*" not in user.permissions

    def test_check_permission(self):
        """Test permission checking"""
        rbac = RBACManager()

        user = rbac.create_user(username="testuser", roles={"user"})
        admin = rbac.create_user(username="admin", roles={"admin"})

        # User permissions
        assert rbac.check_permission(user.id, "read:own") is True
        assert rbac.check_permission(user.id, "admin:system") is False

        # Admin permissions (with wildcards)
        assert rbac.check_permission(admin.id, "read:anything") is True
        assert rbac.check_permission(admin.id, "write:anything") is True
        assert rbac.check_permission(admin.id, "admin:system") is True

    def test_create_permission(self):
        """Test permission creation"""
        rbac = RBACManager()

        permission = rbac.create_permission(
            name="custom:action",
            resource="custom_resource",
            action="action",
            description="Custom permission",
            constraints={"max_size": 100}
        )

        assert permission.name == "custom:action"
        assert permission.resource == "custom_resource"
        assert permission.action == "action"
        assert permission.constraints == {"max_size": 100}


class TestAuthManager:
    """Test Auth Manager"""

    @pytest.mark.asyncio
    async def test_basic_authentication(self):
        """Test basic authentication"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create user
        user = auth_manager.rbac.create_user(
            username="testuser",
            password="SecurePass123!",
            roles={"user"}
        )

        # Authenticate with correct credentials
        session = await auth_manager.authenticate(
            method=AuthMethod.BASIC,
            credentials={
                "username": "testuser",
                "password": "SecurePass123!"
            },
            ip_address="192.168.1.1"
        )

        assert session is not None
        assert session.user_id == user.id
        assert session.access_token is not None
        assert session.refresh_token is not None

        # Authenticate with wrong password
        session = await auth_manager.authenticate(
            method=AuthMethod.BASIC,
            credentials={
                "username": "testuser",
                "password": "WrongPassword"
            }
        )

        assert session is None

    @pytest.mark.asyncio
    async def test_jwt_authentication(self):
        """Test JWT authentication"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create user and get token
        user = auth_manager.rbac.create_user(username="testuser")
        token = auth_manager.jwt_provider.create_access_token(user)

        # Authenticate with token
        session = await auth_manager.authenticate(
            method=AuthMethod.JWT,
            credentials={"token": token}
        )

        assert session is not None
        assert session.user_id == user.id

    @pytest.mark.asyncio
    async def test_api_key_authentication(self):
        """Test API key authentication"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create user and API key
        user = auth_manager.rbac.create_user(username="testuser")
        api_key = auth_manager.api_key_provider.generate_api_key(user.id)

        # Authenticate with API key
        session = await auth_manager.authenticate(
            method=AuthMethod.API_KEY,
            credentials={"api_key": api_key.key}
        )

        assert session is not None
        assert session.user_id == user.id

    @pytest.mark.asyncio
    async def test_session_validation(self):
        """Test session validation"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create session
        user = auth_manager.rbac.create_user(username="testuser", password="Pass123!")
        session = await auth_manager.authenticate(
            method=AuthMethod.BASIC,
            credentials={"username": "testuser", "password": "Pass123!"}
        )

        # Validate valid session
        valid_session = await auth_manager.validate_session(session.id)
        assert valid_session is not None
        assert valid_session.id == session.id

        # Validate non-existent session
        invalid_session = await auth_manager.validate_session("invalid-session-id")
        assert invalid_session is None

    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test token refresh"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create initial session
        user = auth_manager.rbac.create_user(username="testuser", password="Pass123!")
        session = await auth_manager.authenticate(
            method=AuthMethod.BASIC,
            credentials={"username": "testuser", "password": "Pass123!"}
        )

        # Refresh token
        new_session = await auth_manager.refresh_token(
            refresh_token=session.refresh_token
        )

        assert new_session is not None
        assert new_session.user_id == user.id
        assert new_session.access_token != session.access_token  # New token

    def test_logout(self):
        """Test logout functionality"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create mock session
        session = Session(
            id="session-123",
            user_id="user-123",
            access_token="access-token",
            refresh_token="refresh-token",
            expires_at=time.time() + 3600
        )
        auth_manager.sessions[session.id] = session

        # Logout
        success = auth_manager.logout(session.id)
        assert success is True
        assert session.id not in auth_manager.sessions
        assert "access-token" in auth_manager.token_blacklist
        assert "refresh-token" in auth_manager.token_blacklist

    @pytest.mark.asyncio
    async def test_failed_login_rate_limiting(self):
        """Test failed login attempt tracking"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Make multiple failed attempts
        for i in range(5):
            await auth_manager.authenticate(
                method=AuthMethod.BASIC,
                credentials={"username": "testuser", "password": "wrong"},
                ip_address="192.168.1.1"
            )

        # Should be blocked now
        blocked = await auth_manager._check_failed_attempts("testuser", "192.168.1.1")
        assert blocked is True

    def test_mfa_setup(self):
        """Test MFA setup"""
        auth_manager = AuthManager(jwt_secret="test-secret")

        # Create user
        user = auth_manager.rbac.create_user(username="testuser")

        # Enable MFA
        provisioning_uri = auth_manager.enable_mfa(user.id)

        assert provisioning_uri is not None
        assert user.mfa_enabled is True
        assert user.mfa_secret is not None

    def test_audit_logging(self):
        """Test audit logging"""
        auth_manager = AuthManager(jwt_secret="test-secret", enable_audit=True)

        # Generate some audit logs
        auth_manager._audit(
            action="login",
            resource="authentication",
            result="success",
            ip_address="192.168.1.1",
            details={"user_id": "user-123"}
        )

        auth_manager._audit(
            action="permission_check",
            resource="api",
            result="denied",
            ip_address="192.168.1.2",
            details={"permission": "admin:system"}
        )

        # Get audit logs
        logs = auth_manager.get_audit_logs()
        assert len(logs) == 2

        # Filter logs
        login_logs = auth_manager.get_audit_logs(action="login")
        assert len(login_logs) == 1
        assert login_logs[0].action == "login"


class TestSession:
    """Test Session model"""

    def test_session_creation(self):
        """Test session creation"""
        session = Session(
            id="session-123",
            user_id="user-123",
            access_token="access-token",
            refresh_token="refresh-token",
            expires_at=time.time() + 3600,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        assert session.id == "session-123"
        assert session.user_id == "user-123"
        assert session.ip_address == "192.168.1.1"
        assert session.is_expired() is False

    def test_session_expiry(self):
        """Test session expiry"""
        session = Session(
            id="session-123",
            user_id="user-123",
            access_token="token",
            expires_at=time.time() - 1  # Expired
        )

        assert session.is_expired() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])