"""
Authentication & Authorization Integration Tests for ORCHEX

Tests comprehensive auth flows including SSO, MFA, RBAC, and API keys.
"""

import pytest
import asyncio
import time
import jwt
import secrets
import hashlib
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pyotp
import bcrypt

from atlas_core.auth import (
    AuthenticationManager,
    AuthorizationManager,
    TokenManager,
    UserManager,
    RoleManager,
    PermissionManager,
    SessionManager,
    AuditLogger,
    MFAProvider,
    SSOProvider,
)
from atlas_core.engine import ATLASEngine


class TestAuthIntegration:
    """Test authentication and authorization integration."""

    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager."""
        return AuthenticationManager(
            providers=["local", "ldap", "oauth", "saml"],
            mfa_required=True,
            session_timeout=3600,
            password_policy={
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_digits": True,
                "require_special": True,
                "max_age_days": 90,
            },
        )

    @pytest.fixture
    def authz_manager(self):
        """Create authorization manager."""
        return AuthorizationManager(
            rbac_enabled=True,
            abac_enabled=True,
            policy_engine="opa",
            cache_ttl=300,
        )

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return {
            "username": "test_researcher",
            "email": "researcher@example.com",
            "password": "SecureP@ssw0rd123!",
            "roles": ["researcher", "data_analyst"],
            "attributes": {
                "department": "AI Research",
                "clearance_level": "confidential",
                "projects": ["project_a", "project_b"],
            },
        }

    @pytest.mark.integration
    async def test_user_registration_flow(self, auth_manager, test_user):
        """Test complete user registration flow."""
        # Register user
        registration_result = await auth_manager.register_user(
            username=test_user["username"],
            email=test_user["email"],
            password=test_user["password"],
        )

        assert registration_result["success"] is True
        assert "user_id" in registration_result
        assert "verification_token" in registration_result

        # Verify email
        verification_result = await auth_manager.verify_email(
            registration_result["verification_token"]
        )
        assert verification_result["success"] is True

        # Setup MFA
        mfa_setup = await auth_manager.setup_mfa(
            user_id=registration_result["user_id"], method="totp"
        )
        assert "secret" in mfa_setup
        assert "qr_code" in mfa_setup
        assert "backup_codes" in mfa_setup

    @pytest.mark.integration
    async def test_multi_factor_authentication(self, auth_manager, test_user):
        """Test MFA with various methods."""
        # Create user with MFA
        user = await auth_manager.create_user(test_user)

        # Setup TOTP
        totp_secret = pyotp.random_base32()
        await auth_manager.enable_mfa(
            user_id=user["id"], method="totp", secret=totp_secret
        )

        # Test successful MFA
        totp = pyotp.TOTP(totp_secret)
        current_otp = totp.now()

        auth_result = await auth_manager.authenticate(
            username=test_user["username"],
            password=test_user["password"],
            mfa_code=current_otp,
        )

        assert auth_result["success"] is True
        assert "access_token" in auth_result
        assert "refresh_token" in auth_result

        # Test failed MFA
        auth_failed = await auth_manager.authenticate(
            username=test_user["username"],
            password=test_user["password"],
            mfa_code="000000",  # Wrong code
        )

        assert auth_failed["success"] is False
        assert auth_failed["error"] == "invalid_mfa_code"

    @pytest.mark.integration
    async def test_role_based_access_control(self, authz_manager):
        """Test RBAC implementation."""
        # Define roles
        roles = [
            {
                "name": "admin",
                "permissions": ["workflow.*", "user.*", "system.*"],
                "description": "Full system access",
            },
            {
                "name": "researcher",
                "permissions": ["workflow.create", "workflow.read", "agent.invoke"],
                "description": "Research workflow access",
            },
            {
                "name": "viewer",
                "permissions": ["workflow.read", "report.read"],
                "description": "Read-only access",
            },
        ]

        for role in roles:
            await authz_manager.create_role(role)

        # Test permission checks
        user_context = {"user_id": "user_123", "roles": ["researcher"]}

        # Should be allowed
        assert await authz_manager.check_permission(
            user_context, "workflow.create"
        ) is True

        # Should be denied
        assert await authz_manager.check_permission(
            user_context, "user.delete"
        ) is False

        # Test role hierarchy
        await authz_manager.set_role_hierarchy({"admin": ["researcher", "viewer"]})

        admin_context = {"user_id": "admin_123", "roles": ["admin"]}
        assert await authz_manager.check_permission(
            admin_context, "workflow.create"
        ) is True  # Inherited from researcher

    @pytest.mark.integration
    async def test_attribute_based_access_control(self, authz_manager):
        """Test ABAC implementation."""
        # Define ABAC policies
        policies = [
            {
                "name": "project_access",
                "rule": "user.project in resource.allowed_projects",
                "effect": "allow",
            },
            {
                "name": "clearance_required",
                "rule": "user.clearance_level >= resource.required_clearance",
                "effect": "allow",
            },
            {
                "name": "time_restriction",
                "rule": "current_time between resource.allowed_hours",
                "effect": "allow",
            },
        ]

        for policy in policies:
            await authz_manager.add_policy(policy)

        # Test attribute-based access
        user_attributes = {
            "project": "project_a",
            "clearance_level": "secret",
            "department": "research",
        }

        resource_attributes = {
            "allowed_projects": ["project_a", "project_b"],
            "required_clearance": "confidential",
            "owner": "research",
        }

        access_result = await authz_manager.evaluate_access(
            user_attributes, resource_attributes, action="read"
        )

        assert access_result["allowed"] is True
        assert "project_access" in access_result["matched_policies"]

    @pytest.mark.integration
    async def test_oauth_sso_flow(self, auth_manager):
        """Test OAuth 2.0 SSO integration."""
        # Configure OAuth provider
        oauth_config = {
            "provider": "google",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "redirect_uri": "http://localhost:8080/auth/callback",
            "scopes": ["openid", "profile", "email"],
        }

        sso_provider = await auth_manager.configure_sso(oauth_config)

        # Generate authorization URL
        auth_url = await sso_provider.get_authorization_url(state="test_state")
        assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "client_id=test_client_id" in auth_url

        # Simulate callback with code
        mock_code = "test_authorization_code"
        with patch.object(sso_provider, "exchange_code") as mock_exchange:
            mock_exchange.return_value = {
                "access_token": "test_access_token",
                "id_token": "test_id_token",
                "user_info": {
                    "sub": "google_user_123",
                    "email": "user@example.com",
                    "name": "Test User",
                },
            }

            callback_result = await auth_manager.handle_sso_callback(
                provider="google", code=mock_code, state="test_state"
            )

            assert callback_result["success"] is True
            assert "user_id" in callback_result
            assert "session_token" in callback_result

    @pytest.mark.integration
    async def test_saml_sso_flow(self, auth_manager):
        """Test SAML 2.0 SSO integration."""
        # Configure SAML provider
        saml_config = {
            "entity_id": "http://localhost:8080",
            "sso_url": "https://idp.example.com/sso",
            "x509_cert": "test_certificate",
            "attribute_mapping": {
                "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
                "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            },
        }

        saml_provider = await auth_manager.configure_saml(saml_config)

        # Generate SAML request
        saml_request = await saml_provider.create_auth_request()
        assert saml_request["SAMLRequest"] is not None
        assert saml_request["RelayState"] is not None

        # Simulate SAML response
        mock_saml_response = "base64_encoded_saml_response"
        with patch.object(saml_provider, "parse_response") as mock_parse:
            mock_parse.return_value = {
                "name_id": "saml_user_123",
                "attributes": {"email": "user@example.com", "name": "SAML User"},
                "session_index": "session_123",
            }

            response_result = await auth_manager.handle_saml_response(
                saml_response=mock_saml_response
            )

            assert response_result["success"] is True
            assert "user_id" in response_result
            assert "session_token" in response_result

    @pytest.mark.integration
    async def test_api_key_authentication(self, auth_manager):
        """Test API key authentication."""
        # Create API key
        api_key_result = await auth_manager.create_api_key(
            user_id="user_123",
            name="CI/CD Pipeline",
            scopes=["workflow.create", "workflow.read"],
            expires_in_days=30,
        )

        assert "key" in api_key_result
        assert "key_id" in api_key_result
        assert len(api_key_result["key"]) >= 32

        # Authenticate with API key
        auth_result = await auth_manager.authenticate_api_key(api_key_result["key"])
        assert auth_result["success"] is True
        assert auth_result["user_id"] == "user_123"
        assert "workflow.create" in auth_result["scopes"]

        # Revoke API key
        revoke_result = await auth_manager.revoke_api_key(api_key_result["key_id"])
        assert revoke_result["success"] is True

        # Try to use revoked key
        auth_failed = await auth_manager.authenticate_api_key(api_key_result["key"])
        assert auth_failed["success"] is False

    @pytest.mark.integration
    async def test_session_management(self, auth_manager):
        """Test session creation, validation, and revocation."""
        session_manager = SessionManager(
            storage="redis",
            session_timeout=3600,
            sliding_expiration=True,
            max_concurrent_sessions=5,
        )

        # Create session
        session = await session_manager.create_session(
            user_id="user_123",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
            metadata={"login_method": "password"},
        )

        assert "session_id" in session
        assert "expires_at" in session

        # Validate session
        validation_result = await session_manager.validate_session(session["session_id"])
        assert validation_result["valid"] is True
        assert validation_result["user_id"] == "user_123"

        # Update session activity (sliding expiration)
        await session_manager.touch_session(session["session_id"])

        # List user sessions
        user_sessions = await session_manager.get_user_sessions("user_123")
        assert len(user_sessions) == 1
        assert user_sessions[0]["session_id"] == session["session_id"]

        # Revoke session
        await session_manager.revoke_session(session["session_id"])
        validation_after_revoke = await session_manager.validate_session(
            session["session_id"]
        )
        assert validation_after_revoke["valid"] is False

    @pytest.mark.integration
    async def test_password_management(self, auth_manager):
        """Test password policies and management."""
        # Test password validation
        weak_passwords = [
            "short",  # Too short
            "alllowercase123",  # No uppercase
            "ALLUPPERCASE123",  # No lowercase
            "NoNumbers!",  # No digits
            "NoSpecialChars123",  # No special characters
        ]

        for password in weak_passwords:
            validation = await auth_manager.validate_password(password)
            assert validation["valid"] is False
            assert len(validation["errors"]) > 0

        # Test password history
        user_id = "user_123"
        old_passwords = ["OldP@ssw0rd1!", "OldP@ssw0rd2!", "OldP@ssw0rd3!"]

        for pwd in old_passwords:
            await auth_manager.add_password_history(user_id, pwd)

        # Try to reuse old password
        reuse_check = await auth_manager.check_password_reuse(
            user_id, "OldP@ssw0rd2!"
        )
        assert reuse_check["reused"] is True

        # Test password reset flow
        reset_token = await auth_manager.initiate_password_reset("user@example.com")
        assert reset_token is not None

        reset_result = await auth_manager.reset_password(
            reset_token, new_password="NewSecureP@ss123!"
        )
        assert reset_result["success"] is True

    @pytest.mark.integration
    async def test_audit_logging(self, auth_manager):
        """Test security audit logging."""
        audit_logger = AuditLogger(
            storage="elasticsearch",
            retention_days=90,
            real_time_alerts=True,
        )

        # Log various auth events
        events = [
            {
                "event_type": "login_success",
                "user_id": "user_123",
                "ip": "192.168.1.100",
                "timestamp": datetime.now(),
            },
            {
                "event_type": "login_failure",
                "username": "attacker",
                "ip": "10.0.0.1",
                "reason": "invalid_credentials",
                "timestamp": datetime.now(),
            },
            {
                "event_type": "permission_denied",
                "user_id": "user_456",
                "resource": "admin_panel",
                "action": "delete",
                "timestamp": datetime.now(),
            },
        ]

        for event in events:
            await audit_logger.log(event)

        # Query audit logs
        failed_logins = await audit_logger.query(
            filters={"event_type": "login_failure"}, time_range="last_24h"
        )
        assert len(failed_logins) > 0

        # Detect anomalies
        anomalies = await audit_logger.detect_anomalies(
            user_id="user_123", lookback_hours=24
        )

        # Generate compliance report
        report = await audit_logger.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            include_types=["login_success", "login_failure", "permission_denied"],
        )
        assert "summary" in report
        assert "detailed_events" in report

    @pytest.mark.integration
    async def test_token_management(self, auth_manager):
        """Test JWT token creation, validation, and refresh."""
        token_manager = TokenManager(
            secret_key="test_secret_key",
            algorithm="HS256",
            access_token_ttl=900,  # 15 minutes
            refresh_token_ttl=86400,  # 24 hours
        )

        # Create tokens
        tokens = await token_manager.create_token_pair(
            user_id="user_123",
            roles=["researcher"],
            additional_claims={"department": "AI"},
        )

        assert "access_token" in tokens
        assert "refresh_token" in tokens

        # Validate access token
        validation = await token_manager.validate_token(tokens["access_token"])
        assert validation["valid"] is True
        assert validation["user_id"] == "user_123"
        assert "researcher" in validation["roles"]

        # Refresh tokens
        new_tokens = await token_manager.refresh_tokens(tokens["refresh_token"])
        assert new_tokens["access_token"] != tokens["access_token"]
        assert new_tokens["refresh_token"] != tokens["refresh_token"]

        # Revoke refresh token
        await token_manager.revoke_refresh_token(tokens["refresh_token"])

        # Try to use revoked refresh token
        with pytest.raises(InvalidTokenError):
            await token_manager.refresh_tokens(tokens["refresh_token"])

    @pytest.mark.integration
    async def test_permission_inheritance(self, authz_manager):
        """Test complex permission inheritance and delegation."""
        # Create permission hierarchy
        await authz_manager.create_permission_hierarchy({
            "system": ["workflow", "user", "report"],
            "workflow": ["workflow.create", "workflow.read", "workflow.update", "workflow.delete"],
            "user": ["user.create", "user.read", "user.update", "user.delete"],
            "report": ["report.create", "report.read", "report.export"],
        })

        # Test wildcard permissions
        admin_context = {"user_id": "admin", "permissions": ["system.*"]}
        assert await authz_manager.check_permission(admin_context, "workflow.create") is True
        assert await authz_manager.check_permission(admin_context, "user.delete") is True

        # Test permission delegation
        await authz_manager.delegate_permission(
            from_user="admin",
            to_user="delegate_user",
            permission="workflow.create",
            duration=3600,
            constraints={"max_uses": 5},
        )

        delegate_context = {"user_id": "delegate_user"}
        assert await authz_manager.check_permission(delegate_context, "workflow.create") is True