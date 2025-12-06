"""
API Gateway Integration Tests for ORCHEX

Tests API gateway integration, routing, authentication, and rate limiting.
"""

import pytest
import asyncio
import json
import time
import jwt
import hashlib
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from fastapi.testclient import TestClient
import httpx

from atlas_core.api_gateway import (
    ATLASGateway,
    RouteManager,
    AuthenticationMiddleware,
    RateLimiter,
    RequestValidator,
    ResponseCache,
)
from atlas_core.engine import ATLASEngine


class TestAPIGatewayIntegration:
    """Test API gateway integration and functionality."""

    @pytest.fixture
    def gateway(self):
        """Create API gateway instance."""
        return ATLASGateway(
            host="0.0.0.0",
            port=8080,
            auth_enabled=True,
            rate_limiting=True,
            caching=True,
        )

    @pytest.fixture
    def test_client(self, gateway):
        """Create test client for API."""
        return TestClient(gateway.app)

    @pytest.fixture
    def auth_token(self):
        """Generate test authentication token."""
        payload = {
            "sub": "test_user",
            "exp": int(time.time()) + 3600,
            "roles": ["researcher", "admin"],
        }
        secret = "test_secret_key"
        return jwt.encode(payload, secret, algorithm="HS256")

    @pytest.mark.integration
    async def test_api_endpoints_registration(self, gateway):
        """Test automatic endpoint registration."""
        # Register workflow endpoints
        endpoints = await gateway.register_workflow_endpoints()

        # Verify endpoints registered
        assert "/api/v1/workflow/create" in endpoints
        assert "/api/v1/workflow/{id}/status" in endpoints
        assert "/api/v1/workflow/{id}/cancel" in endpoints
        assert "/api/v1/workflow/list" in endpoints

        # Register agent endpoints
        agent_endpoints = await gateway.register_agent_endpoints()
        assert "/api/v1/agents/list" in agent_endpoints
        assert "/api/v1/agents/{name}/invoke" in agent_endpoints

    @pytest.mark.integration
    def test_request_routing(self, test_client, auth_token):
        """Test request routing to appropriate handlers."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Test workflow creation
        response = test_client.post(
            "/api/v1/workflow/create",
            json={"type": "research", "topic": "AI Safety"},
            headers=headers,
        )
        assert response.status_code == 201
        assert "workflow_id" in response.json()

        # Test status check
        workflow_id = response.json()["workflow_id"]
        status_response = test_client.get(
            f"/api/v1/workflow/{workflow_id}/status", headers=headers
        )
        assert status_response.status_code == 200
        assert "status" in status_response.json()

    @pytest.mark.integration
    def test_authentication_middleware(self, test_client):
        """Test authentication and authorization."""
        # Test without token
        response = test_client.get("/api/v1/workflow/list")
        assert response.status_code == 401

        # Test with invalid token
        response = test_client.get(
            "/api/v1/workflow/list", headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

        # Test with expired token
        expired_token = jwt.encode(
            {"sub": "test_user", "exp": int(time.time()) - 3600},
            "test_secret_key",
            algorithm="HS256",
        )
        response = test_client.get(
            "/api/v1/workflow/list", headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401

    @pytest.mark.integration
    def test_role_based_access_control(self, test_client):
        """Test RBAC implementation."""
        # Create tokens with different roles
        admin_token = jwt.encode(
            {"sub": "admin", "exp": int(time.time()) + 3600, "roles": ["admin"]},
            "test_secret_key",
            algorithm="HS256",
        )

        user_token = jwt.encode(
            {"sub": "user", "exp": int(time.time()) + 3600, "roles": ["user"]},
            "test_secret_key",
            algorithm="HS256",
        )

        # Test admin-only endpoint
        admin_response = test_client.delete(
            "/api/v1/workflow/all", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert admin_response.status_code in [200, 204]

        # Test user access to admin endpoint
        user_response = test_client.delete(
            "/api/v1/workflow/all", headers={"Authorization": f"Bearer {user_token}"}
        )
        assert user_response.status_code == 403

    @pytest.mark.integration
    def test_rate_limiting(self, test_client, auth_token):
        """Test rate limiting functionality."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Make requests up to rate limit
        responses = []
        for _ in range(150):
            response = test_client.get("/api/v1/workflow/list", headers=headers)
            responses.append(response.status_code)

        # Check that some requests were rate limited
        rate_limited = responses.count(429)
        assert rate_limited > 0

        # Check rate limit headers
        response = test_client.get("/api/v1/workflow/list", headers=headers)
        if response.status_code == 429:
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.integration
    def test_request_validation(self, test_client, auth_token):
        """Test request validation and sanitization."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Test invalid JSON
        response = test_client.post(
            "/api/v1/workflow/create",
            data="invalid json",
            headers={**headers, "Content-Type": "application/json"},
        )
        assert response.status_code == 400

        # Test missing required fields
        response = test_client.post("/api/v1/workflow/create", json={}, headers=headers)
        assert response.status_code == 422
        assert "validation_error" in response.json()

        # Test invalid field types
        response = test_client.post(
            "/api/v1/workflow/create",
            json={"type": 123, "topic": ["not", "a", "string"]},  # Should be string
            headers=headers,
        )
        assert response.status_code == 422

    @pytest.mark.integration
    def test_response_caching(self, test_client, auth_token):
        """Test response caching for improved performance."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # First request (cache miss)
        start_time = time.time()
        response1 = test_client.get("/api/v1/agents/list", headers=headers)
        first_request_time = time.time() - start_time
        assert response1.status_code == 200

        # Second request (cache hit)
        start_time = time.time()
        response2 = test_client.get("/api/v1/agents/list", headers=headers)
        second_request_time = time.time() - start_time
        assert response2.status_code == 200

        # Cache should make second request faster
        assert second_request_time < first_request_time * 0.5

        # Verify cache headers
        assert "X-Cache" in response2.headers
        assert response2.headers["X-Cache"] == "HIT"

    @pytest.mark.integration
    async def test_websocket_support(self, gateway):
        """Test WebSocket support for real-time updates."""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("ws://localhost:8080/ws/workflow/updates") as ws:
                # Send subscription message
                await ws.send_json({"action": "subscribe", "workflow_id": "test_workflow"})

                # Simulate workflow update
                await gateway.broadcast_update(
                    {"workflow_id": "test_workflow", "status": "running", "progress": 0.5}
                )

                # Receive update
                msg = await ws.receive_json()
                assert msg["workflow_id"] == "test_workflow"
                assert msg["status"] == "running"
                assert msg["progress"] == 0.5

    @pytest.mark.integration
    def test_cors_configuration(self, test_client):
        """Test CORS headers and configuration."""
        # Test preflight request
        response = test_client.options(
            "/api/v1/workflow/list",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    @pytest.mark.integration
    def test_api_versioning(self, test_client, auth_token):
        """Test API versioning support."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Test v1 endpoint
        v1_response = test_client.get("/api/v1/workflow/list", headers=headers)
        assert v1_response.status_code == 200

        # Test v2 endpoint (with different response format)
        v2_response = test_client.get("/api/v2/workflows", headers=headers)
        assert v2_response.status_code == 200

        # Verify different response formats
        v1_data = v1_response.json()
        v2_data = v2_response.json()
        assert "workflows" in v1_data  # v1 format
        assert "data" in v2_data  # v2 format

    @pytest.mark.integration
    def test_request_logging_and_tracing(self, test_client, auth_token):
        """Test request logging and distributed tracing."""
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "X-Request-ID": "test-request-123",
            "X-Trace-ID": "trace-456",
        }

        response = test_client.get("/api/v1/workflow/list", headers=headers)

        # Verify tracing headers in response
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == "test-request-123"
        assert "X-Trace-ID" in response.headers

    @pytest.mark.integration
    def test_error_handling(self, test_client, auth_token):
        """Test error handling and error responses."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Test 404 error
        response = test_client.get("/api/v1/workflow/nonexistent", headers=headers)
        assert response.status_code == 404
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data

        # Test 500 error simulation
        with patch("atlas_core.engine.ATLASEngine.get_workflow") as mock_get:
            mock_get.side_effect = Exception("Database connection failed")
            response = test_client.get("/api/v1/workflow/test_id", headers=headers)
            assert response.status_code == 500
            error_data = response.json()
            assert "error" in error_data
            assert "internal_server_error" in error_data["error"].lower()

    @pytest.mark.integration
    def test_file_upload_endpoint(self, test_client, auth_token):
        """Test file upload functionality."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Create test file content
        file_content = b"test research paper content"
        files = {"file": ("research.pdf", file_content, "application/pdf")}

        response = test_client.post("/api/v1/upload/document", files=files, headers=headers)

        assert response.status_code == 201
        result = response.json()
        assert "file_id" in result
        assert "size" in result
        assert result["size"] == len(file_content)

    @pytest.mark.integration
    def test_batch_request_processing(self, test_client, auth_token):
        """Test batch request processing."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        batch_request = {
            "requests": [
                {"method": "GET", "path": "/api/v1/agents/list"},
                {"method": "GET", "path": "/api/v1/workflow/list"},
                {
                    "method": "POST",
                    "path": "/api/v1/workflow/create",
                    "body": {"type": "research", "topic": "test"},
                },
            ]
        }

        response = test_client.post("/api/v1/batch", json=batch_request, headers=headers)

        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 3
        assert all("status_code" in r for r in results)
        assert all("body" in r for r in results)

    @pytest.mark.integration
    async def test_graphql_endpoint(self, gateway):
        """Test GraphQL endpoint for flexible queries."""
        query = """
        query {
            workflows(status: "running") {
                id
                type
                status
                agents {
                    name
                    status
                }
            }
        }
        """

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/graphql", json={"query": query}
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "data" in data
                assert "workflows" in data["data"]

    @pytest.mark.integration
    def test_api_documentation(self, test_client):
        """Test API documentation endpoints."""
        # Test OpenAPI schema
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "components" in schema

        # Test Swagger UI
        response = test_client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

        # Test ReDoc
        response = test_client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()

    @pytest.mark.integration
    def test_metrics_endpoint(self, test_client, auth_token):
        """Test metrics and monitoring endpoints."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        response = test_client.get("/api/v1/metrics", headers=headers)
        assert response.status_code == 200

        metrics = response.json()
        assert "request_count" in metrics
        assert "average_latency" in metrics
        assert "error_rate" in metrics
        assert "active_workflows" in metrics