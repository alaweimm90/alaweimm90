"""
Unit tests for the example FastAPI application.

Demonstrates testing patterns per governance contract requirements.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app, messages_db


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_db():
    """Clear in-memory database before each test."""
    messages_db.clear()
    yield
    messages_db.clear()


class TestHealth:
    """Health check endpoint tests."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "service" in data


class TestRoot:
    """Root endpoint tests."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data
        assert "health" in data


class TestMessages:
    """Message CRUD endpoint tests."""

    def test_create_message(self, client):
        """Test creating a message."""
        payload = {
            "id": 1,
            "content": "Hello, World!",
            "author": "Test User",
        }
        response = client.post("/api/messages", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["content"] == "Hello, World!"

    def test_get_message(self, client):
        """Test retrieving a message."""
        # Create first
        payload = {
            "id": 1,
            "content": "Test message",
            "author": "Test User",
        }
        client.post("/api/messages", json=payload)

        # Retrieve
        response = client.get("/api/messages/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["content"] == "Test message"

    def test_get_nonexistent_message(self, client):
        """Test retrieving a non-existent message."""
        response = client.get("/api/messages/999")
        assert response.status_code == 404

    def test_delete_message(self, client):
        """Test deleting a message."""
        # Create first
        payload = {"id": 1, "content": "Delete me", "author": "Test User"}
        client.post("/api/messages", json=payload)

        # Delete
        response = client.delete("/api/messages/1")
        assert response.status_code == 200

        # Verify deleted
        response = client.get("/api/messages/1")
        assert response.status_code == 404


class TestCoverage:
    """Tests to achieve coverage targets (85% per governance contract)."""

    def test_message_model_without_author(self, client):
        """Test message creation without optional author field."""
        payload = {"id": 2, "content": "No author"}
        response = client.post("/api/messages", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["author"] is None

    def test_multiple_messages(self, client):
        """Test managing multiple messages."""
        for i in range(5):
            payload = {"id": i, "content": f"Message {i}"}
            client.post("/api/messages", json=payload)

        # Verify all created
        for i in range(5):
            response = client.get(f"/api/messages/{i}")
            assert response.status_code == 200

    def test_delete_nonexistent_message(self, client):
        """Test deleting a non-existent message returns 404."""
        response = client.delete("/api/messages/999")
        assert response.status_code == 404
