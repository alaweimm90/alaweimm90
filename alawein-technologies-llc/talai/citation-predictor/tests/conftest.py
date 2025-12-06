"""Pytest configuration and shared fixtures for CitationPredictor tests"""

import pytest
import random


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility"""
    random.seed(42)
    yield
    # Cleanup after test if needed
