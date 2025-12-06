"""
Shared fixtures for Devil's Advocate tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from self_refutation import Hypothesis, HypothesisDomain


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Identified flaw: Algorithm assumes independence when variables may be correlated."
    )
    return orchestrator


@pytest.fixture
def optimization_hypothesis() -> Hypothesis:
    """Create optimization hypothesis."""
    return Hypothesis(
        claim="Our algorithm improves performance by 40%",
        domain=HypothesisDomain.OPTIMIZATION,
        evidence=["Benchmark tests on 100 instances"],
        assumptions=["Linear scaling", "Independent variables"],
    )


@pytest.fixture
def ml_hypothesis() -> Hypothesis:
    """Create machine learning hypothesis."""
    return Hypothesis(
        claim="Our neural network achieves 99% accuracy",
        domain=HypothesisDomain.MACHINE_LEARNING,
        evidence=["Tested on training set", "Validated on test set"],
        assumptions=["IID data", "No distribution shift"],
    )


@pytest.fixture
def quantum_hypothesis() -> Hypothesis:
    """Create quantum computing hypothesis."""
    return Hypothesis(
        claim="Our quantum algorithm provides quadratic speedup",
        domain=HypothesisDomain.QUANTUM,
        evidence=["Theoretical analysis", "Simulated on 10 qubits"],
        assumptions=["Perfect qubits", "No decoherence"],
    )
