"""
Base strategy class for refutation strategies
"""

from abc import ABC, abstractmethod
from typing import Optional
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy


class BaseRefutationStrategy(ABC):
    """
    Abstract base class for refutation strategies

    All refutation strategies must:
    1. Implement the refute() method
    2. Return a StrategyResult
    3. Use AI Orchestrator if provided
    """

    def __init__(self, orchestrator=None, **kwargs):
        """
        Initialize strategy

        Args:
            orchestrator: Optional AI Orchestrator for LLM calls
            **kwargs: Strategy-specific parameters
        """
        self.orchestrator = orchestrator
        self.config = kwargs

    @abstractmethod
    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """
        Attempt to refute the hypothesis

        Args:
            hypothesis: The hypothesis to test

        Returns:
            StrategyResult indicating pass/fail and reasoning

        Raises:
            StrategyError: If strategy execution fails
        """
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> RefutationStrategy:
        """Return the strategy type"""
        pass

    def _validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Validate hypothesis has required fields"""
        if not hypothesis.claim or len(hypothesis.claim) < 10:
            return False
        return True


class StrategyError(Exception):
    """Raised when strategy execution fails"""
    pass
