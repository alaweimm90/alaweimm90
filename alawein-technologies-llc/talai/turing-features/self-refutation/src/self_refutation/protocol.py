"""
Self-Refutation Protocol

Main orchestrator that coordinates all 5 refutation strategies and produces
a comprehensive strength score for scientific hypotheses.
"""

import asyncio
from typing import Optional, List
from self_refutation.core.models import (
    Hypothesis,
    RefutationResult,
    StrategyResult,
    Confidence,
)
from self_refutation.strategies import (
    LogicalContradictionStrategy,
    EmpiricalCounterExampleStrategy,
    AnalogicalFalsificationStrategy,
    BoundaryViolationStrategy,
    MechanismImplausibilityStrategy,
)


class SelfRefutationProtocol:
    """
    Popperian Falsification System

    Tests hypotheses using 5 orthogonal refutation strategies.
    A hypothesis gains strength by SURVIVING refutation attempts,
    not by accumulating supporting evidence.

    Usage:
        protocol = SelfRefutationProtocol()
        result = await protocol.refute(hypothesis)
        print(f"Strength: {result.strength_score}/100")
    """

    def __init__(self, orchestrator=None, **config):
        """
        Initialize protocol

        Args:
            orchestrator: Optional AI Orchestrator for LLM-powered analysis
            **config: Configuration options
                - parallel: Run strategies in parallel (default: True)
                - fail_fast: Stop on first critical failure (default: False)
                - min_confidence: Minimum confidence threshold (default: Confidence.LOW)
        """
        self.orchestrator = orchestrator
        self.config = config

        # Initialize all 5 strategies
        self.strategies = [
            LogicalContradictionStrategy(orchestrator=orchestrator),
            EmpiricalCounterExampleStrategy(orchestrator=orchestrator),
            AnalogicalFalsificationStrategy(orchestrator=orchestrator),
            BoundaryViolationStrategy(orchestrator=orchestrator),
            MechanismImplausibilityStrategy(orchestrator=orchestrator),
        ]

    async def refute(self, hypothesis: Hypothesis) -> RefutationResult:
        """
        Attempt to refute hypothesis using all strategies

        Args:
            hypothesis: The hypothesis to test

        Returns:
            RefutationResult with strength score and detailed analysis
        """
        # Validate hypothesis
        if not hypothesis.claim or len(hypothesis.claim) < 10:
            raise ValueError("Invalid hypothesis: claim must be at least 10 characters")

        # Run all strategies
        if self.config.get("parallel", True):
            strategy_results = await self._run_parallel(hypothesis)
        else:
            strategy_results = await self._run_sequential(hypothesis)

        # Calculate strength score
        strength_score = self._calculate_strength_score(strategy_results)

        # Determine if refuted
        refuted, refutation_reason = self._determine_refutation(strategy_results)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(strategy_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(hypothesis, strategy_results, strength_score)

        # Build result
        result = RefutationResult(
            hypothesis=hypothesis,
            strength_score=strength_score,
            strategies_passed=sum(1 for r in strategy_results if r.passed),
            total_strategies=len(strategy_results),
            refuted=refuted,
            refutation_reason=refutation_reason,
            confidence=overall_confidence,
            strategy_results=strategy_results,
            recommendations=recommendations,
            metadata={
                "orchestrator_used": self.orchestrator is not None,
                "parallel_execution": self.config.get("parallel", True),
            }
        )

        return result

    async def refute_batch(self, hypotheses: List[Hypothesis]) -> List[RefutationResult]:
        """
        Refute multiple hypotheses in parallel

        Args:
            hypotheses: List of hypotheses to test

        Returns:
            List of RefutationResults
        """
        tasks = [self.refute(h) for h in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                processed_results.append(
                    RefutationResult(
                        hypothesis=hypotheses[i],
                        strength_score=0.0,
                        strategies_passed=0,
                        total_strategies=5,
                        refuted=True,
                        refutation_reason=f"Protocol error: {str(result)}",
                        confidence=Confidence.HIGH,
                        strategy_results=[],
                        recommendations=["Fix protocol error before retesting"],
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_parallel(self, hypothesis: Hypothesis) -> List[StrategyResult]:
        """Run all strategies in parallel"""
        tasks = [strategy.refute(hypothesis) for strategy in self.strategies]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any strategy errors
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for this strategy
                processed_results.append(
                    StrategyResult(
                        strategy=self.strategies[i].strategy_type,
                        passed=True,  # Don't penalize hypothesis for strategy error
                        confidence=Confidence.LOW,
                        reasoning=f"Strategy error: {str(result)}",
                        evidence=[],
                        severity=0.0,
                        metadata={"error": True}
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_sequential(self, hypothesis: Hypothesis) -> List[StrategyResult]:
        """Run strategies sequentially"""
        results = []
        fail_fast = self.config.get("fail_fast", False)

        for strategy in self.strategies:
            try:
                result = await strategy.refute(hypothesis)
                results.append(result)

                # Check for critical failure
                if fail_fast and not result.passed and result.severity >= 0.7:
                    # Stop early on critical failure
                    break
            except Exception as e:
                # Create error result
                results.append(
                    StrategyResult(
                        strategy=strategy.strategy_type,
                        passed=True,
                        confidence=Confidence.LOW,
                        reasoning=f"Strategy error: {str(e)}",
                        evidence=[],
                        severity=0.0,
                        metadata={"error": True}
                    )
                )

        return results

    def _calculate_strength_score(self, strategy_results: List[StrategyResult]) -> float:
        """
        Calculate hypothesis strength score (0-100)

        Algorithm:
        1. Start with base score from pass rate
        2. Apply severity penalties for failures
        3. Apply confidence adjustments
        4. Normalize to 0-100 range
        """
        if not strategy_results:
            return 0.0

        # Base score: percentage of strategies passed
        passed = sum(1 for r in strategy_results if r.passed)
        base_score = (passed / len(strategy_results)) * 100

        # Apply severity penalties
        total_severity_penalty = 0.0
        for result in strategy_results:
            if not result.passed:
                # Higher severity = bigger penalty
                penalty = result.severity * 20  # Max 20 points per failure

                # Weight by confidence
                confidence_multipliers = {
                    Confidence.LOW: 0.5,
                    Confidence.MEDIUM: 0.75,
                    Confidence.HIGH: 1.0,
                    Confidence.VERY_HIGH: 1.25,
                }
                multiplier = confidence_multipliers.get(result.confidence, 1.0)
                total_severity_penalty += penalty * multiplier

        # Apply penalty
        final_score = base_score - total_severity_penalty

        # Clamp to 0-100
        return max(0.0, min(100.0, final_score))

    def _determine_refutation(self, strategy_results: List[StrategyResult]) -> tuple:
        """
        Determine if hypothesis is refuted

        Hypothesis is refuted if:
        - Any strategy fails with HIGH or VERY_HIGH confidence AND severity >= 0.7
        - Multiple strategies fail (3+ out of 5)
        """
        critical_failures = [
            r for r in strategy_results
            if not r.passed
            and r.confidence in [Confidence.HIGH, Confidence.VERY_HIGH]
            and r.severity >= 0.7
        ]

        if critical_failures:
            reasons = [f"{r.strategy.value}: {r.reasoning}" for r in critical_failures]
            return True, "; ".join(reasons)

        # Check for multiple failures
        failures = [r for r in strategy_results if not r.passed]
        if len(failures) >= 3:
            return True, f"Failed {len(failures)}/{len(strategy_results)} refutation strategies"

        return False, None

    def _calculate_overall_confidence(self, strategy_results: List[StrategyResult]) -> Confidence:
        """Calculate overall confidence in result"""
        if not strategy_results:
            return Confidence.LOW

        # Map confidences to numeric values
        confidence_values = {
            Confidence.LOW: 1,
            Confidence.MEDIUM: 2,
            Confidence.HIGH: 3,
            Confidence.VERY_HIGH: 4,
        }

        # Average confidence across strategies
        avg_confidence = sum(
            confidence_values[r.confidence] for r in strategy_results
        ) / len(strategy_results)

        # Map back to confidence enum
        if avg_confidence >= 3.5:
            return Confidence.VERY_HIGH
        elif avg_confidence >= 2.5:
            return Confidence.HIGH
        elif avg_confidence >= 1.5:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW

    def _generate_recommendations(
        self,
        hypothesis: Hypothesis,
        strategy_results: List[StrategyResult],
        strength_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Score-based recommendations
        if strength_score >= 81:
            recommendations.append("✅ Strong hypothesis - Proceed to experimental validation")
            recommendations.append("Consider pre-registering hypothesis to avoid p-hacking")
        elif strength_score >= 61:
            recommendations.append("⚠️ Minor concerns exist - Address them before expensive experiments")
        elif strength_score >= 41:
            recommendations.append("🔴 Moderate concerns - Revise hypothesis addressing specific failures")
        elif strength_score >= 21:
            recommendations.append("🔴 Major issues detected - Substantial revision needed")
        else:
            recommendations.append("❌ Critically flawed - Consider rejecting or completely reformulating")

        # Strategy-specific recommendations
        for result in strategy_results:
            if not result.passed and result.severity >= 0.5:
                recommendations.append(
                    f"Address {result.strategy.value}: {result.reasoning}"
                )

        # Missing elements recommendations
        if not hypothesis.assumptions:
            recommendations.append("Add explicit assumptions to improve clarity")

        if not hypothesis.predictions:
            recommendations.append("Add testable predictions to make hypothesis falsifiable")

        if not hypothesis.context:
            recommendations.append("Provide more context about hypothesis origin and scope")

        return recommendations
