"""
Strategy 1: Logical Contradiction Detection

Searches for internal logical contradictions in the hypothesis.
"""

from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy, Confidence


class LogicalContradictionStrategy(BaseRefutationStrategy):
    """
    Detect logical contradictions in hypothesis

    Examples of contradictions:
    - "X improves Y AND X degrades Y"
    - "Always works AND sometimes fails"
    - "Increases speed AND decreases speed"
    - "Method is deterministic AND produces random results"
    """

    @property
    def strategy_type(self) -> RefutationStrategy:
        return RefutationStrategy.LOGICAL_CONTRADICTION

    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """Search for logical contradictions"""
        if not self._validate_hypothesis(hypothesis):
            raise StrategyError("Invalid hypothesis")

        # Extract claim and context
        claim = hypothesis.claim.lower()
        context = (hypothesis.context or "").lower()
        full_text = f"{claim} {context}"

        # Check for obvious contradictions
        contradictions_found = []

        # Pattern 1: X and not X
        contradiction_patterns = [
            (["improves", "increases", "enhances", "better"], ["degrades", "decreases", "worsens", "worse"]),
            (["always", "all", "every", "universally"], ["sometimes", "some", "occasionally", "never"]),
            (["deterministic", "predictable", "consistent"], ["random", "stochastic", "unpredictable"]),
            (["fast", "quick", "rapid"], ["slow", "sluggish"]),
            (["efficient"], ["inefficient"]),
            (["optimal"], ["suboptimal"]),
        ]

        for positive_terms, negative_terms in contradiction_patterns:
            has_positive = any(term in full_text for term in positive_terms)
            has_negative = any(term in full_text for term in negative_terms)

            if has_positive and has_negative:
                contradictions_found.append(
                    f"Contains both positive ({[t for t in positive_terms if t in full_text]}) "
                    f"and negative ({[t for t in negative_terms if t in full_text]}) claims"
                )

        # If we have AI Orchestrator, do deeper analysis
        if self.orchestrator:
            ai_result = await self._ai_contradiction_check(hypothesis)
            if ai_result:
                contradictions_found.extend(ai_result)

        # Determine result
        if contradictions_found:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=False,
                confidence=Confidence.HIGH,
                reasoning=f"Logical contradictions detected: {len(contradictions_found)} found",
                evidence=contradictions_found,
                severity=min(1.0, len(contradictions_found) * 0.3),
                metadata={"contradiction_count": len(contradictions_found)}
            )
        else:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=True,
                confidence=Confidence.MEDIUM,
                reasoning="No obvious logical contradictions detected",
                evidence=[],
                severity=0.0
            )

    async def _ai_contradiction_check(self, hypothesis: Hypothesis) -> list:
        """Use AI to find subtle contradictions"""
        if not self.orchestrator:
            return []

        from atlas_orchestrator import Task, TaskType

        prompt = f"""Analyze this hypothesis for logical contradictions:

Hypothesis: {hypothesis.claim}
Context: {hypothesis.context or 'None'}
Assumptions: {', '.join(hypothesis.assumptions) if hypothesis.assumptions else 'None'}

Look for:
1. Statements that contradict each other
2. Claims that cannot both be true
3. Logical impossibilities
4. Mutually exclusive assertions

List any contradictions found. If none, say "No contradictions found".
"""

        task = Task(prompt=prompt, task_type=TaskType.ANALYSIS, max_tokens=500)
        result = await self.orchestrator.execute(task)

        if result.success and "no contradictions" not in result.content.lower():
            # Parse AI response for contradictions
            lines = result.content.strip().split('\n')
            contradictions = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            return contradictions[:5]  # Limit to 5

        return []
