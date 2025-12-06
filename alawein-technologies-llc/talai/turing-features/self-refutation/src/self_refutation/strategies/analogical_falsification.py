"""
Strategy 3: Analogical Falsification

Finds similar claims in other domains that were disproven.
"""

from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy, Confidence


class AnalogicalFalsificationStrategy(BaseRefutationStrategy):
    """
    Find analogous claims that failed in other domains

    Examples:
    - "New optimizer always finds global optimum" → Similar claims about Simulated Annealing failed
    - "Algorithm works on all inputs" → Many "universal" algorithms have been disproven
    - "Method guarantees convergence" → Similar guarantees often failed in practice
    """

    @property
    def strategy_type(self) -> RefutationStrategy:
        return RefutationStrategy.ANALOGICAL_FALSIFICATION

    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """Search for analogous failures"""
        if not self._validate_hypothesis(hypothesis):
            raise StrategyError("Invalid hypothesis")

        analogies = []

        # Check for common claim patterns with known failures
        claim = hypothesis.claim.lower()

        # Pattern: Universal claims
        if any(term in claim for term in ["always", "all", "every", "universal", "guaranteed"]):
            analogies.extend(await self._check_universal_claim_analogies(hypothesis))

        # Pattern: Optimization claims
        if any(term in claim for term in ["optimal", "best", "maximum", "minimum"]):
            analogies.extend(await self._check_optimization_analogies(hypothesis))

        # Pattern: Performance claims
        if any(term in claim for term in ["faster", "better", "improves", "outperforms"]):
            analogies.extend(await self._check_performance_analogies(hypothesis))

        # Use AI to find deeper analogies
        if self.orchestrator:
            ai_analogies = await self._ai_analogy_search(hypothesis)
            analogies.extend(ai_analogies)

        # Determine result
        if analogies:
            # Higher severity if multiple analogies found
            severity = min(1.0, len(analogies) * 0.2)

            return StrategyResult(
                strategy=self.strategy_type,
                passed=False,
                confidence=Confidence.MEDIUM,
                reasoning=f"Found {len(analogies)} analogous claims that failed",
                evidence=analogies,
                severity=severity,
                metadata={"analogy_count": len(analogies)}
            )
        else:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=True,
                confidence=Confidence.LOW,  # Absence of analogy != proof
                reasoning="No clear analogous failures found",
                evidence=[],
                severity=0.0
            )

    async def _check_universal_claim_analogies(self, hypothesis: Hypothesis) -> list:
        """Check for analogies to failed universal claims"""
        analogies = []

        claim = hypothesis.claim.lower()

        # No Free Lunch Theorem analogy
        if "all" in claim or "every" in claim:
            if hypothesis.domain.value in ["optimization", "machine_learning"]:
                analogies.append(
                    "No Free Lunch Theorem: Claims of universal superiority in optimization/ML "
                    "have historically failed (Wolpert & Macready, 1997)"
                )

        # Universal solver analogy
        if "always" in claim and "solution" in claim:
            analogies.append(
                "Historical analogy: Many 'universal solver' claims were disproven "
                "(e.g., early genetic algorithm claims, simulated annealing guarantees)"
            )

        return analogies

    async def _check_optimization_analogies(self, hypothesis: Hypothesis) -> list:
        """Check for analogies to failed optimization claims"""
        analogies = []

        claim = hypothesis.claim.lower()

        # Global optimum claims
        if "global" in claim and "optim" in claim:
            analogies.append(
                "Analogy: Simulated Annealing claimed to find global optima 'always' "
                "but was later shown to fail on many problem classes"
            )

        # Best solution claims
        if "best" in claim or "optimal" in claim:
            if hypothesis.domain.value == "optimization":
                analogies.append(
                    "Analogy: Many heuristics claimed 'optimal' or 'best' results "
                    "but were outperformed by later methods (e.g., greedy algorithms vs dynamic programming)"
                )

        return analogies

    async def _check_performance_analogies(self, hypothesis: Hypothesis) -> list:
        """Check for analogies to failed performance claims"""
        analogies = []

        claim = hypothesis.claim.lower()

        # "Always better" claims
        if any(term in claim for term in ["always better", "always faster", "always improves"]):
            analogies.append(
                "Analogy: 'Always better' claims historically fail - context matters "
                "(e.g., QuickSort is not always faster than MergeSort, depends on data)"
            )

        # Speedup claims
        if "faster" in claim or "speedup" in claim:
            if "x" in claim or "%" in claim:
                analogies.append(
                    "Analogy: Many speedup claims failed to generalize "
                    "(e.g., parallel algorithms often don't scale as claimed due to Amdahl's Law)"
                )

        return analogies

    async def _ai_analogy_search(self, hypothesis: Hypothesis) -> list:
        """Use AI to search for deeper analogies"""
        if not self.orchestrator:
            return []

        from atlas_orchestrator import Task, TaskType

        prompt = f"""Search for analogous claims that failed in history:

Hypothesis: {hypothesis.claim}
Domain: {hypothesis.domain.value}
Context: {hypothesis.context or 'None'}

Think about:
1. Similar claims made in other domains that were later disproven
2. Historical precedents where analogous reasoning failed
3. Famous examples of similar hypotheses being refuted
4. Theoretical reasons why analogous claims fail

List specific historical analogies. Focus on well-documented cases.
If no clear analogies exist, say "No strong analogies found".
"""

        task = Task(prompt=prompt, task_type=TaskType.RESEARCH, max_tokens=600)
        result = await self.orchestrator.execute(task)

        if result.success and "no strong analogies" not in result.content.lower():
            lines = result.content.strip().split('\n')
            # Filter for substantive analogies
            analogies = [
                line.strip()
                for line in lines
                if line.strip()
                and len(line.strip()) > 20
                and any(keyword in line.lower() for keyword in ['analog', 'similar', 'precedent', 'historical', 'like', 'comparable'])
            ]
            return analogies[:5]  # Limit to 5

        return []
