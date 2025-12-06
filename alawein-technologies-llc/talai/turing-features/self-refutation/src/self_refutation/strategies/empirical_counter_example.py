"""
Strategy 2: Empirical Counter-Example Search

Searches for known counter-examples in data, literature, or databases.
"""

from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy, Confidence


class EmpiricalCounterExampleStrategy(BaseRefutationStrategy):
    """
    Search for empirical counter-examples

    Examples:
    - "Algorithm works on all problems" → Found failure on specific instance
    - "Method always improves performance" → Found cases where it degrades
    - "Approach never fails" → Found documented failure cases
    """

    @property
    def strategy_type(self) -> RefutationStrategy:
        return RefutationStrategy.EMPIRICAL_COUNTER_EXAMPLE

    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """Search for counter-examples"""
        if not self._validate_hypothesis(hypothesis):
            raise StrategyError("Invalid hypothesis")

        counter_examples = []

        # Check for absolute claims that are easy to refute
        claim = hypothesis.claim.lower()
        absolute_terms = ["all", "always", "never", "every", "none", "100%", "perfect", "guaranteed"]

        has_absolute = any(term in claim for term in absolute_terms)

        if has_absolute:
            counter_examples.append(
                "Hypothesis makes absolute claim (all/always/never) which is easily falsifiable"
            )

        # Check domain-specific counter-examples
        if hypothesis.domain:
            domain_counter_examples = await self._check_domain_counter_examples(hypothesis)
            counter_examples.extend(domain_counter_examples)

        # Use AI to search for counter-examples
        if self.orchestrator:
            ai_counter_examples = await self._ai_counter_example_search(hypothesis)
            counter_examples.extend(ai_counter_examples)

        # Determine result
        if counter_examples:
            severity = 0.5 if has_absolute else 0.3
            severity = min(1.0, severity + len(counter_examples) * 0.15)

            return StrategyResult(
                strategy=self.strategy_type,
                passed=False,
                confidence=Confidence.HIGH if has_absolute else Confidence.MEDIUM,
                reasoning=f"Found {len(counter_examples)} potential counter-examples",
                evidence=counter_examples,
                severity=severity,
                metadata={"has_absolute_claim": has_absolute}
            )
        else:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=True,
                confidence=Confidence.LOW,  # Absence of evidence != evidence of absence
                reasoning="No counter-examples found in available data",
                evidence=[],
                severity=0.0
            )

    async def _check_domain_counter_examples(self, hypothesis: Hypothesis) -> list:
        """Check for domain-specific counter-examples"""
        counter_examples = []

        # Optimization domain
        if hypothesis.domain.value == "optimization":
            if "qap" in hypothesis.claim.lower():
                # Check for known hard instances
                if "all" in hypothesis.claim.lower() or "always" in hypothesis.claim.lower():
                    counter_examples.append(
                        "QAP has known NP-hard instances (tai50a, tai100a) where optimal solutions are difficult"
                    )

        # Machine learning domain
        if hypothesis.domain.value == "machine_learning":
            if "always" in hypothesis.claim.lower() or "all datasets" in hypothesis.claim.lower():
                counter_examples.append(
                    "No Free Lunch Theorem: No algorithm works best on all problems"
                )

        return counter_examples

    async def _ai_counter_example_search(self, hypothesis: Hypothesis) -> list:
        """Use AI to search for counter-examples"""
        if not self.orchestrator:
            return []

        from atlas_orchestrator import Task, TaskType

        prompt = f"""Search for counter-examples to this hypothesis:

Hypothesis: {hypothesis.claim}
Domain: {hypothesis.domain.value}
Context: {hypothesis.context or 'None'}

Think about:
1. Known experimental results that contradict this
2. Edge cases where this would fail
3. Documented failures in literature
4. Similar claims that were disproven

List specific counter-examples. If none found, say "No counter-examples found".
"""

        task = Task(prompt=prompt, task_type=TaskType.RESEARCH, max_tokens=600)
        result = await self.orchestrator.execute(task)

        if result.success and "no counter-examples" not in result.content.lower():
            lines = result.content.strip().split('\n')
            examples = [line.strip() for line in lines if line.strip() and any(c in line.lower() for c in ['fail', 'counter', 'exception', 'not work', 'disproven'])]
            return examples[:5]

        return []
