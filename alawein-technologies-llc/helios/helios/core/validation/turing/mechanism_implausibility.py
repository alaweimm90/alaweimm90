"""
Strategy 5: Mechanism Implausibility

Checks if the proposed causal mechanism is plausible given known science.
"""

from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy, Confidence


class MechanismImplausibilityStrategy(BaseRefutationStrategy):
    """
    Check if causal mechanism is scientifically plausible

    Examples:
    - "X causes Y" → Is there a known mechanism for X→Y?
    - "Algorithm A improves B" → What is the causal pathway?
    - "Method increases speed" → What is the physical/computational mechanism?

    Red flags:
    - No proposed mechanism
    - Mechanism violates known laws (physics, math, computation theory)
    - Correlation claimed as causation without mechanism
    - "Magic happens here" gaps in causal chain
    """

    @property
    def strategy_type(self) -> RefutationStrategy:
        return RefutationStrategy.MECHANISM_IMPLAUSIBILITY

    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """Check mechanism plausibility"""
        if not self._validate_hypothesis(hypothesis):
            raise StrategyError("Invalid hypothesis")

        implausibilities = []

        # Check if mechanism is specified at all
        mechanism_specified = self._check_mechanism_specified(hypothesis)
        if not mechanism_specified:
            implausibilities.append("No causal mechanism specified - correlation vs causation unclear")

        # Check for causal claim patterns
        claim = hypothesis.claim.lower()
        context = (hypothesis.context or "").lower()
        full_text = f"{claim} {context}"

        # Pattern: "X causes Y" or "X improves Y"
        causal_verbs = ["causes", "improves", "increases", "decreases", "enhances", "reduces", "leads to", "results in"]
        has_causal_claim = any(verb in full_text for verb in causal_verbs)

        if has_causal_claim:
            # Check if mechanism is explained
            mechanism_keywords = ["because", "due to", "mechanism", "via", "through", "by means of", "process"]
            has_mechanism_explanation = any(keyword in full_text for keyword in mechanism_keywords)

            if not has_mechanism_explanation:
                implausibilities.append(
                    "Causal claim made without explaining the mechanism "
                    "(e.g., 'X improves Y' without explaining HOW)"
                )

        # Check for domain-specific implausibilities
        domain_issues = await self._check_domain_implausibilities(hypothesis)
        implausibilities.extend(domain_issues)

        # Check for computational complexity violations
        if hypothesis.domain.value in ["optimization", "computer_science"]:
            complexity_issues = self._check_complexity_violations(hypothesis)
            implausibilities.extend(complexity_issues)

        # Use AI for deeper mechanism analysis
        if self.orchestrator:
            ai_issues = await self._ai_mechanism_check(hypothesis)
            implausibilities.extend(ai_issues)

        # Determine result
        if implausibilities:
            # Higher severity if mechanism is completely missing or violates known laws
            has_critical_issue = any(
                keyword in " ".join(implausibilities).lower()
                for keyword in ["no mechanism", "violates", "impossible", "contradicts"]
            )
            severity = 0.7 if has_critical_issue else min(1.0, len(implausibilities) * 0.2)

            return StrategyResult(
                strategy=self.strategy_type,
                passed=False,
                confidence=Confidence.MEDIUM if has_critical_issue else Confidence.LOW,
                reasoning=f"Found {len(implausibilities)} mechanism plausibility issues",
                evidence=implausibilities,
                severity=severity,
                metadata={
                    "issue_count": len(implausibilities),
                    "critical_issue": has_critical_issue
                }
            )
        else:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=True,
                confidence=Confidence.MEDIUM,
                reasoning="Proposed mechanism appears plausible",
                evidence=[],
                severity=0.0
            )

    def _check_mechanism_specified(self, hypothesis: Hypothesis) -> bool:
        """Check if any mechanism is specified"""
        claim = hypothesis.claim.lower()
        context = (hypothesis.context or "").lower()
        assumptions = " ".join(hypothesis.assumptions).lower() if hypothesis.assumptions else ""

        full_text = f"{claim} {context} {assumptions}"

        # Look for mechanism indicators
        mechanism_indicators = [
            "because", "due to", "via", "through", "by",
            "mechanism", "process", "pathway", "means",
            "works by", "operates by", "functions by",
            "reason", "explanation"
        ]

        return any(indicator in full_text for indicator in mechanism_indicators)

    async def _check_domain_implausibilities(self, hypothesis: Hypothesis) -> list:
        """Check for domain-specific implausibilities"""
        implausibilities = []

        claim = hypothesis.claim.lower()

        # Optimization domain
        if hypothesis.domain.value == "optimization":
            # Check for unrealistic optimization claims
            if "global optimum" in claim and "polynomial time" in claim:
                implausibilities.append(
                    "Claims polynomial-time global optimum for NP-hard problem - "
                    "violates computational complexity theory unless P=NP"
                )

            if "always converges" in claim and "no assumptions" in claim:
                implausibilities.append(
                    "Claims convergence without assumptions - all convergence proofs require assumptions"
                )

        # Machine Learning domain
        if hypothesis.domain.value == "machine_learning":
            if "zero error" in claim or "100% accuracy" in claim:
                implausibilities.append(
                    "Claims perfect accuracy - ignores irreducible error (Bayes error rate) "
                    "and overfitting risks"
                )

            if "no training data" in claim and "learns" in claim:
                implausibilities.append(
                    "Claims learning without training data - violates fundamental ML principles"
                )

        # Physics domain
        if hypothesis.domain.value == "physics":
            if any(term in claim for term in ["perpetual", "free energy", "faster than light"]):
                implausibilities.append(
                    "Claim appears to violate fundamental physics laws "
                    "(thermodynamics, relativity)"
                )

        # Mathematics domain
        if hypothesis.domain.value == "mathematics":
            if "proven" in claim and "counterexample exists" in claim:
                implausibilities.append(
                    "Cannot be both proven and have counterexamples - logical contradiction"
                )

        return implausibilities

    def _check_complexity_violations(self, hypothesis: Hypothesis) -> list:
        """Check for computational complexity violations"""
        violations = []

        claim = hypothesis.claim.lower()

        # P vs NP violations
        np_hard_problems = ["tsp", "knapsack", "sat", "clique", "vertex cover", "qap"]
        claims_polynomial = any(term in claim for term in ["polynomial", "o(n)", "linear time", "constant time"])
        mentions_np_hard = any(problem in claim for problem in np_hard_problems)

        if claims_polynomial and mentions_np_hard:
            violations.append(
                "Claims polynomial-time solution to NP-hard problem without specifying "
                "approximation guarantees or heuristic nature - likely unsound"
            )

        # Impossibility results
        if "halting problem" in claim and "solves" in claim:
            violations.append(
                "Claims to solve halting problem - proven impossible by Turing (1936)"
            )

        # Lower bound violations
        if "o(n log n)" in claim and "sorting" in claim and "comparison" in claim:
            violations.append(
                "Claims faster than O(n log n) comparison-based sorting - "
                "violates decision tree lower bound"
            )

        return violations

    async def _ai_mechanism_check(self, hypothesis: Hypothesis) -> list:
        """Use AI to check mechanism plausibility"""
        if not self.orchestrator:
            return []

        from atlas_orchestrator import Task, TaskType

        prompt = f"""Analyze the plausibility of the causal mechanism in this hypothesis:

Hypothesis: {hypothesis.claim}
Domain: {hypothesis.domain.value}
Context: {hypothesis.context or 'None'}
Assumptions: {', '.join(hypothesis.assumptions) if hypothesis.assumptions else 'None'}

Evaluate:
1. Is a causal mechanism specified? (X causes Y because Z)
2. Is the mechanism scientifically plausible?
3. Does it violate known laws of physics/math/computation?
4. Are there gaps in the causal chain?
5. Is correlation being confused with causation?

List specific mechanism implausibilities or violations.
If mechanism appears sound, say "Mechanism appears plausible".
"""

        task = Task(prompt=prompt, task_type=TaskType.ANALYSIS, max_tokens=500)
        result = await self.orchestrator.execute(task)

        if result.success and "mechanism appears plausible" not in result.content.lower():
            lines = result.content.strip().split('\n')
            issues = [
                line.strip()
                for line in lines
                if line.strip()
                and len(line.strip()) > 15
                and any(keyword in line.lower() for keyword in [
                    'mechanism', 'causal', 'implausible', 'violate', 'impossible',
                    'gap', 'missing', 'unclear', 'unsound'
                ])
            ]
            return issues[:5]

        return []
