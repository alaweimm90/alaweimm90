"""
Strategy 4: Boundary Violation Detection

Tests hypothesis at extreme parameter values to check validity.
"""

import re
from self_refutation.strategies.base import BaseRefutationStrategy, StrategyError
from self_refutation.core.models import Hypothesis, StrategyResult, RefutationStrategy, Confidence


class BoundaryViolationStrategy(BaseRefutationStrategy):
    """
    Test hypothesis at extreme parameter values

    Examples:
    - "Improves by 50%" → What if improvement is -50%? 200%? Still valid?
    - "Works for N=100" → What about N=0? N=∞?
    - "Temperature = 0.5 optimal" → What about T=0? T=1000?
    - "Dataset size = 1000" → What about size=1? size=10^9?
    """

    @property
    def strategy_type(self) -> RefutationStrategy:
        return RefutationStrategy.BOUNDARY_VIOLATION

    async def refute(self, hypothesis: Hypothesis) -> StrategyResult:
        """Test at boundary conditions"""
        if not self._validate_hypothesis(hypothesis):
            raise StrategyError("Invalid hypothesis")

        violations = []

        # Extract numerical claims from hypothesis
        claim = hypothesis.claim
        numerical_claims = self._extract_numerical_claims(claim)

        # Test each numerical claim at boundaries
        for claim_text, value, unit in numerical_claims:
            boundary_tests = self._test_boundaries(claim_text, value, unit)
            violations.extend(boundary_tests)

        # Check for missing boundary conditions
        missing_boundaries = self._check_missing_boundaries(hypothesis)
        violations.extend(missing_boundaries)

        # Use AI for deeper boundary analysis
        if self.orchestrator:
            ai_violations = await self._ai_boundary_check(hypothesis)
            violations.extend(ai_violations)

        # Determine result
        if violations:
            severity = min(1.0, len(violations) * 0.25)

            return StrategyResult(
                strategy=self.strategy_type,
                passed=False,
                confidence=Confidence.MEDIUM,
                reasoning=f"Found {len(violations)} boundary violations or missing conditions",
                evidence=violations,
                severity=severity,
                metadata={"violation_count": len(violations)}
            )
        else:
            return StrategyResult(
                strategy=self.strategy_type,
                passed=True,
                confidence=Confidence.MEDIUM,
                reasoning="No obvious boundary violations detected",
                evidence=[],
                severity=0.0
            )

    def _extract_numerical_claims(self, claim: str) -> list:
        """Extract numerical claims from hypothesis"""
        numerical_claims = []

        # Pattern: "X% improvement/better/faster"
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(improv|better|faster|increase|decrease|reduc)'
        for match in re.finditer(percent_pattern, claim.lower()):
            value = float(match.group(1))
            numerical_claims.append((match.group(0), value, "percent"))

        # Pattern: "N=100" or "size=1000"
        assignment_pattern = r'([a-zA-Z_]\w*)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(assignment_pattern, claim):
            param_name = match.group(1)
            value = float(match.group(2))
            numerical_claims.append((match.group(0), value, param_name))

        # Pattern: "X times faster/better"
        multiplier_pattern = r'(\d+(?:\.\d+)?)\s*(?:x|times)\s*(faster|better|more|less)'
        for match in re.finditer(multiplier_pattern, claim.lower()):
            value = float(match.group(1))
            numerical_claims.append((match.group(0), value, "multiplier"))

        return numerical_claims

    def _test_boundaries(self, claim_text: str, value: float, unit: str) -> list:
        """Test numerical claim at boundary conditions"""
        violations = []

        if unit == "percent":
            # Test at 0%, negative, and very large percentages
            if value > 0:  # Positive improvement
                violations.append(
                    f"Claim '{claim_text}': Does this hold at 0% improvement? "
                    "What about negative improvement (degradation)?"
                )
                if value > 10:
                    violations.append(
                        f"Claim '{claim_text}': Large improvement claimed. "
                        "Does this hold for all test cases or just cherry-picked examples?"
                    )

        elif unit == "multiplier":
            # Test at 1x (no improvement), <1x (degradation), and extreme values
            if value > 1:
                violations.append(
                    f"Claim '{claim_text}': What if multiplier is 1x (no improvement) "
                    "or <1x (degradation)? Is hypothesis still valid?"
                )
                if value > 10:
                    violations.append(
                        f"Claim '{claim_text}': Extreme multiplier claimed. "
                        "This often indicates measurement error or cherry-picking."
                    )

        else:  # Parameter assignment
            # Test at 0, 1, and very large values
            if value > 1:
                violations.append(
                    f"Claim '{claim_text}': Does this work when parameter = 0? "
                    "What about parameter = 1? Extreme values?"
                )

        return violations

    def _check_missing_boundaries(self, hypothesis: Hypothesis) -> list:
        """Check for missing boundary specifications"""
        violations = []

        claim = hypothesis.claim.lower()
        context = (hypothesis.context or "").lower()
        full_text = f"{claim} {context}"

        # Check if claim mentions performance without bounds
        if any(term in full_text for term in ["improves", "better", "faster", "optimal"]):
            # Look for boundary specifications
            has_conditions = any(
                term in full_text
                for term in ["when", "if", "for", "given", "assuming", "where", "with"]
            )

            if not has_conditions:
                violations.append(
                    "Hypothesis makes performance claims without specifying boundary conditions "
                    "(e.g., 'works for all N' vs 'works for N > 100')"
                )

        # Check if claim mentions "always" without exceptions
        if "always" in claim and "except" not in full_text and "unless" not in full_text:
            violations.append(
                "Hypothesis uses 'always' without specifying exceptions or boundary conditions"
            )

        # Check if assumptions list boundary conditions
        if hypothesis.assumptions:
            assumption_text = " ".join(hypothesis.assumptions).lower()
            if not any(term in assumption_text for term in ["range", "bound", "limit", "constraint", "condition"]):
                violations.append(
                    "Assumptions do not specify parameter ranges, bounds, or applicability limits"
                )

        return violations

    async def _ai_boundary_check(self, hypothesis: Hypothesis) -> list:
        """Use AI to find subtle boundary violations"""
        if not self.orchestrator:
            return []

        from atlas_orchestrator import Task, TaskType

        prompt = f"""Analyze boundary conditions for this hypothesis:

Hypothesis: {hypothesis.claim}
Context: {hypothesis.context or 'None'}
Domain: {hypothesis.domain.value}

Test at extreme values:
1. What if parameters are 0? Negative? Infinite?
2. What if dataset is empty? Has 1 element? Has 10^9 elements?
3. What if claimed improvement is actually 0%? Or degradation?
4. Are there edge cases where hypothesis breaks down?
5. Are boundary conditions properly specified?

List specific boundary violations or missing conditions.
If boundaries seem well-specified, say "Boundaries appear reasonable".
"""

        task = Task(prompt=prompt, task_type=TaskType.ANALYSIS, max_tokens=500)
        result = await self.orchestrator.execute(task)

        if result.success and "boundaries appear reasonable" not in result.content.lower():
            lines = result.content.strip().split('\n')
            violations = [
                line.strip()
                for line in lines
                if line.strip()
                and len(line.strip()) > 15
                and any(keyword in line.lower() for keyword in ['boundary', 'extreme', 'edge', 'limit', 'condition', 'violation'])
            ]
            return violations[:5]

        return []
