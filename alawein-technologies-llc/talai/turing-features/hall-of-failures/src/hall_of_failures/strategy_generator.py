"""
Strategy generator - Generates prevention strategies from failures
"""

from typing import List
from hall_of_failures.models import Failure


class StrategyGenerator:
    """
    Generate prevention strategies from failures

    Uses lessons learned and failure patterns to create actionable strategies.
    """

    def __init__(self, orchestrator=None):
        """
        Initialize strategy generator

        Args:
            orchestrator: Optional AI Orchestrator for LLM-powered generation
        """
        self.orchestrator = orchestrator

    async def generate_strategies(self, failure: Failure) -> List[str]:
        """
        Generate prevention strategies

        Args:
            failure: Failure to generate strategies for

        Returns:
            List of prevention strategies
        """
        if self.orchestrator:
            return await self._ai_generation(failure)
        else:
            return self._heuristic_generation(failure)

    async def _ai_generation(self, failure: Failure) -> List[str]:
        """AI-powered strategy generation"""
        from atlas_orchestrator import Task, TaskType

        lessons_text = '\n'.join(f"• {lesson}" for lesson in failure.lessons_learned)

        prompt = f"""Generate prevention strategies based on this failure:

Failure Type: {failure.failure_type.value}
Root Causes: {', '.join(failure.root_causes)}

Lessons Learned:
{lessons_text}

Generate 3-5 specific, actionable prevention strategies.
Focus on concrete steps to avoid this failure in future work.

Format: One strategy per line, starting with "•"
"""

        task = Task(prompt=prompt, task_type=TaskType.ANALYSIS, max_tokens=500)

        try:
            result = await self.orchestrator.execute(task)

            if result.success:
                # Parse strategies
                strategies = []
                for line in result.content.strip().split('\n'):
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-')):
                        strategy = line.lstrip('•-').strip()
                        if len(strategy) > 10:
                            strategies.append(strategy)

                return strategies[:5] if strategies else self._heuristic_generation(failure)
        except Exception:
            pass

        # Fallback
        return self._heuristic_generation(failure)

    def _heuristic_generation(self, failure: Failure) -> List[str]:
        """Heuristic-based strategy generation"""
        strategies = []

        # Generate from failure type
        if failure.failure_type == "hypothesis":
            strategies.append("Use Self-Refutation Protocol before experiments")
            strategies.append("Run 200-Question Interrogation for deep validation")
            strategies.append("Have peers review hypothesis for logical flaws")

        elif failure.failure_type == "experimental":
            strategies.append("Conduct power analysis and pre-register studies")
            strategies.append("Use proper randomization and blinding")
            strategies.append("Include manipulation checks and pilot studies")

            if "sample size" in failure.description.lower():
                strategies.append("Calculate minimum sample size using power analysis (aim for 80% power)")

            if "bias" in failure.description.lower():
                strategies.append("Implement double-blinding to reduce bias")

        elif failure.failure_type == "computational":
            strategies.append("Write comprehensive unit tests (>80% coverage)")
            strategies.append("Use continuous integration to catch bugs early")
            strategies.append("Perform code reviews before deployment")

        elif failure.failure_type == "integration":
            strategies.append("Create integration tests for all external interfaces")
            strategies.append("Use API contracts and schema validation")
            strategies.append("Test with realistic data formats")

        elif failure.failure_type == "theoretical":
            strategies.append("Consult domain experts before proposing novel mechanisms")
            strategies.append("Check literature for similar proposed mechanisms")
            strategies.append("Verify mechanism with simulations/toy models")

        # Generate from root causes
        for cause in failure.root_causes:
            if "sample size" in cause.lower():
                strategies.append("Use sequential testing or adaptive designs to optimize sample size")
            elif "control" in cause.lower():
                strategies.append("Include both positive and negative controls")
            elif "confounding" in cause.lower():
                strategies.append("Use propensity score matching or stratification")

        # Deduplicate
        strategies = list(set(strategies))

        return strategies[:5]

    def generate_checklist(self, failures: List[Failure]) -> List[str]:
        """
        Generate a prevention checklist from multiple failures

        Args:
            failures: List of failures to learn from

        Returns:
            Checklist of prevention items
        """
        checklist_items = set()

        # Extract strategies from all failures
        for failure in failures:
            for strategy in failure.prevention_strategies:
                checklist_items.add(strategy)

        # Add general best practices
        checklist_items.add("Have hypothesis reviewed by peers")
        checklist_items.add("Check for similar past failures in Hall of Failures")
        checklist_items.add("Document all assumptions explicitly")
        checklist_items.add("Pre-register analysis plan when appropriate")

        return sorted(list(checklist_items))
