"""
Lesson extractor - Extracts actionable lessons from failures
"""

from typing import List
from hall_of_failures.models import Failure


class LessonExtractor:
    """
    Extract lessons learned from failures

    Uses AI (if available) or heuristic-based extraction.
    """

    def __init__(self, orchestrator=None):
        """
        Initialize lesson extractor

        Args:
            orchestrator: Optional AI Orchestrator for LLM-powered extraction
        """
        self.orchestrator = orchestrator

    async def extract_lessons(self, failure: Failure) -> List[str]:
        """
        Extract lessons from a failure

        Args:
            failure: Failure to analyze

        Returns:
            List of actionable lessons
        """
        if self.orchestrator:
            return await self._ai_extraction(failure)
        else:
            return self._heuristic_extraction(failure)

    async def _ai_extraction(self, failure: Failure) -> List[str]:
        """AI-powered lesson extraction"""
        from atlas_orchestrator import Task, TaskType

        prompt = f"""Analyze this research failure and extract key lessons:

Failure Type: {failure.failure_type.value}
Hypothesis: {failure.hypothesis}
Description: {failure.description}
Root Causes: {', '.join(failure.root_causes) if failure.root_causes else 'Unknown'}

Extract 3-5 specific, actionable lessons learned.
Focus on what went wrong and how to avoid it.

Format: One lesson per line, starting with "•"
"""

        task = Task(prompt=prompt, task_type=TaskType.ANALYSIS, max_tokens=500)

        try:
            result = await self.orchestrator.execute(task)

            if result.success:
                # Parse lessons from output
                lessons = []
                for line in result.content.strip().split('\n'):
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-')):
                        lesson = line.lstrip('•-').strip()
                        if len(lesson) > 10:
                            lessons.append(lesson)

                return lessons[:5] if lessons else self._heuristic_extraction(failure)
        except Exception:
            pass

        # Fallback to heuristic
        return self._heuristic_extraction(failure)

    def _heuristic_extraction(self, failure: Failure) -> List[str]:
        """Heuristic-based lesson extraction"""
        lessons = []

        # Extract lessons based on failure type
        if failure.failure_type == "hypothesis":
            lessons.append("Ensure hypothesis is falsifiable with clear success/failure criteria")
            lessons.append("Define boundary conditions and scope explicitly")

            if "contradiction" in failure.description.lower():
                lessons.append("Check for logical contradictions before proceeding")

        elif failure.failure_type == "experimental":
            lessons.append("Conduct power analysis before starting experiments")

            if "sample size" in failure.description.lower():
                lessons.append("Use adequate sample size based on expected effect size")

            if "control" in failure.description.lower():
                lessons.append("Include proper control groups to isolate causal effects")

            if "bias" in failure.description.lower():
                lessons.append("Identify and control for potential sources of bias")

        elif failure.failure_type == "computational":
            lessons.append("Thoroughly test implementation before running experiments")

            if "bug" in failure.description.lower():
                lessons.append("Add comprehensive unit tests for critical components")

            if "numerical" in failure.description.lower():
                lessons.append("Check for numerical stability issues")

        elif failure.failure_type == "theoretical":
            lessons.append("Verify mechanism doesn't violate known physical/mathematical principles")
            lessons.append("Consult domain literature before proposing new mechanisms")

        # Generic lessons from root causes
        for cause in failure.root_causes:
            if "sample size" in cause.lower():
                lessons.append(f"Address identified root cause: {cause}")

        # Deduplicate and limit
        lessons = list(set(lessons))
        return lessons[:5]
