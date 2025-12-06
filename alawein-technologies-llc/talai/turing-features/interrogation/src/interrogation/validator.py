"""
Validator

LLM-driven validation of answer quality.
"""

from typing import Optional
from interrogation.core.models import Answer


class AnswerValidator:
    """
    Validate answer quality using AI

    Checks:
    - Relevance to question
    - Completeness
    - Evidence-based reasoning
    - Critical thinking
    """

    def __init__(self, orchestrator=None):
        """
        Initialize validator

        Args:
            orchestrator: AI Orchestrator for LLM calls
        """
        self.orchestrator = orchestrator

    async def validate_answer(self, answer: Answer) -> float:
        """
        Validate an answer and return quality score (0-100)

        Args:
            answer: Answer to validate

        Returns:
            Quality score (0-100)
        """
        if not self.orchestrator:
            # Fallback: Use heuristic validation
            return self._heuristic_validation(answer)

        # Use AI for validation
        return await self._ai_validation(answer)

    def _heuristic_validation(self, answer: Answer) -> float:
        """Simple heuristic-based validation"""
        score = 50.0
        text = answer.answer.lower()

        # Length check
        if len(answer.answer) < 30:
            score -= 20
        elif len(answer.answer) > 100:
            score += 10

        # Keywords indicating good answers
        good_keywords = [
            "evidence", "study", "research", "data",
            "specifically", "measured", "tested",
            "because", "therefore", "indicates"
        ]
        score += sum(5 for kw in good_keywords if kw in text)

        # Keywords indicating poor answers
        bad_keywords = [
            "unclear", "insufficient", "cannot determine",
            "unknown", "not specified", "missing"
        ]
        score -= sum(8 for kw in bad_keywords if kw in text)

        # Check for evidence markers
        if any(marker in text for marker in ["shown that", "demonstrated", "confirmed"]):
            score += 10

        # Clamp
        return max(0.0, min(100.0, score))

    async def _ai_validation(self, answer: Answer) -> float:
        """AI-powered validation"""
        from atlas_orchestrator import Task, TaskType

        prompt = f"""Evaluate the quality of this answer to a scientific question.

Question: {answer.question}

Answer: {answer.answer}

Rate the answer quality (0-100) based on:
1. Relevance: Does it answer the question?
2. Completeness: Is the answer thorough?
3. Evidence: Is it evidence-based?
4. Critical thinking: Does it show analytical depth?
5. Clarity: Is it clear and specific?

Respond ONLY with a number between 0-100.
"""

        task = Task(
            prompt=prompt,
            task_type=TaskType.ANALYSIS,
            max_tokens=50,
        )

        try:
            result = await self.orchestrator.execute(task)

            if result.success:
                # Extract numeric score
                score_text = result.content.strip()
                # Try to parse number
                import re
                match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(100.0, score))

        except Exception:
            pass

        # Fallback
        return self._heuristic_validation(answer)

    async def validate_category_answers(
        self,
        answers: list[Answer]
    ) -> dict:
        """
        Validate all answers in a category

        Returns:
            Dict with validation results
        """
        if not answers:
            return {
                "valid_count": 0,
                "invalid_count": 0,
                "avg_quality": 0.0,
                "issues": ["No answers to validate"],
            }

        # Validate each answer
        scores = []
        for answer in answers:
            score = await self.validate_answer(answer)
            scores.append(score)

        avg_quality = sum(scores) / len(scores)
        valid_count = sum(1 for s in scores if s >= 60)
        invalid_count = len(scores) - valid_count

        # Identify issues
        issues = []
        if invalid_count > len(answers) / 2:
            issues.append(f"More than half of answers ({invalid_count}/{len(answers)}) are low quality")

        if avg_quality < 50:
            issues.append(f"Average answer quality is poor ({avg_quality:.1f}/100)")

        low_quality_answers = [
            a.question[:50] + "..."
            for a, s in zip(answers, scores)
            if s < 40
        ]

        if low_quality_answers:
            issues.append(f"Low quality answers for: {', '.join(low_quality_answers[:3])}")

        return {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "avg_quality": avg_quality,
            "quality_scores": scores,
            "issues": issues,
        }
