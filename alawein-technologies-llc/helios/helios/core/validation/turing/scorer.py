"""
Scorer

Calculates weighted scores from interrogation answers.
"""

from typing import List, Dict, Tuple
import re
from interrogation.core.models import Answer, CategoryResult


class InterrogationScorer:
    """
    Calculate scores from interrogation answers

    Scoring algorithm:
    1. Score each answer (0-100) based on quality
    2. Average scores within category → raw_score
    3. Apply category weight → weighted_score
    4. Aggregate across categories → overall_score
    """

    def __init__(self, **config):
        """
        Initialize scorer

        Args:
            **config: Configuration options
                - strict_mode: More critical scoring (default: False)
                - quality_threshold: Minimum quality score (default: 40)
        """
        self.config = config
        self.strict_mode = config.get("strict_mode", False)
        self.quality_threshold = config.get("quality_threshold", 40)

    def score_category(
        self,
        category_name: str,
        category_weight: float,
        answers: List[Answer],
        validator=None
    ) -> CategoryResult:
        """
        Score answers for a single category

        Args:
            category_name: Name of the category
            category_weight: Category weight for scoring
            answers: List of answers for this category
            validator: Optional validator for quality checks

        Returns:
            CategoryResult with scores and analysis
        """
        if not answers:
            return CategoryResult(
                category_name=category_name,
                category_weight=category_weight,
                raw_score=0.0,
                weighted_score=0.0,
                questions_asked=0,
                answers=[],
                strengths=[],
                weaknesses=["No answers provided"],
                key_issues=["Category not evaluated"],
            )

        # Score each answer
        answer_scores = [
            self._score_answer(answer, validator)
            for answer in answers
        ]

        # Calculate raw score (average)
        raw_score = sum(answer_scores) / len(answer_scores)

        # Apply weight
        weighted_score = raw_score  # Weight applied at overall level

        # Analyze strengths and weaknesses
        strengths, weaknesses, key_issues = self._analyze_category(
            answers, answer_scores, category_name
        )

        return CategoryResult(
            category_name=category_name,
            category_weight=category_weight,
            raw_score=raw_score,
            weighted_score=weighted_score,
            questions_asked=len(answers),
            answers=answers,
            strengths=strengths,
            weaknesses=weaknesses,
            key_issues=key_issues,
            metadata={
                "answer_scores": answer_scores,
                "min_score": min(answer_scores),
                "max_score": max(answer_scores),
            }
        )

    def calculate_overall_score(
        self,
        category_results: List[CategoryResult]
    ) -> float:
        """
        Calculate overall weighted score

        Formula:
            overall = Σ(category_score × weight) / Σ(weight)

        Args:
            category_results: List of category results

        Returns:
            Overall score (0-100)
        """
        if not category_results:
            return 0.0

        total_weighted_score = sum(
            result.raw_score * result.category_weight
            for result in category_results
        )

        total_weight = sum(
            result.category_weight
            for result in category_results
        )

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _score_answer(self, answer: Answer, validator=None) -> float:
        """
        Score a single answer (0-100)

        Considers:
        - Answer confidence
        - Answer quality (length, specificity, concerns raised)
        - Validator score (if provided)
        """
        score = 50.0  # Base score

        # Factor 1: Confidence (±30 points)
        confidence_contribution = (answer.confidence - 0.5) * 60
        score += confidence_contribution

        # Factor 2: Answer quality
        quality_score = self._assess_answer_quality(answer.answer)
        score += quality_score

        # Factor 3: Validator (if available)
        if validator:
            try:
                validator_score = validator.validate_answer(answer)
                score = (score + validator_score) / 2  # Average
            except:
                pass  # Skip if validator fails

        # Apply strict mode penalty
        if self.strict_mode:
            score *= 0.9  # 10% penalty in strict mode

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def _assess_answer_quality(self, answer_text: str) -> float:
        """
        Assess answer quality based on content

        Returns score adjustment (-20 to +20)
        """
        score_adj = 0.0
        answer_lower = answer_text.lower()

        # Positive indicators (+)
        if len(answer_text) > 100:  # Detailed answer
            score_adj += 5

        if any(keyword in answer_lower for keyword in ["evidence", "data", "study", "research"]):
            score_adj += 5

        if any(keyword in answer_lower for keyword in ["specifically", "precisely", "quantitative"]):
            score_adj += 5

        # Negative indicators (-)
        if len(answer_text) < 50:  # Too brief
            score_adj -= 5

        if any(keyword in answer_lower for keyword in ["unclear", "insufficient", "unknown", "cannot determine"]):
            score_adj -= 10

        if any(keyword in answer_lower for keyword in ["major concern", "critical flaw", "significant issue"]):
            score_adj -= 10

        if "[error]" in answer_lower or "[placeholder]" in answer_lower:
            score_adj -= 20

        # Vague language
        if answer_text.count("possibly") + answer_text.count("maybe") > 2:
            score_adj -= 5

        return score_adj

    def _analyze_category(
        self,
        answers: List[Answer],
        scores: List[float],
        category_name: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyze category to extract strengths, weaknesses, and key issues

        Returns:
            (strengths, weaknesses, key_issues)
        """
        strengths = []
        weaknesses = []
        key_issues = []

        avg_score = sum(scores) / len(scores)

        # Overall assessment
        if avg_score >= 80:
            strengths.append(f"{category_name} is well-addressed with strong evidence")
        elif avg_score >= 60:
            strengths.append(f"{category_name} is adequately addressed")
        elif avg_score >= 40:
            weaknesses.append(f"{category_name} has moderate concerns")
        else:
            key_issues.append(f"{category_name} has critical weaknesses (score: {avg_score:.1f}/100)")

        # Analyze specific answer patterns
        all_answers_text = " ".join([a.answer.lower() for a in answers])

        # Look for recurring concerns
        concern_patterns = {
            "sample size": ["small sample", "insufficient sample", "sample size", "n="],
            "replication": ["not replicated", "no replication", "replication failed"],
            "controls": ["no control", "insufficient control", "lacking control"],
            "mechanism": ["unclear mechanism", "no mechanism", "mechanism unknown"],
            "statistics": ["not significant", "p >", "underpowered"],
            "bias": ["selection bias", "confirmation bias", "publication bias"],
        }

        for concern_type, patterns in concern_patterns.items:
            if any(pattern in all_answers_text for pattern in patterns):
                weaknesses.append(f"Concern raised regarding {concern_type}")

        # Look for positive indicators
        strength_patterns = {
            "well-designed": ["well-designed", "rigorous", "systematic"],
            "replicated": ["replicated", "reproduced", "confirmed"],
            "evidence": ["strong evidence", "substantial evidence", "clear evidence"],
        }

        for strength_type, patterns in strength_patterns.items():
            if any(pattern in all_answers_text for pattern in patterns):
                strengths.append(f"Positive indication: {strength_type}")

        # Identify critical issues from low-scoring answers
        low_score_threshold = 30
        critical_answers = [
            (a, s) for a, s in zip(answers, scores)
            if s < low_score_threshold
        ]

        if len(critical_answers) >= 3:
            key_issues.append(
                f"Multiple questions ({len(critical_answers)}) scored very low, "
                f"indicating fundamental problems"
            )

        # Deduplicate and limit
        strengths = list(set(strengths))[:5]
        weaknesses = list(set(weaknesses))[:7]
        key_issues = list(set(key_issues))[:5]

        return strengths, weaknesses, key_issues

    def categorize_results(
        self,
        category_results: List[CategoryResult]
    ) -> Dict[str, List[str]]:
        """
        Categorize category results by score ranges

        Returns:
            Dict with keys: strong, adequate, weak, critical
        """
        categorized = {
            "strong": [],      # ≥80
            "adequate": [],    # 60-79
            "weak": [],        # 40-59
            "critical": [],    # <40
        }

        for result in category_results:
            score = result.raw_score
            name = result.category_name

            if score >= 80:
                categorized["strong"].append(name)
            elif score >= 60:
                categorized["adequate"].append(name)
            elif score >= 40:
                categorized["weak"].append(name)
            else:
                categorized["critical"].append(name)

        return categorized
