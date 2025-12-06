"""
Interrogation Protocol

Main orchestrator for 200-question interrogation framework.
"""

import time
import asyncio
from typing import List, Optional
from pathlib import Path

from interrogation.question_loader import QuestionLoader
from interrogation.interrogator import Interrogator
from interrogation.scorer import InterrogationScorer
from interrogation.validator import AnswerValidator
from interrogation.core.models import InterrogationResult, CategoryResult


class InterrogationProtocol:
    """
    200-Question Interrogation Framework

    Systematically tests hypotheses using 200 guided questions
    across 10 categories with weighted scoring.

    Usage:
        protocol = InterrogationProtocol(orchestrator=orchestrator)
        result = await protocol.interrogate(hypothesis)
    """

    def __init__(
        self,
        orchestrator=None,
        database_path: Optional[Path] = None,
        **config
    ):
        """
        Initialize protocol

        Args:
            orchestrator: AI Orchestrator for LLM calls
            database_path: Path to question database (None = auto-find)
            **config: Configuration options
                - use_consensus: Use multi-model consensus (default: False)
                - parallel: Process questions in parallel (default: True)
                - adaptive: Ask more questions in weak areas (default: False)
                - strict_mode: More critical scoring (default: False)
                - categories: List of category names to interrogate (None = all)
                - questions_per_category: Limit questions per category (None = all 20)
        """
        self.orchestrator = orchestrator
        self.config = config

        # Initialize components
        self.loader = QuestionLoader(database_path)
        self.interrogator = Interrogator(orchestrator=orchestrator, **config)
        self.scorer = InterrogationScorer(**config)
        self.validator = AnswerValidator(orchestrator=orchestrator) if orchestrator else None

        # Configuration
        self.use_consensus = config.get("use_consensus", False)
        self.parallel = config.get("parallel", True)
        self.adaptive = config.get("adaptive", False)
        self.selected_categories = config.get("categories", None)
        self.questions_per_category = config.get("questions_per_category", None)

    async def interrogate(self, hypothesis) -> InterrogationResult:
        """
        Interrogate hypothesis with 200 questions

        Args:
            hypothesis: Hypothesis to interrogate (must have .claim attribute)

        Returns:
            InterrogationResult with scores and analysis
        """
        start_time = time.time()

        # Get categories to interrogate
        categories = self._select_categories()

        # Interrogate each category
        category_results = []
        total_tokens = 0

        for category in categories:
            print(f"  Interrogating category: {category.name}...")

            # Get questions for this category
            questions = self._select_questions(category)

            # Ask questions
            answers = await self.interrogator.ask_questions(
                questions=questions,
                hypothesis=hypothesis,
                context=f"Category: {category.name}"
            )

            # Score category
            category_result = self.scorer.score_category(
                category_name=category.name,
                category_weight=category.weight,
                answers=answers,
                validator=self.validator
            )

            category_results.append(category_result)
            total_tokens += sum(a.tokens_used for a in answers)

            print(f"    → Score: {category_result.raw_score:.1f}/100")

        # Calculate overall score
        overall_score = self.scorer.calculate_overall_score(category_results)

        # Categorize results
        categorized = self.scorer.categorize_results(category_results)

        # Generate recommendations and identify failure points
        recommendations = self._generate_recommendations(
            category_results, overall_score, categorized
        )
        failure_points = self._identify_failure_points(category_results)

        # Build result
        result = InterrogationResult(
            hypothesis=hypothesis,
            overall_score=overall_score,
            category_results=category_results,
            strong_categories=categorized["strong"],
            adequate_categories=categorized["adequate"],
            weak_categories=categorized["weak"],
            critical_categories=categorized["critical"],
            failure_points=failure_points,
            recommendations=recommendations,
            total_questions=sum(cr.questions_asked for cr in category_results),
            total_tokens=total_tokens,
            execution_time_seconds=time.time() - start_time,
            models_used=list(set(
                a.model for cr in category_results for a in cr.answers
            )),
            consensus_used=self.use_consensus,
        )

        return result

    async def interrogate_batch(self, hypotheses: List) -> List[InterrogationResult]:
        """
        Interrogate multiple hypotheses

        Args:
            hypotheses: List of hypotheses

        Returns:
            List of InterrogationResults
        """
        tasks = [self.interrogate(h) for h in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                processed.append(self._create_error_result(hypotheses[i], str(result)))
            else:
                processed.append(result)

        return processed

    def _select_categories(self) -> List:
        """Select categories to interrogate"""
        all_categories = self.loader.get_all_categories()

        if self.selected_categories:
            # Filter by selected
            return [
                cat for cat in all_categories
                if cat.name in self.selected_categories
            ]
        else:
            return all_categories

    def _select_questions(self, category) -> List[str]:
        """Select questions from category"""
        questions = category.questions

        if self.questions_per_category:
            # Limit number of questions
            questions = questions[:self.questions_per_category]

        return questions

    def _generate_recommendations(
        self,
        category_results: List[CategoryResult],
        overall_score: float,
        categorized: dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Overall assessment
        if overall_score >= 85:
            recommendations.append("✅ Excellent hypothesis - Ready for publication/grant submission")
            recommendations.append("Consider pre-registering the study to enhance credibility")
        elif overall_score >= 70:
            recommendations.append("✓ Good hypothesis - Address minor weaknesses before proceeding")
        elif overall_score >= 55:
            recommendations.append("⚠ Adequate hypothesis - Moderate revision recommended")
        elif overall_score >= 40:
            recommendations.append("🔴 Weak hypothesis - Substantial rework needed")
        else:
            recommendations.append("❌ Poor hypothesis - Major reformulation or rejection advised")

        # Address weak categories
        if categorized["critical"]:
            recommendations.append(
                f"CRITICAL: Address failures in {', '.join(categorized['critical'])}"
            )

        if categorized["weak"]:
            recommendations.append(
                f"Address weaknesses in: {', '.join(categorized['weak'])}"
            )

        # Category-specific recommendations
        for result in category_results:
            if result.raw_score < 60:
                # Add specific recommendations for weak categories
                if result.category_name == "Falsifiability":
                    recommendations.append(
                        "• Improve falsifiability: Specify exact conditions that would disprove hypothesis"
                    )
                elif result.category_name == "Evidence Quality":
                    recommendations.append(
                        "• Strengthen evidence: Increase sample size, add controls, replicate findings"
                    )
                elif result.category_name == "Mechanism":
                    recommendations.append(
                        "• Clarify mechanism: Explain causal pathway with supporting theory"
                    )
                elif result.category_name == "Experimental Design":
                    recommendations.append(
                        "• Improve design: Add proper controls, blinding, and power analysis"
                    )
                elif result.category_name == "Statistics & Analysis":
                    recommendations.append(
                        "• Strengthen statistics: Perform power analysis, adjust for multiple comparisons"
                    )
                elif result.category_name == "Reproducibility":
                    recommendations.append(
                        "• Enhance reproducibility: Share code, data, and detailed methods"
                    )

        return recommendations[:10]  # Limit to top 10

    def _identify_failure_points(
        self,
        category_results: List[CategoryResult]
    ) -> List[str]:
        """Identify critical failure points"""
        failures = []

        for result in category_results:
            if result.raw_score < 40:
                # Critical failure
                failures.append(
                    f"Critical weakness in {result.category_name} (score: {result.raw_score:.1f}/100)"
                )

                # Add key issues from this category
                failures.extend(result.key_issues[:2])

        # Add any other critical issues
        for result in category_results:
            if result.key_issues and result.raw_score < 60:
                for issue in result.key_issues[:1]:
                    if issue not in failures:
                        failures.append(issue)

        return failures[:8]  # Limit to top 8

    def _create_error_result(self, hypothesis, error_msg: str) -> InterrogationResult:
        """Create error result for failed interrogation"""
        return InterrogationResult(
            hypothesis=hypothesis,
            overall_score=0.0,
            category_results=[],
            strong_categories=[],
            adequate_categories=[],
            weak_categories=[],
            critical_categories=["All"],
            failure_points=[f"Interrogation failed: {error_msg}"],
            recommendations=["Fix protocol error and retry"],
            total_questions=0,
            total_tokens=0,
            execution_time_seconds=0.0,
            models_used=[],
            consensus_used=False,
        )

    def get_database_info(self) -> dict:
        """Get information about question database"""
        return self.loader.get_info()

    def list_categories(self) -> List[str]:
        """List available categories"""
        return self.loader.get_category_names()

    def validate_database(self) -> dict:
        """Validate question database"""
        return self.loader.validate_database()
