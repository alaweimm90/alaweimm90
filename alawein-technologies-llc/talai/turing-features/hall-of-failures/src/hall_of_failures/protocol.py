"""
Hall of Failures Protocol

Main interface for recording, analyzing, and learning from failures.
"""

from typing import List, Optional
from pathlib import Path

from hall_of_failures.models import (
    Failure,
    FailureAnalysis,
    RiskAssessment,
    FailurePattern,
    FailureType,
)
from hall_of_failures.database import FailureDatabase
from hall_of_failures.classifier import FailureClassifier
from hall_of_failures.lesson_extractor import LessonExtractor
from hall_of_failures.similarity_matcher import SimilarityMatcher
from hall_of_failures.strategy_generator import StrategyGenerator


class HallOfFailures:
    """
    Hall of Failures - Learn from failures

    Main interface for:
    - Recording failures
    - Extracting lessons
    - Finding similar past failures
    - Generating prevention strategies
    - Risk assessment
    """

    def __init__(
        self,
        db_path: str = "failures.db",
        orchestrator=None
    ):
        """
        Initialize Hall of Failures

        Args:
            db_path: Path to SQLite database
            orchestrator: Optional AI Orchestrator for enhanced analysis
        """
        self.db = FailureDatabase(db_path)
        self.classifier = FailureClassifier()
        self.lesson_extractor = LessonExtractor(orchestrator=orchestrator)
        self.similarity_matcher = SimilarityMatcher()
        self.strategy_generator = StrategyGenerator(orchestrator=orchestrator)

        # Build similarity index from existing failures
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild similarity index from database"""
        failures = self.db.get_all_failures()
        if failures:
            self.similarity_matcher.build_index(failures)

    async def record_failure(self, failure: Failure) -> FailureAnalysis:
        """
        Record a failure and analyze it

        Args:
            failure: Failure to record

        Returns:
            FailureAnalysis with lessons and similar failures
        """
        # 1. Classify if not already classified
        if not failure.failure_type or not failure.severity:
            failure.failure_type, failure.severity = self.classifier.classify(failure)

        # 2. Identify root causes if not provided
        if not failure.root_causes:
            failure.root_causes = self.classifier.identify_root_causes(failure)

        # 3. Extract lessons
        if not failure.lessons_learned:
            failure.lessons_learned = await self.lesson_extractor.extract_lessons(failure)

        # 4. Generate prevention strategies
        if not failure.prevention_strategies:
            failure.prevention_strategies = await self.strategy_generator.generate_strategies(failure)

        # 5. Compute similarity hash
        if not failure.similarity_hash:
            failure.similarity_hash = failure.compute_similarity_hash()

        # 6. Find similar past failures
        all_failures = self.db.get_all_failures()
        similar = self.similarity_matcher.find_similar(
            failure, all_failures, top_k=5, min_similarity=0.3
        )

        similar_failures = [f for f, score in similar]
        similarity_scores = [score for f, score in similar]

        # 7. Generate insights from similar failures
        new_insights = self._generate_insights(failure, similar_failures)

        # 8. Store in database
        self.db.add_failure(failure)

        # 9. Rebuild index
        self._rebuild_index()

        # 10. Create analysis result
        return FailureAnalysis(
            failure=failure,
            similar_failures=similar_failures,
            similarity_scores=similarity_scores,
            new_insights=new_insights,
            recommended_actions=failure.prevention_strategies,
        )

    async def assess_risk(self, hypothesis: str) -> RiskAssessment:
        """
        Assess risk of a hypothesis based on past failures

        Args:
            hypothesis: Hypothesis text to assess

        Returns:
            RiskAssessment with warnings and recommendations
        """
        # Create temporary failure for matching
        temp_failure = Failure(
            hypothesis=hypothesis,
            failure_type=FailureType.HYPOTHESIS,
            description=hypothesis,
        )

        # Find similar past failures
        all_failures = self.db.get_all_failures()
        similar = self.similarity_matcher.find_similar(
            temp_failure, all_failures, top_k=10, min_similarity=0.2
        )

        similar_failures = [f for f, score in similar]
        similarity_scores = [score for f, score in similar]

        # Calculate risk score
        if not similar_failures:
            risk_score = 0.1  # Low risk if no similar failures
            risk_level = "Low"
        else:
            # Higher score = more risk
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            risk_score = min(1.0, avg_similarity * len(similar_failures) / 5)

            if risk_score >= 0.7:
                risk_level = "High"
            elif risk_score >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"

        # Generate warnings
        warnings = []
        if similar_failures:
            warnings.append(
                f"Found {len(similar_failures)} similar past failures"
            )

            # Extract common issues
            common_causes = {}
            for f in similar_failures:
                for cause in f.root_causes:
                    common_causes[cause] = common_causes.get(cause, 0) + 1

            top_causes = sorted(common_causes.items(), key=lambda x: x[1], reverse=True)[:3]
            for cause, count in top_causes:
                warnings.append(f"Common issue in past failures: {cause} ({count}x)")

        # Generate recommendations
        recommendations = []
        for failure in similar_failures[:3]:
            recommendations.extend(failure.prevention_strategies[:2])

        recommendations = list(set(recommendations))[:5]

        return RiskAssessment(
            hypothesis=hypothesis,
            risk_level=risk_level,
            risk_score=risk_score,
            similar_failures=similar_failures[:5],
            warnings=warnings,
            recommendations=recommendations,
        )

    def find_similar(
        self,
        hypothesis: str,
        top_k: int = 5
    ) -> List[Failure]:
        """
        Find similar past failures

        Args:
            hypothesis: Hypothesis to match
            top_k: Number of results

        Returns:
            List of similar failures
        """
        temp_failure = Failure(
            hypothesis=hypothesis,
            failure_type=FailureType.HYPOTHESIS,
            description=hypothesis,
        )

        all_failures = self.db.get_all_failures()
        similar = self.similarity_matcher.find_similar(
            temp_failure, all_failures, top_k=top_k
        )

        return [f for f, score in similar]

    def query(
        self,
        failure_type: Optional[FailureType] = None,
        limit: int = 20
    ) -> List[Failure]:
        """Query failures from database"""
        return self.db.query_failures(failure_type=failure_type, limit=limit)

    def get_stats(self) -> dict:
        """Get database statistics"""
        return self.db.get_stats()

    def analyze_patterns(self, failures: List[Failure]) -> List[FailurePattern]:
        """
        Analyze patterns across multiple failures

        Args:
            failures: List of failures to analyze

        Returns:
            List of identified patterns
        """
        patterns = []

        # Group by failure type
        by_type = {}
        for f in failures:
            if f.failure_type not in by_type:
                by_type[f.failure_type] = []
            by_type[f.failure_type].append(f)

        # Analyze each type
        for ftype, type_failures in by_type.items():
            if len(type_failures) < 2:
                continue

            # Find common root causes
            cause_counts = {}
            for f in type_failures:
                for cause in f.root_causes:
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1

            common_causes = [
                cause for cause, count in cause_counts.items()
                if count >= len(type_failures) * 0.3  # Appears in 30%+
            ]

            if common_causes:
                # Aggregate prevention strategies
                all_strategies = []
                for f in type_failures:
                    all_strategies.extend(f.prevention_strategies)

                unique_strategies = list(set(all_strategies))[:5]

                pattern = FailurePattern(
                    pattern_name=f"Common {ftype.value} failures",
                    description=f"Pattern of {len(type_failures)} failures in {ftype.value}",
                    failure_count=len(type_failures),
                    failure_types=[ftype],
                    common_root_causes=common_causes,
                    prevention_strategies=unique_strategies,
                    example_failures=[f.id for f in type_failures[:3]],
                )

                patterns.append(pattern)

        return patterns

    def _generate_insights(
        self,
        failure: Failure,
        similar_failures: List[Failure]
    ) -> List[str]:
        """Generate new insights from similar failures"""
        insights = []

        if not similar_failures:
            insights.append("No similar past failures found - this is a novel failure type")
            return insights

        # Check if this is a recurring pattern
        if len(similar_failures) >= 3:
            insights.append(
                f"⚠️ RECURRING PATTERN: {len(similar_failures)} similar failures recorded previously"
            )

        # Extract common themes
        all_lessons = []
        for f in similar_failures:
            all_lessons.extend(f.lessons_learned)

        if all_lessons:
            insights.append(
                f"Similar failures shared these lessons: {'; '.join(list(set(all_lessons))[:2])}"
            )

        return insights

    def generate_checklist(self, domain: Optional[str] = None) -> List[str]:
        """Generate failure prevention checklist"""
        failures = self.db.get_all_failures(limit=100)
        return self.strategy_generator.generate_checklist(failures)
