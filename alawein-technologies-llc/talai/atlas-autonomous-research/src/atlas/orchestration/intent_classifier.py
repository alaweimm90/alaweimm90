"""
ORCHEX Orchestration - Intent Classifier

Classifies user requests to determine problem type and optimal workflow.
Uses keyword matching, pattern analysis, and AI-assisted classification.

Cycle 21-22: Adaptive Orchestration
"""

from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass

from ORCHEX.orchestration.problem_types import (
    ProblemType,
    ProblemCharacteristics,
    INTENT_KEYWORDS,
    get_problem_config
)


@dataclass
class ClassificationResult:
    """Result of intent classification"""

    primary_type: ProblemType
    confidence: float  # 0-1
    secondary_types: List[Tuple[ProblemType, float]]  # Ranked alternatives
    detected_keywords: List[str]
    reasoning: str
    problem_config: ProblemCharacteristics


class IntentClassifier:
    """
    Classifies research requests to determine problem type.

    Uses multiple strategies:
    1. Keyword matching
    2. Pattern recognition
    3. Phrase analysis
    4. AI-assisted classification (if orchestrator available)
    """

    def __init__(self, orchestrator=None):
        """
        Initialize intent classifier

        Args:
            orchestrator: Optional AI Orchestrator for advanced classification
        """
        self.orchestrator = orchestrator

    def classify(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> ClassificationResult:
        """
        Classify a research query to determine problem type

        Args:
            query: User's research question/request
            domain: Research domain (e.g., "machine_learning")
            context: Additional context about the request

        Returns:
            ClassificationResult with problem type and confidence
        """
        query_lower = query.lower()

        # Strategy 1: Keyword matching
        keyword_scores = self._keyword_matching(query_lower)

        # Strategy 2: Pattern recognition
        pattern_scores = self._pattern_recognition(query_lower)

        # Strategy 3: Phrase analysis
        phrase_scores = self._phrase_analysis(query_lower)

        # Combine scores
        combined_scores = self._combine_scores(
            keyword_scores,
            pattern_scores,
            phrase_scores,
            weights=[0.4, 0.3, 0.3]
        )

        # Strategy 4: AI-assisted classification (if available)
        if self.orchestrator:
            ai_scores = self._ai_classification(query, domain, context)
            combined_scores = self._combine_scores(
                combined_scores,
                ai_scores,
                weights=[0.6, 0.4]
            )

        # Get top prediction
        sorted_types = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_type, confidence = sorted_types[0]
        secondary_types = sorted_types[1:4]  # Top 3 alternatives

        # Get detected keywords
        detected = self._get_detected_keywords(query_lower, primary_type)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            query, primary_type, confidence, detected
        )

        # Get problem configuration
        config = get_problem_config(primary_type)

        return ClassificationResult(
            primary_type=primary_type,
            confidence=confidence,
            secondary_types=secondary_types,
            detected_keywords=detected,
            reasoning=reasoning,
            problem_config=config
        )

    def _keyword_matching(self, query: str) -> Dict[ProblemType, float]:
        """Score problem types based on keyword matches"""
        scores = {pt: 0.0 for pt in ProblemType}

        for problem_type, keywords in INTENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query)
            if matches > 0:
                # Normalize by keyword list length
                scores[problem_type] = min(1.0, matches / len(keywords) * 10)

        # Default to GENERAL_RESEARCH if no matches
        if all(score == 0 for score in scores.values()):
            scores[ProblemType.GENERAL_RESEARCH] = 0.5

        return scores

    def _pattern_recognition(self, query: str) -> Dict[ProblemType, float]:
        """Score problem types based on query patterns"""
        scores = {pt: 0.0 for pt in ProblemType}

        # Question patterns
        if re.search(r'\b(is|are) (this|these|it) (new|novel|original)', query):
            scores[ProblemType.NOVELTY_CHECK] += 0.8

        if re.search(r'\b(find|discover|explore) (new|novel|other)', query):
            scores[ProblemType.DISCOVERY] += 0.8

        if re.search(r'\b(how|why) (does|do|can|should)', query):
            scores[ProblemType.EXPLANATION] += 0.7

        if re.search(r'\bcompare .* (to|with|versus|vs)', query):
            scores[ProblemType.COMPARISON] += 0.8

        if re.search(r'\boptimize|improve|enhance|better', query):
            scores[ProblemType.OPTIMIZATION] += 0.7

        if re.search(r'\bbenchmark|measure|evaluate performance', query):
            scores[ProblemType.BENCHMARK] += 0.8

        if re.search(r'\bvalidate|verify|test|prove', query):
            scores[ProblemType.VALIDATION] += 0.7

        if re.search(r'\bdesign|create|build|develop', query):
            scores[ProblemType.DESIGN] += 0.7

        if re.search(r'\bcombine|integrate|merge|synthesize', query):
            scores[ProblemType.SYNTHESIS] += 0.7

        if re.search(r'\bbrainstorm|ideas|think of ways', query):
            scores[ProblemType.IDEATION] += 0.8

        # Command patterns
        if query.startswith(('find ', 'discover ', 'explore ')):
            scores[ProblemType.DISCOVERY] += 0.5

        if query.startswith(('optimize ', 'improve ', 'enhance ')):
            scores[ProblemType.OPTIMIZATION] += 0.5

        if query.startswith(('compare ', 'benchmark ')):
            scores[ProblemType.COMPARISON] += 0.5
            scores[ProblemType.BENCHMARK] += 0.5

        return scores

    def _phrase_analysis(self, query: str) -> Dict[ProblemType, float]:
        """Score problem types based on phrase analysis"""
        scores = {pt: 0.0 for pt in ProblemType}

        # Multi-word phrases (stronger signals)
        strong_signals = {
            ProblemType.NOVELTY_CHECK: [
                "is this new", "is this novel", "has this been done",
                "similar work", "prior art", "state of the art"
            ],
            ProblemType.DISCOVERY: [
                "find new methods", "discover approaches", "what are other",
                "new ways to", "alternatives to"
            ],
            ProblemType.IDEATION: [
                "brainstorm ideas", "come up with", "think of ways",
                "creative solutions", "generate ideas"
            ],
            ProblemType.VALIDATION: [
                "validate hypothesis", "test this claim", "prove that",
                "verify this", "confirm hypothesis"
            ],
            ProblemType.BENCHMARK: [
                "benchmark performance", "measure speed", "evaluate performance",
                "performance comparison", "throughput test"
            ],
            ProblemType.COMPARISON: [
                "compare x and y", "x versus y", "difference between",
                "better than", "advantages of"
            ],
            ProblemType.OPTIMIZATION: [
                "optimize performance", "make it faster", "improve efficiency",
                "reduce latency", "increase throughput"
            ],
            ProblemType.PARAMETER_TUNING: [
                "tune parameters", "find best parameters", "optimal settings",
                "hyperparameter search", "grid search"
            ],
            ProblemType.ANALYSIS: [
                "analyze this", "investigate why", "examine how",
                "study the effect", "understand the behavior"
            ],
            ProblemType.EXPLANATION: [
                "explain why", "explain how", "what causes",
                "why does this", "how does this work"
            ],
            ProblemType.SYNTHESIS: [
                "combine x and y", "integrate approaches", "merge methods",
                "synthesize ideas", "fusion of"
            ],
            ProblemType.DESIGN: [
                "design a system", "create an algorithm", "build a framework",
                "develop a method", "engineer a solution"
            ],
        }

        for problem_type, phrases in strong_signals.items():
            for phrase in phrases:
                if phrase in query:
                    scores[problem_type] += 0.9

        return scores

    def _ai_classification(
        self,
        query: str,
        domain: Optional[str],
        context: Optional[Dict]
    ) -> Dict[ProblemType, float]:
        """Use AI orchestrator for advanced classification"""
        # TODO: Implement AI-assisted classification
        # For now, return neutral scores
        return {pt: 0.5 for pt in ProblemType}

    def _combine_scores(
        self,
        *score_dicts: Dict[ProblemType, float],
        weights: Optional[List[float]] = None
    ) -> Dict[ProblemType, float]:
        """Combine multiple scoring strategies"""
        if weights is None:
            weights = [1.0 / len(score_dicts)] * len(score_dicts)

        combined = {pt: 0.0 for pt in ProblemType}

        for score_dict, weight in zip(score_dicts, weights):
            for pt, score in score_dict.items():
                combined[pt] += score * weight

        # Normalize to 0-1
        max_score = max(combined.values()) if combined.values() else 1.0
        if max_score > 0:
            combined = {pt: score / max_score for pt, score in combined.items()}

        return combined

    def _get_detected_keywords(
        self,
        query: str,
        problem_type: ProblemType
    ) -> List[str]:
        """Get keywords that were detected for this problem type"""
        if problem_type not in INTENT_KEYWORDS:
            return []

        keywords = INTENT_KEYWORDS[problem_type]
        detected = [kw for kw in keywords if kw in query]
        return detected[:5]  # Top 5

    def _generate_reasoning(
        self,
        query: str,
        problem_type: ProblemType,
        confidence: float,
        keywords: List[str]
    ) -> str:
        """Generate human-readable reasoning for classification"""
        reasoning_parts = [
            f"Classified as {problem_type.value} with {confidence:.1%} confidence."
        ]

        if keywords:
            kw_str = ", ".join(f'"{kw}"' for kw in keywords[:3])
            reasoning_parts.append(f"Detected keywords: {kw_str}")

        # Add explanation based on type
        explanations = {
            ProblemType.NOVELTY_CHECK: "Query asks about novelty or existing work.",
            ProblemType.DISCOVERY: "Query seeks to discover new approaches or methods.",
            ProblemType.IDEATION: "Query requests brainstorming or idea generation.",
            ProblemType.VALIDATION: "Query aims to validate a hypothesis or claim.",
            ProblemType.BENCHMARK: "Query requests performance benchmarking.",
            ProblemType.COMPARISON: "Query compares multiple approaches or methods.",
            ProblemType.OPTIMIZATION: "Query seeks to optimize or improve something.",
            ProblemType.PARAMETER_TUNING: "Query focuses on parameter optimization.",
            ProblemType.ANALYSIS: "Query requests analysis or investigation.",
            ProblemType.EXPLANATION: "Query asks for explanation or understanding.",
            ProblemType.SYNTHESIS: "Query aims to combine or synthesize ideas.",
            ProblemType.DESIGN: "Query requests design or creation of a solution.",
            ProblemType.GENERAL_RESEARCH: "Standard research query.",
        }

        if problem_type in explanations:
            reasoning_parts.append(explanations[problem_type])

        return " ".join(reasoning_parts)

    def classify_batch(
        self,
        queries: List[str],
        domain: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Classify multiple queries in batch"""
        return [self.classify(q, domain) for q in queries]

    def get_problem_summary(
        self,
        classification: ClassificationResult
    ) -> str:
        """Get human-readable summary of problem classification"""
        config = classification.problem_config

        summary_parts = [
            f"Problem Type: {classification.primary_type.value}",
            f"Confidence: {classification.confidence:.1%}",
            f"",
            "Pipeline Configuration:",
            f"  • Hypothesis emphasis: {config.hypothesis_emphasis:.0%}",
            f"  • Validation emphasis: {config.validation_emphasis:.0%}",
            f"  • Experiment emphasis: {config.experiment_emphasis:.0%}",
            f"  • Paper emphasis: {config.paper_emphasis:.0%}",
            f"",
            f"Max hypotheses: {config.max_hypotheses}",
            f"Max validation rounds: {config.max_validation_rounds}",
            f"Max experiment variants: {config.max_experiment_variants}",
        ]

        if config.enable_brainstorming:
            summary_parts.append("✓ Brainstorming enabled")
        if config.enable_novelty_scoring:
            summary_parts.append("✓ Novelty scoring enabled")
        if config.parallel_hypothesis_generation:
            summary_parts.append("✓ Parallel hypothesis generation")

        if config.preferred_agents:
            summary_parts.append("")
            summary_parts.append("Preferred Agents:")
            for role, priority in sorted(
                config.preferred_agents.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                summary_parts.append(f"  • {role}: {priority:.0%} priority")

        return "\n".join(summary_parts)


# Convenience function
def classify_intent(
    query: str,
    domain: Optional[str] = None,
    orchestrator=None
) -> ClassificationResult:
    """
    Classify a research query to determine problem type

    Args:
        query: User's research question/request
        domain: Research domain
        orchestrator: Optional AI Orchestrator

    Returns:
        ClassificationResult
    """
    classifier = IntentClassifier(orchestrator=orchestrator)
    return classifier.classify(query, domain)
