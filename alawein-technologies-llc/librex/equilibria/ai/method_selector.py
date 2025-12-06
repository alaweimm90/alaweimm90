"""
AI-powered method selection for optimization problems

This module provides intelligent method selection based on problem characteristics.
It uses a combination of heuristic rules and optional machine learning models.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Librex.ai.features import ProblemFeatureExtractor, ProblemFeatures
from Librex.core.interfaces import StandardizedProblem, UniversalOptimizationInterface

logger = logging.getLogger(__name__)


@dataclass
class MethodRecommendation:
    """A method recommendation with confidence and explanation"""

    method_name: str
    config: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    explanation: str
    supporting_evidence: List[str]
    method: Optional[str] = None
    expected_time: Optional[float] = None
    expected_quality: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceRecord:
    """Record of method performance on a problem"""

    timestamp: str
    problem_features: Dict[str, Any]
    method: str
    config: Dict[str, Any]
    objective_value: float
    iterations: int
    runtime_seconds: float
    success: bool


class MethodSelector:
    """
    AI-powered method selector for optimization problems

    Uses heuristic rules and optional ML models to recommend the best
    optimization method for a given problem.
    """

    # Available methods with their characteristics
    METHODS = {
        'random_search': {
            'complexity': 'O(n)',
            'exploration': 'high',
            'exploitation': 'low',
            'memory': 'low',
            'parallelizable': True,
            'good_for': ['quick baseline', 'no structure', 'initial exploration'],
            'bad_for': ['structured problems', 'precision required']
        },
        'simulated_annealing': {
            'complexity': 'O(n)',
            'exploration': 'medium-high',
            'exploitation': 'medium',
            'memory': 'low',
            'parallelizable': False,
            'good_for': ['discrete optimization', 'escaping local optima', 'QAP', 'TSP'],
            'bad_for': ['very large problems', 'tight time constraints']
        },
        'local_search': {
            'complexity': 'O(n^2)',
            'exploration': 'low',
            'exploitation': 'high',
            'memory': 'low',
            'parallelizable': True,
            'good_for': ['refinement', 'good initial solution', 'small problems'],
            'bad_for': ['many local optima', 'large search space']
        },
        'genetic_algorithm': {
            'complexity': 'O(n*pop_size)',
            'exploration': 'high',
            'exploitation': 'medium',
            'memory': 'high',
            'parallelizable': True,
            'good_for': ['complex fitness landscape', 'population diversity', 'parallel search'],
            'bad_for': ['small problems', 'memory constraints', 'quick solutions']
        },
        'tabu_search': {
            'complexity': 'O(n*tabu_size)',
            'exploration': 'medium',
            'exploitation': 'high',
            'memory': 'medium',
            'parallelizable': False,
            'good_for': ['avoiding cycles', 'systematic exploration', 'combinatorial problems'],
            'bad_for': ['continuous optimization', 'very large neighborhoods']
        },
        # Advanced methods (will be supported in recommendations if available)
        'ant_colony': {
            'complexity': 'O(n^2*ants)',
            'exploration': 'high',
            'exploitation': 'medium-high',
            'memory': 'medium-high',
            'parallelizable': True,
            'good_for': ['path problems', 'TSP', 'routing', 'graph problems'],
            'bad_for': ['small problems', 'non-graph problems']
        },
        'particle_swarm': {
            'complexity': 'O(n*particles)',
            'exploration': 'medium-high',
            'exploitation': 'medium',
            'memory': 'medium',
            'parallelizable': True,
            'good_for': ['continuous optimization', 'multi-modal functions', 'real-valued problems'],
            'bad_for': ['discrete problems', 'permutation problems']
        },
        'variable_neighborhood': {
            'complexity': 'O(n^2*k)',
            'exploration': 'medium-high',
            'exploitation': 'high',
            'memory': 'low-medium',
            'parallelizable': False,
            'good_for': ['local optima escape', 'systematic neighborhood exploration'],
            'bad_for': ['very large problems', 'time-critical applications']
        },
        'iterated_local_search': {
            'complexity': 'O(n^2*restarts)',
            'exploration': 'medium',
            'exploitation': 'high',
            'memory': 'low',
            'parallelizable': True,
            'good_for': ['refinement with restarts', 'avoiding local optima', 'QAP'],
            'bad_for': ['problems needing global exploration']
        },
        'grasp': {
            'complexity': 'O(n^2*iterations)',
            'exploration': 'medium',
            'exploitation': 'medium-high',
            'memory': 'low',
            'parallelizable': True,
            'good_for': ['construction heuristics', 'combinatorial problems', 'QAP'],
            'bad_for': ['continuous optimization', 'very large problems']
        }
    }

    def __init__(
        self,
        use_ml_model: bool = False,
        performance_log_path: Optional[str] = None
    ):
        """
        Initialize the method selector

        Args:
            use_ml_model: Whether to use ML model for predictions (requires training data)
            performance_log_path: Path to store performance records for training
        """
        self.feature_extractor = ProblemFeatureExtractor()
        self.use_ml_model = use_ml_model
        self.ml_model = None

        # Performance tracking for future ML training
        if performance_log_path:
            self.performance_log_path = Path(performance_log_path)
            self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.performance_log_path = None

        # Load ML model if requested
        if use_ml_model:
            self._load_ml_model()

    def recommend_method(
        self,
        problem: Any,
        adapter: Optional[UniversalOptimizationInterface] = None,
        standardized_problem: Optional[StandardizedProblem] = None,
        time_budget: Optional[float] = None,
        quality_requirement: str = 'balanced'  # 'fast', 'balanced', 'best'
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Recommend the best optimization method for a problem

        Args:
            problem: The optimization problem
            adapter: Domain adapter (required if standardized_problem not provided)
            standardized_problem: Already standardized problem (optional)
            time_budget: Maximum time in seconds (affects method choice)
            quality_requirement: Quality vs speed preference

        Returns:
            Tuple of (method_name, config, confidence)
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            problem, adapter, standardized_problem
        )

        # Get recommendation
        if self.use_ml_model and self.ml_model is not None:
            recommendation = self._ml_based_recommendation(
                features, time_budget, quality_requirement
            )
        else:
            recommendation = self._heuristic_recommendation(
                features, time_budget, quality_requirement
            )

        logger.info(
            f"Recommended: {recommendation.method_name} "
            f"(confidence: {recommendation.confidence:.2f})"
        )

        return recommendation.method_name, recommendation.config, recommendation.confidence

    def recommend_top_k(
        self,
        problem: Any,
        adapter: Optional[UniversalOptimizationInterface] = None,
        standardized_problem: Optional[StandardizedProblem] = None,
        k: int = 3,
        time_budget: Optional[float] = None,
        quality_requirement: str = 'balanced'
    ) -> List[MethodRecommendation]:
        """
        Get top-k method recommendations

        Args:
            problem: The optimization problem
            adapter: Domain adapter
            standardized_problem: Already standardized problem (optional)
            k: Number of recommendations
            time_budget: Maximum time in seconds
            quality_requirement: Quality vs speed preference

        Returns:
            List of top k recommendations sorted by confidence
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            problem, adapter, standardized_problem
        )

        # Get all recommendations
        recommendations = self._get_all_recommendations(
            features, time_budget, quality_requirement
        )

        # Sort by confidence and return top k
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:k]

    def explain_recommendation(
        self,
        problem: Any,
        adapter: Optional[UniversalOptimizationInterface] = None,
        standardized_problem: Optional[StandardizedProblem] = None,
        method: Optional[str] = None
    ) -> str:
        """
        Explain why a method was recommended (or why a specific method would perform)

        Args:
            problem: The optimization problem
            adapter: Domain adapter
            standardized_problem: Already standardized problem (optional)
            method: Specific method to explain (if None, explains top recommendation)

        Returns:
            Human-readable explanation
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            problem, adapter, standardized_problem
        )

        # Get recommendation if method not specified
        if method is None:
            recommendation = self._heuristic_recommendation(features, None, 'balanced')
            method = recommendation.method_name
        else:
            # Get recommendation for specific method
            recommendation = self._evaluate_method(method, features, None, 'balanced')

        # Build explanation
        explanation_parts = [
            f"Method: {method.upper()}",
            f"Confidence: {recommendation.confidence:.1%}",
            "",
            "Problem Characteristics:",
            f"- Dimension: {features.dimension}",
            f"- Type: {features.problem_type}",
            f"- Difficulty: {features.estimated_difficulty}",
            f"- Sparsity: {features.matrix_sparsity:.1%}",
            f"- Has constraints: {'Yes' if features.has_constraints else 'No'}",
            "",
            "Recommendation Reasoning:",
            recommendation.explanation,
            "",
            "Supporting Evidence:"
        ]
        explanation_parts.extend([f"- {evidence}" for evidence in recommendation.supporting_evidence])

        return "\n".join(explanation_parts)

    def record_performance(
        self,
        problem: Any,
        adapter: Optional[UniversalOptimizationInterface],
        method: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        runtime_seconds: float
    ):
        """
        Record method performance for future training

        Args:
            problem: The optimization problem
            adapter: Domain adapter
            method: Method used
            config: Configuration used
            result: Optimization result
            runtime_seconds: Runtime in seconds
        """
        if self.performance_log_path is None:
            return

        # Extract features
        features = self.feature_extractor.extract_features(problem, adapter)

        # Create record
        record = PerformanceRecord(
            timestamp=datetime.now().isoformat(),
            problem_features=asdict(features),
            method=method,
            config=config,
            objective_value=result.get('objective', float('inf')),
            iterations=result.get('iterations', 0),
            runtime_seconds=runtime_seconds,
            success=result.get('is_valid', False)
        )

        # Append to log file
        with open(self.performance_log_path, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')

        logger.debug(f"Recorded performance: {method} achieved {record.objective_value}")

    def _heuristic_recommendation(
        self,
        features: ProblemFeatures,
        time_budget: Optional[float],
        quality_requirement: str
    ) -> MethodRecommendation:
        """Generate recommendation using heuristic rules"""
        recommendations = self._get_all_recommendations(
            features, time_budget, quality_requirement
        )

        # Return the best recommendation
        return max(recommendations, key=lambda x: x.confidence)

    def _get_all_recommendations(
        self,
        features: ProblemFeatures,
        time_budget: Optional[float],
        quality_requirement: str
    ) -> List[MethodRecommendation]:
        """Get recommendations for all methods"""
        recommendations = []

        for method in self.METHODS:
            rec = self._evaluate_method(method, features, time_budget, quality_requirement)
            recommendations.append(rec)

        return recommendations

    def _evaluate_method(
        self,
        method: str,
        features: ProblemFeatures,
        time_budget: Optional[float],
        quality_requirement: str
    ) -> MethodRecommendation:
        """Evaluate a specific method for the problem"""
        confidence = 0.5  # Base confidence
        evidence = []
        config = {}

        # Problem size considerations
        if features.dimension < 20:
            if method == 'local_search':
                confidence += 0.2
                evidence.append("Small problem size favors exhaustive local search")
            elif method == 'genetic_algorithm':
                confidence -= 0.1
                evidence.append("GA overhead not justified for small problems")
        elif features.dimension > 100:
            if method == 'random_search':
                confidence += 0.1
                evidence.append("Large search space benefits from random exploration")
            elif method == 'local_search':
                confidence -= 0.2
                evidence.append("Large neighborhood makes local search expensive")

        # Problem type specific rules
        if features.problem_type == 'QAP':
            if method == 'simulated_annealing':
                confidence += 0.3
                evidence.append("SA is proven effective for QAP problems")
                config = {
                    'iterations': min(10000 * features.dimension, 1000000),
                    'initial_temp': 100.0 * features.value_range,
                    'cooling_rate': 0.95
                }
            elif method == 'tabu_search':
                confidence += 0.2
                evidence.append("Tabu search works well for QAP")
                config = {
                    'iterations': min(5000 * features.dimension, 500000),
                    'tabu_tenure': max(7, features.dimension // 10)
                }

        elif features.problem_type == 'TSP':
            if method in ['simulated_annealing', 'genetic_algorithm']:
                confidence += 0.25
                evidence.append(f"{method} is effective for TSP")

        # Structure-based recommendations
        if features.matrix_sparsity > 0.7:
            if method == 'local_search':
                confidence += 0.1
                evidence.append("Sparse structure suits local search")
        else:
            if method == 'genetic_algorithm':
                confidence += 0.1
                evidence.append("Dense problems benefit from population-based search")

        # Constraint handling
        if features.has_constraints:
            if method == 'genetic_algorithm':
                confidence += 0.15
                evidence.append("GA handles constraints well through population diversity")
                config['population_size'] = min(100, features.dimension * 2)
            elif method == 'random_search':
                confidence -= 0.2
                evidence.append("Random search struggles with constraints")

        # Symmetry exploitation
        if features.matrix_symmetry > 0.8:
            if method in ['local_search', 'tabu_search']:
                confidence += 0.1
                evidence.append("Symmetric structure can be exploited by intelligent search")

        # Quality requirements
        if quality_requirement == 'fast':
            if method == 'random_search':
                confidence += 0.2
                evidence.append("Fast results required")
                config['iterations'] = min(1000, features.dimension * 10)
            elif method == 'genetic_algorithm':
                confidence -= 0.2
                evidence.append("GA too slow for quick results")
        elif quality_requirement == 'best':
            if method in ['simulated_annealing', 'genetic_algorithm']:
                confidence += 0.15
                evidence.append("Quality prioritized over speed")
                if 'iterations' not in config:
                    config['iterations'] = features.dimension * 1000

        # Time budget constraints
        if time_budget is not None:
            if time_budget < 1.0:
                if method in ['random_search', 'local_search']:
                    confidence += 0.1
                    evidence.append("Short time budget favors simple methods")
                else:
                    confidence -= 0.1
            elif time_budget > 60:
                if method == 'genetic_algorithm':
                    confidence += 0.1
                    evidence.append("Sufficient time for population-based search")

        # Difficulty-based adjustments
        if features.estimated_difficulty == 'very_hard':
            if method in ['simulated_annealing', 'genetic_algorithm']:
                confidence += 0.1
                evidence.append("Hard problems need sophisticated methods")
            elif method == 'local_search':
                confidence -= 0.2
                evidence.append("Local search may get stuck in hard problems")
        elif features.estimated_difficulty == 'easy':
            if method == 'local_search':
                confidence += 0.2
                evidence.append("Simple methods suffice for easy problems")

        # Connectivity considerations
        if features.connectivity > 0.5:
            if method == 'tabu_search':
                confidence += 0.1
                evidence.append("High connectivity suits systematic exploration")

        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))

        # Generate explanation
        method_info = self.METHODS[method]
        explanation = (
            f"{method.replace('_', ' ').title()} is recommended because it has "
            f"{method_info['exploration']} exploration and {method_info['exploitation']} "
            f"exploitation capabilities. "
        )

        if features.problem_type != 'generic':
            explanation += f"It is particularly good for {features.problem_type} problems. "

        explanation += f"The problem is estimated to be {features.estimated_difficulty}."

        # Set default config if not specified
        if not config:
            config = self._get_default_config(method, features)

        return MethodRecommendation(
            method_name=method,
            config=config,
            confidence=confidence,
            explanation=explanation,
            supporting_evidence=evidence if evidence else ["Default heuristic rules applied"]
        )

    def _get_default_config(self, method: str, features: ProblemFeatures) -> Dict[str, Any]:
        """Get default configuration for a method based on problem features"""
        base_iterations = features.dimension * 100

        configs = {
            'random_search': {
                'iterations': min(base_iterations, 10000),
                'seed': None
            },
            'simulated_annealing': {
                'iterations': min(base_iterations * 10, 100000),
                'initial_temp': 100.0,
                'cooling_rate': 0.95,
                'seed': None
            },
            'local_search': {
                'iterations': min(base_iterations * 5, 50000),
                'restarts': max(5, features.dimension // 20),
                'seed': None
            },
            'genetic_algorithm': {
                'population_size': min(100, features.dimension * 2),
                'generations': min(base_iterations // 10, 1000),
                'crossover_rate': 0.8,
                'mutation_rate': 1.0 / features.dimension if features.dimension > 0 else 0.1,
                'seed': None
            },
            'tabu_search': {
                'iterations': min(base_iterations * 5, 50000),
                'tabu_tenure': max(7, features.dimension // 10),
                'seed': None
            }
        }

        return configs.get(method, {'iterations': base_iterations})

    def _ml_based_recommendation(
        self,
        features: ProblemFeatures,
        time_budget: Optional[float],
        quality_requirement: str
    ) -> MethodRecommendation:
        """Generate recommendation using ML model (when available)"""
        # For now, fall back to heuristic
        # This is where ML model predictions would go
        logger.info("ML model not available, using heuristic rules")
        return self._heuristic_recommendation(features, time_budget, quality_requirement)

    def _load_ml_model(self):
        """Load pre-trained ML model if available"""
        # This is a placeholder for future ML model loading
        # Would load from Librex.ai.models
        try:
            from Librex.ai.models import load_pretrained_selector
            self.ml_model = load_pretrained_selector()
            logger.info("ML model loaded successfully")
        except ImportError:
            logger.info("ML model not available, will use heuristic rules")
            self.ml_model = None
