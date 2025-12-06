"""
Intelligent Algorithm Auto-Selection

Selects:
- Best algorithm by quality/speed trade-off
- Domain-specific optimal algorithms
- Fallback algorithms under constraints
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ALGORITHM PERFORMANCE PROFILE
# ============================================================================

@dataclass
class AlgorithmProfile:
    """Performance profile for an algorithm."""

    id: str
    name: str
    domain: str
    quality: float  # 0-1
    speedup: float  # relative to baseline
    memory_mb: float
    latency_ms: float
    reliability: float  # success rate
    avg_execution_count: int


# ============================================================================
# SELECTION CRITERIA
# ============================================================================

@dataclass
class SelectionCriteria:
    """Criteria for algorithm selection."""

    domain: str
    quality_weight: float = 0.4  # How important quality is
    speed_weight: float = 0.4  # How important speed is
    memory_weight: float = 0.1  # How important low memory is
    reliability_weight: float = 0.1  # How important reliability is
    max_latency_ms: Optional[float] = None  # Hard constraint
    min_quality: Optional[float] = None  # Hard constraint


# ============================================================================
# ALGORITHM SELECTOR
# ============================================================================

class AlgorithmSelector:
    """Intelligently select algorithms based on performance metrics."""

    # Sample algorithm profiles (in production, these come from monitoring)
    ALGORITHM_PROFILES = {
        "quantum": [
            AlgorithmProfile("q1", "Grover", "quantum", 0.85, 2.0, 150, 800, 0.95, 100),
            AlgorithmProfile("q2", "VQE", "quantum", 0.88, 2.3, 180, 1200, 0.92, 85),
            AlgorithmProfile("q3", "Annealing", "quantum", 0.82, 1.8, 120, 600, 0.98, 120),
        ],
        "materials": [
            AlgorithmProfile("m1", "Crystal", "materials", 0.81, 1.9, 200, 900, 0.94, 95),
            AlgorithmProfile("m2", "GNN", "materials", 0.86, 2.2, 250, 1500, 0.89, 70),
            AlgorithmProfile("m3", "Stability", "materials", 0.84, 2.0, 220, 1100, 0.91, 80),
        ],
        "optimization": [
            AlgorithmProfile("o1", "SA", "optimization", 0.78, 1.5, 100, 500, 0.99, 150),
            AlgorithmProfile("o2", "GA", "optimization", 0.79, 1.6, 110, 550, 0.97, 140),
            AlgorithmProfile("o3", "Hybrid", "optimization", 0.88, 2.4, 300, 2000, 0.88, 50),
        ],
        "ml": [
            AlgorithmProfile("ml1", "ResNet", "ml", 0.89, 2.1, 400, 1800, 0.96, 120),
            AlgorithmProfile("ml2", "Transformer", "ml", 0.91, 2.5, 500, 2500, 0.93, 80),
            AlgorithmProfile("ml3", "Attention", "ml", 0.90, 2.3, 450, 2200, 0.94, 90),
        ],
        "nas": [
            AlgorithmProfile("nas1", "DARTS", "nas", 0.85, 1.9, 350, 1400, 0.92, 60),
            AlgorithmProfile("nas2", "Diff", "nas", 0.87, 2.2, 380, 1800, 0.89, 50),
        ],
        "synthesis": [
            AlgorithmProfile("s1", "Retro", "synthesis", 0.82, 1.8, 180, 1000, 0.90, 75),
            AlgorithmProfile("s2", "GenGraph", "synthesis", 0.86, 2.1, 220, 1300, 0.88, 65),
        ],
        "graph": [
            AlgorithmProfile("g1", "GNN", "graph", 0.87, 2.0, 200, 1100, 0.91, 85),
            AlgorithmProfile("g2", "PageRank", "graph", 0.81, 1.7, 150, 800, 0.95, 110),
        ],
    }

    def __init__(self):
        self.selection_history: List[Tuple[str, str]] = []  # (domain, selected_algo)

    def select(self, criteria: SelectionCriteria) -> AlgorithmProfile:
        """
        Select best algorithm matching criteria.

        Args:
            criteria: Selection criteria

        Returns:
            Selected algorithm profile

        Raises:
            ValueError: If no suitable algorithm found
        """
        # Get candidates for domain
        candidates = self.ALGORITHM_PROFILES.get(criteria.domain, [])

        if not candidates:
            raise ValueError(f"No algorithms available for domain: {criteria.domain}")

        # Apply hard constraints
        filtered = candidates.copy()

        if criteria.max_latency_ms:
            filtered = [a for a in filtered if a.latency_ms <= criteria.max_latency_ms]

        if criteria.min_quality:
            filtered = [a for a in filtered if a.quality >= criteria.min_quality]

        if not filtered:
            logger.warning(
                f"No algorithms match hard constraints for {criteria.domain}. "
                f"Using least-constrained option."
            )
            filtered = [min(candidates, key=lambda a: a.latency_ms)]

        # Score remaining candidates
        scores = {}
        for algo in filtered:
            # Normalize metrics to 0-1
            norm_quality = algo.quality
            norm_speed = min(algo.speedup / 3.0, 1.0)  # Normalize by max expected speedup
            norm_memory = max(1.0 - (algo.memory_mb / 1000), 0.0)  # Lower is better
            norm_reliability = algo.reliability

            # Weighted score
            score = (
                criteria.quality_weight * norm_quality +
                criteria.speed_weight * norm_speed +
                criteria.memory_weight * norm_memory +
                criteria.reliability_weight * norm_reliability
            )

            scores[algo.id] = score

        # Select highest scoring
        best_id = max(scores, key=scores.get)
        best_algo = next(a for a in filtered if a.id == best_id)

        self.selection_history.append((criteria.domain, best_algo.name))

        logger.info(
            f"Selected {best_algo.name} for {criteria.domain} "
            f"(score: {scores[best_id]:.3f})"
        )

        return best_algo

    def select_for_quality(self, domain: str) -> AlgorithmProfile:
        """Select algorithm optimizing for quality."""
        criteria = SelectionCriteria(
            domain=domain,
            quality_weight=0.7,
            speed_weight=0.2,
            memory_weight=0.05,
            reliability_weight=0.05,
        )
        return self.select(criteria)

    def select_for_speed(self, domain: str) -> AlgorithmProfile:
        """Select algorithm optimizing for speed."""
        criteria = SelectionCriteria(
            domain=domain,
            quality_weight=0.2,
            speed_weight=0.7,
            memory_weight=0.05,
            reliability_weight=0.05,
        )
        return self.select(criteria)

    def select_for_balanced(self, domain: str) -> AlgorithmProfile:
        """Select algorithm with balanced trade-off."""
        criteria = SelectionCriteria(
            domain=domain,
            quality_weight=0.4,
            speed_weight=0.4,
            memory_weight=0.1,
            reliability_weight=0.1,
        )
        return self.select(criteria)

    def get_recommendations(self, domain: str) -> Dict[str, Any]:
        """Get algorithm recommendations for a domain."""
        candidates = self.ALGORITHM_PROFILES.get(domain, [])

        if not candidates:
            return {"error": f"No algorithms for domain {domain}"}

        # Sort by overall score
        ranked = sorted(
            candidates,
            key=lambda a: a.quality * 0.4 + min(a.speedup / 3.0, 1.0) * 0.4 +
                          max(1.0 - a.memory_mb / 1000, 0.0) * 0.1 + a.reliability * 0.1,
            reverse=True
        )

        return {
            "domain": domain,
            "total_algorithms": len(candidates),
            "recommendations": [
                {
                    "rank": i + 1,
                    "name": algo.name,
                    "quality": algo.quality,
                    "speedup": algo.speedup,
                    "latency_ms": algo.latency_ms,
                    "reliability": algo.reliability,
                }
                for i, algo in enumerate(ranked[:5])
            ]
        }

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get algorithm selection statistics."""
        if not self.selection_history:
            return {"message": "No selections made yet"}

        from collections import Counter

        domain_counts = Counter(d for d, _ in self.selection_history)
        algo_counts = Counter(a for _, a in self.selection_history)

        return {
            "total_selections": len(self.selection_history),
            "by_domain": dict(domain_counts),
            "by_algorithm": dict(algo_counts),
        }


# ============================================================================
# GLOBAL SELECTOR
# ============================================================================

_global_selector: Optional[AlgorithmSelector] = None


def select_algorithm(criteria: SelectionCriteria) -> AlgorithmProfile:
    """Select algorithm using global selector."""
    global _global_selector
    if _global_selector is None:
        _global_selector = AlgorithmSelector()

    return _global_selector.select(criteria)
