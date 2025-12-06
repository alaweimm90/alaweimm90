"""
Query Analyzer - Optimize algorithm queries

Analyzes:
- Query patterns
- Cache effectiveness
- Optimal algorithm combinations
- Performance predictions
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY ANALYSIS
# ============================================================================

class QueryAnalysis:
    """Analysis of a query pattern."""

    def __init__(self, query_id: str, domain: str, complexity: str = "unknown"):
        self.query_id = query_id
        self.domain = domain
        self.complexity = complexity  # simple, moderate, complex
        self.execution_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.recommended_algorithm: Optional[str] = None
        self.created_at = datetime.utcnow()

    def add_execution(self, duration_ms: float):
        """Record execution time."""
        self.execution_times.append(duration_ms)

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        if not self.execution_times:
            return {}

        times = self.execution_times
        total_requests = self.cache_hits + self.cache_misses

        return {
            "query_id": self.query_id,
            "domain": self.domain,
            "complexity": self.complexity,
            "total_executions": len(times),
            "average_time_ms": round(sum(times) / len(times), 2),
            "min_time_ms": round(min(times), 2),
            "max_time_ms": round(max(times), 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(
                (self.cache_hits / total_requests * 100) if total_requests > 0 else 0, 2
            ),
            "recommended_algorithm": self.recommended_algorithm,
        }


# ============================================================================
# QUERY ANALYZER
# ============================================================================

class QueryAnalyzer:
    """Analyze and optimize queries."""

    def __init__(self):
        self.queries: Dict[str, QueryAnalysis] = {}
        self.query_patterns: Dict[str, int] = {}  # Count of each pattern

    def analyze_query(
        self,
        query_id: str,
        domain: str,
        parameters: Dict[str, Any],
    ) -> QueryAnalysis:
        """
        Analyze a query and create analysis record.

        Args:
            query_id: Unique query identifier
            domain: Research domain
            parameters: Query parameters

        Returns:
            QueryAnalysis object
        """
        # Determine complexity from parameters
        complexity = self._estimate_complexity(parameters)

        analysis = QueryAnalysis(query_id, domain, complexity)

        # Pattern matching
        pattern = f"{domain}:{complexity}"
        self.query_patterns[pattern] = self.query_patterns.get(pattern, 0) + 1

        self.queries[query_id] = analysis
        return analysis

    def _estimate_complexity(self, parameters: Dict[str, Any]) -> str:
        """Estimate query complexity from parameters."""
        # Simple heuristic based on parameter count and values
        param_count = len(parameters)

        if param_count <= 2:
            return "simple"
        elif param_count <= 5:
            return "moderate"
        else:
            return "complex"

    def get_optimization_suggestions(self, query_id: str) -> List[str]:
        """Get optimization suggestions for a query."""
        if query_id not in self.queries:
            return []

        analysis = self.queries[query_id]
        stats = analysis.get_stats()
        suggestions = []

        # Cache effectiveness
        if stats.get("cache_hit_rate", 0) < 30:
            suggestions.append(
                "Low cache hit rate. This query may benefit from caching similar results."
            )

        # Execution time
        avg_time = stats.get("average_time_ms", 0)
        if avg_time > 2000:
            suggestions.append(
                f"Query takes {avg_time:.0f}ms on average. "
                "Consider using a faster algorithm or increasing parallelization."
            )

        # Pattern optimization
        pattern = f"{analysis.domain}:{analysis.complexity}"
        if self.query_patterns.get(pattern, 0) > 10:
            suggestions.append(
                f"Pattern '{pattern}' is common. Pre-compute results for frequently used parameters."
            )

        return suggestions or ["Query is performing well."]

    def get_performance_forecast(self, query_id: str) -> Dict[str, Any]:
        """Forecast future performance based on trend."""
        if query_id not in self.queries:
            return {}

        analysis = self.queries[query_id]
        times = analysis.execution_times

        if len(times) < 3:
            return {"warning": "Insufficient data for forecast"}

        # Simple trend analysis
        recent = sum(times[-3:]) / 3
        older = sum(times[:-3]) / len(times[:-3]) if len(times) > 3 else recent

        trend = "improving" if recent < older else ("degrading" if recent > older else "stable")
        trend_percent = ((recent - older) / older * 100) if older > 0 else 0

        return {
            "current_avg_ms": round(recent, 2),
            "previous_avg_ms": round(older, 2),
            "trend": trend,
            "trend_percent": round(trend_percent, 1),
            "recommendation": self._get_trend_recommendation(trend, trend_percent),
        }

    def _get_trend_recommendation(self, trend: str, percent: float) -> str:
        """Get recommendation based on trend."""
        if trend == "improving":
            return "Performance is improving. Continue current strategy."
        elif trend == "degrading":
            if percent > 20:
                return "Significant degradation detected. Investigate algorithm efficiency."
            else:
                return "Slight degradation. Monitor closely."
        else:
            return "Performance is stable."

    def get_common_patterns(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most common query patterns."""
        sorted_patterns = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "pattern": pattern,
                "count": count,
                "percentage": round(
                    (count / sum(self.query_patterns.values()) * 100)
                    if self.query_patterns else 0, 2
                ),
            }
            for pattern, count in sorted_patterns[:top_n]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get analyzer summary."""
        if not self.queries:
            return {"message": "No queries analyzed yet"}

        # Aggregate stats
        total_executions = sum(
            len(q.execution_times) for q in self.queries.values()
        )
        total_cache_hits = sum(q.cache_hits for q in self.queries.values())
        total_cache_misses = sum(q.cache_misses for q in self.queries.values())

        return {
            "total_queries": len(self.queries),
            "total_executions": total_executions,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "overall_cache_hit_rate": round(
                (total_cache_hits / (total_cache_hits + total_cache_misses) * 100)
                if (total_cache_hits + total_cache_misses) > 0 else 0, 2
            ),
            "common_patterns": self.get_common_patterns(),
        }

    def export_report(self) -> Dict[str, Any]:
        """Export complete analysis report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_summary(),
            "common_patterns": self.get_common_patterns(10),
            "queries": [
                {
                    "query_id": query_id,
                    "stats": query.get_stats(),
                    "suggestions": self.get_optimization_suggestions(query_id),
                    "forecast": self.get_performance_forecast(query_id),
                }
                for query_id, query in list(self.queries.items())[:20]  # Top 20
            ],
        }


# ============================================================================
# GLOBAL ANALYZER
# ============================================================================

_global_analyzer: Optional[QueryAnalyzer] = None


def analyze_query(
    query_id: str,
    domain: str,
    parameters: Dict[str, Any],
) -> QueryAnalysis:
    """Analyze query using global analyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = QueryAnalyzer()

    return _global_analyzer.analyze_query(query_id, domain, parameters)
