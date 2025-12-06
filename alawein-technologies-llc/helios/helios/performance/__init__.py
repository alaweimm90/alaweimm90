"""
HELIOS Performance Optimization - Profiling and Auto-Selection

Provides:
- Dynamic performance profiling
- Bottleneck detection
- Algorithm auto-selection
- Adaptive caching
- Query optimization
"""

from .profiler import DynamicProfiler, get_profiler
from .bottleneck_detector import BottleneckDetector, detect_bottlenecks
from .algorithm_selector import AlgorithmSelector, select_algorithm
from .cache_strategy import AdaptiveCache, get_cache
from .query_analyzer import QueryAnalyzer, analyze_query

__all__ = [
    'DynamicProfiler',
    'get_profiler',
    'BottleneckDetector',
    'detect_bottlenecks',
    'AlgorithmSelector',
    'select_algorithm',
    'AdaptiveCache',
    'get_cache',
    'QueryAnalyzer',
    'analyze_query',
]
