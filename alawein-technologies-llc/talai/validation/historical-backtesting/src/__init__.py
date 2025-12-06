"""
TalAI Historical Backtesting System

Validates TalAI's ability to predict past scientific discoveries through time-travel validation.
Tests against a comprehensive database of 50+ major historical discoveries across multiple domains.
"""

from .backtesting_engine import HistoricalBacktestingEngine
from .discovery_database import DiscoveryDatabase
from .time_travel_validator import TimeTravelValidator
from .metrics_analyzer import MetricsAnalyzer
from .domain_benchmarks import DomainBenchmarks

__version__ = "1.0.0"

__all__ = [
    "HistoricalBacktestingEngine",
    "DiscoveryDatabase",
    "TimeTravelValidator",
    "MetricsAnalyzer",
    "DomainBenchmarks",
]