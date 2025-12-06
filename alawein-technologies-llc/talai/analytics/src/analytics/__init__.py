"""
TalAI Analytics and Intelligence System

Advanced analytics, ML-powered insights, recommendations, and visualizations
for hypothesis validation and research optimization.
"""

from .engine.analytics_engine import AnalyticsEngine
from .engine.time_series import TimeSeriesAnalyzer
from .engine.scoring import QualityScorer
from .ml_insights.embeddings import EmbeddingProcessor
from .ml_insights.classification import DomainClassifier
from .ml_insights.prediction import OutcomePredictor
from .recommendations.recommendation_engine import RecommendationEngine
from .visualization.dashboard import DashboardGenerator

__version__ = "1.0.0"
__all__ = [
    "AnalyticsEngine",
    "TimeSeriesAnalyzer",
    "QualityScorer",
    "EmbeddingProcessor",
    "DomainClassifier",
    "OutcomePredictor",
    "RecommendationEngine",
    "DashboardGenerator",
]