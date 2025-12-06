"""LitReviewBot - Automated Literature Review Generator

Generate comprehensive literature reviews from paper collections.
Cluster by theme, identify gaps, visualize citation networks.
"""

__version__ = "1.0.0"
__author__ = "TalAI"

from .main import LitReviewBot, Paper, PaperCluster, ResearchGap, LiteratureReview

__all__ = ["LitReviewBot", "Paper", "PaperCluster", "ResearchGap", "LiteratureReview"]
