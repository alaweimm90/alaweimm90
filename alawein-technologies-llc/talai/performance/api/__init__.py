"""
TalAI Performance: API Rate Limiting & Optimization
=====================================================

Intelligent rate limiting, request batching, deduplication,
streaming, GraphQL, and comprehensive API optimization.

© 2024 AlaweinOS. All rights reserved.
"""

from .rate_limiter import RateLimiter, RateLimitConfig
from .request_batcher import RequestBatcher
from .response_streaming import ResponseStreamer
from .api_optimizer import APIOptimizer
from .graphql_api import GraphQLAPI

__all__ = [
    'RateLimiter',
    'RateLimitConfig',
    'RequestBatcher',
    'ResponseStreamer',
    'APIOptimizer',
    'GraphQLAPI'
]

__version__ = "1.0.0"