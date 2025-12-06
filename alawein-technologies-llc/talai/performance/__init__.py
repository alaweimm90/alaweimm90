"""
TalAI Performance Optimization Infrastructure
==============================================

High-performance infrastructure for TalAI with multi-level caching,
async task queues, database optimization, and API rate limiting.

Components:
- Cache: Multi-level caching with Redis and in-memory stores
- Queue: Async task processing with Celery and priority management
- Database: Connection pooling, query optimization, and sharding
- API: Rate limiting, request batching, and GraphQL

© 2024 AlaweinOS. All rights reserved.
"""

from .cache import (
    CacheManager,
    LLMResponseCache,
    InMemoryCache,
    RedisCache,
    CacheAnalytics,
    CacheWarmer,
    CacheInvalidator
)

from .queue import (
    TaskQueue,
    TaskManager,
    TaskScheduler,
    TaskMonitor,
    DeadLetterQueue,
    CeleryBackend,
    TaskPriority,
    TaskStatus
)

from .database import (
    ConnectionPool,
    PoolConfig,
    QueryOptimizer,
    MigrationManager,
    ShardingManager,
    DatabaseMonitor,
    ReplicaManager
)

from .api import (
    RateLimiter,
    RateLimitConfig,
    RequestBatcher,
    ResponseStreamer,
    APIOptimizer,
    GraphQLAPI
)

__version__ = "1.0.0"

__all__ = [
    # Cache
    'CacheManager',
    'LLMResponseCache',
    'InMemoryCache',
    'RedisCache',
    'CacheAnalytics',
    'CacheWarmer',
    'CacheInvalidator',

    # Queue
    'TaskQueue',
    'TaskManager',
    'TaskScheduler',
    'TaskMonitor',
    'DeadLetterQueue',
    'CeleryBackend',
    'TaskPriority',
    'TaskStatus',

    # Database
    'ConnectionPool',
    'PoolConfig',
    'QueryOptimizer',
    'MigrationManager',
    'ShardingManager',
    'DatabaseMonitor',
    'ReplicaManager',

    # API
    'RateLimiter',
    'RateLimitConfig',
    'RequestBatcher',
    'ResponseStreamer',
    'APIOptimizer',
    'GraphQLAPI'
]