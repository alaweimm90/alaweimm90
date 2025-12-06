"""
TalAI Performance: Database Optimization Infrastructure
========================================================

Connection pooling, query optimization, migrations, sharding,
and comprehensive database performance monitoring.

© 2024 AlaweinOS. All rights reserved.
"""

from .connection_pool import ConnectionPool, PoolConfig
from .query_optimizer import QueryOptimizer
from .migration_manager import MigrationManager
from .sharding_manager import ShardingManager
from .db_monitor import DatabaseMonitor
from .replica_manager import ReplicaManager

__all__ = [
    'ConnectionPool',
    'PoolConfig',
    'QueryOptimizer',
    'MigrationManager',
    'ShardingManager',
    'DatabaseMonitor',
    'ReplicaManager'
]

__version__ = "1.0.0"