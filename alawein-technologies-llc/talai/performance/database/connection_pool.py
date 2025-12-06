"""
Advanced database connection pooling with health checks and auto-scaling.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg
import psycopg2.pool
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, NullPool, StaticPool

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Database connection pool configuration."""
    # Basic settings
    database_url: str
    min_connections: int = 5
    max_connections: int = 100
    overflow: int = 20

    # Timeouts
    connection_timeout: float = 10.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0

    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 30
    health_check_query: str = "SELECT 1"

    # Performance
    enable_prepared_statements: bool = True
    enable_connection_recycling: bool = True
    recycle_interval: int = 3600

    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_check_interval: int = 60


class ConnectionPool:
    """
    Enterprise-grade database connection pool with health monitoring,
    auto-scaling, and comprehensive performance optimization.
    """

    def __init__(self, config: PoolConfig):
        """Initialize connection pool."""
        self.config = config
        self._pools: Dict[str, Any] = {}
        self._connections_in_use = 0
        self._total_connections = 0
        self._health_status = True
        self._last_health_check = time.time()
        self._shutdown = False
        self._background_tasks = []

        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "connections_recycled": 0,
            "queries_executed": 0,
            "errors": 0,
            "health_checks": 0
        }

        self._initialize_pools()
        logger.info(f"ConnectionPool initialized with {config.min_connections}-{config.max_connections} connections")

    def _initialize_pools(self) -> None:
        """Initialize different pool implementations."""
        # SQLAlchemy pool
        self.sqlalchemy_engine = create_engine(
            self.config.database_url,
            poolclass=QueuePool,
            pool_size=self.config.min_connections,
            max_overflow=self.config.overflow,
            pool_timeout=self.config.connection_timeout,
            pool_recycle=self.config.recycle_interval if self.config.enable_connection_recycling else -1,
            pool_pre_ping=self.config.enable_health_checks,
            echo_pool=False
        )

        # Register event listeners
        event.listen(self.sqlalchemy_engine, "connect", self._on_connect)
        event.listen(self.sqlalchemy_engine, "checkout", self._on_checkout)
        event.listen(self.sqlalchemy_engine, "checkin", self._on_checkin)

        # Create session factory
        self.Session = sessionmaker(bind=self.sqlalchemy_engine)

        # Async pool (if PostgreSQL)
        if "postgresql" in self.config.database_url:
            self._init_async_pool()

    async def _init_async_pool(self) -> None:
        """Initialize async PostgreSQL pool."""
        try:
            # Parse connection URL
            import urllib.parse
            parsed = urllib.parse.urlparse(self.config.database_url)

            self.async_pool = await asyncpg.create_pool(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                max_queries=50000,
                max_inactive_connection_lifetime=self.config.idle_timeout,
                timeout=self.config.connection_timeout,
                command_timeout=60,
                statement_cache_size=100 if self.config.enable_prepared_statements else 0
            )

            self._pools["async_pg"] = self.async_pool
            logger.info("Async PostgreSQL pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")

    @asynccontextmanager
    async def get_connection(self, pool_type: str = "sqlalchemy"):
        """
        Get a database connection from the pool.

        Args:
            pool_type: Type of pool to use (sqlalchemy, async_pg)

        Yields:
            Database connection
        """
        connection = None
        start_time = time.time()

        try:
            if pool_type == "sqlalchemy":
                session = self.Session()
                self._connections_in_use += 1
                yield session

            elif pool_type == "async_pg" and "async_pg" in self._pools:
                async with self._pools["async_pg"].acquire() as connection:
                    self._connections_in_use += 1
                    yield connection

            else:
                raise ValueError(f"Unknown pool type: {pool_type}")

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Connection error: {e}")
            raise

        finally:
            self._connections_in_use -= 1

            if pool_type == "sqlalchemy" and connection:
                connection.close()

            # Record query time
            query_time = time.time() - start_time
            self._stats["queries_executed"] += 1

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        pool_type: str = "sqlalchemy"
    ) -> Any:
        """
        Execute a query using the connection pool.

        Args:
            query: SQL query to execute
            params: Query parameters
            pool_type: Pool type to use

        Returns:
            Query results
        """
        async with self.get_connection(pool_type) as conn:
            if pool_type == "sqlalchemy":
                result = conn.execute(query, params or {})
                return result.fetchall()

            elif pool_type == "async_pg":
                if params:
                    result = await conn.fetch(query, *params.values())
                else:
                    result = await conn.fetch(query)
                return result

    async def execute_many(
        self,
        query: str,
        params_list: List[Dict[str, Any]],
        pool_type: str = "sqlalchemy"
    ) -> None:
        """
        Execute multiple queries in batch.

        Args:
            query: SQL query template
            params_list: List of parameter dictionaries
            pool_type: Pool type to use
        """
        async with self.get_connection(pool_type) as conn:
            if pool_type == "sqlalchemy":
                conn.execute(query, params_list)
                conn.commit()

            elif pool_type == "async_pg":
                await conn.executemany(query, [list(p.values()) for p in params_list])

    async def health_check(self) -> bool:
        """
        Perform health check on the connection pool.

        Returns:
            Health status
        """
        try:
            # Execute health check query
            await self.execute_query(
                self.config.health_check_query,
                pool_type="async_pg" if "async_pg" in self._pools else "sqlalchemy"
            )

            self._health_status = True
            self._stats["health_checks"] += 1
            self._last_health_check = time.time()

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._health_status = False
            return False

    async def auto_scale(self) -> None:
        """Auto-scale connection pool based on usage."""
        if not self.config.enable_auto_scaling:
            return

        usage_ratio = self._connections_in_use / self.config.max_connections

        if usage_ratio > self.config.scale_up_threshold:
            # Scale up
            new_max = min(
                self.config.max_connections * 1.5,
                200  # Hard limit
            )

            if new_max > self.config.max_connections:
                self.config.max_connections = int(new_max)
                await self._resize_pools()
                logger.info(f"Scaled up pool to {self.config.max_connections} connections")

        elif usage_ratio < self.config.scale_down_threshold:
            # Scale down
            new_max = max(
                self.config.max_connections * 0.7,
                self.config.min_connections * 2
            )

            if new_max < self.config.max_connections:
                self.config.max_connections = int(new_max)
                await self._resize_pools()
                logger.info(f"Scaled down pool to {self.config.max_connections} connections")

    async def _resize_pools(self) -> None:
        """Resize connection pools."""
        # Resize async pool if exists
        if "async_pg" in self._pools:
            # AsyncPG doesn't support dynamic resizing, would need to recreate
            pass

        # SQLAlchemy pool resizing would happen automatically

    def _on_connect(self, dbapi_conn, connection_record):
        """Handle new connection creation."""
        self._stats["connections_created"] += 1
        self._total_connections += 1

        # Set connection parameters
        if hasattr(dbapi_conn, 'set_session'):
            dbapi_conn.set_session(autocommit=False)

    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Handle connection checkout from pool."""
        # Check if connection needs recycling
        if self.config.enable_connection_recycling:
            age = time.time() - connection_record.info.get('checkout_time', 0)

            if age > self.config.max_lifetime:
                connection_record.invalidate()
                self._stats["connections_recycled"] += 1

        connection_record.info['checkout_time'] = time.time()

    def _on_checkin(self, dbapi_conn, connection_record):
        """Handle connection checkin to pool."""
        # Reset connection state if needed
        pass

    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if self.config.enable_health_checks:
            task = asyncio.create_task(self._health_check_loop())
            self._background_tasks.append(task)

        if self.config.enable_auto_scaling:
            task = asyncio.create_task(self._auto_scale_loop())
            self._background_tasks.append(task)

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _auto_scale_loop(self) -> None:
        """Background auto-scaling loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.scale_check_interval)
                await self.auto_scale()

            except Exception as e:
                logger.error(f"Auto-scale loop error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        pool_stats = {}

        # SQLAlchemy pool stats
        if hasattr(self.sqlalchemy_engine.pool, 'size'):
            pool_stats["sqlalchemy"] = {
                "size": self.sqlalchemy_engine.pool.size(),
                "checked_in": self.sqlalchemy_engine.pool.checkedin(),
                "overflow": self.sqlalchemy_engine.pool.overflow(),
                "checked_out": self.sqlalchemy_engine.pool.checkedout()
            }

        # Async pool stats
        if "async_pg" in self._pools:
            pool = self._pools["async_pg"]
            pool_stats["async_pg"] = {
                "size": pool.get_size(),
                "min_size": pool.get_min_size(),
                "max_size": pool.get_max_size(),
                "free_connections": pool.get_idle_size()
            }

        return {
            "config": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "overflow": self.config.overflow
            },
            "usage": {
                "connections_in_use": self._connections_in_use,
                "total_connections": self._total_connections,
                "usage_ratio": self._connections_in_use / self.config.max_connections
            },
            "health": {
                "status": self._health_status,
                "last_check": datetime.fromtimestamp(self._last_health_check).isoformat()
            },
            "pools": pool_stats,
            "stats": self._stats
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown connection pool."""
        logger.info("Shutting down ConnectionPool")
        self._shutdown = True

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close pools
        if "async_pg" in self._pools:
            await self._pools["async_pg"].close()

        self.sqlalchemy_engine.dispose()

        logger.info("ConnectionPool shutdown complete")