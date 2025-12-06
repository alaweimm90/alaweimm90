"""
TalAI Enterprise Database Configuration
Production-ready database setup with async support
"""

import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from datetime import datetime, UTC

from sqlalchemy import MetaData, create_engine, event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base, DeclarativeMeta
from sqlalchemy.pool import NullPool
from pydantic_settings import BaseSettings

# Configuration
class DatabaseSettings(BaseSettings):
    """Database configuration with environment variable support"""

    # Primary database
    database_url: str = "postgresql+asyncpg://user:password@localhost/talai_enterprise"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_echo: bool = False

    # Read replica (for scaling)
    read_replica_url: Optional[str] = None

    # Multi-tenancy
    enable_row_level_security: bool = True
    enable_schema_isolation: bool = False

    # Performance
    enable_query_cache: bool = True
    slow_query_threshold_ms: int = 100

    class Config:
        env_prefix = "TALAI_DB_"
        case_sensitive = False

settings = DatabaseSettings()

# Metadata configuration for multi-tenancy
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Base class for all models
class TimestampMixin:
    """Mixin for automatic timestamp management"""

    from sqlalchemy import Column, DateTime, func

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

class TenantMixin:
    """Mixin for multi-tenant models"""

    from sqlalchemy import Column, String, Index

    tenant_id = Column(
        String(64),
        nullable=False,
        index=True
    )

Base: DeclarativeMeta = declarative_base(metadata=metadata)

# Engine creation with production configurations
def create_async_db_engine(
    url: str,
    is_read_replica: bool = False
) -> AsyncEngine:
    """Create async database engine with production settings"""

    pool_class = NullPool if is_read_replica else pool.AsyncAdaptedQueuePool

    engine = create_async_engine(
        url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size if not is_read_replica else 5,
        max_overflow=settings.database_max_overflow if not is_read_replica else 5,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        poolclass=pool_class,
        connect_args={
            "server_settings": {
                "application_name": "talai_enterprise",
                "jit": "off"
            },
            "command_timeout": 60,
            "timeout": 30,
        }
    )

    # Add performance monitoring
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start_time", []).append(datetime.now(UTC))

    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        total_time = datetime.now(UTC) - conn.info["query_start_time"].pop(-1)
        if total_time.total_seconds() * 1000 > settings.slow_query_threshold_ms:
            # Log slow queries (integrate with monitoring)
            print(f"Slow query detected ({total_time.total_seconds():.2f}s): {statement[:100]}")

    return engine

# Primary database engine
engine = create_async_db_engine(settings.database_url)

# Read replica engine (if configured)
read_engine = (
    create_async_db_engine(settings.read_replica_url, is_read_replica=True)
    if settings.read_replica_url
    else engine
)

# Session factories
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

async_read_session = async_sessionmaker(
    read_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Dependency injection for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for write operations"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def get_read_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for read operations"""
    async with async_read_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Database initialization
async def init_database():
    """Initialize database with tables and extensions"""
    async with engine.begin() as conn:
        # Enable required extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS uuid_ossp")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

        # Create tables
        await conn.run_sync(Base.metadata.create_all)

        # Enable row-level security if configured
        if settings.enable_row_level_security:
            # This would be expanded with actual RLS policies
            pass

# Health check
async def check_database_health() -> dict:
    """Check database connectivity and health"""
    try:
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            await conn.close()

        return {
            "status": "healthy",
            "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else None,
            "connections_in_use": engine.pool.checked_in_connections() if hasattr(engine.pool, 'checked_in_connections') else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Context manager for transactions
@asynccontextmanager
async def transaction_context():
    """Provide transactional context across multiple operations"""
    async with async_session() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            else:
                await session.commit()