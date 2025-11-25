"""
Database connection and initialization for REPZ Workflow System
"""

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Global database engine and session
_engine: Optional[create_async_engine] = None
_async_session: Optional[async_sessionmaker] = None


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


async def init_database(database_url: str):
    """Initialize database connection and create tables"""
    global _engine, _async_session

    try:
        # Create async engine
        _engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging in development
            pool_pre_ping=True,
            pool_recycle=300
        )

        # Create async session factory
        _async_session = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create all tables
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def get_db_session() -> AsyncSession:
    """Get database session"""
    if _async_session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_database():
    """Close database connections"""
    global _engine, _async_session

    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session = None
        logger.info("Database connections closed")


# Dependency for FastAPI
async def get_db():
    """FastAPI dependency for database session"""
    async for session in get_db_session():
        yield session