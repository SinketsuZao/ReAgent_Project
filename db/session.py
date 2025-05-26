"""
Database session management for ReAgent.

This module provides session factories and context managers
for both synchronous and asynchronous database operations.
"""

import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from reagent.config import settings
from db.models import Base

# Database URLs
def get_database_url(async_mode: bool = False) -> str:
    """Get database URL based on environment variables."""
    user = os.getenv('POSTGRES_USER', 'reagent_user')
    password = os.getenv('POSTGRES_PASSWORD', 'changeme')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'reagent_db')
    
    if async_mode:
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    else:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Synchronous engine and session factory
engine = create_engine(
    get_database_url(async_mode=False),
    pool_pre_ping=True,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    echo=settings.environment == "development",
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=30,
    connect_args={
        "connect_timeout": 10,
        "application_name": "reagent_sync",
        "options": "-c statement_timeout=300000"  # 5 minute statement timeout
    }
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Asynchronous engine and session factory
async_engine = create_async_engine(
    get_database_url(async_mode=True),
    pool_pre_ping=True,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    echo=settings.environment == "development",
    pool_recycle=3600,
    pool_timeout=30,
    connect_args={
        "server_settings": {
            "application_name": "reagent_async",
            "jit": "off"
        },
        "command_timeout": 300,  # 5 minutes
        "timeout": 10
    }
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Context managers
@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get a synchronous database session.
    
    Usage:
        with get_db() as db:
            result = db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an asynchronous database session.
    
    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(Model))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Dependency injection for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Alias for backwards compatibility
get_async_session = get_async_db

# Database initialization
async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        # Create schema if it doesn't exist
        await conn.execute("CREATE SCHEMA IF NOT EXISTS reagent")
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

async def drop_db():
    """Drop all database tables. USE WITH CAUTION!"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Connection management
async def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False

async def close_database_connections():
    """Close all database connections."""
    await async_engine.dispose()

# Utility functions
def create_session_factory(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False
) -> sessionmaker:
    """
    Create a custom session factory.
    
    Args:
        database_url: Database connection URL
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        echo: Whether to echo SQL statements
        
    Returns:
        Session factory
    """
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo
    )
    
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

async def create_async_session_factory(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False
) -> async_sessionmaker:
    """
    Create a custom async session factory.
    
    Args:
        database_url: Async database connection URL
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        echo: Whether to echo SQL statements
        
    Returns:
        Async session factory
    """
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo
    )
    
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )

# Transaction management
class DatabaseTransaction:
    """Context manager for explicit transaction control."""
    
    def __init__(self, session: Session):
        self.session = session
        self.transaction = None
        
    def __enter__(self):
        self.transaction = self.session.begin()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.transaction.rollback()
        else:
            self.transaction.commit()
        return False

class AsyncDatabaseTransaction:
    """Async context manager for explicit transaction control."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.transaction = None
        
    async def __aenter__(self):
        self.transaction = await self.session.begin()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.transaction.rollback()
        else:
            await self.transaction.commit()
        return False

# Export commonly used items
__all__ = [
    'engine',
    'async_engine',
    'SessionLocal',
    'AsyncSessionLocal',
    'get_db',
    'get_async_db',
    'get_db_session',
    'get_async_session',
    'init_db',
    'drop_db',
    'check_database_connection',
    'close_database_connections',
    'create_session_factory',
    'create_async_session_factory',
    'DatabaseTransaction',
    'AsyncDatabaseTransaction',
]
