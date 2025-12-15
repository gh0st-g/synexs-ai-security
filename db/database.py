"""
Synexs Database Connection Manager
Handles PostgreSQL connections with connection pooling
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)

# Database configuration from environment
DB_USER = os.getenv('POSTGRES_USER', 'synexs')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'synexs_secure_pass_2024')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'synexs')

# Connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL debug logging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Thread-safe scoped session
db_session = scoped_session(SessionLocal)


@contextmanager
def get_db():
    """
    Context manager for database sessions

    Usage:
        with get_db() as db:
            db.add(attack)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


def get_db_session():
    """
    Get database session (for dependency injection)
    Remember to close session after use
    """
    return SessionLocal()


def init_db():
    """
    Initialize database (create all tables)
    """
    from db.models import Base
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def test_connection():
    """
    Test database connection
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def close_db():
    """
    Close all database connections
    """
    db_session.remove()
    engine.dispose()
    logger.info("Database connections closed")
