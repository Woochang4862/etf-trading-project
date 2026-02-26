from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
import os

# Remote MySQL Engine (for reading stock data)
remote_engine = create_engine(
    settings.remote_db_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# Processed features MySQL Engine (for reading ML features)
processed_engine = create_engine(
    settings.processed_db_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# Local SQLite Engine (for storing predictions)
os.makedirs(os.path.dirname(settings.local_db_path), exist_ok=True)
local_engine = create_engine(
    f"sqlite:///{settings.local_db_path}",
    connect_args={"check_same_thread": False}
)

# Session factories
RemoteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=remote_engine)
ProcessedSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=processed_engine)
LocalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=local_engine)

# Base for local models
Base = declarative_base()


def get_remote_db():
    """원격 MySQL DB 세션 (etf2_db)"""
    db = RemoteSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_processed_db():
    """Processed features DB 세션 (etf2_db_processed)"""
    db = ProcessedSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_local_db():
    """로컬 SQLite DB 세션"""
    db = LocalSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_local_db():
    """로컬 DB 테이블 생성"""
    from app.models import Prediction, ETFMonthlySnapshot, ETFComposition  # noqa
    Base.metadata.create_all(bind=local_engine)
