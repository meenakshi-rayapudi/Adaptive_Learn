import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .schema import Base

DB_PATH = os.getenv("ADAPTIVELEARN_DB_PATH", os.path.join("data", "adaptivelearn.db"))

_engine = None
_SessionFactory = None


def get_engine():
    """Lazy singleton SQLAlchemy engine bound to the local SQLite file."""
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def init_db():
    """Creates all tables if they do not already exist. Safe to call on every app startup."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()
