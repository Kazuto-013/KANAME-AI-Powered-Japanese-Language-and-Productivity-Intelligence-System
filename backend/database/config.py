import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# KANAME uses SQLite as its default persistence layer so the stack runs with zero
# external services. Set DATABASE_URL in .env to point at another backend if needed.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kaname.db")

# SQLite needs check_same_thread=False because FastAPI serves requests from a
# thread pool. Other dialects use the default connection settings.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
