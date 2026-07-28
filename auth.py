"""
BeatHunter AI - Authentication & Security Module (v1.5)
Handles user registration, database session management (SQLite), 
password hashing (bcrypt), JWT-based authentication, and user data models.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import bcrypt
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# SECRET_KEY .env dosyasından okunur. Üretmek için:
#   python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY = os.getenv("SECRET_KEY", "beathunter-fallback-key-CHANGE-IN-PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
SQLALCHEMY_DATABASE_URL = "sqlite:///./beathunter_users.db"

# --- DATABASE SETUP (SQLAlchemy) ---
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """Database model for registered users."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Relationships linking the user to their search history and favorite beats
    history = relationship("SearchHistory", back_populates="owner")
    favorites = relationship("Favorite", back_populates="owner")

class SearchHistory(Base):
    """Database model for tracking user search history."""
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    query = Column(String, index=True)          # e.g., "Travis Scott Type Beat"
    source = Column(String)                     # e.g., "youtube" or "local"
    task_id = Column(String)                    # The cache key (e.g., "yt_...")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship linking back to the user
    owner = relationship("User", back_populates="history")

class Favorite(Base):
    """Database model for a user's favorite beats."""
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    query = Column(String)                      # Song name or search query
    task_id = Column(String)                    # To retrieve analysis from cache
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship linking back to the user
    owner = relationship("User", back_populates="favorites")

# Generate the SQLite database and tables if they don't exist
Base.metadata.create_all(bind=engine)


def run_migrations():
    """
    Applies schema migrations that can't be handled by create_all.
    Safe to run multiple times — checks before altering.
    """
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(users)"))
        columns = [row[1] for row in result]

        if "is_admin" not in columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0"))
            # Mevcut 'admin' kullanıcısına otomatik olarak admin yetkisi ver
            conn.execute(text("UPDATE users SET is_admin = 1 WHERE username = 'admin'"))
            conn.commit()
            print("✅ Migration: is_admin kolonu eklendi, 'admin' kullanıcısına yetki verildi.")


run_migrations()

# --- SECURITY & CRYPTOGRAPHY ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compares a plain text password against its bcrypt hashed version."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    """Hashes a password using the raw bcrypt algorithm."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Generates a JSON Web Token (JWT) for authenticated sessions."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_refresh_token(data: dict):
    """
    Generates a long-lived refresh token.
    Used to obtain a new access token without requiring re-login.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Add a specific type to differentiate from access tokens
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- DEPENDENCIES ---
def get_db():
    """Yields a database session and safely closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Extracts and validates the current user from the provided JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user