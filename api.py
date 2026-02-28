"""
BeatHunter AI - Core API Engine (v1.1 Enterprise Edition)
This module serves as the backend REST API, handling audio processing, 
source separation (Demucs), ML inference, and now JWT-based Authentication.
"""

import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time

# Import custom inference engine modules
from inference_engine import (
    download_youtube_audio, 
    separate_audio, 
    analyze_beat, 
    TEMP_FOLDER
)

# Import our newly created Security Module
import auth

app = FastAPI(
    title="BeatHunter AI Core API",
    description="Music Information Retrieval, Multi-Task ML Inference & Auth Engine",
    version="1.1.0"
)

# --- PYDANTIC SCHEMAS (Data Validation) ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# --- HEALTH CHECK ---
@app.get("/")
def read_root():
    """Health check endpoint to verify API status."""
    return {
        "status": "online", 
        "message": "BeatHunter AI Backend Engine v1.1 is Running 🚀"
    }

# --- RATE LIMITING (SECURITY SHIELD) ---
RATE_LIMIT_DB = {}
MAX_REQUESTS_PER_MINUTE = 3

def check_rate_limit(current_user: auth.User = Depends(auth.get_current_user)):

    username = current_user.username
    
    if username.lower() == "admin":
        return current_user

    current_time = time.time()
    
    if username not in RATE_LIMIT_DB:
        RATE_LIMIT_DB[username] = []
        
    RATE_LIMIT_DB[username] = [t for t in RATE_LIMIT_DB[username] if current_time - t < 60]
    
    if len(RATE_LIMIT_DB[username]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded! You can only analyze 3 songs per minute. Please wait."
        )
        
    RATE_LIMIT_DB[username].append(current_time)
    return current_user

# --- AUTHENTICATION ROUTES ---
@app.post("/register", response_model=Token, tags=["Authentication"])
def register_user(user: UserCreate, db: Session = Depends(auth.get_db)):
    """Registers a new user, hashes the password, and returns a JWT token."""
    db_user = db.query(auth.User).filter(
        (auth.User.username == user.username) | (auth.User.email == user.email)
    ).first()
    
    if db_user:
        raise HTTPException(status_code=400, detail="Error: Username or email is already registered.")
    
    hashed_pwd = auth.get_password_hash(user.password)
    new_user = auth.User(username=user.username, email=user.email, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = auth.create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token, tags=["Authentication"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(auth.get_db)):
    """Authenticates a user and returns a secure JWT access token."""
    user = db.query(auth.User).filter(auth.User.username == form_data.username).first()
    
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- ENDPOINT 1: LOCAL FILE ANALYSIS (PROTECTED) ---
@app.post("/analyze_file/")
async def analyze_file_endpoint(
    file: UploadFile = File(...), 
    skip_demucs: bool = Form(False),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Accepts an uploaded audio file, optionally applies vocal separation, 
    and returns ML-based feature analysis. Requires JWT Authentication.
    """
    try:
        print(f"👤 Secure Local Analysis requested by user: {current_user.username}")
        if not os.path.exists(TEMP_FOLDER): 
            os.makedirs(TEMP_FOLDER)
            
        file_path = os.path.join(TEMP_FOLDER, "api_uploaded_song.mp3")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not skip_demucs:
            beat_path = separate_audio(file_path)
        else:
            beat_path = file_path

        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        tags = formatted_result.split(" / ")

        return JSONResponse(content={
            "status": "success",
            "user": current_user.username,
            "source": "local_file",
            "filename": file.filename,
            "analysis": {
                "artist_type": tags[0].replace(" Type", ""),
                "mood": tags[1],
                "genre": tags[2],
                "bpm": bpm,
                "key": key
            },
            "recommendations": recommendations
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: YOUTUBE LINK/QUERY ANALYSIS (PROTECTED) ---
@app.post("/analyze_youtube/")
async def analyze_youtube_endpoint(
    query: str = Form(...),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Takes a YouTube URL/query, downloads audio, applies vocal separation,
    and returns ML-based feature analysis. Requires JWT Authentication.
    """
    try:
        print(f"🔍 [API] Incoming YouTube Request from {current_user.username}: {query}")
        
        file_path = download_youtube_audio(query)
        beat_path = separate_audio(file_path)
        
        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        tags = formatted_result.split(" / ")

        return JSONResponse(content={
            "status": "success",
            "user": current_user.username,
            "source": "youtube",
            "query": query,
            "analysis": {
                "artist_type": tags[0].replace(" Type", ""),
                "mood": tags[1],
                "genre": tags[2],
                "bpm": bpm,
                "key": key
            },
            "recommendations": recommendations
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ADMIN ROUTES (SUPER SECRET) ---
@app.get("/admin/users", tags=["Admin Dashboard"])
def get_all_users(
    current_user: auth.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db)
):
    """Fetches all registered users. Restricted to 'admin' account only."""
    if current_user.username.lower() != "admin":
        raise HTTPException(status_code=403, detail="Access forbidden: Admins only! 🛑")

    users = db.query(auth.User).all()
    safe_users_data = [
        {"ID": u.id, "Username": u.username, "Email": u.email, "Active": u.is_active} 
        for u in users
    ]
    return safe_users_data

if __name__ == "__main__":
    print("🚀 Starting BeatHunter Enterprise API: http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)