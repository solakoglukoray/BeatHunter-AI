"""
BeatHunter AI - Core API Engine (v1.1 Enterprise Edition)
This module serves as the backend REST API, handling audio processing, 
source separation (Demucs), ML inference, and now JWT-based Authentication.
"""

import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time
import hashlib
from diskcache import Cache
import psutil

api_cache = Cache("api_cache_dir")

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

# --- ENDPOINT 1: LOCAL FILE ANALYSIS (PROTECTED & CACHED) ---
@app.post("/analyze_file/")
async def analyze_file_endpoint(
    file: UploadFile = File(...), 
    skip_demucs: bool = Form(False),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Accepts an uploaded audio file, optionally applies vocal separation, 
    and returns ML-based feature analysis. Uses DiskCache for extreme speed.
    """
    try:
        print(f"👤 Secure Local Analysis requested by user: {current_user.username}")
        
        # 1. Read file content to generate a unique mathematical footprint (Hash)
        file_content = await file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        cache_key = f"file_{file_hash}_{skip_demucs}"

        # 2. Check if this exact file was analyzed before (Cache Hit)
        cached_result = api_cache.get(cache_key)
        if cached_result:
            print(f"⚡ [CACHE HIT] Exact file found in memory! Serving in 0.01s.")
            cached_result["user"] = current_user.username # Update user info
            return JSONResponse(content=cached_result)

        # 3. If not in cache, reset file cursor and proceed with normal ML inference
        await file.seek(0)
        
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

        # 4. Prepare the final response data
        response_data = {
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
        }
        
        # 5. Save the result to cache for 7 days (604800 seconds)
        api_cache.set(cache_key, response_data, expire=604800)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- THE HEAVY WORKER FUNCTION (RUNS IN BACKGROUND) ---
def background_youtube_worker(query: str, cache_key: str, username: str):
    """
    Runs completely in the background. Does the heavy ML lifting (Demucs & Model) 
    and updates the DiskCache when finished, preventing API timeouts.
    """
    try:
        # 1. Mark task as processing
        api_cache.set(cache_key, {"status": "processing", "query": query}, expire=3600)
        
        # 2. Execute heavy lifting
        file_path = download_youtube_audio(query)
        beat_path = separate_audio(file_path)
        
        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        tags = formatted_result.split(" / ")

        # 3. Prepare final success payload
        response_data = {
            "status": "completed",
            "user": username,
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
        }
        
        # 4. Save the final result to cache for 7 days
        api_cache.set(cache_key, response_data, expire=604800)
        print(f"✅ [WORKER] Background task finished successfully for: {query}")
        
    except Exception as e:
        # 5. Handle failures gracefully
        print(f"❌ [WORKER] Task failed for {query}: {str(e)}")
        api_cache.set(cache_key, {"status": "failed", "error": str(e)}, expire=3600)


# --- ENDPOINT 2: YOUTUBE LINK/QUERY ANALYSIS (ASYNC TICKET SYSTEM) ---
@app.post("/analyze_youtube/")
async def analyze_youtube_endpoint(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Takes a YouTube query. If not cached, issues a Ticket (Task ID) 
    and delegates heavy ML inference to a background worker.
    """
    try:
        print(f"🔍 [API] Incoming YouTube Request from {current_user.username}: {query}")
        
        # 1. Create a unique Task ID (Cache Key) based on the search query
        cache_key = f"yt_{query.strip().lower()}"
        
        # 2. Check current state in memory
        cached_result = api_cache.get(cache_key)
        
        if cached_result:
            if cached_result.get("status") == "completed":
                print(f"⚡ [CACHE HIT] Query already completed! Serving instantly.")
                cached_result["user"] = current_user.username
            else:
                print(f"⏳ [CACHE QUEUE] Task is already processing or pending.")
            # Return the existing task state
            return JSONResponse(content={"task_id": cache_key, "current_state": cached_result})
        
        # 3. If completely new, issue a ticket and start the background worker
        print(f"🎟️ [NEW TASK] Issuing ticket for: {query}")
        
        initial_state = {"status": "pending", "query": query}
        api_cache.set(cache_key, initial_state, expire=3600)
        
        # Fire and forget! (Delegating to background)
        background_tasks.add_task(background_youtube_worker, query, cache_key, current_user.username)
        
        return JSONResponse(content={
            "task_id": cache_key, 
            "current_state": initial_state,
            "message": "Task queued successfully. Please poll for status."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3: TASK STATUS POLLING (LIMIT-FREE) ---
@app.get("/task_status/{task_id}")
async def get_task_status(
    task_id: str, 
    current_user: auth.User = Depends(auth.get_current_user)
):
    """
    Check task status without hitting the rate limit. 
    Only authentication is required.
    """
    clean_id = task_id.strip()
    cached_result = api_cache.get(clean_id)
    
    if not cached_result:
        return JSONResponse(content={"status": "pending", "message": "Initializing..."}, status_code=200)
        
    return JSONResponse(content=cached_result)

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

@app.get("/admin/stats")
async def get_admin_stats(current_user: auth.User = Depends(auth.get_current_user)):
    """
    Admin-only endpoint to retrieve system health, cache metrics, 
    and recent analysis activities.
    """
    if current_user.username.lower() != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admins only.")
    
    # 1. Filter and count song-related keys in DiskCache
    all_keys = [key for key in api_cache.iterkeys() if key.startswith("yt_")]
    total_cached_songs = len(all_keys)
    
    # 2. Extract recent activities (last 5 tasks)
    recent_tasks = []
    # Taking the last 5 keys to show the most recent submissions
    for k in all_keys[-5:]:
        data = api_cache.get(k)
        if data and isinstance(data, dict):
            recent_tasks.append({
                "query": data.get("query", "Unknown"),
                "user": data.get("user", "System"),
                "status": data.get("status", "unknown")
            })

    # 3. Collect system performance metrics
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    
    # 4. Compile final statistics payload
    stats = {
        "total_queries": total_cached_songs,
        "system_health": {
            "cpu": f"%{cpu_usage}",
            "ram": f"%{ram_usage}",
            "status": "Healthy" if cpu_usage < 80 else "Overloaded"
        },
        "cache_size": f"{len(api_cache)} items",
        "recent_activities": recent_tasks  # Added for the new Dashboard UI
    }
    
    return stats

if __name__ == "__main__":
    print("🚀 Starting BeatHunter Enterprise API: http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)