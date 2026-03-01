"""
BeatHunter AI - Core API Engine (v1.4 Enterprise Edition)
This module serves as the backend REST API, handling audio processing, 
source separation (Demucs), ML inference, and JWT-based Authentication.

v1.4 Changes:
- Each background task now uses a unique temp folder (UUID-based) to prevent
  concurrent users from overwriting each other's downloaded/separated files.
- worker_semaphore added: limits concurrent heavy ML tasks to 2 at a time,
  preventing CPU overload when many users analyze simultaneously.
"""

import os
import shutil
import uvicorn
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time
import hashlib
from diskcache import Cache
import psutil
import asyncio
from jose import jwt, JWTError, ExpiredSignatureError

api_cache = Cache("api_cache_dir")

# Import custom inference engine modules
from inference_engine import (
    download_youtube_audio,
    separate_audio,
    analyze_beat,
    TEMP_FOLDER
)

import auth

app = FastAPI(
    title="BeatHunter AI Core API",
    description="Music Information Retrieval, Multi-Task ML Inference & Auth Engine",
    version="1.2.0"
)

# ==========================================
# 🚦 CONCURRENCY LIMITER
# Max 2 heavy ML tasks run simultaneously.
# Others wait in queue — no CPU overload.
# ==========================================
worker_semaphore = asyncio.Semaphore(2)

# ==========================================
# 🛡️ GLOBAL EXCEPTION HANDLER
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches ALL unhandled exceptions across the entire API.
    Always returns a clean JSON structure instead of raw HTML error pages.
    """
    error_message = str(exc)
    print(f"🚨 [GLOBAL ERROR] {request.method} {request.url.path} -> {error_message}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "failed",
            "error": "An unexpected system error occurred.",
            "details": error_message,
            "path": request.url.path
        }
    )

# --- PYDANTIC SCHEMAS ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

# --- HEALTH CHECK ---
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "BeatHunter AI Backend Engine v1.2 is Running 🚀"
    }

# --- RATE LIMITING ---
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
    """Registers a new user, hashes the password, and returns JWT tokens."""
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

    access_token  = auth.create_access_token(data={"sub": new_user.username})
    refresh_token = auth.create_refresh_token(data={"sub": new_user.username})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@app.post("/login", response_model=Token, tags=["Authentication"])
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(auth.get_db)
):
    """Authenticates a user and returns a JWT access token + refresh token."""
    user = db.query(auth.User).filter(auth.User.username == form_data.username).first()

    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token  = auth.create_access_token(data={"sub": user.username})
    refresh_token = auth.create_refresh_token(data={"sub": user.username})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@app.post("/refresh", tags=["Authentication"])
async def refresh_access_token(refresh_token: str = Form(...)):
    """Issues a new access token if the provided refresh token is valid."""
    try:
        payload = jwt.decode(refresh_token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])

        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type.")

        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload.")

        new_access_token = auth.create_access_token(data={"sub": username})
        return {"access_token": new_access_token, "token_type": "bearer"}

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired. Please log in again.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")


# --- WEBSOCKET CONNECTION MANAGER ---
class ConnectionManager:
    """Maps active WebSocket connections to their Task IDs."""
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]

    async def send_update(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(message)
            except Exception as e:
                print(f"⚠️ [WS] Failed to send update for {task_id}: {e}")

manager = ConnectionManager()

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """Real-time communication tunnel for a specific analysis task."""
    await manager.connect(websocket, task_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id)
        print(f"🔌 [WS] Client disconnected from task: {task_id}")


# --- ENDPOINT 1: LOCAL FILE ANALYSIS ---
@app.post("/analyze_file/")
async def analyze_file_endpoint(
    file: UploadFile = File(...),
    skip_demucs: bool = Form(False),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Accepts an uploaded audio file, optionally applies vocal separation,
    and returns ML-based analysis. Uses DiskCache to serve repeated files instantly.
    """
    try:
        print(f"👤 Local analysis requested by: {current_user.username}")

        file_content = await file.read()
        file_hash    = hashlib.md5(file_content).hexdigest()
        cache_key    = f"file_{file_hash}_{skip_demucs}"

        # Cache hit — serve instantly
        cached_result = api_cache.get(cache_key)
        if cached_result:
            print(f"⚡ [CACHE HIT] Serving instantly.")
            cached_result["user"] = current_user.username
            return JSONResponse(content=cached_result)

        await file.seek(0)

        if not os.path.exists(TEMP_FOLDER):
            os.makedirs(TEMP_FOLDER)

        file_path = os.path.join(TEMP_FOLDER, "api_uploaded_song.mp3")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        beat_path = separate_audio(file_path) if not skip_demucs else file_path

        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        tags = formatted_result.split(" / ")

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

        api_cache.set(cache_key, response_data, expire=604800)
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 🔧 BACKGROUND WORKER
# Each task gets its own UUID-based temp folder.
# This prevents concurrent users from overwriting
# each other's downloaded/separated audio files.
# worker_semaphore limits simultaneous heavy tasks
# to 2, preventing CPU overload.
# ==========================================
async def background_youtube_worker(query: str, cache_key: str, username: str):
    # Create a unique temp folder for this specific task
    task_id     = uuid.uuid4().hex[:8]
    task_folder = os.path.join(TEMP_FOLDER, f"task_{task_id}")
    os.makedirs(task_folder, exist_ok=True)

    try:
        api_cache.set(cache_key, {"status": "processing", "query": query}, expire=3600)

        # Acquire semaphore — if 2 workers are already running, this one waits
        async with worker_semaphore:
            print(f"⚙️ [WORKER] Starting task {task_id} for: {query}")
            loop = asyncio.get_event_loop()

            # Download to unique folder
            file_path = await loop.run_in_executor(
                None, _download_to_folder, query, task_folder
            )

            # Separate vocals
            beat_path = await loop.run_in_executor(
                None, _separate_to_folder, file_path, task_folder
            )

            # Run ML inference
            formatted_result, recommendations, bpm, key = await loop.run_in_executor(
                None, analyze_beat, beat_path
            )

        tags = formatted_result.split(" / ")

        response_data = {
            "status": "completed",
            "user": username,
            "source": "youtube",
            "query": query,
            "beat_path": beat_path,
            "analysis": {
                "artist_type": tags[0].replace(" Type", ""),
                "mood": tags[1],
                "genre": tags[2],
                "bpm": bpm,
                "key": key
            },
            "recommendations": recommendations
        }

        # Save the separated beat to a stable location for audio preview
        preview_folder = os.path.join(TEMP_FOLDER, "previews")
        os.makedirs(preview_folder, exist_ok=True)
        stable_beat_path = os.path.join(preview_folder, f"{task_id}_no_vocals.mp3")
        shutil.copy2(beat_path, stable_beat_path)

        # Update response_data with the stable path
        response_data["beat_path"] = stable_beat_path
        api_cache.set(cache_key, response_data, expire=604800)
        await manager.send_update(cache_key, response_data)
        print(f"✅ [WORKER] Task {task_id} completed for: {query}")

    except Exception as e:
        print(f"❌ [WORKER] Task {task_id} failed: {e}")
        error_data = {"status": "failed", "error": str(e)}
        api_cache.set(cache_key, error_data, expire=3600)
        await manager.send_update(cache_key, error_data)
        
    finally:
        # Always clean up the unique temp folder after task finishes
        if os.path.exists(task_folder):
            shutil.rmtree(task_folder, ignore_errors=True)
            print(f"🧹 [WORKER] Cleaned up temp folder: {task_folder}")


def _download_to_folder(query: str, folder: str) -> str:
    """
    Downloads YouTube audio into the task's unique temp folder.
    Wraps download_youtube_audio to use a custom output path.
    """
    import yt_dlp
    output_path = os.path.join(folder, "downloaded_song.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'quiet': True, 'noplaylist': True, 'default_search': 'ytsearch1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])
    return os.path.join(folder, "downloaded_song.mp3")


def _separate_to_folder(input_path: str, folder: str) -> str:
    """
    Runs Demucs vocal separation, outputting into the task's unique temp folder.
    """
    import sys, subprocess
    command = (
        f'"{sys.executable}" -m demucs -n htdemucs --two-stems=vocals '
        f'--mp3 --mp3-bitrate 320 "{input_path}" -o "{folder}"'
    )
    subprocess.run(command, shell=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    beat_path = os.path.join(folder, "htdemucs", base_name, "no_vocals.mp3")
    if os.path.exists(beat_path):
        return beat_path
    else:
        raise Exception(f"Source separation failed. Expected: {beat_path}")


# --- ENDPOINT 2: YOUTUBE ANALYSIS (ASYNC TICKET SYSTEM) ---
@app.post("/analyze_youtube/")
async def analyze_youtube_endpoint(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    current_user: auth.User = Depends(check_rate_limit)
):
    """
    Issues a task ticket for a YouTube query and delegates
    heavy ML inference to an isolated background worker.
    """
    try:
        print(f"🔍 [API] YouTube request from {current_user.username}: {query}")

        cache_key     = f"yt_{query.strip().lower()}"
        cached_result = api_cache.get(cache_key)

        if cached_result:
            if cached_result.get("status") == "completed":
                print(f"⚡ [CACHE HIT] Serving instantly.")
                cached_result["user"] = current_user.username
            else:
                print(f"⏳ [QUEUE] Task already in progress.")
            return JSONResponse(content={"task_id": cache_key, "current_state": cached_result})

        print(f"🎟️ [NEW TASK] Issuing ticket for: {query}")
        initial_state = {"status": "pending", "query": query}
        api_cache.set(cache_key, initial_state, expire=3600)

        asyncio.create_task(background_youtube_worker(query, cache_key, current_user.username))

        return JSONResponse(content={
            "task_id": cache_key,
            "current_state": initial_state,
            "message": "Task queued. Connect via WebSocket for real-time callback."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3: TASK STATUS POLLING ---
@app.get("/task_status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: auth.User = Depends(auth.get_current_user)
):
    """Check task status without consuming the rate limit."""
    cached_result = api_cache.get(task_id.strip())
    if not cached_result:
        return JSONResponse(content={"status": "pending", "message": "Initializing..."})
    return JSONResponse(content=cached_result)


# --- ADMIN ROUTES ---
@app.get("/admin/users", tags=["Admin"])
def get_all_users(
    current_user: auth.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db)
):
    """Returns all registered users. Admin only."""
    if current_user.username.lower() != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admins only.")
    users = db.query(auth.User).all()
    return [{"ID": u.id, "Username": u.username, "Email": u.email, "Active": u.is_active} for u in users]


@app.get("/admin/stats", tags=["Admin"])
async def get_admin_stats(current_user: auth.User = Depends(auth.get_current_user)):
    """Returns system health, cache metrics, and recent activities. Admin only."""
    if current_user.username.lower() != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admins only.")

    all_keys         = [key for key in api_cache.iterkeys() if key.startswith("yt_")]
    total_cached     = len(all_keys)
    recent_tasks     = []

    for k in all_keys[-5:]:
        data = api_cache.get(k)
        if data and isinstance(data, dict):
            recent_tasks.append({
                "query":  data.get("query", "Unknown"),
                "user":   data.get("user", "System"),
                "status": data.get("status", "unknown")
            })

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    return {
        "total_queries": total_cached,
        "system_health": {
            "cpu":    f"%{cpu_usage}",
            "ram":    f"%{ram_usage}",
            "status": "Healthy" if cpu_usage < 80 else "Overloaded"
        },
        "cache_size":        f"{len(api_cache)} items",
        "recent_activities": recent_tasks
    }


@app.delete("/admin/cache", tags=["Admin"])
async def clear_system_cache(current_user: auth.User = Depends(auth.get_current_user)):
    """Clears the entire DiskCache. Admin only."""
    if current_user.username.lower() != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admins only.")
    api_cache.clear()
    return {"status": "success", "message": "Cache completely cleared."}


if __name__ == "__main__":
    print("🚀 Starting BeatHunter Enterprise API: http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)