"""
BeatHunter AI - Core API Engine
This module serves as the backend REST API, handling audio processing, 
source separation (Demucs), and ML inference via FastAPI.
"""

import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Import custom inference engine modules
from inference_engine import (
    download_youtube_audio, 
    separate_audio, 
    analyze_beat, 
    TEMP_FOLDER
)

app = FastAPI(
    title="BeatHunter AI Core API",
    description="Music Information Retrieval & Multi-Task ML Inference Engine",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Health check endpoint to verify API status."""
    return {
        "status": "online", 
        "message": "BeatHunter AI Backend Engine is Running 🚀"
    }

# --- ENDPOINT 1: LOCAL FILE ANALYSIS ---
@app.post("/analyze_file/")
async def analyze_file_endpoint(
    file: UploadFile = File(...), 
    skip_demucs: bool = Form(False)
):
    """
    Accepts an uploaded audio file, optionally applies vocal separation, 
    and returns ML-based feature analysis and vector recommendations.
    """
    try:
        # Ensure temporary directory exists
        if not os.path.exists(TEMP_FOLDER): 
            os.makedirs(TEMP_FOLDER)
            
        file_path = os.path.join(TEMP_FOLDER, "api_uploaded_song.mp3")
        
        # Save the uploaded file to local disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Apply source separation if not skipped
        if not skip_demucs:
            beat_path = separate_audio(file_path)
        else:
            beat_path = file_path

        # Run inference and extract features
        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        
        # Parse the formatted tags (Artist / Mood / Genre)
        tags = formatted_result.split(" / ")

        return JSONResponse(content={
            "status": "success",
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
        # Catch and return any processing errors
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: YOUTUBE LINK/QUERY ANALYSIS ---
@app.post("/analyze_youtube/")
async def analyze_youtube_endpoint(
    query: str = Form(...)
):
    """
    Takes a YouTube URL or search query, downloads the audio, applies vocal separation,
    and returns ML-based feature analysis and vector recommendations.
    """
    try:
        print(f"🔍 [API] Incoming YouTube Request: {query}")
        
        # 1. Download audio from YouTube
        file_path = download_youtube_audio(query)
        
        # 2. Force vocal separation (Demucs) for YouTube sources to ensure clean beats
        beat_path = separate_audio(file_path)
        
        # 3. Run inference and extract features
        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)
        tags = formatted_result.split(" / ")

        return JSONResponse(content={
            "status": "success",
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

if __name__ == "__main__":
    print("🚀 Starting MLOps Server: http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)