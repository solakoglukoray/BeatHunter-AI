# 🎧 BeatHunter AI: MLOps Edition

**BeatHunter AI** is an end-to-end Music Information Retrieval (MIR) and vectorized "Type Beat" recommendation system built for music producers, artists, and listeners. 

Instead of relying on off-the-shelf pre-trained image models, it features a custom-built **Multi-Head CNN** architecture specifically designed for audio spectrograms. The system listens to audio tracks, isolates vocals using AI, predicts the Genre, Mood, and Sonic Brand (Producer Style), and recommends the most similar beats in seconds using latent vector space matching.

## 🚀 Tech Stack (Microservice Architecture)
This project is built with a production-ready microservice architecture and is fully **Dockerized**.
* **Backend:** FastAPI, Uvicorn, Python-Multipart
* **AI & DSP:** TensorFlow 2.15 (GPU), Librosa, Demucs, Scikit-learn
* **Frontend:** Streamlit
* **Data Mining:** yt-dlp (Autonomous data miner)
* **Deployment:** Docker & Docker Compose (with NVIDIA GPU Support)

## 🧠 Core Features
1. **Autonomous Data Mining:** Automatically scrapes, downloads, and labels thousands of tracks from YouTube based on a curated sonic taxonomy.
2. **AI Vocal Separation:** Utilizes the Demucs deep learning model to strip vocals and isolate the raw instrumental beat.
3. **Majority Voting Inference:** Slices the track into 8 uniform segments and analyzes each independently to prevent intro/outro biases, ensuring highly robust predictions.
4. **Vectorized Audio DNA (Recommendation Engine):** Extracts 128-dimensional latent vectors from the CNN and uses Cosine Similarity to find the closest matching beats within a 4000-track vector database.

## 🐳 Deployment (Dockerized)
The project is packaged for production environments. To spin up the entire microservice ecosystem, simply run:

```bash
docker-compose up -d
```

* **Frontend UI:** `http://localhost:8501`
* **Backend API Docs (Swagger):** `http://localhost:8000/docs`