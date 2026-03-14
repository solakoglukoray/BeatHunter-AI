# 🎧 BeatHunter AI

> An end-to-end MLOps pipeline for music analysis and "Type Beat" recommendation using a custom Multi-Head CNN and vector embeddings.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)](https://docker.com)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat)]()

---

## What It Does

BeatHunter AI listens to a music track, strips the vocals using AI, and then:
1. **Classifies** the genre, mood, and producer style (sonic brand)
2. **Recommends** the most similar beats from a 4000-track database using cosine similarity on 128-dimensional latent vectors

Instead of using a pre-trained image model, the system uses a **custom Multi-Head CNN** built specifically for audio spectrograms.

---

## Architecture

```
Audio Input
    │
    ▼
Demucs (Vocal Separation)
    │
    ▼
Mel-Spectrogram Extraction  ←── Librosa
    │
    ▼
Multi-Head CNN (EfficientNetB3 backbone)
    ├── Head 1: Genre Classification
    ├── Head 2: Mood Classification
    └── Head 3: Sonic Brand (Producer Style)
    │
    ▼
128-dim Latent Vector
    │
    ▼
Cosine Similarity → Top-N Recommendations
```

**Majority Voting Inference:** The track is sliced into 8 uniform segments. Each segment is analyzed independently and predictions are aggregated via majority vote — preventing intro/outro bias and improving robustness.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| AI / DSP | TensorFlow 2.15 (GPU), Librosa, Demucs, Scikit-learn |
| Frontend | Streamlit |
| Data Mining | yt-dlp (autonomous scraper) |
| Auth | Custom JWT (auth.py) |
| Deployment | Docker & Docker Compose (NVIDIA GPU support) |
| Training | WSL2 + RTX 4080 Laptop |

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers (for training)

### Run with Docker

```bash
git clone https://github.com/solakoglukoray/BeatHunter-AI.git
cd BeatHunter-AI
docker-compose up -d
```

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

### Run Locally

```bash
pip install -r requirements-backend.txt
uvicorn api:app --reload
# In a new terminal:
pip install -r requirements-frontend.txt
streamlit run app.py
```

---

## Project Structure

```
BeatHunter-AI/
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit frontend
├── auth.py                 # JWT authentication
├── config.py               # Configuration & hyperparameters
├── data_collector.py       # Autonomous YouTube data miner
├── data_processor.py       # Audio preprocessing pipeline
├── data_validator.py       # Dataset validation utilities
├── inference_engine.py     # Multi-head inference + majority voting
├── model_trainer.py        # CNN training pipeline
├── models/                 # Saved model weights
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml
```

---

## Status

🔧 **Active development** — Currently resolving EfficientNetB3 input normalization and optimizing the multi-head output layer. Training on RTX 4080 Laptop via WSL2.

---

## Author

**Koray Solakoğlu** · [LinkedIn](https://www.linkedin.com/in/koray-solakoglu/) · [GitHub](https://github.com/solakoglukoray)
