# 🎧 BeatHunter AI

> An end-to-end MLOps pipeline for music analysis and "Type Beat" recommendation, built on a fine-tuned audio embedding model and vector search.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)](https://docker.com)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat)]()

---

## What It Does

BeatHunter AI listens to a track, strips the vocals, and then:

1. **Classifies** the producer/artist style and genre as a consistent pair
2. **Measures** tempo, key and three character axes straight from the signal
3. **Recommends** similar beats from a 14,660-track catalogue by cosine similarity on 128-dimensional vectors
4. **Suggests what to type into YouTube** to find more of the same

---

## Architecture

```
Audio Input
    │
    ▼
Demucs (Vocal Separation)
    │
    ▼
PANNs Cnn14 — AudioSet-pretrained, fine-tuned on this corpus
    │
    ▼
2048-dim embedding ──► PCA-128 ──► StandardScaler
    │                                   │
    ├───────────────┐                   │
    ▼               ▼                   ▼
LogisticRegression  LogisticRegression  Cosine similarity
   (artist)            (genre)          → Top-5 beats
    │                   │
    └──► Joint decoding ┘
         score(a) = log P(artist=a) + log P(genre=G[a])
```

**Joint decoding** reads genre off the taxonomy instead of arguing it out independently,
so the pair is always one the data contains. It measures better than predicting genre
directly, and it cannot emit an artist/genre combination that does not exist.

Tempo, key and mood never touch the classifier — they are measured from the signal, so
they carry no confidence caveat.

---

## Measured Results

Held-out tracks, split by track (`GroupShuffleSplit`, `random_state=42`). Every number
here comes from that split.

| representation | artist top-1 | vs chance | top-5 | genre |
|---|---|---|---|---|
| EfficientNetB3 (ImageNet) | ~0.7% | 0.8x | — | ~1.3% |
| classical MFCC 102-dim | 9.45% | 16.7x | 24.13% | 17.32% |
| frozen PANNs (AudioSet) | 12.96% | 22.9x | 32.03% | 24.17% |
| **fine-tuned PANNs** | **17.90%** | **34.5x** | **39.33%** | **30.77%** |

The first row's lesson is domain, not capacity: EfficientNetB3 was the largest model
tried and landed *below chance*, because it transfers from photographs. PANNs transfers
from 2M audio clips. (The last row is measured on a later, larger corpus — 213 artists
against 194 — so it is not directly comparable to the rows above it.)

**Absolute accuracy is still low, and the UI does not hide it.** A single artist name is
shown only above a calibrated confidence bar — 150x chance, which covers 13.2% of tracks
and is right 47.0% of the time in that slice. Otherwise the top-5 candidate list leads.
The bar is re-measured on every training run, never copied between models.

---

## Beyond Classification

**Mood is measured, not predicted.** Three axes — energy, brightness, aggression — scored
against corpus percentiles rather than hand-picked thresholds. An earlier hand-assigned
mood label was dropped after three independent tests found it unrelated to the audio
(Cohen d +0.023 / +0.094, k-NN 1.4x chance).

**The query suggestion was validated by downloading.** The app suggests a YouTube search,
and that suggestion was checked by running it: 275 downloaded results scored against a
control arm that keeps the query shape but swaps in a random label. Genre queries beat
their control by +0.076 (3.1 SE), artist by +0.047 (2.2 SE). Mood queries were
indistinguishable from random (+0.018, 1.1 SE), so mood is shown as character but never
suggested as a search.

**Known data ceiling.** The corpus was scraped by searching `"<artist> <genre> Type Beat"`,
and 9.7% of it carried the wrong artist — the label recorded what was *asked for*, not
what came back. 1,171 tracks have since been re-labelled from their video titles, and the
collector now verifies titles before downloading. Fixing them moved the headline by 0.1
points, so label noise was real but was not the ceiling.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Audio / ML | PyTorch + PANNs, Librosa, Demucs, scikit-learn |
| Frontend | Streamlit |
| Data Mining | yt-dlp (autonomous scraper with failure guards) |
| Auth | Custom JWT (`auth.py`) |
| Deployment | Docker & Docker Compose (NVIDIA GPU support) |

TensorFlow appears only in the retired EfficientNetB3 training path
(`model_trainer.py`). Serving does not import it.

---

## Getting Started

### Prerequisites
- Docker & Docker Compose, or Python 3.13
- NVIDIA GPU + CUDA drivers (optional; inference runs on CPU)

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
python get_encoder.py            # check / fetch encoder weights first
uvicorn api:app --reload
# in a second terminal
pip install -r requirements-frontend.txt
streamlit run app.py
```

> `torch` and `torchaudio` must be the **same version**. Demucs fails with an opaque
> "exit 1" when they differ, because torchaudio's compiled extension links a specific
> torch ABI. Install both from the same index.

---

## Project Structure

```
BeatHunter-AI/
├── api.py                        # FastAPI backend
├── app.py                        # Streamlit frontend
├── auth.py                       # JWT authentication
├── config.py                     # Taxonomy: 213 artists / 60 genres
├── corpus.py                     # Corpus + train/held-out split — single source
├── audio_features.py             # PANNs embedding, mood axes, BPM folding
├── artist_match.py               # Artist-name matching, shared by collector and audit
├── inference_engine.py           # Classification, similarity, BPM/key
├── label_recommender.py          # "Search this on YouTube", by measured query utility
├── train_classic.py              # Classifier + catalogue + confidence calibration
├── data_collector.py             # YouTube miner with failure guard + title verification
├── calibrate_mood.py             # Derives and validates the mood axes
├── validate_query_suggestions.py # Downloads search results to test the suggestion
├── measure_label_noise.py        # Audits title against assigned label
├── models/                       # Weights and artefacts — generated, not hand-edited
└── docker-compose.yml
```

`CLAUDE.md` carries the working context: what was measured, what was tried and rejected,
and the traps that cost the most time.

---

## Status

🔧 **Active development.** Current focus is the accuracy ceiling. Label noise and taxonomy
coverage have been addressed and measured; the remaining gap looks to be in the
representation rather than the data.

---

## Author

**Koray Solakoğlu** · [LinkedIn](https://www.linkedin.com/in/koray-solakoglu/) · [GitHub](https://github.com/solakoglukoray)
