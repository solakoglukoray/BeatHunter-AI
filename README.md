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

## Kurulum — encoder ağırlıkları

Yeni bir makinede **ilk iş**:

```bash
pip install -r requirements-backend.txt
python get_encoder.py        # ne var ne yok raporlar, stock ağırlıkları indirir
```

Sınıflandırma ve benzerlik araması, AudioSet'te ön-eğitilip bu veri setiyle fine-tune
edilmiş bir PANNs (Cnn14) encoder'ına dayanıyor. **İki ayrı ağırlık dosyası var ve
cinsleri farklı:**

| dosya | boyut | ne | nasıl elde edilir |
|---|---|---|---|
| `models/panns_ft_encoder.pt` | 327 MB | **bizim** fine-tune ettiğimiz encoder | **indirilemez** — build artefaktı |
| `~/panns_data/Cnn14_mAP=0.431.pth` | 327 MB | stock AudioSet ağırlıkları | Zenodo'dan otomatik indirilir |

Fine-tuned ağırlık hiçbir yerde yayınlanmış değil; `panns_finetune.py`'nin çıktısı.
Üç yol var:

1. **Kopyala** (günlük kullanımda bu) — elinde olan bir makineden
   `models/panns_ft_encoder.pt` olarak koy, `python get_encoder.py --verify` ile doğrula.
2. **Yeniden üret** — `python panns_finetune.py` (~2.8 sa GPU), ardından
   `python panns_ft_frozen_experiment.py --extract-encoder`, sonra
   `python train_classic.py --extract` (~45 dk). Sonuç bit-özdeş olmaz; hash değişir,
   yeniden üretim uyumlu bir çift yazar.
3. **Onsuz çalış** — dosyayı koymazsan stock ağırlıklar otomatik devreye girer.
   Ölçülen maliyet: artist top-1 %15.35 → %12.96, top-5 %37.72 → %32.03.
   Ardından `python train_classic.py` çalıştır (dakikalar, embedding'ler cache'te).

Git ikisini de takip etmiyor: her yeniden eğitimde değişen 327 MB binary LFS işi ve
repoda LFS kurulu değil.

⚠️ **Katalog ve sorgu aynı encoder'dan gelmek zorunda.** Model bundle'ı `encoder_id` ve
ağırlık hash'ini saklar; `inference_engine` yüklemede ikisini de doğrular ve uyuşmazlıkta
hata verir. Encoder'ı değiştirdiysen `train_classic.py` çalıştırman gerekir.

## Ölçülen doğruluk

2,847 held-out parça (`GroupShuffleSplit`, track bazlı):

| temsil | artist top-1 | top-5 | genre |
|---|---|---|---|
| EfficientNetB3 (ImageNet) | ~%0.7 *(şansın altında)* | — | ~%1.3 |
| klasik MFCC 102-dim | %9.45 | %24.13 | %17.32 |
| donuk PANNs | %12.96 | %32.03 | %24.17 |
| **fine-tuned PANNs** | **%15.35** | **%37.72** | **%28.21** |

Mutlak seviye düşük: top-1 yaklaşık 7 denemede 1 doğru. UI bunu saklamıyor — tek bir
isim ancak ölçülmüş bir güven eşiğini (150x şans, parçaların %11'i, o dilimde doğruluk
%43.5) geçtiğinde gösterilir, aksi halde top-5 aday listesi sunulur.
