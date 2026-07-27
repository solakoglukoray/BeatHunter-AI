# BeatHunter AI — Proje Bağlamı

## Ne yapıyor
Ses girdisi alır → Demucs ile vokal ayırır → mel-spektrogram çıkarır (Librosa) →
özel Multi-Head CNN (EfficientNetB3 backbone) ile genre/mood/sonic-brand sınıflandırır →
128-dim (gerçekte 1536-dim, bkz. Bilinen Tuzaklar) latent vektör çıkarır →
FAISS ile ~14K track'lik veritabanından cosine similarity ile benzer beat önerir.

Segment bazlı çıkarım: track 8 eşit segmente bölünür, her segment ayrı analiz edilip
majority voting ile birleştirilir (intro/outro bias'ını önlemek için).

## Stack
- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- ML: TensorFlow 2.15 (GPU), Librosa, Demucs, Scikit-learn, FAISS
- Auth: Custom JWT (`auth.py`)
- Deploy: Docker Compose (NVIDIA GPU support)
- Training: WSL2 + RTX 4080 Laptop

## Dosya haritası
| Dosya | Görev |
|---|---|
| `api.py` | FastAPI backend, endpoint'ler |
| `app.py` | Streamlit frontend |
| `auth.py` | JWT auth |
| `config.py` | Config & hiperparametreler |
| `data_collector.py` | YouTube veri madenciliği (yt-dlp) |
| `data_processor.py` | Ses → spektrogram pipeline (eğitim tarafı) |
| `data_validator.py` | Dataset doğrulama |
| `inference_engine.py` | Çıkarım + majority voting (inference tarafı) |
| `model_trainer.py` | CNN eğitim pipeline'ı |
| `models/` | Kaydedilmiş model ağırlıkları — **elle değiştirme** |

## ⚠️ Bilinen Tuzaklar (kritik — yeni kod yazmadan önce oku)

1. **Train/inference spektrogram uyuşmazlığı (KÖK NEDEN — ✅ DÜZELTİLDİ 2026-07-27)**
   `data_processor.py` v2.0'a güncellendi (n_mels=256, fmin/fmax eklendi) ama diskteki
   873K eğitim görüntüsü hâlâ eski parametrelerle (n_mels=128, fmin/fmax yok) üretilmiş
   ve **hiç yeniden üretilmedi**. Düzeltme = inference'ı gerçek eğitim parametrelerine
   geri hizalamak, modele değil — `inference_engine.py` artık n_mels=128 / fmin-fmax yok
   kullanıyor ve ölçülen korelasyon 0.31 → **0.9749**.
   → ⚠️ **Parite kırılgan.** `render_spectrogram_png()` içindeki her argüman kritik:
   `melspectrogram`'a `fmin/fmax` **verme**, `specshow`'a `cmap` **verme** (librosa
   otomatik seçiyor; `cmap='viridis'` pariteyi bozar). `data_processor.py` hâlâ
   commit edilmemiş v2.0 halinde — **kod ile diskteki veri farklı**, bunu çalıştırma.
   → `data_processor.py:102`'deki `if os.path.exists(img_path): continue` satırı bu
   dosyayı normal koşullarda **sessizce resume/skip** eder ve "ALL CONVERSIONS
   COMPLETED" yazdırır — dataset'i gerçekten yeniden üretmek istiyorsan bu satırı
   bypass etmen gerekir.

2. **Layer adı hardcode edilmiş (✅ DÜZELTİLDİ 2026-07-27)**
   `model.get_layer('global_average_pooling2d')` çağrısı `ValueError` veriyordu:
   `model_trainer.py` aynı oturumda `build_efficientnet_multihead`'i iki kez çağırıyor,
   Keras ikinci çağrıda katman adına `_1` ekliyor — kaydedilen model bu ikincisi
   (`global_average_pooling2d_1`). Artık `build_embedding_model()` katmanı **tipinden**
   buluyor. Yeniden eğitim sonrası ad yine değişebilir; isme geri dönme.

3. **FAISS index / metadata farklı zamanlardan, boyut uyuşmuyor**
   Index d=256 ama modelin GAP çıktısı 1536. `vector_meta.csv` (214,506 satır, 28 Şub)
   ile `metadata_processed.csv` (873,924 satır, 8 Mart) ile model (13 Mart) üç ayrı
   zaman diliminden. **Index'i yeniden inşa etmeden önce Tuzak 1 ve 2 düzeltilmiş
   olmalı**, yoksa yanlış veriyle kalıcı hatalı bir index kurulur. Rebuild **track
   seviyesinde** yapılmalı (slice değil) — hem boyut (5.37GB → 87MB), hem "5'ten az
   öneri dönme" sorununu, hem sorgu/index temsil uyumunu çözüyor.

4. **Train/val veri sızıntısı (eğitim gerektirir, şimdilik ERTELENDİ)**
   `model_trainer.py`'de `train_test_split` slice seviyesinde rastgele. Aynı şarkının
   ~61 slice'ı hem train hem val'e dağılıyor → model şarkıyı ezberliyor, val accuracy
   yanıltıcı yüksek. Düzeltme: `GroupShuffleSplit`, grup anahtarı `filename.split('_slice')[0]`.
   **Bir sonraki eğitim turuna kadar bu val accuracy sayılarına güvenme.**

5. **Etiket taksonomisi dejenere (eğitim gerektirir, şimdilik ERTELENDİ)**
   194 sanatçının hepsi tek bir genre + tek bir mood'a sahip — genre/mood head'leri
   pratikte artist head'inin birebir lookup'ı, bağımsız bir şey öğrenmiyor. Mood etiketi
   sesten değil `config.py` içinde elle atanmış. Kısa vadeli çözüm (eğitim gerektirmez):
   **joint decoding** — `skor(a) = log P(artist=a) + log P(genre=G[a]) + log P(mood=M[a])`,
   üç head'i tutarlı tek karara birleştirir, imkânsız kombinasyonları engeller.

6. **Sınıf dengesizliği** — class_weight yok, genre'de 46x, artist'te 28x oran var.
   Nadir sınıflar pratikte hiç tahmin edilmiyor. (Eğitim gerektirir, ertelendi.)

7. **Eşzamanlılık riskleri** — sabit `temp_inference/temp_spec.png` ve
   `downloaded_song.mp3` yolları (paralel istekler birbirini eziyor); `plt`/pyplot
   global state thread-safe değil ama `run_in_executor` ile thread'de çağrılıyor;
   `subprocess.run(shell=True)` (Demucs) returncode/timeout kontrolsüz.

## Kurallar
- **Şu an eğitim yapılamıyor** (süre kısıtı) — eğitim gerektiren değişiklikleri
  (`model_trainer.py`, dataset yeniden üretimi, taksonomi değişikliği) bir sonraki
  eğitim turuna kadar sadece **planla, uygulama**.
- Veri dosyaları ve model ağırlıkları proje klasörünün içinde — büyük/binary
  dosyaları (spektrogram PNG'leri, `.h5`/`.keras` ağırlıklar, FAISS index) elle
  düzenlemeye çalışma, sadece üreten kodu düzelt.
- Bir dosyayı "v2.0" / güncellenmiş olarak işaretlemek yetmez — **üretilen veri
  (spektrogramlar, index) fiilen yeniden üretilmediyse kod ile veri arasında sessiz
  bir uyuşmazlık olabilir.** Değişiklik yapınca "veri de yeniden üretildi mi?" diye sor.
- `beathunter_users_old.db` çöp, silinebilir.

## Şu anki durum
Faz 0 tamam (Tuzak 1 ve 2 düzeltildi, 2026-07-27). Sınıflandırma yolu uçtan uca
çalışıyor. `analyze_beat` hâlâ düşüyor ama artık **tek** noktada: FAISS
`AssertionError` (index d=256 ≠ sorgu 1536) — Tuzak 3, yani sıradaki iş Faz 1.1
(track-level FAISS rebuild). Sonra Faz 2.1 (joint decoding) → Faz 3
(performans/eşzamanlılık/temizlik). Detaylar ve gerekçeler için `TODO.md`.
