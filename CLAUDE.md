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

## 🚨 EN ÖNEMLİ: Kaydedilmiş model sınıflandırma yapmıyor (2026-07-28)

`models/beathunter_model.keras` **kullanılabilir durumda değil.** Sınıflandırma
head'leri hiç öğrenmemiş. Ölçümler (eğitim görüntülerinin **ta kendisi**, n=400):

| | Model | Rastgele |
|---|---|---|
| artist | **%0.5** | %0.5 |
| genre | **%2.2** | %1.5 |
| mood | %25.0 | %16.7 (çoğunluk sınıfı %22) |

Ortalama max softmax: genre 0.032, artist **0.014** (uniform = 0.005). Model neredeyse
düzgün dağılım üretiyor. 400 görüntüde sadece **4 farklı sanatçı** tahmin edildi
(gerçekte 164). Bunlar eğitim verisi — sızıntı (Tuzak 4) yüzünden ezberlenmiş olmalıydı.

Ağırlık imzası:
- **Backbone sağlam:** 208 katmanın 191'i taze ImageNet ile **birebir aynı**, 17'si
  Faz 2'de ince ayarlanmış. → Retrieval'in kısmen çalışmasının sebebi bu: FAISS
  **ImageNet doku özellikleri** üzerinde çalışıyor, öğrenilmiş müzik özellikleri değil.
- **Gövde dense katmanları:** std Glorot'un **0.03-0.06 katı** → sıfıra büzülmüş
- **Çıkış head'leri:** std Glorot'un 0.7-1.5 katı → **başlangıç değerinde**, öğrenmemiş

→ ⚠️ **Decode-time hiçbir düzeltme bunu kurtaramaz** (joint decoding dahil).
   Sınıflandırma çıktısına güvenme, kullanıcıya sunma.
→ ⚠️ Ağırlıklar dosyadan **doğru yükleniyor** (birebir doğrulandı), yükleme hatası yok.
   `best_checkpoint.keras` farklı bir dosya ama gövde ağırlıkları aynı
   (EarlyStopping `restore_best_weights=True` ile tutarlı) — yedek değil.

**Elenen nedenler** (tekrar araştırıp zaman harcama):
- ~~BatchNorm `training=True` grafiğe gömülü~~ → config'de `training: false`,
  çıktılar batch'ten bağımsız
- ~~Faz 2'deki `set_weights` sırayı karıştırıyor~~ → iki build'in ağırlık sırası ve
  şekilleri birebir aynı, sadece isimler farklı
- ~~`mixed_float16` gradyan ölçekleme yok~~ → Keras 3 optimizer'ı otomatik
  `LossScaleOptimizer`'a sarıyor (doğrulandı)

Gerçek neden **bilinmiyor** ve tespiti bir eğitim koşusu gerektiriyor. Bir sonraki
eğitim turunda **önce küçük bir alt kümede train accuracy'nin yükseldiğini doğrula** —
bu model tam eğitim harcanmadan önce yakalanabilirdi.

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

3. **FAISS index (✅ DÜZELTİLDİ 2026-07-28)**
   Eski index d=256 iken modelin GAP çıktısı 1536'ydı ve `analyze_beat` her istekte
   `AssertionError` veriyordu. Artık **track seviyesinde** yeni bir index kullanılıyor:
   `models/vector_db_track.index` + `vector_meta_track.csv` (14,235 × 1536, 87 MB).
   Eski `vector_db.index` / `vector_meta.csv` fallback olarak duruyor, **silme**.
   → ⚠️ **Retrieval embedding'i index-hizalı 4 segment kullanır**
   (`compute_retrieval_embedding`: tüm 3s grid üzerinden `linspace`, 4 segment).
   Sınıflandırma ise ayrı — 8 segment, intro/outro atlanarak. **İkisini birleştirme.**
   Sorgu ve index örneklemesi birebir aynı olmalı; segment *sayısı* da dahil
   (hizalı-4 → medyan self-match rank 1.5, hizalı-8 → 186, hizasız → 4,856).
   Index farklı slice sayısıyla yeniden kurulursa `RETRIEVAL_SEGMENTS` de değişmeli.
   → `rebuild_vector_database_if_needed()` **devre dışı** (slice seviyesinde index
   üretiyordu). Rebuild WSL2'deki track-level builder ile yapılır, bkz. TODO.md 1.1.
   → ⚠️ **Retrieval çalışıyor ama güvenilir değil** (n=40 ölçümü, TODO.md 1.3):
   medyan rank 1.5, ama **top-1 sadece %50**, top-10 %70, %20'si ilk 100'e bile
   girmiyor. Top-1 iken ikinciyle fark ortalama 4.4 puan — sıralama kırılgan.
   → ⚠️ **D=%75 tavanı:** aynı şarkının farklı bölümleri arası embedding benzerliği
   ~%75, rastgele parçalar %55±10. Öneri kalitesinin üst sınırı budur ve Tuzak 4/5'ten
   gelir — **Faz 4'te yeniden eğitim yapılana kadar geçerli.** Kod tarafında
   yapılacak bir şey kalmadı; öneri sayılarını kullanıcıya kesinlik gibi sunma.

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
Faz 0, 1 ve 2 tamam. Pipeline **mekanik olarak uçtan uca çalışıyor** — `analyze_beat`
düşmüyor, 5 öneri dönüyor, çıktı üçlüsü artık taksonomiye uygun.

Ama **sınıflandırma işlevsel değil**: model eğitilmemiş (yukarıdaki 🚨 bölümü).
Retrieval "çalışır ama zayıf" — top-1 %50 — ve bunu sağlayan şey ImageNet
özellikleri, öğrenilmiş müzik temsili değil.

Bu, kalan yol haritasını değiştiriyor: **kod tarafında yapılacak anlamlı bir kalite
iyileştirmesi kalmadı.** Faz 3 (performans/eşzamanlılık/temizlik) hâlâ değerli ve
eğitim gerektirmiyor, ama modeli düzeltmiyor. Gerçek blokaj **Faz 4 — yeniden eğitim**;
ve o tur başlamadan önce eğitimin neden başarısız olduğu bulunmalı, yoksa aynı sonuç
tekrarlanır. Detaylar ve gerekçeler için `TODO.md`.
