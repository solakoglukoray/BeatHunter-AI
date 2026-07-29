# BeatHunter AI — Proje Bağlamı

## Ne yapıyor
Ses girdisi alır → Demucs ile vokal ayırır → **klasik akustik özellikler** çıkarır
(MFCC + delta, spectral contrast, centroid/rolloff/ZCR/RMS — 102 boyut) →
LogisticRegression ile artist/genre sınıflandırır (joint decoding ile tutarlı çift) →
aynı 102-boyutlu vektörle 14,235 parçalık katalogda kosinüs benzerliği ile
benzer beat önerir. BPM ve key librosa'dan (modelden bağımsız).

**⚠️ CNN yolu emekliye ayrıldı (2026-07-29).** EfficientNetB3, görülmemiş parçalarda
şansın *altında* kalıyordu (artist 0.8x, genre 0.7x); klasik özellikler aynı held-out
parçalarda **16.7x / 9.7x** veriyor. Detaylar aşağıda.

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

## 🚨 EN ÖNEMLİ: CNN yolu ölçümle elendi (2026-07-29)

Aynı held-out parçalarda (`GroupShuffleSplit`, track bazlı, `random_state=42`):

| | artist | genre |
|---|---|---|
| EfficientNetB3 (saatlerce GPU, 700K dilim) | **0.8x şans** | **0.7x şans** |
| 102 klasik özellik + LogReg (15 sn CPU) | **16.7x** | **9.7x** |

CNN **şansın altındaydı** ve soft voting hiçbir şey eklemiyordu (dilim ve parça
seviyesi aynı) — yani seyreltilen bir sinyal yoktu, sinyal hiç yoktu.

Elenen açıklamalar (tekrar araştırma):
- ~~Veri/etiket tavanı~~ → veri artist'te en az 6.0x sinyal taşıyor (91 saniyede,
  eğitim olmadan, MFCC k-NN ile ölçüldü)
- ~~Eğitim kurulumu bozuk~~ → `overfit_test.py` 8 varyantın hepsinde %95+ train
  accuracy verdi; mimari küçük ölçekte sorunsuz öğreniyor
- ~~BatchNorm / set_weights / mixed_float16~~ → üçü de ayrı ayrı ölçülüp çürütüldü

Kalan hipotez (**doğrulanmadı**): 3 saniyelik dilim → viridis PNG → 224×224 →
ImageNet backbone zinciri fazla bilgi kaybediyor, ve ağır augmentation (spec_augment
%60 frekans + %60 zaman maskeleme) MFCC'nin koruduğu tınısal ipuçlarını siliyor.
Bu, "spektrogram CNN'leri çalışmaz" demek **değil** — bu pipeline'ın naif bir
baseline'ın altında kaldığı demek.

→ `model_trainer.py` duruyor ve çalışır durumda (Faz B için). `inference_engine.py`
  artık onu kullanmıyor.

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

3. **Retrieval artık FAISS değil (2026-07-29)**
   `models/classic_catalog.npz` (14,235 × 102, 5.6 MB) + düz numpy kosinüs, sorgu
   **0.04 ms**. Eski `vector_db_track.index` (87 MB) ve CNN embedding'leri
   kullanılmıyor.
   → ⚠️ **Katalog ve sorgu aynı fonksiyonu kullanmalı:** `audio_features.extract_features`.
   Bu kural Tuzak 1'in dersi — orada eğitim ve inference ayrı kod bloklarıydı, birbirinden
   uzaklaştılar ve aylarca fark edilmedi. `FEATURE_DIM` değişirse hem katalog hem
   sınıflandırıcı yeniden üretilmeli (`train_classic.py --extract`); kod bunu yükleme
   sırasında kontrol edip hata veriyor.

4. **Train/val veri sızıntısı (eğitim gerektirir, şimdilik ERTELENDİ)**
   `model_trainer.py`'de `train_test_split` slice seviyesinde rastgele. Aynı şarkının
   ~61 slice'ı hem train hem val'e dağılıyor → model şarkıyı ezberliyor, val accuracy
   yanıltıcı yüksek. Düzeltme: `GroupShuffleSplit`, grup anahtarı `filename.split('_slice')[0]`.
   **Bir sonraki eğitim turuna kadar bu val accuracy sayılarına güvenme.**

5. **Mood hedef olmaktan çıkarıldı (✅ 2026-07-29)**
   Üç bağımsız ölçüm: Cohen d +0.023 (CNN embedding), +0.094 (MFCC), k-NN 1.4x şans.
   Mood sesle ilişkili değil — `config.py`'de sanatçı başına elle atanmış.
   `analyze_beat` artık mood döndürmüyor, API ve UI'da da yok.
   → Genre hâlâ tahmin ediliyor ama **joint decoding** ile: `skor(a) = log P(artist=a)
   + log P(genre=G[a])`. Ölçüldü — doğrudan tahminden iyi (%17.32 vs %16.37) ve
   taksonomide var olmayan çift üretemiyor.

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
**Klasik özellik yolu üretimde** (2026-07-29). `analyze_beat` uçtan uca çalışıyor:
artist/genre sınıflandırma, 5 öneri, BPM/key. GPU'ya hiç ihtiyaç yok — bu aynı zamanda
"Windows'ta TF GPU desteği yok" sorununu tamamen ortadan kaldırdı.

Ölçülen (held-out, track bazlı split): artist top-1 %9.45 / top-5 %24.13 (16.7x şans),
genre top-1 %17.32 (9.7x). Öncesi: CNN, şansın altında.

⚠️ **Mutlak seviye hâlâ düşük.** Top-1 %9.45 çözülmüş bir problem değil. UI bunu
saklamıyor: tek isim ancak **100x şans** eşiğini geçince gösteriliyor (parçaların
~%9'u, o dilimde doğruluk %26), aksi halde top-5 aday listesi sunuluyor. Eşik
ölçülmüş doğruluk tablosundan seçildi, tahminle değil.

**Sıradaki karar Faz B:** temsili yeniden tasarlayıp (uzun segment, PNG round-trip yok,
hafif augmentation) CNN'e dönmek mantıklı mı? Klasik yol artık **16.7x'lik bir taban**
kuruyor — B'nin başarı ölçütü bu. Detaylar `TODO.md`.
