# BeatHunter AI — Proje Bağlamı

## Ne yapıyor
Ses girdisi alır → Demucs ile vokal ayırır → **PANNs embedding'i** çıkarır
(AudioSet üzerinde önceden eğitilmiş CNN, 2048 → PCA-128) → LogisticRegression ile
artist/genre sınıflandırır (joint decoding ile tutarlı çift) → aynı 128-boyutlu
vektörle 14,235 parçalık katalogda kosinüs benzerliği ile benzer beat önerir.
BPM ve key librosa'dan (modelden bağımsız).

**Ölçülen (2,847 held-out parça, `GroupShuffleSplit` track bazlı, `random_state=42` —
projedeki her sayı bu split'ten):**

| temsil | artist top-1 | oran | top-5 | genre (joint) |
|---|---|---|---|---|
| EfficientNetB3 (ImageNet) | ~%0.7 | 0.8x | — | ~%1.3 |
| klasik MFCC 102-dim | %9.45 | 16.7x | %24.13 | %17.32 |
| **PANNs PCA-128** | **%12.96** | **22.9x** | **%32.03** | **%24.17** |

Bu üç satırın dersi **kapasite değil domain**: EfficientNetB3 üçünün en büyüğü ve
şansın altında kaldı, çünkü fotoğraftan transfer ediyor. PANNs 2M ses klibinden
transfer ediyor.

## Stack
- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- Serving ML: PyTorch + PANNs (AudioSet), scikit-learn, Librosa, Demucs
- Eğitim ML (emekli CNN yolu): TensorFlow — serving artık import etmiyor
- Auth: Custom JWT (`auth.py`)
- Deploy: Docker Compose (NVIDIA GPU support)
- Training: WSL2 + RTX 4080 (yalnızca CNN yolu için; PANNs Windows'ta CUDA ile çalışır)

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
| `audio_features.py` | **Temsil — TEK KAYNAK.** PANNs embedding + klasik fallback |
| `train_classic.py` | Sınıflandırıcı + katalog + eşik kalibrasyonu üretir |
| `inference_engine.py` | Çıkarım: sınıflandırma, benzerlik, BPM/key |
| `model_trainer.py` | EfficientNetB3 eğitimi — **emekli**, yeniden denenirse duruyor |
| `overfit_test.py` | Eğitim kurulumu teşhisi (tam koşudan önce çalıştır) |
| `thermal_guard.py` | GPU sıcaklık koruması (uzun koşular) |
| `panns_features_experiment.py` | PANNs vs klasik karşılaştırma deneyi |
| `pca_sweep_experiment.py` | PCA boyut taraması (negatif sonuç, kayıtlı) |
| `models/` | Ağırlıklar ve artefaktlar — **elle değiştirme** |

## 🚨 EN ÖNEMLİ: iki temsil elendi, üçüncüsü üretimde

**1. EfficientNetB3 (ImageNet) — şansın altında.** Soft voting hiçbir şey eklemiyordu
(dilim ve parça seviyesi aynı), yani seyreltilen bir sinyal yoktu.

Elenen açıklamalar (tekrar araştırma):
- ~~Veri/etiket tavanı~~ → veri artist'te en az 6.0x sinyal taşıyor (MFCC k-NN, eğitim yok)
- ~~Eğitim kurulumu bozuk~~ → `overfit_test.py` 8 varyantın hepsinde %95+ train accuracy
- ~~BatchNorm / set_weights / mixed_float16~~ → üçü de ölçülüp çürütüldü
- ~~L2 / LR ayarı~~ → 7 katlık farkı kapatmaz, denenmedi

**2. Klasik MFCC 102-dim — çalıştı ama aşıldı.** 16.7x. `audio_features.py` içinde
`extract_classical_features` olarak **duruyor** — PyTorch gerektirmiyor, hem fallback
hem gelecek temsillerin geçmesi gereken ucuz referans. Silme.

**3. PANNs PCA-128 — üretimde.** 22.9x. Klasik tabana karşı +3.51p artist top-1
(4.2 SE), +7.90p top-5 (6.7 SE), +6.85p genre (6.4 SE).

→ ⚠️ **PCA-128, ham 2048'i her metrikte geçiyor** (2048: %11.52 / 20.3x / %28.13).
2048 boyut ~11K eğitim parçasına karşı sınıflandırıcının gürültüye oturmasına izin
veriyor; projeksiyon düzenlileştiriyor. İki özellik setini birleştirmeyi de geçiyor.
PCA yalnızca eğitim split'inde fit edilir ve model bundle'ında saklanır.

→ ✅ **Boyut taraması yapıldı (2026-07-30) — 128 kalıyor, TEKRAR TARAMA.**
Seçim, held-out'a dokunmadan train'den ayrılan inner-val'de yapıldı (7 aday arasından
test setinde seçmek kazananı şişirir). Held-out sonuçları:

| boyut | artist top-1 | top-5 | genre joint |
|---|---|---|---|
| 64 | %12.89 | %33.37 | %22.97 |
| 96 *(inner-val'de seçilen)* | %13.17 | %32.81 | %23.78 |
| **128** *(üretimde)* | %12.96 | %32.03 | **%24.17** |
| 256 | %11.10 | %28.21 | %20.58 |
| 384 | %10.57 | %26.27 | %19.85 |

96 vs 128 = +0.21p, SE ~0.89p → **0.2 SE, ayrıştırılamaz**; genre'de 128 daha iyi.
`PCA_DIM` değiştirilmedi — model/katalog/eşiği ölçüm gürültüsü için yeniden üretmek
maliyeti hak etmiyor. **64-128 geniş bir plato**, bu eksende kazanç bitmiş; daha ince
tarama (80, 112) zaman kaybı. Boyutla düşüş ise gerçek: 384 vs 96 arası 2.6p (~2.9 SE).

→ `model_trainer.py` duruyor ve çalışır durumda (CNN yolunu yeniden denemek istenirse).
  `inference_engine.py` artık TensorFlow import etmiyor.

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
   `models/classic_catalog.npz` (14,235 × 128, 6.9 MB) + düz numpy kosinüs.
   FAISS ve CNN embedding'leri kullanılmıyor.
   → ⚠️ **Katalog ve sorgu aynı yoldan geçmeli:** `inference_engine._project()`
   (embed → PCA → scale). Sıra önemli ve `train_classic.py` ile aynı olmalı.
   Bu kural Tuzak 1'in dersi — orada eğitim ve inference ayrı kod bloklarıydı,
   birbirinden uzaklaştılar ve aylarca fark edilmedi. Boyut değişirse hem katalog hem
   sınıflandırıcı yeniden üretilmeli; kod bunu yüklemede kontrol edip hata veriyor.

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
   + log P(genre=G[a])`. Ölçüldü — doğrudan tahminden iyi (%24.17 vs %22.87) ve
   taksonomide var olmayan çift üretemiyor.

6. **Sınıf dengesizliği** — class_weight yok, genre'de 46x, artist'te 28x oran var.
   Nadir sınıflar pratikte hiç tahmin edilmiyor. (Eğitim gerektirir, ertelendi.)

7. **Eşzamanlılık (✅ DÜZELTİLDİ 2026-07-28)**
   Spektrogram görseli istek başına benzersiz yola yazılıyor (eskiden sabit yol →
   eşzamanlı iki analiz birbirini eziyor, kullanıcıya başkasının görseli
   gösterilebiliyordu). Demucs `shell=True` olmadan, returncode kontrollü, 15 dk
   timeout'lu. Render `pyplot` yerine `Figure` nesnesi kullanıyor.

8. **Güven eşiği modele özel, KOPYALANAMAZ**
   `train_classic.py` her eğitimde ölçüp bundle'a yazıyor, `inference_engine` oradan
   okuyor. Klasik modelde 100x %26.1 hassasiyet veriyordu; PANNs modelinde 100x %25.9
   ve %35 barajı için 150x gerekiyor. Eşiği kodda sabitleme.

## Kurallar
- **CNN eğitimi emekli** — `model_trainer.py` duruyor ama üretim yolunda değil.
  Yeniden denenecekse önce `overfit_test.py`, ve başarı ölçütü PANNs'in 22.9x'i.
- Veri dosyaları ve model ağırlıkları proje klasörünün içinde — büyük/binary
  dosyaları (spektrogram PNG'leri, `.keras` ağırlıklar, PANNs `.pth`, katalog `.npz`) elle
  düzenlemeye çalışma, sadece üreten kodu düzelt.
- Bir dosyayı "v2.0" / güncellenmiş olarak işaretlemek yetmez — **üretilen veri
  (spektrogramlar, index) fiilen yeniden üretilmediyse kod ile veri arasında sessiz
  bir uyuşmazlık olabilir.** Değişiklik yapınca "veri de yeniden üretildi mi?" diye sor.
- `beathunter_users_old.db` çöp, silinebilir.

## Şu anki durum
**PANNs PCA-128 yolu üretimde** (2026-07-29). `analyze_beat` uçtan uca çalışıyor:
artist/genre sınıflandırma, 5 öneri, BPM/key. Serving TensorFlow import etmiyor.

Ölçülen: artist top-1 %12.96 / top-5 %32.03 (22.9x şans), genre %24.17 (14.4x).
Yol boyunca: EfficientNetB3 0.8x (şansın altında) → klasik MFCC 16.7x → PANNs 22.9x.

⚠️ **Mutlak seviye hâlâ düşük.** Top-1 %12.96, yani 8 denemede 1. UI bunu saklamıyor:
tek isim ancak **150x** eşiğini geçince gösteriliyor (parçaların %6.8'i, o dilimde
doğruluk %38.7), aksi halde top-5 aday listesi (%32.03 isabet) sunuluyor. Eşik her
eğitimde yeniden ölçülüyor.

**Sıradaki fikirler** (hiçbiri ölçülmedi): PANNs'i fine-tune etmek (şu an encoder
donmuş), daha uzun veya çoklu pencere, daha büyük PANNs varyantı.

~~PCA boyut taraması~~ → **yapıldı, kazanç yok** (yukarıdaki tablo, 64-128 platosu).
Tekrar denemeye değmez.
