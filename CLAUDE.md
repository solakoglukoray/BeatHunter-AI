# BeatHunter AI — Proje Bağlamı

## Ne yapıyor
Ses girdisi alır → Demucs ile vokal ayırır → **PANNs embedding'i** çıkarır
(AudioSet'te ön-eğitilmiş, kendi verimizle fine-tune edilmiş CNN, 2048 → PCA-128)
→ LogisticRegression ile
artist/genre sınıflandırır (joint decoding ile tutarlı çift) → aynı 128-boyutlu
vektörle 16,169 parçalık katalogda kosinüs benzerliği ile benzer beat önerir.
BPM ve key librosa'dan (modelden bağımsız).

**Ölçülen — GÜNCEL (2026-08-04, 3,234 held-out parça, 213 sanatçı / 60 genre,
başlıktan yeniden etiketleme sonrası):**

| | artist top-1 | oran | top-5 | genre (joint) |
|---|---|---|---|---|
| **fine-tuned PANNs, temizlenmiş korpus** | **%17.90** | **34.5x** | **%39.33** | **%30.77** |

**Temsil karşılaştırması (eski 2,847 parçalık split, 194 sanatçı — kendi içinde geçerli):**

| temsil | artist top-1 | oran | top-5 | genre (joint) |
|---|---|---|---|---|
| EfficientNetB3 (ImageNet) | ~%0.7 | 0.8x | — | ~%1.3 |
| klasik MFCC 102-dim | %9.45 | 16.7x | %24.13 | %17.32 |
| donuk PANNs (AudioSet) | %12.96 | 22.9x | %32.03 | %24.17 |
| fine-tuned PANNs | %16.16 | 28.5x | %37.72 | %28.21 |

🚨 **İki tablo BİRBİRİYLE KARŞILAŞTIRILAMAZ.** Taksonomi değişiminde hem korpus
(14,235→16,169), hem sınıf sayısı (194→213 sanatçı, 65→60 genre), hem de split
değişti. Şans seviyesi %0.57'den %0.50'ye indiği için oran, ham doğruluktan daha
çok arttı. Üstteki satır bugünkü sistemi, alttaki tablo temsil seçimini anlatır.

⚠️ Top-1 **joint decoding** ile ölçülür (2026-07-31'de düzeltildi). `Pa.argmax`
%16.88 verirdi ama `inference_engine.classify()` sanatçıyı joint skorla seçiyor —
o sayı üretimde çalışmayan bir sistemi tarif ederdi. `train_classic.py` ikisini de
basıyor ki bir daha sessizce ayrışmasınlar.

İlk üç satırın dersi **kapasite değil domain**: EfficientNetB3 en büyüğüydü ve şansın
altında kaldı, çünkü fotoğraftan transfer ediyor. PANNs 2M ses klibinden transfer ediyor.

Dördüncü satırın dersi ayrı: **fine-tuning'in kazancı temsilde, head'de değil.** Encoder
donuk özellik çıkarıcı olarak kullanıldığında, tam uçtan uca eğitimin top-1'deki
kazancının **%94'ünü**, genre'dekinin **%106'sını** veriyor (Deney C). Bu yüzden
PCA+LogReg+joint decoding mimarisi hiç değişmedi — sadece embedding kaynağı değişti.
Tek istisna top-5: uçtan uca model orada 2.7p (~3 SE) daha iyi, çünkü kendi head'i
2048 boyutun tamamını görüyor, PCA ise %10 varyansı atıyor.

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
| `audio_features.py` | **Temsil — TEK KAYNAK.** PANNs embedding + klasik fallback + mood eksenleri + `fold_bpm` |
| `calibrate_mood.py` | Mood eşiklerini korpustan türetir ve **doğrular** (bağımsızlık, tür, kulak) |
| `corpus.py` | **Korpus + split + taksonomi — TEK KAYNAK.** Bayat artefaktları yeniden üretir |
| `label_recommender.py` | "YouTube'a şunu yaz" sorgusunu kurar — ölçülen sorgu faydasına göre |
| `calibrate_label_choice.py` | Etiket türlerinin güvenilirlik eğrilerini ölçer |
| `demo_label_choice.py` | Seçiciyi held-out örnekler + toplu sayılarla gösterir |
| `artist_match.py` | **Sanatçı adı eşleştirme — TEK KAYNAK.** Toplayıcı ve denetleyici aynı kodu kullanır |
| `measure_label_noise.py` | Başlık vs etiket uyuşmazlığını ölçer (bkz. Tuzak 9) |
| `validate_query_suggestions.py` | **Önerilen sorguyu YouTube'da fiilen test eder** (indirir, benzerlik ölçer) |
| `train_classic.py` | Sınıflandırıcı + katalog + eşik kalibrasyonu üretir |
| `inference_engine.py` | Çıkarım: sınıflandırma, benzerlik, BPM/key |
| `model_trainer.py` | EfficientNetB3 eğitimi — **emekli**, yeniden denenirse duruyor |
| `overfit_test.py` | Eğitim kurulumu teşhisi (tam koşudan önce çalıştır) |
| `thermal_guard.py` | GPU sıcaklık koruması (uzun koşular) |
| `panns_features_experiment.py` | PANNs vs klasik karşılaştırma deneyi |
| `pca_sweep_experiment.py` | PCA boyut taraması (negatif sonuç, kayıtlı) |
| `panns_finetune.py` | PANNs fine-tuning (kesinti-toleranslı, termal korumalı) |
| `panns_ft_frozen_experiment.py` | Deney C: kazanç temsilde mi head'de mi |
| `get_encoder.py` | **Yeni makinede İLK bunu çalıştır** — ağırlık kurulumu/doğrulaması |
| `models/` | Ağırlıklar ve artefaktlar — **elle değiştirme** |

## 🚨 EN ÖNEMLİ: üç temsil denendi ve aşıldı, dördüncüsü üretimde

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

**3. Donuk PANNs (stock AudioSet) PCA-128 — aşıldı, fallback olarak duruyor.** 22.9x.
Klasik tabana karşı +3.51p artist top-1 (4.2 SE), +7.90p top-5 (6.7 SE), +6.85p genre
(6.4 SE). Fine-tuned ağırlık yoksa `audio_features` otomatik buna düşer.

**4. Fine-tuned PANNs PCA-128 — ÜRETİMDE.** 27.1x. Donuk PANNs'e karşı +2.39p artist
top-1, +5.69p top-5, +4.04p genre. Fine-tuning 12 epoch / ~2.8 sa GPU sürdü; en iyi
epoch 11. Uçtan uca modelin kendi head'i top-5'te 2.7p daha iyiydi ama inference'ı
baştan yazmayı gerektiriyordu ve kazancın %94'ü zaten donuk encoder'la geliyor.

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

5. **Elle atanmış mood silindi, yerine ölçülen mood geldi (✅ 2026-07-30)**
   Eski etiket üç bağımsız ölçümde sesle ilişkisiz çıktı: Cohen d +0.023 (CNN
   embedding), +0.094 (MFCC), k-NN 1.4x şans. **Hedef olmaktan çıkarıldı ve
   `config.py`'den de kaldırıldı** — oradaki string `search_hint` olarak duruyor
   çünkü aslında hiç etiket değildi, YouTube arama sorgusunun bir kelimesiydi
   (silinirse `dataset/` yeniden üretilemez). Etiket değil, tahmin edilmiyor.
   → Genre hâlâ tahmin ediliyor ama **joint decoding** ile: `skor(a) = log P(artist=a)
   + log P(genre=G[a])`. Ölçüldü — doğrudan tahminden iyi (%24.17 vs %22.87) ve
   taksonomide var olmayan çift üretemiyor.
   → **Yeni mood tahmin değil, ölçüm.** `audio_features.classify_mood()` üç eksende
   korpus yüzdeliği veriyor (`calibrate_mood.py` → `models/mood_calibration.json`):

   | eksen | sinyaller | ayrım (2026-08-04) |
   |---|---|---|
   | energy | onset_density + rms_norm | **+0.188** |
   | brightness | centroid − minor_lean | +0.042 — **tür bu ekseni doğrulamıyor** |
   | aggression | low_ratio + hf_ratio | **+0.178** |

   Eksen korelasyonu en fazla 0.42 (bağımsız). Kalibrasyon 2,500 parça. Tür **girdi değil**; agresyon tepesi
   Georgia Trap/Rage/Trap/Crunk/Chicago Drill, dibi Lofi %31/Chillhop/Trip Hop.
   Drill enerjide düşük (.34) ama agresyonda yüksek (.59) — çelişki değil, tek
   eksenin söyleyemeyeceği şey.
   → ⚠️ **Hiçbir eşik elle seçilmedi.** Etiket kesimleri eksen skoru dağılımının
   yüzdelikleri — sabit 1/3-2/3 kullanmak parçaların %70'ini "Moderate" yapıyordu
   (yüzdelik ortalaması merkezi limit yüzünden 0.5 civarında toplanır).
   → ⚠️ Elenmiş sinyaller **tekrar eklenmesin**: `perc_ratio` (hiçbir türü ayırmıyor,
   0.47-0.54, üstelik en pahalısı), `tempo` (enerjide +0.01 — bu korpusta tüm türler
   120-150 BPM), `onset_contrast` (ters yönde), `dynamic_range` (-0.00),
   `rolloff_mean` (centroid ile r=0.97). Gerekçeler `audio_features.py`'de.

6. **Sınıf dengesizliği — ⚠️ `class_weight` DENENDİ VE REDDEDİLDİ (2026-08-04)**
   Bu maddenin öncülü bu boru hattı için **yanlış**. Aynı held-out split'te, tek değişken:

   | class_weight | top-1 | top-5 | genre | tahmin edilen sınıf |
   |---|---|---|---|---|
   | **None** *(üretimde)* | **%17.90** | **%39.33** | **%30.77** | 213/213 |
   | balanced | %17.84 | %38.78 | %30.24 | 213/213 |

   Üç metrikte de kötü, **ve "nadir sınıflar hiç tahmin edilmiyor" iddiası geçersiz** —
   iki ayarda da 213 sınıfın hepsi tahmin ediliyor, joint decoding tahminleri zaten
   yayıyor. Bu tuzak emekli EfficientNetB3'ün üç-head'li mimarisinde ölçülmüştü.
   Yeniden ölçmeden geri ekleme. (Eski metin aşağıda, tarihsel kayıt olarak.)

   ~~Eski:~~  — class_weight yok, genre'de 46x, artist'te 28x oran var.
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

9. **🚨 ETİKET GÜRÜLTÜSÜ — bilinen ve ÖLÇÜLMÜŞ veri tavanı (2026-08-04)**
   Korpus `"<artist> <hint> <genre> Type Beat"` aramasının ilk sonuçları alınarak
   toplandı ve **dönen şeyin sorulan şey olduğu hiç doğrulanmadı**. Etiket, *istediğimizi*
   söylüyor; *aldığımızı* değil. `measure_label_noise.py`, 700 parçalık örneklemde
   video başlıklarını çekip ölçtü:

   | | n | oran |
   |---|---|---|
   | başlıkta atanan sanatçı geçiyor | 423 | %60.4 |
   | **başlıkta BAŞKA bir hedef sanatçı geçiyor** | **71** | **%10.1** |
   | başlık hiçbir hedef sanatçıyı anmıyor | 185 | %26.4 |
   | video özel/silinmiş | 21 | %3.0 |

   **Başlıkla karar verilebilen 494 parçada uyuşmazlık %14.4 (±1.6).** Gerçek oran
   %10.1 (alt sınır) ile %14.4 arasında — "undetermined" temiz değil, *bilinmiyor*.

   Kırılım: **tür-içi %5.9** (aynı genre'den komşu sanatçı; genre etiketi doğru kalır,
   yalnızca artist kayar — Wiley↔Skepta, OZ↔Travis Scott) / **tür-dışı %8.5** (ikisi de
   bozulur — Common↔J. Cole, J Dilla↔MF DOOM).
   → ⚠️ Tür-dışı rakamı **şişkin okunmalı**: taksonomi drill'i beşe, hip-hop'ı düzinelerce
   alt türe bölüyor, o yüzden sesçe bitişik çiftler bile tür-dışı sayılıyor.

   En çok etkilenenler **yapımcılar ve efsane isimler**: J Dilla 4/4, Wiley 4/4,
   Pharrell 3/4, The Roots 3/4, 808Melo 3/5, Metro Boomin 2/4. Bu isimler bir sanatçıdan
   çok bir *stili* tarif ediyor ve YouTube da öyle yorumluyor — "J Dilla type beat"
   araması MF DOOM ve Joey Bada$$ getiriyor.

   → **Bu, top-1 %17.81'in tavanının bir kısmını açıklıyor.** Sınıflandırıcı bozuk değil;
   etiketlerin en az %10'u yanlış ve model onları öğrenmek zorunda.
   → Daha önce ölçtüğümüz "**yanlış sanatçı bile +0.126 sorgu faydası veriyor**" bulgusu
   da buradan geliyor: model komşu sanatçıyla karıştırıyordu çünkü **etiketler zaten öyle
   karışmış**.
   → ✅ **İleriye dönük düzeltildi**: `data_collector.py` artık başlığı kaydediyor
   (`metadata.csv`'de `title` kolonu) ve başlığında *başka* bir hedef sanatçı geçen
   sonuçları indirmeden önce reddediyor (`title_mismatch`). Hiç isim anmayan başlıklar
   KABUL ediliyor — sonuçların %26'sı öyle ve onları atmak her aramanın dörtte birini
   kanıtsız çöpe atardı. Eşleştirme `artist_match.py`'de, toplayıcı ve denetleyici
   aynı kodu kullanıyor.
   → ✅ **Mevcut korpus da temizlendi (2026-08-04).** `relabel_from_titles.py` bütün
   16,169 başlığı çekti (oEmbed, ~7 dk) ve **1,171 parçayı** başlıkta adı geçen sanatçıya
   yeniden etiketledi — silmedi. Genre otomatik düzeldi, çünkü `corpus.py` genre'yi
   sanatçıdan türetiyor.
   → ⚠️ Yeniden etiketleme kapsam DIŞI bırakılanlar, ikisi de kasıtlı:
   **grup üyeleri** (44 parça — "ANDRE 3000 x ERYKAH BADU" bir Outkast etiketiyle
   çelişmez, André 3000 zaten Outkast'tır) ve **type beat olmayan içerik**
   (119 parça — resmi klipler, karaoke). `undetermined` (%31) de dokunulmadı:
   doğrulanamamış olmak kirli olmak değil.
   → ⚠️ Etki tek yönlü değil: dengesizlik **5.0x → 6.0x** yükseldi, çünkü jenerik ve
   yapımcı sorguları popüler isimlere akıyor (808Melo 75→27, çünkü "808Melo type beat"
   Pop Smoke getiriyor; Drake 75→138, J. Cole 75→141).
   → 🚨 **Sayılar neredeyse hiç değişmedi**: top-1 %17.81→%17.90, top-5 %38.81→%39.33,
   genre %28.21→%30.77. Yani etiket gürültüsü tavanın **tek** açıklaması değildi;
   1,171 etiketi düzeltmek 0.1 puan getirdi. Kalan tavan başka yerde.

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

## 🔑 Encoder ağırlıkları (yeni makinede ilk iş)

```
python get_encoder.py        # ne var ne yok, stock ağırlıkları indirir
```

İki dosya var ve **cinsleri farklı**:

| dosya | ne | nasıl elde edilir |
|---|---|---|
| `models/panns_ft_encoder.pt` (327 MB) | **bizim** fine-tune ettiğimiz encoder | indirilemez — build artefaktı |
| `~/panns_data/Cnn14_mAP=0.431.pth` (327 MB) | stock AudioSet | Zenodo'dan indirilir |

Fine-tuned ağırlık için üç yol: **(1)** başka makineden kopyala (günlük kullanımda bu),
**(2)** yeniden üret (`panns_finetune.py` ~2.8 sa GPU + `--extract-encoder`, sonra
`train_classic.py --extract` ~45 dk), **(3)** onsuz çalış — stock ağırlıklar otomatik
devreye girer, maliyeti artist top-1 %16.16 → %12.96; ardından `train_classic.py`
(dakikalar, embedding'ler cache'te).

⚠️ Git ikisini de takip etmiyor: her yeniden eğitimde değişen 327 MB binary LFS işi ve
repo'da LFS yok. Gerekirse `panns_ft_encoder.pt` LFS'e alınır, cache'lenmiş
embedding'ler dışarıda kalır.

⚠️ **Katalog ve sorgu aynı encoder'dan gelmek ZORUNDA.** Model bundle'ı `encoder_id` ve
ağırlık hash'ini saklıyor; `inference_engine` yüklemede ikisini de doğrulayıp uyuşmazlıkta
**hata veriyor**. İkisi de negatif testle kanıtlandı (yanlış encoder → hata, değişmiş
dosya → hata). Bu kontrolü kaldırma; Tuzak 1'in tam deseni.

## Şu anki durum
**Fine-tuned PANNs + PCA-128 üretimde.** Korpus **16,169 parça / 213 sanatçı / 60 genre**
(2026-08-04 taksonomi revizyonu). `analyze_beat` uçtan uca çalışıyor.

Ölçülen (3,234 held-out parça): artist top-1 **%17.81** / top-5 **%38.81** (35.4x şans),
genre **%30.33** (16.5x).

⚠️ **Mutlak seviye hâlâ düşük.** Top-1 %17.81, yani ~6 denemede 1. UI bunu saklamıyor:
tek isim ancak **150x** eşiğini geçince gösteriliyor (parçaların %13.8'i, o dilimde
doğruluk %44.3), aksi halde top-5 aday listesi (%38.81). Eşik her eğitimde
`train_classic.py` tarafından yeniden ölçülüyor — kopyalanmıyor.
→ Yeni bir basamak açıldı: **200x'te %64.9 hassasiyet**, ama parçaların yalnızca %3.5'i.

🚨 **`corpus.py` KORPUSUN VE SPLIT'İN TEK KAYNAĞI (2026-08-04).** Öncesinde
`models/vector_meta_track.csv` 8 script tarafından okunuyor, **hiçbiri tarafından
üretilmiyordu**; taksonomi değişince eski 14,235 parçalık korpusu tarif etmeye devam etti.
Split ise `dataset_processed/metadata_processed.csv`'den (eski spektrogram dilimleri)
geliyordu — yeni parçalar orada olmadığı için hepsi eğitime giderdi ve 213 sınıflı model
yalnızca eski 194 sanatçıdan oluşan bir test setinde ölçülürdü. `taxonomy.json` de aynı
durumdaydı. Üçü de artık `config.py` + `dataset/metadata.csv`'den türetiliyor.
→ ⚠️ **Genre, CSV'den okunmaz, sanatçıdan TÜRETİLİR.** Her satır indirildiği andaki
genre'yi taşıyor; "Mexican Drill → Corridos Tumbados" yeniden adlandırması 225 satırı
eski isimde bıraktı ve joint decoding veride olmayan bir genre'yi arayıp çöktü.
→ ⚠️ Embedding cache append-only: kırpılan parçaların vektörleri içinde kalıyor.
`train_classic` korpus dışı olanları eleyip devam ediyor (son koşuda 691 tane).

**Sorgu önericisi üretimde** (2026-07-31). Kullanıcıya "YouTube'a şunu yaz" diyor:
`label_recommender.recommend_label()`, `analyze_beat` içinde
`confidence["recommendation"]` alanına yazılıyor (7'li tuple bozulmadı).

→ 🚨 **Hedef "en güvenilir etiket" DEĞİL, "en iyi arama terimi".** İkisi farklı şeyler
ve bu ayrımı ölçüm öğretti. `validate_query_suggestions.py` 45 parça / 275 indirilen
YouTube sonucu ile, **aynı biçimde ama rastgele etiketli** kontrol sorgusuna karşı:

| sorgu türü | 2026-08-01 | **2026-08-04 (güncel)** | |
|---|---|---|---|
| artist | +0.147 (5.7 SE) | **+0.114** | 4.7 SE |
| genre | +0.109 (3.8 SE) | **+0.052** | 2.6 SE |
| mood | +0.018 (1.1 SE) | *aday değil* | |

Taban (rastgele iki katalog parçası) ~0.00. Kontaminasyon: sonuçların %23.5'i (önerilen)
/ %20.2'si (kontrol) korpusta — fark 1.2 SE, gürültü içinde; sorgulanan parçanın kendisi
hiç dönmedi.
→ ⚠️ **İki fayda da düştü, genre yarı yarıya.** Genre sayısı 65→60'a inince bir genre
etiketi daha geniş bir alanı kapsıyor ve sesi daha az sabitliyor. Artist hâlâ açık ara
önde, `QUERY_UPLIFT_PER_PRECISION` bu yüzden onu daha ağır tartıyor (k: 0.375 vs 0.176).

→ ⚠️ **Mood sorgu adayı DEĞİL, ama UI'da duruyor.** Etiket doğru — sorun onu arama
terimi yapmaktı. "chill type beat" gerçekten chill beat getiriyor ama chill beat'ler
tüm türlere yayıldığı için tek bir parçaya benzemiyor. Genre çıpası yardım ediyor
("chill Southern Hip-Hop type beat", +0.045 / 2.8 SE) ama yine artist/genre'nin
altında. **Mood'u sorguya geri ekleme.**
→ ⚠️ **Güven barajı yok, kasıtlı.** 150x'i geçmeyen etiketten de sorgu kuruluyor:
genre kolu %49 hassasiyetle çalışırken bile kontrolü 3.8 SE geçti. Bir ismi *iddia*
olarak göstermek ayrı karar, hâlâ `confidence["reliable"]`.
→ ⚠️ **Yanlış sanatçı bile işe yarıyor** (+0.126; yanlış genre +0.078). Sınıflandırıcı
sanatçıyı karıştırdığında sesçe komşu sanatçıyla karıştırıyor.
→ ⚠️ `QUERY_UPLIFT_PER_PRECISION` ağırlıklandırması **artık etkili** (2026-08-04).
Eski eğrilerde `query_utility` ile `reliability` 2,847 parçanın hiçbirinde ayrışmıyordu;
yeni eğrilerde üçte birinde ayrışıyor (artist 154 vs 86 seçim). Hangisinin doğru olduğu
buradan çıkmıyor: `reliability` yer gerçeğinde daha iyi (%27.3 vs %25.3), ama
`query_utility` **aramanın işe yaraması**nı optimize ediyor ve varsayılan o.
→ Kapsam %13.0 → **%100** (artist 154 / genre 352 seçim, 506 parçalık held-out alt
kümesinde). Yer gerçeğiyle ölçülebilen isabet %25.3 (eski politika %43.9 ama yalnızca
%13 kapsamda).
→ ⚠️ Mood **pencereye bağlı**: 0-30sn vs 30-60sn tutarlılığı energy %41 (kayma +0.231,
intro etkisi), brightness %57, aggression %67. Çoklu pencere ortalaması düzeltir —
yapılmadı.
→ ⚠️ Ölçülen fayda rakamları, eski ölçütün o türü seçtiği güven aralığında toplandı
(artist ~184x, genre ~42x). Çok daha düşük güvende ekstrapolasyon; sınıflandırıcı
değişirse `validate_query_suggestions.py` yeniden koşulmalı.

**Mood artık ölçülüyor ve UI'da** (2026-07-30, Tuzak 5). Sınıflandırıcıdan bağımsız —
BPM/key gibi sinyalden geliyor, o yüzden güven uyarısı taşımıyor. `analyze_beat` 7'li
tuple döndürüyor; `models/mood_calibration.json` yoksa mood `None` olur ve UI hiçbir şey
göstermez (uydurma nötr değer yok). Çıkarım ~1.0 sn/parça.

**Denenmiş ve kapanmış:** PCA boyut taraması (64-128 platosu), tam uçtan uca fine-tune
(top-5'te +2.7p daha iyi ama inference'ı baştan yazmayı gerektiriyor; temsil kazancının
%94'ü zaten donuk encoder'la geliyor).

**Sıradaki fikirler** (ölçülmedi): daha uzun/çoklu pencere, daha büyük PANNs varyantı
(Cnn14 yerine ResNet38/Wavegram), ikinci bir fine-tune turu daha düşük LR ile.
