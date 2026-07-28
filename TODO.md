# BeatHunter AI — Roadmap

Kurallar: Her madde işaretlendiğinde tarih ve kısa not eklenir. Eğitim gerektiren
maddeler `[EĞİTİM GEREKİR]` ile işaretli — şu an ertelenmiş durumda.

---

## FAZ 0 — Sistemi çalışır hale getir (eğitim gerekmiyor, ~1 saat)

- [x] **0.1 — Inference spektrogram parametrelerini hizala** — *2026-07-27*
  `inference_engine.py`: `n_mels` 256 → 128, `fmin`/`fmax` kaldır. Render yolunu
  `matplotlib.specshow` → `io.BytesIO` (diske değil) olarak değiştir — hem disk I/O
  kazancı korunur hem gerçek eğitim pikselleriyle parite sağlanır. Resize:
  PIL LANCZOS → `tf.image.resize` bilinear (eğitimle aynı). `pyplot` yerine `Figure`
  nesnesi kullan (bu aynı zamanda Faz 4'teki thread-safety sorununu da çözer).
  *Neden önce:* kök neden burada; hiçbir şey buna bağımlı olmadan doğru çalışmaz.

  **Not:** Yeni `render_spectrogram_png()` ortak renderer'ı eklendi; hem model girdisi
  hem UI görseli onu kullanıyor. Diskteki gerçek eğitim görüntülerine karşı ölçüldü:
  **4 slice'ta ortalama korelasyon 0.9749** (bozuk halde 0.31 idi), render boyutu
  231×232 ile birebir eşleşiyor. ⚠️ Paritenin kırılgan olduğu iki nokta kodda yorumlandı:
  `melspectrogram`'a `fmin/fmax` **verilmemeli** ve `specshow`'a `cmap` **verilmemeli**
  (librosa otomatik seçiyor; `cmap='viridis'` pariteyi bozar).

- [x] **0.2 — Layer adı bug'ını düzelt** — *2026-07-27*
  `inference_engine.py:212` ve `:301`'deki hardcoded `'global_average_pooling2d'`
  yerine katmanı tipinden (`GlobalAveragePooling2D`) bul.
  *Neden önce:* bu olmadan `analyze_beat` hiç sonuç dönmüyor.

  **Not:** `build_embedding_model(model)` yardımcısı eklendi, her iki çağrı yeri de
  ona bağlandı. Kaydedilmiş modelde doğrulandı — üst seviye katman listesi:
  `['input_layer_2', 'efficientnetb3', 'global_average_pooling2d_1', ...]`, yani eski
  hardcoded ad gerçekten `ValueError` veriyordu. Yeni yol **1536-dim** vektör dönüyor.

**Faz 0 sonucu (2026-07-27):** Sınıflandırma yolu artık uçtan uca çalışıyor.
Bir eğitim parçasında ölçüm: BPM/Key 1.5s, 8-segment soft voting 8.2s, embedding (1,1536),
head çıktıları (65 / 194 / 6) — `classes.json` ile birebir uyumlu.
`analyze_beat` **hâlâ düşüyor**, ama artık tek bir noktada: FAISS `AssertionError`
(index d=256 ≠ sorgu 1536). Yani kalan tek engel Faz 1.1 — beklendiği gibi.

---

## FAZ 1 — Öneri sistemini doğrula

- [x] **1.1 — FAISS index'i track seviyesinde yeniden inşa et** — *2026-07-28*
  Slice değil, parça ortalaması (~14,235 track × 1536-dim, ~87MB). Forward pass
  gerektirir (eğitim değil).
  *Ön koşul:* 0.1 ve 0.2 tamamlanmadan yapma — yanlış spektrogramlarla index kurmak
  hatayı kalıcılaştırır.

  **Not:** Windows'ta TF'in GPU desteği olmadığı için (bkz. 3.6) rebuild **WSL2'de**
  yapıldı: 4 slice/parça → 56,940 görüntü, batch 64, 10.7 dk, ~89 gör/s.
  Parça başına 16 yerine **4 slice** seçildi — termal kısıt (makine 96-97°C'de
  kapanmıştı); index boyutu zaten parça sayısına bağlı olduğundan 87 MB değişmedi.
  Builder script'i termal korumalı (80°C durakla / 70°C devam / 90°C iptal) ve
  60s'de bir atomik checkpoint yazıyor — koşu sırasında 1 duraklama oldu (80→69°C,
  65s), tepe 80°C.
  Çıktı: `models/vector_db_track.index` + `models/vector_meta_track.csv`.
  Eski `vector_db.index` / `vector_meta.csv` **silinmedi**, fallback olarak duruyor.

  ⚠️ `rebuild_vector_database_if_needed()` devre dışı bırakıldı — slice seviyesinde
  index üretiyordu, yeni yollara yazsa `analyze_beat`'in metadata lookup'ını bozardı.

- [x] **1.2 — Index/metadata hizasızlığını doğrula** — *2026-07-28*
  ntotal=14,235 = meta satır sayısı, d=1536 = modelin GAP çıktısı, normalizasyon
  tam 1.0000, parça adları tekil. İkisi de aynı çalıştırmadan.

  **Kritik bulgu — sorgu/index temsil uyumu.** Index doğru kurulmuştu ama ilk
  doğrulamada *self-match başarısızdı*: index'te bulunan bir parça kendi sesinden
  sorgulanınca ilk 10'a girmiyordu (medyan rank 4,856/14,235; rastgele ~7,117).
  Kaybın ayrıştırması (aynı parça, 4 farklı yoldan embedding):

  | Ölçüm | Sonuç | Anlamı |
  |---|---|---|
  | index'i kuran disk PNG'leri | **100.00%** | inşa kodu doğru, bug yok |
  | canlı render, aynı zaman konumları | 78-90% | render paritesi kaybı (%10-22) |
  | 8-segment sorgu yolu | 46-67% | **segment seçimi kaybı (%23-34)** |
  | aynı parçanın başka slice'ları | ~75% | **modelin tavanı** (rastgele %55±10) |

  Düzeltme: sorgu embedding'i index ile **birebir aynı** örneklemeyi kullanır oldu
  (`compute_retrieval_embedding`, tüm 3s grid üzerinden linspace, 4 segment).
  Sonuç — medyan self-match rank **4,856 → 1.5**, top-10 **%0 → %70**.
  (Buradaki n=6 sayıları iyimserdi; güvenilir ölçüm için bkz. **1.3**.)

  ⚠️ Segment **sayısı** da eşleşmeli, sadece strateji değil: hizalı-8 medyan rank 186
  veriyor (aynı 4 konumu içeriyor ama 4 fazlası ortalamayı sulandırıyor). Index farklı
  bir slice sayısıyla yeniden kurulursa `RETRIEVAL_SEGMENTS` de değişmeli.

  Render paritesi kaybı (%10-22) **kabul edildi** — hizalama sonrası benzerlik %78-90'a
  çıkıyor ve top-1 için fazlasıyla yetiyor, dolayısıyla index'i canlı render'la kurmanın
  ~1 saatlik CPU maliyetine değmiyor. (Kalan fark muhtemelen kütüphane sürüm kayması:
  PNG'ler Mart'ta üretildi, bugünkü librosa 0.11 / matplotlib 3.10 biraz farklı render ediyor.)

- [x] **1.3 — Self-match doğrulamasını n=40 parçayla tekrarla** — *2026-07-28*

  **n=6 iyimserdi.** Gerçek sayılar (index içinden rastgele 40 parça):

  | | n=6 | **n=40** | hizasız (referans) |
  |---|---|---|---|
  | medyan rank | 1 | **1.5** | 4,856 |
  | top-1 | 67% | **50%** | 0% |
  | top-10 | 100% | **70%** | 0% |
  | top-100 | 100% | **80%** | 17% |
  | en kötü rank | — | **1,894** | 9,980 |

  self benzerlik: ort %83.2 (min %69.7, max %95.3) — rastgele %55±10'a karşı ~2.8σ.
  Top-1 olduğunda ikinciyle fark **ortalama 4.4 puan** (ilk testteki 20 puanlık örnek
  istisnaymış) → sıralama kırılgan, küçük girdi değişimleri sırayı değiştirebilir.

  **Yorum:** hizalama düzeltmesi gerçek ve büyük (medyan 4,856 → 1.5, ~3,200x), ama
  sistem güvenilir değil: **parçaların yarısında** aynı şarkı 1. sırada bile çıkmıyor,
  **%20'sinde ilk 100'e girmiyor**. Bu, D=%75 tavanıyla tutarlı — embedding zayıf
  ayırt edici. Kod tarafında yapılacak bir şey kalmadı; asıl çözüm Faz 4.1/4.3.

---

## FAZ 2 — Tutarlılık kazancı (eğitim gerektirmez)

- [x] **2.1 — Joint decoding** — *2026-07-28* — **uygulandı, ama doğrulanamıyor**

  `models/taxonomy.json` üretildi (194 sanatçı → genre+mood, determinizmi ve
  `classes.json` ile birebir uyumu doğrulandı). `joint_decode()` eklendi,
  `predict_with_majority_voting(..., joint=True)` varsayılan; `joint=False` eski
  bağımsız argmax'ı koruyor (A/B için).

  **A/B sonucu (n=60, eğitim setinden):**

  | | genre | artist | mood | 3'ü birden |
  |---|---|---|---|---|
  | bağımsız argmax | %1.7 | **%0.0** | %28.3 | %0.0 |
  | joint decoding | %1.7 | %1.7 | %18.3 | %1.7 |

  İmkânsız üçlü: bağımsız **60/60 (%100)** → joint **0/60** (yapısal olarak üretilemez).
  Çeşitlilik: 60 parça için bağımsız 3 farklı artist, joint 4 (gerçekte 51).

  ⚠️ **Bu sayılar joint decoding hakkında bir şey söylemiyor** — model sınıflandırma
  yapmıyor (bkz. CLAUDE.md'deki "Kaydedilmiş model sınıflandırma yapmıyor").
  Birleştirilecek sinyal yok. Joint decoding doğru uygulandı ve taksonomi tutarlılığını
  garanti ediyor; **faydası ancak model yeniden eğitildikten sonra ölçülebilir.**
  Mood'daki −10 puan da anlamlı değil (bağımsız argmax orada sadece çoğunluk sınıfını
  söylüyordu, %22 taban).

  *Faz 2.1'in özgün gerekçesi ("anında gözle görülür iyileşme") **geçersiz** — sorunun
  taksonomi değil, eğitilmemiş model olduğu ancak bu ölçümle anlaşıldı.*

- [ ] ~~**2.1 — Joint decoding**~~ *(özgün madde, referans için)*
  `skor(a) = log P(artist=a) + log P(genre=G[a]) + log P(mood=M[a])`. Üç head'in
  çıktısını tek tutarlı karara birleştirir, imkânsız kombinasyonları (ör. "Wu-Tang /
  Trap / Aggressive") engeller. Taksonomi sorununa (Faz 4) kalıcı çözüm değil ama
  anında gözle görülür iyileşme sağlar.

  **Faz 0'da gözlenen somut kanıt (2026-07-27):** `jkdp4q0cwEY.mp3` (gerçek etiket:
  Tyler, The Creator / Chill / Alternative Hip-Hop) için model
  **"Philanthrope / Dark / Alternative Hip-Hop"** dedi. Ama taksonomide Philanthrope =
  *Lofi Hip-Hop / Melancholic* — yani bu üçlü veri setinde **imkânsız**. Üç head birbiriyle
  çelişiyor. Dikkat çekici olan: genre head'i **doğru** olanı (Alternative Hip-Hop)
  söylemiş, yani joint decoding artist tahminini muhtemelen Tyler'a çekerdi.
  *Uyarı:* bu tek bir örnek — doğruluk ölçümü değil, tutarsızlığın kanıtı.

---

## FAZ 3 — Performans, eşzamanlılık, temizlik ✅ TAMAM (2026-07-28)

**⚠️ 3.1'deki "~8-10x" tahmini YANLIŞTI.** Ölçüm: 8 segment için render (matplotlib)
0.49s = toplamın **%6**'sı, tek batch predict 7.55s = **%94**. Darboğaz graph çağrı
yükü değil, **CPU'daki forward pass**. Batch'leme 9.99s → 8.04s, yani **~%20**.
Gerçek kaldıraç başka: Windows'ta TF'in GPU desteği yok (bkz. 3.6), serving'i WSL2'ye
taşımak ölçülen farka göre ~9x olurdu.

- [x] **3.1 — Inference'ı batch'le** — *2026-07-28* — segment başına 2 ayrı `model.predict()` yerine
  tek `(8,224,224,3)` batch çağrısı. `embed_model`'i cache'le (her istekte yeniden
  inşa edilmiyor). Beklenen kazanç ~8-10x.
- [x] **3.2 — BPM clamp'i while döngüsüne çevir** — *2026-07-28* + hip-hop hedef aralığına (120-170)
  göre yarı-tempo/çift-tempo katlama mantığı ekle (140→70 algılama klasik hata).
- [x] **3.3 — Key tespiti** — *2026-07-28* — `chroma_stft` yerine `librosa.effects.harmonic()` +
  `chroma_cqt`.
- [x] **3.4 — Öneri sayısı garantisi** — *Faz 1.1'de çözüldü (track-level index)* — `k=10` yerine `k=100` komşu çek, tekilleştirme
  sonrası 5 öneri garantiye alınsın.
- [x] **3.5 — Eşzamanlılık** — *2026-07-28* — sabit `temp_inference/temp_spec.png` ve
  `downloaded_song.mp3` yollarını istek/session bazlı yap. `subprocess.run(shell=True)`
  (Demucs) için returncode kontrolü + timeout ekle.
- [x] **3.6 — Küçük temizlik** — *2026-07-28* — `time.sleep(0.3)` (data_processor.py:124, ~12dk boşa
  gidiyor) kaldır; yanlış yorumları düzelt (`"22 worker"` → gerçek `cpu_count()//4`);
  kullanılmayan `compile_model(num_artists=...)` parametresini kaldır; ölü
  `img_to_array` importunu sil; `beathunter_users_old.db`'yi sil;
  `requirements-backend.txt`'e sürüm pinle, eksik `mutagen`/`h5py` ekle.

  **⚠️ TF sürüm uyuşmazlığı — üç ayrı sürüm (ölçüldü 2026-07-28):**

  | Ortam | TF | Not |
  |---|---|---|
  | `requirements-backend.txt` | **2.15.0** (pinli) | hiçbir yerde kurulu değil |
  | Windows (API/Streamlit çalıştığı yer) | **2.20.0** | `is_built_with_cuda()=False` → GPU **yok** |
  | WSL2 `~/beathunter_env` (modelin eğitildiği yer) | **2.21.0** | CUDA build ✅, RTX 4080 görüyor |

  Yani pin gerçeği yansıtmıyor ve iki çalışma ortamı da birbirinden farklı. Model
  WSL2/2.21'de eğitildi, serving Windows/2.20'de yapılıyor — Keras 3 model formatı
  şimdilik ikisinde de yükleniyor (yalnızca zararsız bir optimizer-değişkeni uyarısı
  veriyor), ama bu garanti değil. Karar gerekiyor: pini gerçek serving ortamına göre
  güncelle, yoksa serving'i de WSL2'ye taşı.
  WSL2 diğer paketler: keras 3.13.2, numpy 2.4.3, librosa 0.11.0, faiss-cpu 1.13.2.

  **Ek bulgu:** `~/BeatHunter/dataset` ve `~/BeatHunter/models` symlink'leri **kırık** —
  `/mnt/c/.../Desktop/TypeSheat/...` gösteriyorlar ama proje
  `Desktop/projeler/TypeSheat`'e taşınmış. `dataset_processed/` ise ext4'te gerçek
  dizin ve 873,924 PNG içeriyor; Windows'taki kopyalarla md5 olarak **birebir aynı**.

---

## FAZ 4 — Bir sonraki eğitim turu `[EĞİTİM GEREKİR]`

Şu an ertelendi (süre kısıtı). Eğitim mümkün olduğunda bu sırayla:

- [ ] **4.0 — `[EĞİTİM GEREKİR]` ÖNCE: eğitimin neden başarısız olduğunu bul** ⚠️ *en öncelikli*
  Mevcut model sınıflandırma yapmıyor (CLAUDE.md 🚨 bölümü): head'ler başlangıç
  değerinde, gövde sıfıra büzülmüş, eğitim verisinde bile artist doğruluğu %0.5
  (= rastgele). **Bu bulunmadan yeniden eğitim aynı sonucu verir.**

  Elenen nedenler (tekrar bakma): BatchNorm `training=True`, `set_weights` sıra
  karışması, `mixed_float16` gradyan ölçekleme — üçü de ölçümle çürütüldü.

  **✅ OVERFIT TESTİ ÇALIŞTIRILDI (2026-07-28) — kurulum küçük ölçekte SAĞLAM.**

  | varyant | ep1 | ep20 | sonuç |
  |---|---|---|---|
  | baseline | %56.0 | **%97.5** | ÖĞRENİYOR |
  | no_mixed_precision | %54.2 | %97.0 | ÖĞRENİYOR |
  | no_l2 | %54.1 | %97.7 | ÖĞRENİYOR |
  | no_label_smoothing | %55.2 | %95.0 | ÖĞRENİYOR |
  | no_dropout | %72.0 | %98.9 | ÖĞRENİYOR |
  | higher_lr (1e-2) | %35.0 | %82.7 | kısmen |
  | lower_lr (1e-4) | %29.1 | %93.8 | ÖĞRENİYOR |
  | unfrozen_base | %28.9 | %99.6 | ÖĞRENİYOR |

  Yani **hiçbir hiperparametre tek başına suçlu değil.** Mimari, veri, spektrogramlar
  ve etiketler öğrenilebilir durumda. Sorun ölçeğe bağlı. GPU tepe 80°C, 4 termal
  duraklama (guard çalıştı).

  **🎯 ÖNCELİKLİ HİPOTEZ — `val_loss` izlemek ağırlık büzülmesini ödüllendiriyor.**
  Keras'ın raporladığı `loss`/`val_loss` **L2 regularizasyon terimini içerir.**
  `EarlyStopping` ve `ModelCheckpoint` ikisi de `val_loss` izliyor ve
  `restore_best_weights=True` kullanıyor. Sınıflandırma sinyali zorlaştığında toplam
  kaybı düşürmenin en ucuz yolu ağırlıkları küçültmektir — L2 terimi düşer, `val_loss`
  "iyileşir", callback'ler bunu ödüllendirip o ağırlıkları geri yükler ve kaydeder.

  Gözlenen ağırlık imzası bununla **birebir örtüşüyor**: gövde 0.03-0.06x init'e
  büzülmüş, head'ler init'te kalmış. Küçük ölçekte görünmüyor çünkü sınıflandırma
  kolay ve CCE terimi baskın; 873K görüntü + 40 epoch'ta değil.

  Dikkat: `model_trainer.py` docstring'i bunu **bilinçli bir tercih** olarak anlatıyor —
  *"Monitors val_loss instead of val_genre_out_accuracy (more stable signal)"*. Ölümcül
  olan değişiklik bu olabilir.

  **✅ TUR 2 ÇALIŞTIRILDI (2026-07-28) — düzeltilmiş trainer, 30 sanatçı × 500 slice
  (15,000 görüntü), 12+12 epoch, GroupShuffleSplit + sample weight + accuracy-based
  callback'ler.**

  | head | train | val | şans | val/şans |
  |---|---|---|---|---|
  | genre | %45.8 | %24.3 | %5.00 | **4.9x** |
  | artist | %42.7 | %21.1 | %3.33 | **6.3x** |
  | mood | %55.6 | %31.0 | %16.67 | 1.9x |

  Karşılaştırma: kaydedilmiş üretim modeli kendi **eğitim** görüntülerinde artist
  %0.5 = **1.0x şans**. Yani düzeltilmiş pipeline gerçekten öğreniyor.
  Termal: tepe 83°C, 11 duraklama, guard çalıştı.

  ⚠️ **Ama `val_loss` hipotezi DOĞRULANMADI.** Accuracy-based callback'ler hata modunu
  en baştan engellediği için ıraksamayı (loss düşerken doğruluk sabit) gözlemleyemedik;
  görünen epoch'larda ikisi de plato yapmış. Yani "düzeltme işe yaradı" diyebiliriz ama
  "sebep buydu" diyemeyiz. Özgün başarısızlığın kesin nedeni **hâlâ bilinmiyor**.

  ⚠️ **Ölçek farkı duruyor:** bu koşu 30 sanatçı / 15K görüntü. Gerçek koşu 194 sanatçı /
  873K görüntü — şans tabanı %3.33 değil %0.5. Başarısızlık o ölçekte hâlâ mümkün.
  Tam koşuda `TrainingMonitor`'ün epoch çıktısı ve tripwire'ı bunu yakalayacak.

  train %42.7 vs val %21.1 arasındaki 2x fark gerçek aşırı-öğrenme — ve artık
  ölçülebilir, çünkü grup bazlı ayrım sızıntıyı kaldırdı.

  **Tur 3 (opsiyonel, gerekirse):** overfit testi ile gerçek eğitim
  arasındaki farkları kapat ve hangisinin kırdığını bul. Farklar: (a) tek head vs
  3 head + `loss_weights`, (b) augmentation yok vs var, (c) 8 dengeli sınıf vs
  194/65/6 ve 28x/46x dengesizlik, (d) **callback'ler yok vs `val_loss` izleyen
  EarlyStopping/ModelCheckpoint**. (d) önce test edilmeli.

  **Araçlar:** `overfit_test.py` — 8 varyantı
  (baseline, mixed precision kapalı, L2 yok, label smoothing yok, dropout yok, LR
  yüksek/düşük, backbone çözülmüş) ~2000 görüntüde deneyip hangisinin öğrendiğini
  raporluyor. Her varyant baseline'dan **tek bir ayarı** değiştiriyor, böylece öğrenen
  varyant doğrudan suçluyu gösteriyor. WSL2'de çalıştırılmalı (Windows'ta GPU yok).

  Bir sonraki turda uygulanacak yöntem:
  1. **Önce ~2000 görüntülük bir alt kümede aşırı-öğrenme (overfit) testi yap.**
     Model küçük bir kümeyi ezberleyemiyorsa sorun veride/pipeline'da değil,
     eğitim kurulumunda. Bu test dakikalar sürer ve tam eğitimi harcamadan yakalar.
  2. Faz 1'in kendisi çalışıyor mu ayrı doğrula (mevcut checkpoint Faz 2'den, Faz 1
     hakkında bilgi vermiyor).
  3. L2 (0.0005) + label smoothing kombinasyonunu sorgula — gövde ağırlıklarının
     Glorot'un 0.03-0.06 katına inmesi, kaybın regularizasyon terimine hâkim olduğuna
     işaret ediyor olabilir.
  4. Her epoch'ta **train** accuracy'yi de logla — sadece val değil. Train accuracy
     düşük kalıyorsa öğrenme hiç başlamamış demektir.

- [ ] **4.1 — `[EĞİTİM GEREKİR]` Train/val sızıntısını düzelt**
  `train_test_split` → `GroupShuffleSplit`, grup anahtarı `filename.split('_slice')[0]`.
  *Uyarı:* düzeltme sonrası val accuracy'nin belirgin düşmesini bekle — o gerçek sayı.
- [ ] **4.2 — `[EĞİTİM GEREKİR]` Dataset'i 256-mel'e yükselt (opsiyonel)**
  Eğer gerçekten n_mels=256'ya geçmek isteniyorsa (Faz 0'da geçici olarak 128'e
  dönüldü), 873K spektrogramı yeniden üret + yeniden eğit. 4.1 ile aynı eğitim turunda
  birleştir.
- [ ] **4.3 — `[EĞİTİM GEREKİR]` Taksonomi kararı**
  Mood'u sesten türetmek mi (librosa özellikleri / yeniden etiketleme), yoksa tek head
  (artist) + genre/mood'u lookup ile türetmek mi — bu bir mimari karar, Faz 2'deki
  joint decoding yeterli görülürse ertelenebilir.
- [ ] **4.4 — `[EĞİTİM GEREKİR]` Class weight ekle**
  Genre'de 46x, artist'te 28x dengesizlik var; nadir sınıflar pratikte hiç tahmin
  edilmiyor.
- [ ] **4.5 — `[EĞİTİM GEREKİR]` `random_flip_left_right` augmentation'ını gözden geçir**
  Spektrogramda zaman eksenini ters çeviriyor — ritim yönü müzikte anlamlı, tartışmalı.

---

## Notlar
- Faz 0-3 arası hiçbir adım yeniden eğitim gerektirmiyor — hepsi mevcut modelle
  yapılabilir.
- Faz 0.1'den sonra çıktılar belirgin şekilde değişecek — "önce/sonra" karşılaştırması
  için birkaç bilinen şarkıyı elde tut.
- Faz 4'e geçmeden önce Faz 0-3'ün tamamının rahatça biteceğini varsayma; her fazdan
  sonra bu dosyayı güncelle.
