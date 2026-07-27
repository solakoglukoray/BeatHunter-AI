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

- [ ] **1.1 — FAISS index'i track seviyesinde yeniden inşa et**
  Slice değil, parça ortalaması (~14,235 track × 1536-dim, ~87MB). Forward pass
  gerektirir (eğitim değil). Parça başına ~16 slice örneklemesi yeterli.
  *Ön koşul:* 0.1 ve 0.2 tamamlanmadan yapma — yanlış spektrogramlarla index kurmak
  hatayı kalıcılaştırır.

- [ ] **1.2 — Index/metadata hizasızlığını doğrula**
  Rebuild sonrası `vector_meta.csv` ve `metadata_processed.csv`'nin aynı çalıştırmadan
  geldiğini teyit et.

---

## FAZ 2 — Tutarlılık kazancı (eğitim gerektirmez)

- [ ] **2.1 — Joint decoding**
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

## FAZ 3 — Performans, eşzamanlılık, temizlik (eğitim gerektirmez, paralel gidebilir)

- [ ] **3.1 — Inference'ı batch'le** — segment başına 2 ayrı `model.predict()` yerine
  tek `(8,224,224,3)` batch çağrısı. `embed_model`'i cache'le (her istekte yeniden
  inşa edilmiyor). Beklenen kazanç ~8-10x.
- [ ] **3.2 — BPM clamp'i while döngüsüne çevir** + hip-hop hedef aralığına (120-170)
  göre yarı-tempo/çift-tempo katlama mantığı ekle (140→70 algılama klasik hata).
- [ ] **3.3 — Key tespiti** — `chroma_stft` yerine `librosa.effects.harmonic()` +
  `chroma_cqt`.
- [ ] **3.4 — Öneri sayısı garantisi** — `k=10` yerine `k=100` komşu çek, tekilleştirme
  sonrası 5 öneri garantiye alınsın.
- [ ] **3.5 — Eşzamanlılık** — sabit `temp_inference/temp_spec.png` ve
  `downloaded_song.mp3` yollarını istek/session bazlı yap. `subprocess.run(shell=True)`
  (Demucs) için returncode kontrolü + timeout ekle.
- [ ] **3.6 — Küçük temizlik** — `time.sleep(0.3)` (data_processor.py:124, ~12dk boşa
  gidiyor) kaldır; yanlış yorumları düzelt (`"22 worker"` → gerçek `cpu_count()//4`);
  kullanılmayan `compile_model(num_artists=...)` parametresini kaldır; ölü
  `img_to_array` importunu sil; `beathunter_users_old.db`'yi sil;
  `requirements-backend.txt`'e sürüm pinle, eksik `mutagen`/`h5py` ekle.

---

## FAZ 4 — Bir sonraki eğitim turu `[EĞİTİM GEREKİR]`

Şu an ertelendi (süre kısıtı). Eğitim mümkün olduğunda bu sırayla:

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
