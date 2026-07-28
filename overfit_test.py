"""
BeatHunter AI — Overfit Diagnostic (TODO 4.0)

WHY THIS EXISTS
───────────────────────────────────────────────────────────────
The saved model classifies at chance level even on the images it was trained on
(artist 0.5% where chance is 0.5%; mean max softmax 0.014 against a uniform 0.005).
Its trunk weights sit at 0.03-0.06x their initialisation and its output heads are
still at initialisation — it never learned. A full retrain costs hours, so it must
not be started until we know why.

This script answers one question: CAN this architecture memorise a tiny dataset?

A model that cannot overfit ~2000 images from a handful of classes has a broken
training setup, not a data problem. The check takes minutes rather than hours, and
would have caught the current model before its full training run was spent.

Each variant changes exactly ONE thing from the current model_trainer.py settings, so
a variant that learns while the baseline does not identifies the culprit directly.

USAGE (WSL2 — Windows TensorFlow has no GPU)
───────────────────────────────────────────────────────────────
    source ~/beathunter_env/bin/activate
    python overfit_test.py                 # all variants
    python overfit_test.py --only baseline no_mixed_precision
    python overfit_test.py --artists 6 --per-artist 300 --epochs 25

READING THE RESULT
───────────────────────────────────────────────────────────────
TRAIN accuracy is what matters here, not validation — the point is memorisation.
A healthy run reaches >90% train accuracy on this subset within ~20 epochs.

  every variant fails      → problem is upstream: data, labels, or the input pipeline
  baseline fails, some pass → the differing setting is the cause
  everything passes        → the setup is fine at small scale; suspect scale-dependent
                             causes (LR schedule over 873K images, class imbalance,
                             or the phase-1 → phase-2 transition)
"""

import os, sys, json, time, threading, subprocess, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB3

CSV_PATH = "dataset_processed/metadata_processed.csv"
IMG_DIR  = "dataset_processed/images"
IMG_SIZE = (224, 224)

# ============================================================
# THERMAL GUARD
# ============================================================
# Same thresholds as the FAISS rebuild script. This machine thermally shut down at
# 96-97C earlier, and a soak test showed the GPU still climbing 4-6 C/min after three
# minutes at only 58W — cooling is marginal, so sustained load needs supervision.
#
# CPU temperature is NOT readable (WSL2 exposes no thermal_zone, and the Windows WMI
# thermal class needs admin), so this guards on GPU temperature only.
ABORT_C, COOL_C, RESUME_C = 90, 80, 70
COOL_TIMEOUT = 600
TEMP_LOG = os.path.expanduser("~/bench_out/temps_overfit.csv")

_therm = {"gpu": None, "max": 0, "abort": False, "reason": None, "pauses": 0}
_stop_monitor = threading.Event()


def read_gpu():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return int(o[0]), float(o[1]), float(o[2])
    except Exception:
        return None, None, None


def _monitor():
    """Polls every 2s so the training loop only ever reads a cached value."""
    os.makedirs(os.path.dirname(TEMP_LOG), exist_ok=True)
    with open(TEMP_LOG, "w") as f:
        f.write("t,gpu_temp_c,gpu_power_w,gpu_clock_mhz\n")
        t0 = time.time()
        while not _stop_monitor.is_set():
            t, p, c = read_gpu()
            if t is not None:
                f.write(f"{time.time()-t0:.1f},{t},{p},{c}\n"); f.flush()
                _therm["gpu"] = t
                _therm["max"] = max(_therm["max"], t)
                if t >= ABORT_C:
                    _therm["abort"] = True
                    _therm["reason"] = f"GPU {t}C >= {ABORT_C}C abort esigi"
            _stop_monitor.wait(2)


class ThermalGuard(tf.keras.callbacks.Callback):
    """
    Pauses training when the GPU gets hot and stops it outright if it gets dangerous.

    Reads the cached temperature rather than shelling out to nvidia-smi, so the
    per-batch cost is a dict lookup. Checked every `every` batches and at each epoch
    end; the pause blocks inside the callback, which idles the GPU without losing state.
    """

    def __init__(self, every=20):
        super().__init__()
        self.every = every

    def _check(self):
        if _therm["abort"]:
            self.model.stop_training = True
            return
        t = _therm["gpu"]
        if t is None or t < COOL_C:
            return

        _therm["pauses"] += 1
        print(f"\n  ~ GPU {t}C >= {COOL_C}C — duraklatildi, <{RESUME_C}C bekleniyor...",
              flush=True)
        waited = 0
        while True:
            time.sleep(5); waited += 5
            t = _therm["gpu"]
            if t is None or t < RESUME_C:
                print(f"  ~ devam ({t}C, {waited}s duraklama)", flush=True)
                return
            if t >= ABORT_C or waited > COOL_TIMEOUT:
                _therm["abort"] = True
                _therm["reason"] = f"soguma basarisiz (GPU {t}C, {waited}s)"
                self.model.stop_training = True
                return

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.every == 0:
            self._check()

    def on_epoch_end(self, epoch, logs=None):
        self._check()

# Each variant = the current settings with ONE field overridden.
BASE = dict(mixed_precision=True, l2=0.0005, label_smoothing=0.05,
            lr=1e-3, freeze_base=True, dropout=True)

VARIANTS = {
    "baseline":            {},                          # exactly model_trainer.py today
    "no_mixed_precision":  dict(mixed_precision=False),  # float32 throughout
    "no_l2":               dict(l2=0.0),                 # trunk collapsed to ~0 -> suspect
    "no_label_smoothing":  dict(label_smoothing=0.0),
    "no_dropout":          dict(dropout=False),
    "higher_lr":           dict(lr=1e-2),
    "lower_lr":            dict(lr=1e-4),
    "unfrozen_base":       dict(freeze_base=False, lr=1e-4),
}


def build(n_classes, cfg):
    """Phase-1 architecture from model_trainer.py, single head (artist only)."""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = EfficientNetB3(include_top=False, weights='imagenet', pooling=None)
    base.trainable = not cfg['freeze_base']

    x = base(inputs, training=not cfg['freeze_base'])
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    reg = regularizers.l2(cfg['l2']) if cfg['l2'] else None
    x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    if cfg['dropout']: x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    if cfg['dropout']: x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation='softmax', name='artist_out', dtype='float32')(x)
    return models.Model(inputs, out)


def make_ds(paths, labels, batch, shuffle):
    def load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), y      # EfficientNet rescales internally
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=6).batch(batch).prefetch(2)


def run(name, cfg, paths, labels, n_classes, epochs, batch):
    print("\n" + "=" * 66)
    print(f"VARIANT: {name}")
    changed = {k: v for k, v in cfg.items() if BASE.get(k) != v}
    print(f"  changed from baseline: {changed if changed else '(nothing — this IS the baseline)'}")
    print("=" * 66, flush=True)

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        'mixed_float16' if cfg['mixed_precision'] else 'float32')

    model = build(n_classes, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg['label_smoothing']),
        metrics=['accuracy'])

    # Same data for fit and eval — memorisation is the whole point.
    ds = make_ds(paths, labels, batch, shuffle=True)
    t0 = time.time()
    hist = model.fit(ds, epochs=epochs, verbose=2, callbacks=[ThermalGuard()])
    took = time.time() - t0

    acc   = hist.history['accuracy']
    final = acc[-1]
    best  = max(acc)

    if _therm["abort"]:
        verdict = "IPTAL(termal)"
    elif len(acc) < epochs:
        verdict = "EKSIK(termal)"          # stop_training fired before the last epoch
    else:
        verdict = "OGRENIYOR" if best > 0.90 else ("KISMEN" if best > 0.40 else "OGRENMIYOR")

    print(f"  -> train acc: baslangic {acc[0]*100:.1f}%  son {final*100:.1f}%  "
          f"en iyi {best*100:.1f}%  ({took/60:.1f} dk, {len(acc)}/{epochs} epoch)")
    print(f"  -> GPU tepe {_therm['max']}C, {_therm['pauses']} duraklama")
    print(f"  -> SONUC: {verdict}", flush=True)

    return {"variant": name, "changed": changed, "final_acc": float(final),
            "best_acc": float(best), "verdict": verdict,
            "epochs_done": len(acc), "epochs_target": epochs,
            "gpu_peak_c": _therm["max"], "pauses": _therm["pauses"],
            "acc_curve": [round(float(a), 4) for a in acc], "minutes": round(took / 60, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists",    type=int, default=8)
    ap.add_argument("--per-artist", type=int, default=250)
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--only",       nargs="*", default=None)
    ap.add_argument("--out", default="overfit_results.json")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    print(f"TF {tf.__version__} | GPU: {gpus if gpus else 'YOK (CPU — cok yavas olur)'}")
    if not gpus:
        print("UYARI: GPU yok. WSL2'de ~/beathunter_env icinden calistir.")

    df = pd.read_csv(CSV_PATH)
    # Pick the artists with the most slices, so the subset is dense and easy to fit.
    top = df['artist_type'].value_counts().head(args.artists).index.tolist()
    sub = (df[df['artist_type'].isin(top)]
           .groupby('artist_type', group_keys=False)
           .apply(lambda g: g.head(args.per_artist), include_groups=True))

    classes = sorted(sub['artist_type'].unique())
    idx = {c: i for i, c in enumerate(classes)}
    labels = np.eye(len(classes), dtype=np.float32)[[idx[a] for a in sub['artist_type']]]
    paths  = [os.path.join(IMG_DIR, f) for f in sub['filename']]

    print(f"\nalt kume: {len(paths):,} goruntu / {len(classes)} sinif "
          f"(sansa esit dogruluk = {100/len(classes):.1f}%)")
    print(f"siniflar: {classes}")
    print(f"hedef: train accuracy >90% — bu kadar kucuk bir kume EZBERLENEBILMELI")

    names = args.only if args.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"bilinmeyen varyant: {unknown}\nmevcut: {list(VARIANTS)}")

    t, _, _ = read_gpu()
    print(f"baslangic GPU: {t}C | termal guard: {COOL_C}C durakla / {RESUME_C}C devam / {ABORT_C}C iptal")
    threading.Thread(target=_monitor, daemon=True).start()
    time.sleep(2)   # monitor'un ilk okumayi yapmasini bekle

    results = []
    for n in names:
        if _therm["abort"]:
            print(f"\n!! TERMAL IPTAL: {_therm['reason']} — kalan varyantlar atlandi", flush=True)
            results.append({"variant": n, "verdict": "ATLANDI(termal)"})
            json.dump(results, open(args.out, "w"), indent=2)
            break
        cfg = {**BASE, **VARIANTS[n]}
        try:
            results.append(run(n, cfg, paths, labels, len(classes), args.epochs, args.batch))
        except Exception as e:
            print(f"  !! {n} basarisiz: {type(e).__name__}: {e}", flush=True)
            results.append({"variant": n, "verdict": "HATA", "error": str(e)})
        json.dump(results, open(args.out, "w"), indent=2)   # her varyanttan sonra yaz

    print("\n" + "=" * 66)
    print("OZET")
    print("=" * 66)
    print(f"{'varyant':22s} {'en iyi train acc':>17s}  sonuc")
    for r in results:
        acc = f"{r['best_acc']*100:.1f}%" if 'best_acc' in r else "-"
        print(f"{r['variant']:22s} {acc:>17s}  {r['verdict']}")

    _stop_monitor.set()
    print(f"\nGPU tepe {_therm['max']}C | toplam {_therm['pauses']} termal duraklama")
    print(f"sicaklik logu: {TEMP_LOG}")

    ok   = [r['variant'] for r in results if r.get('verdict') == "OGRENIYOR"]
    base = next((r for r in results if r['variant'] == 'baseline'), None)
    print()
    if _therm["abort"]:
        print(f"⚠️ TERMAL IPTAL: {_therm['reason']}")
        print("   Sonuclar eksik — asagidaki teshis guvenilir olmayabilir.")
    if base and base.get('verdict') != "OGRENIYOR" and ok:
        print(f"TESHIS: baseline ogrenmiyor ama {ok} ogreniyor.")
        print("        -> farki yaratan ayar suclu. model_trainer.py'de onu duzelt.")
    elif not ok:
        print("TESHIS: hicbir varyant ogrenmiyor.")
        print("        -> sorun egitim ayarlarinda degil; veri/etiket/pipeline'a bak.")
        print("        -> once spektrogramlarin gercekten sinifa ozgu bilgi tasidigini dogrula.")
    else:
        print("TESHIS: baseline kucuk olcekte ogreniyor.")
        print("        -> sorun olcege bagli: 873K goruntude LR programi, sinif dengesizligi,")
        print("           veya Faz1->Faz2 gecisi. Tam egitimde train accuracy'yi epoch basina logla.")
    print(f"\nsonuclar: {args.out}")


if __name__ == "__main__":
    main()
