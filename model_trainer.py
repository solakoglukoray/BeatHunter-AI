"""
BeatHunter AI - Model Trainer (v2.0)

Key changes from v1.x:
─────────────────────────────────────────────────────────────
1. EfficientNetB3 Transfer Learning
   - Replaces the custom 4-block CNN.
   - Pre-trained on ImageNet (1.2M images) → converges in 15-25 epochs instead of 60.
   - Dramatically better genre/mood accuracy out of the box.

2. Mixed Precision (float16)
   - Halves VRAM usage, doubles training speed on RTX 4080.
   - GPU stays at ~70-75% VRAM, never hits 100%.

3. Two-Phase Training Strategy
   - Phase 1 (5-8 epochs): Only the custom heads are trained. EfficientNet is frozen.
     Fast, low compute, establishes a good baseline before fine-tuning.
   - Phase 2 (10-15 epochs): Last 30 layers of EfficientNet unfrozen and fine-tuned
     with a very low LR (1e-5). This is where accuracy peaks.

4. Controlled Hardware Usage
   - GPU memory growth enabled → VRAM never spikes to 100%.
   - Batch size 128 → stable gradients on 570K dataset, GPU at ~75%.
   - Shuffle buffer 5000 → RAM stays comfortable.
   - AUTOTUNE for all tf.data ops → CPU at ~60-70%, never maxed.

5. Callbacks select on ACCURACY, not loss
   - EarlyStopping / ModelCheckpoint monitor val_combined_accuracy (mode='max').
   - ⚠️ An earlier version monitored val_loss "as a more stable signal". That is
     believed to be what cost the previous run: Keras folds L2 into the reported loss,
     so shrinking weights lowers val_loss without learning anything, and
     restore_best_weights then locked that in. Do not switch back.
   - TrainingMonitor prints train AND val accuracy each epoch and aborts if the model
     is still at chance after 5 epochs.

6. Spectrogram parameter alignment
   - IMG_SIZE = (224, 224) — matches n_mels=256 output resized in inference_engine.py
─────────────────────────────────────────────────────────────
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import GroupShuffleSplit
import thermal_guard

# ============================================================
# GPU CONFIGURATION — Must run before any TF operations
# memory_growth=True prevents VRAM from being fully allocated
# upfront, keeping usage at ~70-75% instead of 100%.
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {len(gpus)} device(s). Memory growth enabled.")
else:
    print("⚠️  No GPU detected. Training will use CPU (very slow).")

# Mixed precision is applied below, after USE_MIXED_PRECISION is defined, so the
# overfit test's verdict can switch it from a single place.

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH        = "dataset_processed/metadata_processed.csv"
IMG_DIR         = "dataset_processed/images"
MODEL_DIR       = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "beathunter_model.keras")
BEST_CKPT_PATH  = os.path.join(MODEL_DIR, "best_checkpoint.keras")
# Diagnostic subset runs must never touch the real checkpoint.
SUBSET_CKPT_PATH = os.path.join(MODEL_DIR, "subset_diagnostic.keras")
CLASSES_PATH    = os.path.join(MODEL_DIR, "classes.json")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 64
EPOCHS_P1  = 15     # Phase 1: heads only (EfficientNet frozen)
EPOCHS_P2  = 40     # Phase 2: fine-tune top layers (EarlyStopping will stop this early)

# ============================================================
# ⚠️ RUN overfit_test.py BEFORE STARTING A FULL TRAINING RUN
# ============================================================
# The previously saved model never learned to classify: on its OWN training images it
# scored 0.5% artist accuracy where chance is 0.5%, with output heads still at their
# initialisation and trunk weights decayed to 0.03-0.06x init. Hours of GPU time
# produced nothing usable, and nothing in the logs said so at the time.
#
# overfit_test.py trains this architecture on ~2000 images and checks it can memorise
# them. It takes minutes. If the baseline variant does not reach >90% TRAIN accuracy,
# do not start a full run — fix the setup first and re-check.
#
# Three causes have already been tested and ruled out (do not re-investigate):
# BatchNorm training=True baked into the graph; set_weights reordering between the two
# phase builds; mixed_float16 without loss scaling (Keras 3 auto-wraps).
#
# These four are the remaining suspects, hoisted here so the overfit test's verdict can
# be applied in one place:
L2_REG          = 0.0005   # trunk collapsed toward zero — regularisation may dominate
LABEL_SMOOTHING = 0.05     # artist head used 0.08
LR_PHASE1       = 1e-3
USE_MIXED_PRECISION = True

# Reverses the time axis. Rhythm direction is meaningful in music, so this augmentation
# is questionable; disabled pending evidence it helps (TODO.md 4.5).
AUGMENT_TIME_FLIP = False

# float16 compute with float32 variables: ~2x faster, ~50% less VRAM on the RTX 4080.
# Keras 3 wraps the optimizer in LossScaleOptimizer automatically (verified).
if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed Precision policy set: mixed_float16")
else:
    print("ℹ️  Mixed precision disabled — running in float32")


# ============================================================
# TRAINING MONITOR
# ============================================================
# The previous run saved a model that predicts at chance, and nothing in the logs said
# so — only val_loss was reported, and val_loss went down the whole time.
#
# It went down because Keras includes regularisation losses in it. Once the
# classification signal gets hard, the cheapest way to reduce total loss is to shrink
# weights and drop the L2 term. EarlyStopping and ModelCheckpoint were both monitoring
# val_loss with restore_best_weights=True, so they rewarded exactly that, restored those
# weights and saved them. The saved model's trunk sits at 0.03-0.06x its initialisation
# with the heads still at init — the signature of that failure.
#
# So: monitor accuracy, not loss, and print both side by side every epoch. If the two
# diverge — loss improving while accuracy does not — that is the failure happening, and
# now it is visible while the run is still going.
class TrainingMonitor(tf.keras.callbacks.Callback):
    """Prints train vs val accuracy per epoch and aborts if training never leaves chance."""

    def __init__(self, n_genre, n_artist, n_mood, chance_grace=5):
        super().__init__()
        self.chance = {'genre': 1.0 / n_genre, 'artist': 1.0 / n_artist, 'mood': 1.0 / n_mood}
        self.chance_grace = chance_grace

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        heads = ('genre', 'artist', 'mood')

        # Composite score so callbacks have a single classification metric to monitor.
        val_accs = [logs.get(f'val_{h}_out_accuracy', 0.0) for h in heads]
        logs['val_combined_accuracy'] = float(np.mean(val_accs))

        print(f"\n  epoch {epoch+1}  val_loss={logs.get('val_loss', float('nan')):.4f}"
              f"  (loss alone is NOT evidence of learning — compare the accuracies)")
        for h in heads:
            tr = logs.get(f'{h}_out_accuracy', 0.0)
            va = logs.get(f'val_{h}_out_accuracy', 0.0)
            ch = self.chance[h]
            print(f"    {h:7s} train {tr*100:5.1f}%  val {va*100:5.1f}%  "
                  f"(chance {ch*100:4.1f}% -> val is {va/ch if ch else 0:4.1f}x chance)")

        # Tripwire: if the model is still at chance after a few epochs, stop. A full run
        # costs hours; this costs one epoch of patience.
        if epoch + 1 >= self.chance_grace:
            tr_artist = logs.get('artist_out_accuracy', 0.0)
            if tr_artist < 2 * self.chance['artist']:
                print(f"\n  ⛔ TRAIN artist accuracy is {tr_artist*100:.2f}% after "
                      f"{epoch+1} epochs — chance is {self.chance['artist']*100:.2f}%.")
                print("     The model is not learning. Stopping rather than burning hours.")
                print("     Run overfit_test.py to isolate the cause before retrying.")
                self.model.stop_training = True


# ============================================================
# CRASH-SAFE CHECKPOINTING
# ============================================================
# This laptop travels daily and sometimes runs out of battery in a bag. Training takes
# hours, so it WILL be killed mid-run without warning.
#
# ModelCheckpoint alone does not survive that. It writes only at epoch ends, and only
# when the monitored metric improves, so a kill can cost a full epoch — and what it
# writes carries no optimizer moments, no epoch counter and no phase, so a resume would
# restart Adam from zero and re-run the learning-rate schedule from step 0.
#
# So: save the WHOLE model (weights + optimizer slots + LR schedule position, 46
# optimizer variables measured) on a wall-clock timer from inside the epoch, plus a
# small JSON recording which phase and epoch we were on.
#
# Ordering matters. The model is written first, the state second. A crash between them
# leaves the state pointing at an older epoch than the weights, so the resume redoes a
# little work — harmless. The reverse order would claim progress the weights do not
# have and silently skip training.
#
# Both writes go to a temp file and are then os.replace()d, which is atomic on the same
# filesystem: a kill mid-write leaves the previous checkpoint intact rather than a
# half-written file.
#
# Resume granularity: the epoch counter comes from the last save, the weights from the
# last timed save inside that epoch. Resuming therefore repeats part of an epoch with
# weights that are already ahead. Data order is not restored — the training set is
# reshuffled every epoch regardless, so there is nothing meaningful to restore.
CKPT_MODEL_PATH = os.path.join(MODEL_DIR, "training_checkpoint.keras")
CKPT_STATE_PATH = os.path.join(MODEL_DIR, "training_state.json")
# 120s costs ~1.3% of runtime (a full save measured 1.5s / 70 MB) and caps the worst
# case loss at two minutes of training. Lower it for testing; do not raise it.
CKPT_INTERVAL_S = 120


class CrashSafeCheckpoint(tf.keras.callbacks.Callback):
    """Timed, intra-epoch, atomic full-model checkpoint. See the note above."""

    def __init__(self, phase, total_epochs, interval=CKPT_INTERVAL_S):
        super().__init__()
        self.phase = phase
        self.total_epochs = total_epochs
        self.interval = interval
        self.epoch = 0
        self._last = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self._last >= self.interval:
            self._save(reason=f"epoch {self.epoch+1} icinde")

    def on_epoch_end(self, epoch, logs=None):
        # Record the NEXT epoch to run, so resume does not repeat a finished one.
        self.epoch = epoch + 1
        self._save(reason=f"epoch {epoch+1} sonu")

    def _save(self, reason=""):
        t0 = time.time()
        try:
            tmp_model = CKPT_MODEL_PATH + ".tmp.keras"
            self.model.save(tmp_model)
            os.replace(tmp_model, CKPT_MODEL_PATH)      # atomic

            state = {
                "phase": self.phase,
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
            }
            tmp_state = CKPT_STATE_PATH + ".tmp"
            with open(tmp_state, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_state, CKPT_STATE_PATH)      # atomic

            self._last = time.time()
            # self.epoch is the epoch to RESUME at (0-indexed), which mid-epoch is the
            # current one — so report both, or "epoch 6" while inside epoch 7 reads
            # like a bug.
            print(f"\n  💾 checkpoint: faz {self.phase} — {reason}, "
                  f"resume noktasi epoch {self.epoch + 1}/{self.total_epochs} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            # A failed checkpoint must never kill a multi-hour run.
            print(f"\n  ⚠️ checkpoint yazilamadi ({e}) — kosu devam ediyor", flush=True)


def load_resume_state():
    """Returns the saved state dict, or None if there is nothing to resume from."""
    if not (os.path.exists(CKPT_MODEL_PATH) and os.path.exists(CKPT_STATE_PATH)):
        return None
    try:
        with open(CKPT_STATE_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ checkpoint durumu okunamadi ({e}) — sifirdan baslanacak")
        return None


def clear_checkpoint():
    for p in (CKPT_MODEL_PATH, CKPT_STATE_PATH):
        if os.path.exists(p):
            os.remove(p)


def make_callbacks(best_path, patience, n_genre, n_artist, n_mood, reduce_lr=False,
                   phase=None, total_epochs=None):
    """
    Callbacks that select on classification quality rather than on loss.

    monitor='val_combined_accuracy' (mode='max') — the mean of the three head
    accuracies, injected by TrainingMonitor. Weight shrinkage cannot game it.
    """
    cbs = [
        TrainingMonitor(n_genre, n_artist, n_mood),
        thermal_guard.make_keras_callback(tf),
        EarlyStopping(monitor='val_combined_accuracy', mode='max',
                      patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=best_path, monitor='val_combined_accuracy', mode='max',
                        save_best_only=True, verbose=1),
    ]
    if phase is not None:
        cbs.append(CrashSafeCheckpoint(phase, total_epochs))
    if reduce_lr:
        cbs.append(ReduceLROnPlateau(monitor='val_combined_accuracy', mode='max',
                                     factor=0.5, patience=3, min_lr=1e-6, verbose=1))
    return cbs


# ============================================================
# tf.data PIPELINE
# ============================================================
def create_dataset(df, img_dir, genre_cols, artist_cols, mood_cols,
                   batch_size=64, augment=False, class_weights=None):
    """
    High-performance tf.data pipeline.
    - Parallel loading with AUTOTUNE
    - Shuffle before map (correct order)
    - Augmentation only on training set
    - Prefetch keeps GPU fed at all times
    """
    img_paths     = [os.path.join(img_dir, f) for f in df['filename'].values]
    genre_labels  = df[genre_cols].values.astype(np.float32)
    artist_labels = df[artist_cols].values.astype(np.float32)
    mood_labels   = df[mood_cols].values.astype(np.float32)

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img

    def augment_image(img):
        # Temporal flip — reverses the time axis (see AUGMENT_TIME_FLIP)
        if AUGMENT_TIME_FLIP:
            img = tf.image.random_flip_left_right(img)
        # Brightness — simulates volume variation (delta scaled for [0, 255] range)
        img = tf.image.random_brightness(img, max_delta=30.0)
        # Contrast — simulates dynamic range variation
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img

    def spec_augment(img):
        # Frequency masking — 60% probability per sample
        freq_mask_size = tf.random.uniform([], 1, 25, dtype=tf.int32)
        max_freq_start = tf.maximum(224 - freq_mask_size, 1)
        freq_start     = tf.random.uniform([], 0, max_freq_start, dtype=tf.int32)
        freq_end       = tf.minimum(freq_start + freq_mask_size, 224)
        freq_mask = tf.concat([
            tf.ones ([freq_start, 224, 3]),
            tf.zeros([freq_end - freq_start, 224, 3]),
            tf.ones ([224 - freq_end, 224, 3])
        ], axis=0)
        img = tf.cond(
            tf.random.uniform([]) < 0.6,
            lambda: img * freq_mask,
            lambda: img
        )

        # Time masking — 60% probability per sample
        time_mask_size = tf.random.uniform([], 1, 35, dtype=tf.int32)
        max_time_start = tf.maximum(224 - time_mask_size, 1)
        time_start     = tf.random.uniform([], 0, max_time_start, dtype=tf.int32)
        time_end       = tf.minimum(time_start + time_mask_size, 224)
        time_mask = tf.concat([
            tf.ones ([224, time_start, 3]),
            tf.zeros([224, time_end - time_start, 3]),
            tf.ones ([224, 224 - time_end, 3])
        ], axis=1)
        img = tf.cond(
            tf.random.uniform([]) < 0.6,
            lambda: img * time_mask,
            lambda: img
        )

        return img

    path_ds   = tf.data.Dataset.from_tensor_slices(img_paths)
    genre_ds  = tf.data.Dataset.from_tensor_slices(genre_labels)
    artist_ds = tf.data.Dataset.from_tensor_slices(artist_labels)
    mood_ds   = tf.data.Dataset.from_tensor_slices(mood_labels)

    label_ds = tf.data.Dataset.zip((genre_ds, artist_ds, mood_ds))
    label_ds = label_ds.map(
        lambda g, a, m: {'genre_out': g, 'artist_out': a, 'mood_out': m}
    )

    # Class rebalancing via SAMPLE weights, not class_weight.
    # Keras 3 rejects class_weight on multi-output models ("only supported for Models
    # with a single output"), so each sample carries the weight of its own true class
    # per head, supplied as the dataset's third element.
    #
    # ⚠️ The weights must be a TUPLE in the model's output order, not a dict. Keras
    # resolves sample weights positionally against outputs=[genre, artist, mood]; a dict
    # here fails with a bare "KeyError: 0". Labels stay a dict — those resolve by name.
    if class_weights:
        def per_sample(head, cols, onehot):
            w = np.array([class_weights[head][i] for i in range(len(cols))], np.float32)
            return w[onehot.argmax(axis=1)]

        weight_ds = tf.data.Dataset.from_tensor_slices((
            per_sample('genre_out',  genre_cols,  genre_labels),
            per_sample('artist_out', artist_cols, artist_labels),
            per_sample('mood_out',   mood_cols,   mood_labels),
        ))
        ds = tf.data.Dataset.zip((path_ds, label_ds, weight_ds))
    else:
        ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle BEFORE map — correct order for proper randomization
    if augment:
        ds = ds.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

    if class_weights:
        ds = ds.map(lambda p, y, w: (load_image(p), y, w), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y, w: (augment_image(x), y, w), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y, w: (spec_augment(x),  y, w), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (spec_augment(x),  y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# MODEL ARCHITECTURE — EfficientNetB3 + Multi-Head
# ============================================================
def build_efficientnet_multihead(num_genres, num_artists, num_moods, freeze_base=True):
    inputs = layers.Input(shape=(224, 224, 3))

    # EfficientNetB3 — input_tensor yerine direkt çağırıyoruz
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        pooling=None
    )

    # Phase 1: Freeze entire base
    base_model.trainable = not freeze_base

    # Phase 2: Unfreeze only the last 30 layers for fine-tuning
    if not freeze_base:
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

    # Phase 1: training=False → BN uses frozen ImageNet stats (correct for frozen base)
    # Phase 2: training=True → BN updates its stats during fine-tuning
    features = base_model(inputs, training=not freeze_base)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.BatchNormalization()(features)

    # Shared dense trunk
    shared = layers.Dense(
        512, activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG)
    )(features)
    shared = layers.Dropout(0.4)(shared)
    shared = layers.Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG)
    )(shared)
    shared = layers.Dropout(0.3)(shared)

    # HEAD 1 — GENRE
    g = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(shared)
    g = layers.Dropout(0.25)(g)
    genre_out = layers.Dense(num_genres, activation='softmax', name='genre_out',
                             dtype='float32')(g)

    # HEAD 2 — ARTIST
    a = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(shared)
    a = layers.Dropout(0.35)(a)
    a = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(a)
    a = layers.Dropout(0.25)(a)
    artist_out = layers.Dense(num_artists, activation='softmax', name='artist_out',
                              dtype='float32')(a)

    # HEAD 3 — MOOD
    m = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(shared)
    m = layers.Dropout(0.2)(m)
    mood_out = layers.Dense(num_moods, activation='softmax', name='mood_out',
                            dtype='float32')(m)

    return models.Model(inputs=inputs, outputs=[genre_out, artist_out, mood_out])


def compile_model(model, learning_rate):
    """
    Compiles the model with task-specific loss functions and weights.
    Label smoothing fights overfitting from noisy YouTube metadata.
    Artist weight is lower because it has 135 noisy classes.
    """

    genre_loss  = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    artist_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    mood_loss   = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'genre_out':  genre_loss,
            'artist_out': artist_loss,
            'mood_out':   mood_loss
        },
        # Loss weights düzelt
        loss_weights={
            'genre_out':  1.0,
            'artist_out': 0.5,
            'mood_out':   0.8
        },
        metrics={
            'genre_out':  ['accuracy'],
            'artist_out': ['accuracy'],
            'mood_out':   ['accuracy']
        }
    )
    return model


def _summarise(label, history, n_genre, n_artist, n_mood):
    """
    Reports TRAIN alongside validation. Train accuracy is the one that reveals a model
    that never learned — the previous run's summary showed only validation numbers,
    which were inflated by the slice-level split anyway.
    """
    h = history.history
    chance = {'genre': 100.0 / n_genre, 'artist': 100.0 / n_artist, 'mood': 100.0 / n_mood}

    print(f"\n📊 {label} RESULTS (peak over epochs)")
    print(f"   {'head':8s} {'train':>8s} {'val':>8s} {'chance':>8s} {'val/chance':>11s}")
    for head in ('genre', 'artist', 'mood'):
        tr = max(h.get(f'{head}_out_accuracy', [0])) * 100
        va = max(h.get(f'val_{head}_out_accuracy', [0])) * 100
        ch = chance[head]
        print(f"   {head:8s} {tr:7.1f}% {va:7.1f}% {ch:7.2f}% {va/ch:10.1f}x")

    tr_artist = max(h.get('artist_out_accuracy', [0])) * 100
    if tr_artist < 2 * chance['artist']:
        print(f"\n   ⛔ TRAIN artist accuracy peaked at {tr_artist:.2f}% against a "
              f"{chance['artist']:.2f}% chance baseline.")
        print("      This model did not learn. Do NOT ship it — run overfit_test.py.")
    print()


def train_model(subset_artists=0, subset_per_artist=0, epochs_p1=None, epochs_p2=None,
                resume=False):
    print("🚀 BeatHunter AI — Model Training v2.0 (EfficientNetB3 + Mixed Precision)\n")
    is_subset = bool(subset_artists or subset_per_artist)
    resume_state = load_resume_state() if resume else None
    if resume and not resume_state:
        print('ℹ️  --resume verildi ama checkpoint yok — sifirdan baslaniyor')
    epochs_p1 = epochs_p1 or EPOCHS_P1
    epochs_p2 = epochs_p2 or EPOCHS_P2

    os.makedirs(MODEL_DIR, exist_ok=True)
    thermal_guard.start_monitor(os.path.expanduser("~/bench_out/temps_training.csv"))

    # ── 1. Load dataset ──────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)

    # Optional subset, for diagnostic runs that need to be minutes rather than hours.
    # A subset run must NOT overwrite the real model or classes.json.
    if subset_artists or subset_per_artist:
        top = df['artist_type'].value_counts().head(subset_artists or 999).index
        df = (df[df['artist_type'].isin(top)]
              .groupby('artist_type', group_keys=False)
              .apply(lambda g: g.head(subset_per_artist) if subset_per_artist else g,
                     include_groups=True))
        print(f"⚠️  SUBSET RUN — {len(df):,} slices / {df['artist_type'].nunique()} artists")
        print("   Diagnostic only. Model and classes.json will NOT be written.\n")

    print(f"📊 Total spectrograms: {len(df):,}")
    print(f"   Genre classes:  {df['genre'].nunique()}")
    print(f"   Artist classes: {df['artist_type'].nunique()}")
    print(f"   Mood classes:   {df['mood'].nunique()}\n")

    # ── 2. One-Hot Encoding (global, before split) ────────────────
    df_genre  = pd.get_dummies(df['genre'],       prefix='genre')
    df_artist = pd.get_dummies(df['artist_type'], prefix='artist')
    df_mood   = pd.get_dummies(df['mood'],        prefix='mood')

    genre_cols  = list(df_genre.columns)
    artist_cols = list(df_artist.columns)
    mood_cols   = list(df_mood.columns)

    df = pd.concat([df, df_genre, df_artist, df_mood], axis=1)

    # ── 3. GROUPED train/val split — split by track, never by slice ──
    # Each track yields ~61 slices. A random slice-level split scatters one song's
    # slices across both sides, so the model can memorise the song and score high on
    # validation without generalising at all. Every val accuracy reported before this
    # change was inflated for that reason.
    df['track'] = df['filename'].str.split('_slice').str[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df['track']))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    overlap = set(train_df['track']) & set(val_df['track'])
    assert not overlap, f"track leak between train and val: {len(overlap)} tracks"

    print(f"Training:   {len(train_df):,} slices / {train_df['track'].nunique():,} tracks")
    print(f"Validation: {len(val_df):,} slices / {val_df['track'].nunique():,} tracks")
    print("   (grouped by track — no song appears on both sides)")
    print("   ⚠️ Expect val accuracy well below previous runs. Those numbers were leaking.\n")

    # ── 3b. Class weights — genre is 46x imbalanced, artist 28x ──
    # Without these the rare classes are never predicted; the loss is minimised by
    # ignoring them entirely.
    def weights_for(col, cols):
        counts = train_df[col].value_counts()
        n, k = len(train_df), len(counts)
        names = [c.split('_', 1)[1] for c in cols]
        return {i: n / (k * counts.get(name, 1)) for i, name in enumerate(names)}

    class_weights = {
        'genre_out':  weights_for('genre',       genre_cols),
        'artist_out': weights_for('artist_type', artist_cols),
        'mood_out':   weights_for('mood',        mood_cols),
    }
    for head, w in class_weights.items():
        vals = list(w.values())
        print(f"   class_weight[{head}]: min {min(vals):.2f}  max {max(vals):.2f}")
    print()

    # ── 4. tf.data pipelines ──────────────────────────────────────
    # Weights go on the training set only. Weighting validation would distort the
    # metrics we use to judge the run.
    train_ds = create_dataset(train_df, IMG_DIR, genre_cols, artist_cols, mood_cols,
                              batch_size=BATCH_SIZE, augment=True,
                              class_weights=class_weights)
    val_ds   = create_dataset(val_df,   IMG_DIR, genre_cols, artist_cols, mood_cols,
                              batch_size=BATCH_SIZE, augment=False)

    # ── 5. Save class taxonomy ────────────────────────────────────
    class_names = {
        'genre':  [c.replace('genre_',  '') for c in genre_cols],
        'artist': [c.replace('artist_', '') for c in artist_cols],
        'mood':   [c.replace('mood_',   '') for c in mood_cols]
    }
    if is_subset:
        print("   (subset run — classes.json NOT written)\n")
    else:
        with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
            json.dump(class_names, f, indent=4, ensure_ascii=False)
        print("💾 classes.json saved.\n")

    # ════════════════════════════════════════════════════════════
    # PHASE 1 — HEAD TRAINING (EfficientNet frozen)
    # Fast convergence without touching pretrained weights.
    # Establishes a strong baseline before fine-tuning.
    # ════════════════════════════════════════════════════════════
    print("=" * 55)
    print("PHASE 1 — Training heads only (EfficientNet frozen)")
    print("=" * 55)

    p1_initial_epoch = 0
    model = None

    if resume_state and resume_state["phase"] == 1:
        # Reload rather than rebuild+recompile: recompiling would discard the optimizer
        # moments and reset the LR schedule to step 0, which is most of what makes a
        # resume worth having.
        print(f"⏩ RESUME: faz 1, epoch {resume_state['epoch']}/{resume_state['total_epochs']}"
              f" (kayit: {resume_state['saved_at']})")
        model = tf.keras.models.load_model(CKPT_MODEL_PATH)
        p1_initial_epoch = resume_state["epoch"]

    if model is None and not (resume_state and resume_state["phase"] == 2):
        model = build_efficientnet_multihead(
            num_genres=len(genre_cols),
            num_artists=len(artist_cols),
            num_moods=len(mood_cols),
            freeze_base=True
        )
        model = compile_model(model, learning_rate=LR_PHASE1)
        model.summary()

    # Phase 1 callbacks — select on accuracy, not loss (see TrainingMonitor)
    p1_callbacks = make_callbacks((SUBSET_CKPT_PATH if is_subset else BEST_CKPT_PATH), patience=5,
                                  n_genre=len(genre_cols), n_artist=len(artist_cols),
                                  n_mood=len(mood_cols), reduce_lr=True,
                                  phase=1, total_epochs=epochs_p1)

    if resume_state and resume_state["phase"] == 2:
        print("⏩ RESUME: kesinti faz 2'deydi — faz 1 atlaniyor")
        history_p1 = None
    else:
        print("\n🏋️ Phase 1 Training...\n")
        history_p1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_p1,
            initial_epoch=p1_initial_epoch,
            callbacks=p1_callbacks
        )
        _summarise("PHASE 1", history_p1, len(genre_cols), len(artist_cols), len(mood_cols))

    # ════════════════════════════════════════════════════════════
    # PHASE 2 — FINE-TUNING (Top 30 EfficientNet layers unfrozen)
    # Very low LR to avoid catastrophic forgetting of ImageNet weights.
    # EarlyStopping patience=7 — stops well before 40 epochs.
    # ════════════════════════════════════════════════════════════
    print("=" * 55)
    print("PHASE 2 — Fine-tuning top EfficientNet layers")
    print("=" * 55)

    # Unfreeze in place rather than rebuilding. The old code built the architecture a
    # second time and copied weights across with set_weights(). That worked, but it made
    # Keras uniquify the second build's layer names ('global_average_pooling2d_1'), which
    # is what silently broke inference_engine's hardcoded get_layer() lookup for months.
    # Mutating the existing model keeps one set of names and skips the copy entirely.
    p2_initial_epoch = 0

    if resume_state and resume_state["phase"] == 2:
        # The checkpoint already holds the unfrozen model, its Adam moments and the
        # CosineDecay position (carried in optimizer.iterations). Recompiling here would
        # throw all three away, so we load and continue untouched.
        print(f"⏩ RESUME: faz 2, epoch {resume_state['epoch']}/{resume_state['total_epochs']}"
              f" (kayit: {resume_state['saved_at']})")
        model = tf.keras.models.load_model(CKPT_MODEL_PATH)
        p2_initial_epoch = resume_state["epoch"]
        base = next(l for l in model.layers if l.name == 'efficientnetb3')
        print(f"   {sum(1 for l in base.layers if l.trainable)} backbone katmani trainable "
              f"(checkpoint'ten), optimizer adimi {int(model.optimizer.iterations.numpy()):,}")
    else:
        base = next(l for l in model.layers if l.name == 'efficientnetb3')
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        trainable_ct = sum(1 for l in base.layers if l.trainable)
        print(f"   Unfroze the top {trainable_ct} backbone layers in place (no rebuild)")

        # Cosine decay: 1e-5 down to 1e-8 across all of phase 2. Predictable, unlike
        # ReduceLROnPlateau's plateau hunting.
        p2_steps = (len(train_df) // BATCH_SIZE) * epochs_p2
        p2_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-5,
            decay_steps=p2_steps,
            alpha=1e-8
        )
        model = compile_model(model, learning_rate=p2_lr)

    # Phase 2 callbacks — same accuracy-based selection, more patience
    p2_callbacks = make_callbacks((SUBSET_CKPT_PATH if is_subset else BEST_CKPT_PATH), patience=7,
                                  n_genre=len(genre_cols), n_artist=len(artist_cols),
                                  n_mood=len(mood_cols))

    print("\n🔥 Phase 2 Fine-tuning...\n")
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_p2,
        callbacks=p2_callbacks
    )

    # ── 9. Save final model ───────────────────────────────────────
    if is_subset:
        print("\n(subset run — model NOT saved; this was a diagnostic)")
    else:
        model.save(MODEL_SAVE_PATH)
        print(f"\n✅ Final model saved: {MODEL_SAVE_PATH}")

    # ── 10. Training summary ─────────────────────────────────────
    thermal_guard.stop_monitor()
    print(f"🌡️  {thermal_guard.summary()}")
    if thermal_guard.state["abort"]:
        print(f"   ⚠️ TERMAL IPTAL: {thermal_guard.state['reason']} — kosu eksik kaldi")

    _summarise("FINAL", history_p2, len(genre_cols), len(artist_cols), len(mood_cols))
    print("💡 Target: Genre & Mood > 90%, Artist > 75%")
    print("   Compare against the chance column above — a number that looks high but")
    print("   sits near chance means the model has not learned anything.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists",    type=int, default=0, help="diagnostic subset: top N artists")
    ap.add_argument("--per-artist", type=int, default=0, help="diagnostic subset: slices per artist")
    ap.add_argument("--epochs-p1",  type=int, default=None)
    ap.add_argument("--epochs-p2",  type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="son checkpoint'ten devam et (optimizer durumu + epoch + faz)")
    a = ap.parse_args()
    train_model(subset_artists=a.artists, subset_per_artist=a.per_artist,
                epochs_p1=a.epochs_p1, epochs_p2=a.epochs_p2, resume=a.resume)