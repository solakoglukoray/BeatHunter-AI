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

5. Smarter Callbacks
   - EarlyStopping patience reduced from 10 → 7 (stops sooner if stuck).
   - ReduceLROnPlateau patience reduced from 4 → 3.
   - Monitors val_loss instead of val_genre_out_accuracy (more stable signal).
   - ModelCheckpoint saves only the best model during training.

6. Spectrogram parameter alignment
   - IMG_SIZE = (224, 224) — matches n_mels=256 output resized in inference_engine.py
─────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import GroupShuffleSplit

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


def train_model():
    print("🚀 BeatHunter AI — Model Training v2.0 (EfficientNetB3 + Mixed Precision)\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load dataset ──────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
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

    model = build_efficientnet_multihead(
        num_genres=len(genre_cols),
        num_artists=len(artist_cols),
        num_moods=len(mood_cols),
        freeze_base=True
    )

    model = compile_model(model, learning_rate=LR_PHASE1)
    model.summary()

    # Phase 1 callbacks — generous patience, let heads converge fully
    p1_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='min',
            verbose=1
        ),
        ModelCheckpoint(
            filepath=BEST_CKPT_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]

    print("\n🏋️ Phase 1 Training...\n")
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1,
        callbacks=p1_callbacks
    )

    p1_genre  = max(history_p1.history.get('val_genre_out_accuracy',  [0])) * 100
    p1_mood   = max(history_p1.history.get('val_mood_out_accuracy',   [0])) * 100
    p1_artist = max(history_p1.history.get('val_artist_out_accuracy', [0])) * 100
    print(f"\n📊 Phase 1 Results — Genre: {p1_genre:.1f}% | Mood: {p1_mood:.1f}% | Artist: {p1_artist:.1f}%\n")

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
    base = next(l for l in model.layers if l.name == 'efficientnetb3')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    trainable_ct = sum(1 for l in base.layers if l.trainable)
    print(f"   Unfroze the top {trainable_ct} backbone layers in place (no rebuild)")

    # Cosine decay: 1e-5 down to 1e-8 across all of phase 2. Predictable, unlike
    # ReduceLROnPlateau's plateau hunting.
    p2_steps = (len(train_df) // BATCH_SIZE) * EPOCHS_P2
    p2_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=p2_steps,
        alpha=1e-8
    )
    model = compile_model(model, learning_rate=p2_lr)

    p2_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        ModelCheckpoint(
            filepath=BEST_CKPT_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]

    print("\n🔥 Phase 2 Fine-tuning...\n")
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P2,
        callbacks=p2_callbacks
    )

    # ── 9. Save final model ───────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Final model saved: {MODEL_SAVE_PATH}")

    # ── 10. Training summary ─────────────────────────────────────
    best_genre  = max(history_p2.history.get('val_genre_out_accuracy',  [0])) * 100
    best_mood   = max(history_p2.history.get('val_mood_out_accuracy',   [0])) * 100
    best_artist = max(history_p2.history.get('val_artist_out_accuracy', [0])) * 100

    print("\n📊 FINAL PEAK VALIDATION METRICS:")
    print(f"   Genre  Accuracy: {best_genre:.1f}%")
    print(f"   Mood   Accuracy: {best_mood:.1f}%")
    print(f"   Artist Accuracy: {best_artist:.1f}%")
    print("\n💡 Target: Genre & Mood > 90%, Artist > 75%")
    print("   If Artist is below 70%, consider reducing to 100 artists in config.py.")


if __name__ == "__main__":
    train_model()