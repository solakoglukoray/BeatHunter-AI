"""
BeatHunter AI - Multi-Task Deep Learning Model Trainer
This module defines a custom multi-head Convolutional Neural Network (CNN) architecture 
designed specifically for audio spectrograms. It utilizes highly optimized tf.data pipelines,
advanced data augmentation (SpecAugment), and strict regularization techniques to 
prevent overfitting on complex, noisy YouTube metadata.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
CSV_PATH = "dataset_processed/metadata_processed.csv"
IMG_DIR = "dataset_processed/images"
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "beathunter_model.keras")
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, "classes.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64       # Increased from 32 to 64 for more stable gradients on a 200K dataset
EPOCHS = 60           # Max epochs; EarlyStopping will likely halt training sooner


# ============================================================
# HIGH-PERFORMANCE tf.data PIPELINE
# Replaces the legacy Sequence Generator. Processes up to 200K 
# images in parallel, preventing GPU starvation (3-4x faster).
# ============================================================
def create_dataset(df, img_dir, genre_cols, artist_cols, mood_cols,
                   batch_size=64, augment=False):

    img_paths = [os.path.join(img_dir, f) for f in df['filename'].values]
    genre_labels  = df[genre_cols].values.astype(np.float32)
    artist_labels = df[artist_cols].values.astype(np.float32)
    mood_labels   = df[mood_cols].values.astype(np.float32)

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def augment_image(img):
        # Temporal augmentation: Randomly flip along the time axis
        img = tf.image.random_flip_left_right(img)
        # Volume simulation: Random brightness adjustments
        img = tf.image.random_brightness(img, max_delta=0.15)
        # Dynamic range simulation: Random contrast adjustments
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        # Ensure pixel values remain strictly within [0, 1] bounds
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img

    # SpecAugment: Frequency and Time masking applied directly on the TF graph
    def spec_augment(img):
        # Frequency Masking (Horizontal bands)
        # Min set to 1 to avoid boundary overlap issues
        freq_mask_size = tf.random.uniform([], 1, 25, dtype=tf.int32)
        max_freq_start = tf.maximum(224 - freq_mask_size, 1)
        freq_start     = tf.random.uniform([], 0, max_freq_start, dtype=tf.int32)
        freq_end       = tf.minimum(freq_start + freq_mask_size, 224)
        freq_mask = tf.concat([
            tf.ones ([freq_start, 224, 3]),
            tf.zeros([freq_end - freq_start, 224, 3]),
            tf.ones ([224 - freq_end, 224, 3])
        ], axis=0)
        img = img * freq_mask

        # Time Masking (Vertical bands)
        time_mask_size = tf.random.uniform([], 1, 35, dtype=tf.int32)
        max_time_start = tf.maximum(224 - time_mask_size, 1)
        time_start     = tf.random.uniform([], 0, max_time_start, dtype=tf.int32)
        time_end       = tf.minimum(time_start + time_mask_size, 224)
        time_mask = tf.concat([
            tf.ones ([224, time_start, 3]),
            tf.zeros([224, time_end - time_start, 3]),
            tf.ones ([224, 224 - time_end, 3])
        ], axis=1)
        img = img * time_mask

        return img

    # Construct the dataset from tensor slices
    path_ds   = tf.data.Dataset.from_tensor_slices(img_paths)
    genre_ds  = tf.data.Dataset.from_tensor_slices(genre_labels)
    artist_ds = tf.data.Dataset.from_tensor_slices(artist_labels)
    mood_ds   = tf.data.Dataset.from_tensor_slices(mood_labels)

    label_ds = tf.data.Dataset.zip((genre_ds, artist_ds, mood_ds))
    label_ds = label_ds.map(lambda g, a, m: {'genre_out': g, 'artist_out': a, 'mood_out': m})

    ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle BEFORE mapping to ensure different buffer orders per epoch
    if augment:
        ds = ds.shuffle(buffer_size=8000, reshuffle_each_iteration=True)

    # Parallel image loading
    ds = ds.map(
        lambda p, y: (load_image(p), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        ds = ds.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(
            lambda x, y: (spec_augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# CUSTOM ARCHITECTURE
# - 4 Convolutional blocks to handle the complexity of 200K samples.
# - Dedicated dense layers injected before each prediction head.
# - Progressive Dropout (0.2 → 0.3 → 0.4 → 0.5) to combat overfitting.
# - L2 Regularization tuned down to 0.001 (0.01 was too aggressive).
# ============================================================
def build_multihead_model(num_genres, num_artists, num_moods):

    inputs = layers.Input(shape=(224, 224, 3))

    # BLOCK 1: Low-frequency extraction (Bass lines, Kick drums)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # BLOCK 2: Mid-low frequency extraction (Snares, fundamental melodies)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # BLOCK 3: Mid-high frequency extraction (Lead synths, vocal remnants)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # BLOCK 4: High-frequency extraction (Hi-hats, cymbals, air/presence)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # SHARED TRUNK: Latent Audio DNA representation
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    shared = layers.Dense(256, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(x)

    # HEAD 1: GENRE (~30 classes)
    g = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(shared)
    g = layers.Dropout(0.3)(g)
    genre_out = layers.Dense(num_genres, activation='softmax', name='genre_out')(g)

    # HEAD 2: ARTIST / SONIC BRAND (~80+ classes, highest complexity)
    a = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(shared)
    a = layers.Dropout(0.3)(a)
    a = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(a)
    a = layers.Dropout(0.2)(a)
    artist_out = layers.Dense(num_artists, activation='softmax', name='artist_out')(a)

    # HEAD 3: MOOD (6 classes, lowest complexity)
    m = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(shared)
    m = layers.Dropout(0.2)(m)
    mood_out = layers.Dense(num_moods, activation='softmax', name='mood_out')(m)

    return models.Model(inputs=inputs, outputs=[genre_out, artist_out, mood_out])


def train_model():
    print("🚀 Initializing BeatHunter Multi-Head Model Training Sequence...\n")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load and inspect the dataset
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Total spectrograms loaded: {len(df):,}")
    print(f"   Unique Genre classes:  {df['genre'].nunique()}")
    print(f"   Unique Artist classes: {df['artist_type'].nunique()}")
    print(f"   Unique Mood classes:   {df['mood'].nunique()}\n")

    # 2. One-Hot Encoding (Applied globally prior to splitting)
    df_genre  = pd.get_dummies(df['genre'],       prefix='genre')
    df_artist = pd.get_dummies(df['artist_type'], prefix='artist')
    df_mood   = pd.get_dummies(df['mood'],        prefix='mood')

    genre_cols  = list(df_genre.columns)
    artist_cols = list(df_artist.columns)
    mood_cols   = list(df_mood.columns)

    df = pd.concat([df, df_genre, df_artist, df_mood], axis=1)

    # 3. Train / Validation Split
    # Stratified by genre to ensure proportional representation in both sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42,
        shuffle=True, stratify=df['genre']
    )
    print(f"Training Set: {len(train_df):,} | Validation Set: {len(val_df):,}\n")

    # 4. Construct high-performance tf.data pipelines
    train_ds = create_dataset(train_df, IMG_DIR, genre_cols, artist_cols, mood_cols,
                               batch_size=BATCH_SIZE, augment=True)
    val_ds   = create_dataset(val_df,   IMG_DIR, genre_cols, artist_cols, mood_cols,
                               batch_size=BATCH_SIZE, augment=False)

    # 5. Extract and persist class taxonomies for inference mapping
    class_names = {
        'genre':  [c.replace('genre_',  '') for c in genre_cols],
        'artist': [c.replace('artist_', '') for c in artist_cols],
        'mood':   [c.replace('mood_',   '') for c in mood_cols]
    }
    with open(CLASSES_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)
    print("💾 classes.json successfully saved.")

    # 6. Instantiate the Multi-Head Architecture
    model = build_multihead_model(
        num_genres=len(genre_cols),
        num_artists=len(artist_cols),
        num_moods=len(mood_cols)
    )

    # 7. Model Compilation
    # Label smoothing acts as regularization against noisy YouTube metadata.
    # Higher smoothing is applied to the Artist head due to its high variance.
    genre_loss  = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    artist_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15)
    mood_loss   = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'genre_out':  genre_loss,
            'artist_out': artist_loss,
            'mood_out':   mood_loss
        },
        loss_weights={
            'genre_out':  1.0,
            'artist_out': 0.4,  # Penalize artist loss influence due to 80+ noisy classes
            'mood_out':   0.8
        },
        metrics={
            'genre_out':  ['accuracy'],
            'artist_out': ['accuracy'],
            'mood_out':   ['accuracy']
        }
    )

    model.summary()

    # 8. Training Callbacks (Dynamic LR Reduction & Early Stopping based on Genre accuracy)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_genre_out_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_genre_out_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    print("\n🏋️ Commencing Neural Network Training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stop]
    )

    # 9. Persist the trained model weights and architecture
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model successfully saved to: {MODEL_SAVE_PATH}")

    # 10. Training Summary & Performance Metrics
    best_genre  = max(history.history.get('val_genre_out_accuracy',  [0])) * 100
    best_mood   = max(history.history.get('val_mood_out_accuracy',   [0])) * 100
    best_artist = max(history.history.get('val_artist_out_accuracy', [0])) * 100

    print("\n📊 PEAK VALIDATION METRICS:")
    print(f"   Genre  Accuracy: {best_genre:.1f}%")
    print(f"   Mood   Accuracy: {best_mood:.1f}%")
    print(f"   Artist Accuracy: {best_artist:.1f}%")
    print("\n💡 Readiness Indicator: If Genre and Mood > 70%, the model is production-ready.")
    print("   Note: Lower accuracy on the Artist head is expected due to the 80+ class complexity.")


if __name__ == "__main__":
    train_model()