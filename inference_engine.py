"""
BeatHunter AI - Inference Engine
This is the core execution module. It handles audio downloading, vocal separation (Demucs),
BPM/Key extraction, Mel-Spectrogram generation, and applies a Majority Voting scheme 
using the trained Multi-Head CNN model to accurately predict Genre, Mood, and Sonic Brand.
It also orchestrates the vector database for cosine-similarity based recommendations.
"""

import os
import sys
import json
import subprocess
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from collections import Counter

# --- CONFIGURATION ---
TEMP_FOLDER = "temp_inference"
MODEL_PATH = "models/beathunter_model.keras"
CLASSES_PATH = "models/classes.json"

PROCESSED_CSV = "dataset_processed/metadata_processed.csv"
IMG_DIR = "dataset_processed/images"
VECTOR_DB_PATH = "models/vector_db.npy"
VECTOR_META_PATH = "models/vector_meta.csv"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  # seconds
# The number of uniform segments to extract from the track for Majority Voting
NUM_SEGMENTS = 8


def extract_bpm_and_key(y, sr):
    """Analyzes the audio signal to extract the Tempo (BPM) and Musical Key."""
    print("🥁 Analyzing BPM and Musical Key...")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else round(tempo)

    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)

    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    maj_correlations = [stats.pearsonr(mean_chroma, np.roll(maj_profile, i))[0] for i in range(12)]
    min_correlations = [stats.pearsonr(mean_chroma, np.roll(min_profile, i))[0] for i in range(12)]

    best_maj_idx = np.argmax(maj_correlations)
    best_min_idx = np.argmax(min_correlations)

    if maj_correlations[best_maj_idx] > min_correlations[best_min_idx]:
        key = f"{chroma_to_key[best_maj_idx]} Major"
    else:
        key = f"{chroma_to_key[best_min_idx]} Minor"

    return bpm, key


def audio_to_spectrogram_array(y_segment, sr):
    """Converts a 3-second audio segment into a pre-processed numpy array ready for the CNN."""
    mels = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    # Save to a temporary file and reload to ensure consistency with the training pipeline
    tmp_path = os.path.join(TEMP_FOLDER, "_tmp_seg.png")
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(mels_db, sr=sr)
    plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = tf.keras.utils.load_img(tmp_path, target_size=(224, 224))
    return np.expand_dims(img_to_array(img) / 255.0, 0)


def predict_with_majority_voting(model, y_full, sr, class_names):
    """
    Extracts NUM_SEGMENTS uniformly across the track, runs inference on each,
    and determines the final labels via Majority Voting.
    
    Why? Skipping the intro (often misleading) and evaluating multiple sections 
    provides a highly robust representation of the track's true sonic fingerprint.
    """
    samples_per_segment = int(SEGMENT_DURATION * sr)
    total_samples = len(y_full)

    # Skip the first 30 seconds (intro) and the last 30 seconds (outro/fade-out)
    skip_start = min(int(30 * sr), total_samples // 4)
    skip_end   = max(total_samples - int(30 * sr), total_samples * 3 // 4)
    usable_audio = y_full[skip_start:skip_end]

    if len(usable_audio) < samples_per_segment:
        # Fallback for very short tracks: use the entire available audio
        usable_audio = y_full

    # Calculate uniform starting points for the segments across the usable audio
    max_start = len(usable_audio) - samples_per_segment
    if max_start <= 0:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, NUM_SEGMENTS, dtype=int)

    genre_votes  = []
    artist_votes = []
    mood_votes   = []
    all_vectors  = []

    # Create a sub-model to extract the 128-dimensional embedding vector
    embed_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('global_average_pooling2d').output
    )

    print(f"🔍 Analyzing {len(starts)} uniform segments (Majority Voting Protocol)...")

    for i, start in enumerate(starts):
        segment = usable_audio[start: start + samples_per_segment]
        if len(segment) < samples_per_segment:
            continue

        img_array = audio_to_spectrogram_array(segment, sr)

        # Predict multi-head labels
        preds = model.predict(img_array, verbose=0)
        genre_votes.append(np.argmax(preds[0][0]))
        artist_votes.append(np.argmax(preds[1][0]))
        mood_votes.append(np.argmax(preds[2][0]))

        # Extract latent vector representation
        vec = embed_model.predict(img_array, verbose=0)
        all_vectors.append(vec[0])

    # Determine the most frequent prediction for each label (Majority Vote)
    predicted_genre  = class_names['genre'] [Counter(genre_votes).most_common(1)[0][0]]
    predicted_artist = class_names['artist'][Counter(artist_votes).most_common(1)[0][0]]
    predicted_mood   = class_names['mood']  [Counter(mood_votes).most_common(1)[0][0]]

    # Calculate the mean vector across all segments for a highly stable track embedding
    mean_vector = np.mean(all_vectors, axis=0, keepdims=True)

    return predicted_genre, predicted_artist, predicted_mood, mean_vector


def download_youtube_audio(song_name):
    """Downloads audio from YouTube using yt-dlp."""
    import yt_dlp
    print(f"🔍 Searching YouTube for: {song_name}")
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{TEMP_FOLDER}/downloaded_song.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'quiet': True, 'noplaylist': True, 'default_search': 'ytsearch1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([song_name])
    return os.path.join(TEMP_FOLDER, "downloaded_song.mp3")


def separate_audio(input_path):
    """Uses the Demucs AI model to isolate the instrumental beat from vocals."""
    print("✂️ AI Source Separation in progress (Demucs - Isolating Beat)...")
    command = (
        f'"{sys.executable}" -m demucs -n htdemucs --two-stems=vocals '
        f'--mp3 --mp3-bitrate 320 "{input_path}" -o "{TEMP_FOLDER}"'
    )
    subprocess.run(command, shell=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    beat_path = os.path.join(TEMP_FOLDER, "htdemucs", base_name, "no_vocals.mp3")
    if os.path.exists(beat_path):
        return beat_path
    else:
        raise Exception("Source separation failed.")


def build_vector_database():
    """Extracts and saves latent vectors for the entire dataset in optimized batches."""
    if os.path.exists(VECTOR_DB_PATH):
        return

    print("🧬 Building Vector Database (Initial run only)...")
    model = tf.keras.models.load_model(MODEL_PATH)
    embed_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('global_average_pooling2d').output
    )

    df = pd.read_csv(PROCESSED_CSV)
    vectors = []
    BATCH = 256  # Process 256 images per batch instead of individual loops for extreme speed

    img_paths = [os.path.join(IMG_DIR, f) for f in df['filename']]

    def load_img_tf(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return tf.cast(img, tf.float32) / 255.0

    # Build an optimized TensorFlow data pipeline using AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    ds = ds.map(load_img_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    total = len(img_paths)
    processed = 0
    for batch in ds:
        vecs = embed_model.predict(batch, verbose=0)
        vectors.extend(vecs)
        processed += len(batch)
        if processed % 10000 == 0:
            print(f"   ⏳ Processed {processed:,} / {total:,} images...")

    np.save(VECTOR_DB_PATH, np.array(vectors))
    df.to_csv(VECTOR_META_PATH, index=False)
    print("✅ Vector Database is fully operational!")


def analyze_beat(audio_path):
    """
    Main inference controller.
    Executes Majority Voting inference and queries the Vector DB for similarity matching.
    """
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    # Load the entire audio file
    print("🎵 Loading audio matrix...")
    y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Extract global BPM and Key
    bpm, key = extract_bpm_and_key(y_full, sr)

    # Execute Majority Voting inference
    predicted_genre, predicted_artist, predicted_mood, mean_vector = \
        predict_with_majority_voting(model, y_full, sr, class_names)

    formatted_output = f"{predicted_artist} Type / {predicted_mood} / {predicted_genre}"

    # Generate and save the mid-segment spectrogram for UI visualization
    mid_start = len(y_full) // 2
    mid_seg = y_full[mid_start: mid_start + int(SEGMENT_DURATION * sr)]
    mels = librosa.feature.melspectrogram(y=mid_seg, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(mels_db, sr=sr)
    plt.savefig(f"{TEMP_FOLDER}/temp_spec.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Query the Vector Database for Cosine Similarity matches
    build_vector_database()
    db_vectors = np.load(VECTOR_DB_PATH)
    db_meta    = pd.read_csv(VECTOR_META_PATH)

    similarities = cosine_similarity(mean_vector, db_vectors)[0]
    top_indices  = similarities.argsort()[-10:][::-1]  # Fetch top 10 to ensure 5 unique matches

    recommendations = []
    seen = set()
    for idx in top_indices:
        raw_name = db_meta.iloc[idx]['filename'].split('_slice')[0]
        if raw_name in seen:
            continue
        score  = float(similarities[idx] * 100)
        artist = db_meta.iloc[idx]['artist_type']
        mood   = db_meta.iloc[idx]['mood']
        genre  = db_meta.iloc[idx]['genre']
        recommendations.append({
            "id":    raw_name,
            "title": f"{artist} / {mood} / {genre}",
            "score": score
        })
        seen.add(raw_name)
        if len(recommendations) == 5:
            break

    return formatted_output, recommendations, int(bpm), str(key)


if __name__ == "__main__":
    song_input = input("🎵 Enter the target song query (e.g., Metro Boomin Type Beat): ")
    try:
        mp3_path  = download_youtube_audio(song_input)
        beat_path = separate_audio(mp3_path)

        print("\n🧠 Initiating Deep Neural Analysis...")
        formatted_result, recommendations, bpm, key = analyze_beat(beat_path)

        print("\n" + "=" * 50)
        print(f"🎤 Target Track: {song_input}")
        print(f"🎯 Inference: {formatted_result}")
        print(f"⏱️ BPM: {bpm} | 🎵 Key: {key}")
        print("-" * 50)
        print("🎧 TOP SIMILAR BEAT RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['title']} — Similarity Score: {rec['score']:.1f}%")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Execution Error: {e}")