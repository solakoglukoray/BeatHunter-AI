"""
BeatHunter AI - Audio Signal Processor
This module reads the downloaded raw audio files, slices them into uniform segments,
extracts their Mel-Spectrograms using Librosa, and saves them as images to be 
used for CNN model training. It also generates the corresponding metadata.
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
AUDIO_DIR = "dataset/audio"
METADATA_FILE = "dataset/metadata.csv"

PROCESSED_DIR = "dataset_processed"
IMG_DIR = os.path.join(PROCESSED_DIR, "images")
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "metadata_processed.csv")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_spectrogram(y, sr, save_path):
    """
    Converts an audio time-series array into a Mel-Spectrogram 
    and saves it as an image file without axes or borders.
    """
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(mels_db, sr=sr, hop_length=512)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_v2_data():
    """Main function to process the dataset and generate spectrogram images."""
    print("🚀 Starting Audio-to-Image Conversion Process...")
    
    # Create necessary output directories
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        
    # Read the raw metadata (Automatically drop duplicated rows to ensure data integrity)
    df = pd.read_csv(METADATA_FILE).drop_duplicates()
    print(f"📊 Unique tracks to process: {len(df)}")
    
    # List to hold our new processed dataset metadata
    processed_data = []

    for index, row in df.iterrows():
        audio_filename = row['filename']
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        # Check if the downloaded audio file actually exists on disk
        if not os.path.exists(audio_path):
            print(f"   ⚠️ File not found, skipping: {audio_filename}")
            continue
            
        print(f"   🎵 Processing: {audio_filename}")
        
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            num_segments = int(len(y) / SAMPLES_PER_TRACK)
            
            # Slice the track into uniform 3-second segments
            for i in range(num_segments):
                start = i * SAMPLES_PER_TRACK
                end = start + SAMPLES_PER_TRACK
                slice_y = y[start:end]
                
                # Only process segments that are exactly the required duration
                if len(slice_y) == SAMPLES_PER_TRACK:
                    # Generate the image filename (e.g., nxhOctrjPEk_slice_0.png)
                    base_name = os.path.splitext(audio_filename)[0]
                    img_filename = f"{base_name}_slice_{i}.png"
                    img_path = os.path.join(IMG_DIR, img_filename)
                    
                    # Save the spectrogram image
                    save_spectrogram(slice_y, sr, img_path)
                    
                    # Create the row to be appended to the new metadata
                    processed_data.append({
                        "filename": img_filename,
                        "genre": row['genre'],
                        "artist_type": row['artist_type'],
                        "mood": row['mood']
                    })
        except Exception as e:
            print(f"   ❌ Error processing ({audio_filename}): {e}")

    # Save the new dataset metadata as a CSV
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_csv(PROCESSED_CSV, index=False)
    
    print("\n✅ ALL CONVERSIONS COMPLETED SUCCESSFULLY!")
    print(f"📂 Images saved to: {IMG_DIR}")
    print(f"📄 New Metadata saved to: {PROCESSED_CSV}")
    print(f"🧬 Total generated training samples (Images): {len(df_processed)}")

if __name__ == "__main__":
    process_v2_data()