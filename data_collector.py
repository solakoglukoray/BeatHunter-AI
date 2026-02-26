"""
BeatHunter AI - Autonomous Data Collector
This module acts as an automated data mining pipeline. It searches YouTube based 
on the taxonomy defined in config.py, downloads audio, converts it to MP3 via FFmpeg,
and dynamically generates a labeled dataset (CSV metadata) for model training.
"""

import os
import csv
import uuid
import yt_dlp
import config

# Global variable to track the active category during the asynchronous yt-dlp hook execution
CURRENT_TARGET = None

def initialize_dataset():
    """
    Initializes the dataset directory structure and creates the master metadata CSV 
    file with appropriate headers if it doesn't already exist.
    """
    if not os.path.exists(config.AUDIO_DIR):
        os.makedirs(config.AUDIO_DIR)
        print(f"📁 Dataset directory created: {config.AUDIO_DIR}")

    if not os.path.exists(config.METADATA_FILE):
        with open(config.METADATA_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Define the CSV schema
            writer.writerow(["filename", "genre", "artist_type", "mood"])
        print(f"📄 Metadata schema initialized: {config.METADATA_FILE}")

def download_hook(d):
    """
    A callback hook triggered by yt-dlp upon the successful download of an audio file.
    It automatically appends the file details and its corresponding labels to the metadata CSV.
    """
    if d['status'] == 'finished':
        # Extract the base filename and ensure the extension is recorded as .mp3 
        # (since FFmpeg will process it immediately after download)
        filename = os.path.basename(d['filename'])
        filename_mp3 = os.path.splitext(filename)[0] + ".mp3"
        
        # Append the successfully downloaded and processed file to the dataset metadata
        with open(config.METADATA_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                filename_mp3, 
                CURRENT_TARGET['genre'], 
                CURRENT_TARGET['artist'], 
                CURRENT_TARGET['mood']
            ])
        print(f"   ✅ Downloaded & Labeled: {filename_mp3}")

def fetch_target_audio(target):
    """
    Executes a YouTube search query for a specific taxonomy combination, downloads 
    the top results, and triggers the post-processing pipeline.
    
    Args:
        target (dict): A dictionary containing 'artist', 'mood', and 'genre' keys.
    """
    global CURRENT_TARGET
    CURRENT_TARGET = target
    
    # Construct the search string specifically optimized for Type Beat discovery
    search_term = f"{target['artist']} {target['mood']} {target['genre']} Type Beat"
    
    # We fetch 2x the required amount to account for dead links, errors, or short/long videos
    search_query = f"ytsearch{config.FILES_PER_TARGET * 2}:{search_term}"
    print(f"\n🔍 Mining Data For: {search_term}")

    # yt-dlp configuration payload
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{config.AUDIO_DIR}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192', # Standardize audio bitrate for consistent ML input
        }],
        'noplaylist': True,
        'ignoreerrors': True,
        'max_downloads': config.FILES_PER_TARGET,
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [download_hook],
        # Only accept audio tracks between 1 minute and 10 minutes long
        'match_filter': yt_dlp.utils.match_filter_func("duration > 60 & duration < 600"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([search_query])
        except yt_dlp.utils.MaxDownloadsReached:
            print("   ✅ Target quota reached. Moving to the next category...")
        except Exception as e:
            print(f"   ❌ Unexpected pipeline error: {e}")

def main():
    """Main execution function for the Data Mining Pipeline."""
    print("🚀 BeatHunter Data Mining Pipeline Initiated...")
    initialize_dataset()
    
    # Iterate through the curated taxonomy and execute the mining job for each
    for target in config.TARGETS:
        fetch_target_audio(target)
        
    print("\n🎉 DATA COLLECTION AND LABELING PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()