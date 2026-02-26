"""
BeatHunter AI - Data Validator
This module performs Data Quality Assurance (QA) by scanning the dataset 
for corrupted or unreadable audio files and automatically removing them 
to prevent training pipeline crashes.
"""

import os
import librosa
import config

def validate_data():
    """
    Checks the integrity of downloaded audio files.
    Automatically deletes any corrupted files that cannot be processed.
    """
    dataset_path = config.DATASET_ROOT
    total_files = 0
    corrupted_files = 0
    
    print(f"🕵️ Data Validation Initiated: Scanning {dataset_path}...\n")

    # Traverse all directories and files in the dataset (os.walk)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    # Attempt to load the audio file
                    # sr=None: Preserve the original sampling rate
                    # duration=1: Only read the first second to optimize processing speed
                    librosa.load(file_path, sr=None, duration=1)
                    
                    # If no exception is raised up to this point, the file is intact.
                    # print(f"✅ Intact: {file}") # Commented out to prevent console spam
                    
                except Exception as e:
                    print(f"❌ CORRUPTED FILE DETECTED: {file}")
                    print(f"   Reason: {e}")
                    
                    # Delete the corrupted file
                    try:
                        os.remove(file_path)
                        print("   🗑️ File deleted.")
                        corrupted_files += 1
                    except:
                        print("   ⚠️ Failed to delete file (Check folder permissions).")

    print("-" * 30)
    print("📊 VALIDATION REPORT")
    print(f"Total Files Scanned: {total_files}")
    print(f"Corrupted Files Deleted: {corrupted_files}")
    
    # Calculate and display the data integrity rate
    if total_files > 0:
        integrity_rate = ((total_files - corrupted_files) / total_files) * 100
        print(f"Data Integrity Rate: {integrity_rate:.1f}%")
    else:
        print("No Data Found")
    print("-" * 30)

if __name__ == "__main__":
    validate_data()