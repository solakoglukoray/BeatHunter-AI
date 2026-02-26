"""
BeatHunter AI - Frontend (Streamlit)
This module acts as the user interface, communicating with the FastAPI backend
to upload files, send queries, and display the ML analysis results.
"""

import streamlit as st
import requests
import os

# --- API CONFIGURATION ---
# The URL of the backend inference engine (configurable via environment variables)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="BeatHunter AI", page_icon="🎵", layout="wide")

st.title("🎵 BeatHunter AI : MLOps Edition")
st.markdown("AI-Powered Music Information Retrieval & Vectorized Beat Recommendation (Microservice Architecture).")
st.divider()

# --- API HEALTH CHECK ---
try:
    # Ping the API to ensure the backend is running
    requests.get(API_URL)
except requests.exceptions.ConnectionError:
    st.error("🚨 Red Alert: Backend API server is unreachable! Please run `python api.py` first.")
    st.stop()  # Halt UI execution if the API is offline

# --- INPUT METHODS (TABS) ---
tab1, tab2 = st.tabs(["🔗 Analyze from YouTube", "📁 Upload Local File"])

# Initialize a variable to store the API JSON response
api_response = None

with tab1:
    song_query = st.text_input("Search on YouTube (e.g., Travis Scott Type Beat):")
    if st.button("🚀 Fetch and Analyze from YouTube", type="primary"):
        if song_query:
            with st.spinner("📡 Sending request to API (Model is analyzing)..."):
                try:
                    # Send POST request to the YouTube endpoint
                    res = requests.post(f"{API_URL}/analyze_youtube/", data={"query": song_query})
                    if res.status_code == 200:
                        api_response = res.json()
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

with tab2:
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file:", type=["mp3", "wav"])
    is_beat = st.checkbox("This file is already a vocal-free Beat (Skip Demucs separation)")
    
    if st.button("📁 Analyze Uploaded File", type="primary"):
        if uploaded_file is not None:
            with st.spinner("📦 Uploading file to API and analyzing..."):
                try:
                    # Prepare the file payload and form data for the API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"skip_demucs": str(is_beat).lower()}
                    
                    # Send POST request to the local file endpoint
                    res = requests.post(f"{API_URL}/analyze_file/", files=files, data=data)
                    if res.status_code == 200:
                        api_response = res.json()
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please upload a file first!")

# --- MAIN PROCESS (DISPLAYING API JSON RESPONSE) ---
if api_response and api_response.get("status") == "success":
    analysis = api_response["analysis"]
    recs = api_response["recommendations"]
    
    st.divider()
    st.header("🎯 Analysis Report")
    
    # 5-Column Metrics Display (Populated from JSON data)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.info(f"🎧 **Style:**\n### {analysis['artist_type']} Type")
    c2.warning(f"🎭 **Mood:**\n### {analysis['mood']}")
    c3.success(f"🎹 **Genre:**\n### {analysis['genre']}")
    c4.error(f"⏱️ **Tempo:**\n### {analysis['bpm']} BPM")
    c5.info(f"🎵 **Key:**\n### {analysis['key']}") 
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_img, col_rec = st.columns([1, 1])
    with col_img:
        # Note: Accessing local paths directly assumes the frontend and backend 
        # are running on the same machine and sharing the filesystem.
        st.image("temp_inference/temp_spec.png", use_container_width=True, caption="Mel-Spectrogram")
        
        # Determine the correct audio path based on the input source
        if api_response.get("source") == "local_file":
            audio_path = "temp_inference/api_uploaded_song.mp3"
        else:
            audio_path = "temp_inference/htdemucs/downloaded_song/no_vocals.mp3"
            
        st.audio(audio_path, format="audio/mp3")

    with col_rec:
        st.subheader("🧬 Vector-Based Recommendations")
        for i, rec in enumerate(recs):
            # Ensure the progress bar score is strictly between 0 and 100
            safe_score = min(100, max(0, int(rec['score'])))
            st.markdown(f"**{i+1}. {rec['title']}**")
            st.caption(f"🔗 [Listen on YouTube](https://youtu.be/{rec['id']})")
            st.progress(safe_score, text=f"Similarity Score: {rec['score']:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)