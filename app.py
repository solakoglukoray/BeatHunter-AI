"""
BeatHunter AI - Frontend (Streamlit) v1.1 Enterprise Edition
Now equipped with Session State Management, User Authentication UI,
and JWT Bearer Token authorization for backend API requests.
"""

import streamlit as st
import requests
import os

# --- API CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="BeatHunter AI", page_icon="🎵", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# --- API HEALTH CHECK ---
try:
    requests.get(API_URL)
except requests.exceptions.ConnectionError:
    st.error("🚨 Red Alert: Backend API server is unreachable! Please run `python api.py` first.")
    st.stop()

# ==========================================
# 🔒 AUTHENTICATION & LOGIN SCREEN
# ==========================================
if not st.session_state.token:
    st.title("🔒 Welcome to BeatHunter AI Enterprise")
    st.markdown("Please log in or register to access the AI Inference Engine.")
    st.divider()
    
    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with auth_tab1:
        with st.form("login_form"):
            l_user = st.text_input("Username")
            l_pass = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Log In", type="primary")
            
            if submit_login:
                # OAuth2 form expects data (x-www-form-urlencoded), not json
                res = requests.post(f"{API_URL}/login", data={"username": l_user, "password": l_pass})
                if res.status_code == 200:
                    st.session_state.token = res.json().get("access_token")
                    st.session_state.username = l_user
                    st.rerun()
                else:
                    st.error("Invalid username or password!")

    with auth_tab2:
        with st.form("register_form"):
            r_user = st.text_input("Choose a Username")
            r_email = st.text_input("Email Address")
            r_pass = st.text_input("Create Password", type="password")
            submit_register = st.form_submit_button("Create Account", type="primary")
            
            if submit_register:
                # Pydantic schema expects JSON body
                res = requests.post(f"{API_URL}/register", json={"username": r_user, "email": r_email, "password": r_pass})
                if res.status_code == 200:
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error(f"Registration failed: {res.json().get('detail')}")
                    
    st.stop()


# ==========================================
# 🚀 MAIN APP (AUTHENTICATED AREA)
# ==========================================

# --- SIDEBAR (User Profile) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8302/8302196.png", width=100)
    st.success(f"**Logged in as:** {st.session_state.username}")
    
    if st.session_state.username.lower() == "admin":
        st.divider()
        show_admin = st.checkbox("👑 Enter Admin Panel")
        st.divider()
    else:
        show_admin = False

    if st.button("🚪 Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}

# ==========================================
# 👑 ADMIN DASHBOARD RENDER LOGIC
# ==========================================
if show_admin:
    st.title("👑 Executive Admin Dashboard")
    st.markdown("System overview, user management, and metrics.")
    st.divider()
    
    with st.spinner("Fetching secure user data..."):
        res = requests.get(f"{API_URL}/admin/users", headers=HEADERS)
        if res.status_code == 200:
            users_data = res.json()
            st.subheader("👥 Registered Users")
            st.dataframe(users_data, use_container_width=True)
        else:
            st.error(f"Access Denied or API Error: {res.text}")
            
    st.stop()


# ==========================================
# 🚀 MAIN APP
# ==========================================
st.title("🎵 BeatHunter AI : MLOps Edition")
st.markdown("AI-Powered Music Information Retrieval & Vectorized Beat Recommendation.")
st.divider()

# --- INPUT METHODS (TABS) ---
tab1, tab2 = st.tabs(["🔗 Analyze from YouTube", "📁 Upload Local File"])

api_response = None

with tab1:
    song_query = st.text_input("Search on YouTube (e.g., Travis Scott Type Beat):")
    if st.button("🚀 Fetch and Analyze from YouTube", type="primary"):
        if song_query:
            with st.spinner("📡 Sending secure request to API (Model is analyzing)..."):
                try:
                    res = requests.post(f"{API_URL}/analyze_youtube/", data={"query": song_query}, headers=HEADERS)
                    if res.status_code == 200:
                        api_response = res.json()
                    elif res.status_code == 401:
                        st.error("Session expired! Please log out and log in again.")
                    elif res.status_code == 429:
                        st.warning("⏳ **Whoa there, speedster!**\n\nTo ensure top performance for everyone, you can only analyze 3 songs per minute. Please take a short breather and try again.", icon="🛑")
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

with tab2:
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file:", type=["mp3", "wav"])
    is_beat = st.checkbox("This file is already a vocal-free Beat (Skip Demucs separation)")
    
    if st.button("📁 Analyze Uploaded File", type="primary"):
        if uploaded_file is not None:
            with st.spinner("📦 Uploading secure file to API and analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"skip_demucs": str(is_beat).lower()}
                    
                    res = requests.post(f"{API_URL}/analyze_file/", files=files, data=data, headers=HEADERS)
                    if res.status_code == 200:
                        api_response = res.json()
                    elif res.status_code == 401:
                        st.error("Session expired! Please log out and log in again.")
                    elif res.status_code == 429:
                        st.warning("⏳ **Whoa there, speedster!**\n\nTo ensure top performance for everyone, you can only analyze 3 songs per minute. Please take a short breather and try again.", icon="🛑")
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
    
    # 5-Column Metrics Display
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.info(f"🎧 **Style:**\n### {analysis['artist_type']} Type")
    c2.warning(f"🎭 **Mood:**\n### {analysis['mood']}")
    c3.success(f"🎹 **Genre:**\n### {analysis['genre']}")
    c4.error(f"⏱️ **Tempo:**\n### {analysis['bpm']} BPM")
    c5.info(f"🎵 **Key:**\n### {analysis['key']}") 
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_img, col_rec = st.columns([1, 1])
    with col_img:
        st.image("temp_inference/temp_spec.png", use_container_width=True, caption="Mel-Spectrogram")
        
        if api_response.get("source") == "local_file":
            audio_path = "temp_inference/api_uploaded_song.mp3"
        else:
            audio_path = "temp_inference/htdemucs/downloaded_song/no_vocals.mp3"
            
        st.audio(audio_path, format="audio/mp3")

    with col_rec:
        st.subheader("🧬 Vector-Based Recommendations")
        for i, rec in enumerate(recs):
            safe_score = min(100, max(0, int(rec['score'])))
            st.markdown(f"**{i+1}. {rec['title']}**")
            st.caption(f"🔗 [Listen on YouTube](https://youtu.be/{rec['id']})")
            st.progress(safe_score, text=f"Similarity Score: {rec['score']:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)