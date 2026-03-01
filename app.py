"""
BeatHunter AI - Frontend (Streamlit) v1.2 Enterprise Edition

Changes in v1.2:
- Audio preview now uses beat_path from API response (works with UUID temp folders)
- asyncio.run() replaced with nest_asyncio to fix Streamlit event loop conflict
- WebSocket correctly skipped on cache hits
- Silent token refresh integrated into both tabs
"""

import streamlit as st
import requests
import os
import time
import asyncio
import websockets
import json
import nest_asyncio

# Fix: Streamlit has its own event loop — nest_asyncio allows asyncio.run() inside it
nest_asyncio.apply()

# --- API CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="BeatHunter AI", page_icon="🎵", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "token" not in st.session_state:
    st.session_state.token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
if "username" not in st.session_state:
    st.session_state.username = None


async def wait_for_analysis_ws(task_id):
    """
    Connects to the FastAPI WebSocket endpoint for a given task_id
    and blocks until the worker sends back the final result JSON.
    """
    ws_url = API_URL.replace("http", "ws") + f"/ws/{task_id}"
    try:
        async with websockets.connect(ws_url) as websocket:
            message = await websocket.recv()
            return json.loads(message)
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# --- API HEALTH CHECK ---
try:
    requests.get(API_URL, timeout=3)
except requests.exceptions.ConnectionError:
    st.error("🚨 Red Alert: Backend API server is unreachable! Please run `python api.py` first.")
    st.stop()


# ==========================================
# 🔒 AUTHENTICATION SCREEN
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
                res = requests.post(f"{API_URL}/login", data={"username": l_user, "password": l_pass})
                if res.status_code == 200:
                    token_data = res.json()
                    st.session_state.token         = token_data.get("access_token")
                    st.session_state.refresh_token = token_data.get("refresh_token")
                    st.session_state.username      = l_user
                    st.rerun()
                else:
                    st.error("Invalid username or password!")

    with auth_tab2:
        with st.form("register_form"):
            r_user  = st.text_input("Choose a Username")
            r_email = st.text_input("Email Address")
            r_pass  = st.text_input("Create Password", type="password")
            submit_register = st.form_submit_button("Create Account", type="primary")

            if submit_register:
                res = requests.post(
                    f"{API_URL}/register",
                    json={"username": r_user, "email": r_email, "password": r_pass}
                )
                if res.status_code == 200:
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error(f"Registration failed: {res.json().get('detail')}")

    st.stop()


# --- SILENT TOKEN REFRESH ---
def attempt_silent_refresh():
    """
    Tries to obtain a new access token using the stored refresh token.
    Returns True on success, False if the session is fully expired.
    """
    if not st.session_state.refresh_token:
        return False
    try:
        res = requests.post(
            f"{API_URL}/refresh",
            data={"refresh_token": st.session_state.refresh_token}
        )
        if res.status_code == 200:
            st.session_state.token = res.json().get("access_token")
            return True
        else:
            # Refresh token also expired — force full logout
            st.session_state.token         = None
            st.session_state.refresh_token = None
            st.session_state.username      = None
            return False
    except Exception:
        return False


# ==========================================
# 🚀 MAIN APP (AUTHENTICATED)
# ==========================================

# --- SIDEBAR ---
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
        st.session_state.token         = None
        st.session_state.refresh_token = None
        st.session_state.username      = None
        st.rerun()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}


# ==========================================
# 👑 ADMIN DASHBOARD
# ==========================================
if show_admin:
    st.title("👑 Executive Admin Dashboard")
    st.markdown("Real-time system monitoring, user management, and cache metrics.")
    st.divider()

    stats = {}

    try:
        stats_res = requests.get(f"{API_URL}/admin/stats", headers=HEADERS)
        if stats_res.status_code == 200:
            stats = stats_res.json()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Analyzed", stats.get("total_queries", 0), "🎵 Songs")
            m2.metric("CPU Usage",    stats["system_health"]["cpu"],    help="Server CPU consumption")
            m3.metric("RAM Usage",    stats["system_health"]["ram"],    help="Server memory consumption")

            status_color = "normal" if stats["system_health"]["status"] == "Healthy" else "inverse"
            m4.metric("System Status", stats["system_health"]["status"], delta_color=status_color)

            st.divider()
    except Exception as e:
        st.warning(f"⚠️ Could not fetch real-time stats: {e}")

    # User Table
    with st.spinner("Fetching secure user data..."):
        res = requests.get(f"{API_URL}/admin/users", headers=HEADERS)
        if res.status_code == 200:
            users_data = res.json()
            st.subheader(f"👥 Registered Users ({len(users_data)})")
            st.dataframe(users_data, use_container_width=True)
        else:
            st.error(f"Access Denied or API Error: {res.text}")

    st.divider()

    col_act, col_btns = st.columns([2, 1])

    with col_act:
        st.subheader("🕒 Recent Analysis Tasks")
        if "recent_activities" in stats and stats["recent_activities"]:
            for act in reversed(stats["recent_activities"]):
                st.write(f"👤 **{act['user']}** searched for: *{act['query']}* `{act['status']}`")
        else:
            st.info("No recent activity found or API unreachable.")

    with col_btns:
        st.subheader("⚙️ System Actions")
        if st.button("🧹 Clear All Cache", type="secondary", use_container_width=True):
            with st.spinner("Nuking the cache..."):
                try:
                    res = requests.delete(f"{API_URL}/admin/cache", headers=HEADERS)
                    if res.status_code == 200:
                        st.success("✅ " + res.json().get("message"))
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Failed: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.stop()


# ==========================================
# 🎵 MAIN ANALYSIS APP
# ==========================================
st.title("🎵 BeatHunter AI : MLOps Edition")
st.markdown("AI-Powered Music Information Retrieval & Vectorized Beat Recommendation.")
st.divider()

tab1, tab2 = st.tabs(["🔗 Analyze from YouTube", "📁 Upload Local File"])

api_response = None

# --- TAB 1: YOUTUBE ANALYSIS ---
with tab1:
    song_query = st.text_input("Search on YouTube (e.g., Travis Scott Type Beat):")
    if st.button("🚀 Fetch and Analyze from YouTube", type="primary"):
        if song_query:
            with st.spinner("📡 Sending secure request to API..."):
                try:
                    res = requests.post(
                        f"{API_URL}/analyze_youtube/",
                        data={"query": song_query},
                        headers=HEADERS
                    )

                    if res.status_code == 200:
                        data = res.json()

                        if "task_id" in data:
                            current_state = data.get("current_state", {})

                            # Cache hit — result already ready, no WebSocket needed
                            if current_state.get("status") == "completed":
                                st.success("⚡ Cache Hit! Serving instantly.")
                                api_response = current_state

                            else:
                                # New task — connect WebSocket and wait for worker callback
                                task_id = data["task_id"]
                                st.info("🎟️ Ticket booked! Establishing real-time WebSocket connection...")

                                with st.spinner("🎸 AI is analyzing the beat... Waiting for callback!"):
                                    ws_response = asyncio.run(wait_for_analysis_ws(task_id))

                                if ws_response and ws_response.get("status") == "completed":
                                    st.success("✅ Analysis received via WebSocket!")
                                    api_response = ws_response
                                elif ws_response and ws_response.get("status") == "failed":
                                    st.error(f"❌ Analysis failed: {ws_response.get('error')}")
                                else:
                                    st.error("⚠️ Connection lost or unknown WebSocket error.")
                        else:
                            # Direct response (no task system)
                            st.success("⚡ Instant response received.")
                            api_response = data

                    elif res.status_code == 401:
                        # Try silent token refresh before forcing logout
                        if attempt_silent_refresh():
                            st.warning("🔄 Session silently refreshed! Please click Analyze again.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Session completely expired. Please log out and log in again.")

                    elif res.status_code == 429:
                        st.warning(
                            "⏳ **Rate limit reached.**\n\nYou can analyze 3 songs per minute. Please wait.",
                            icon="🛑"
                        )
                    else:
                        st.error(f"API Error: {res.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")


# --- TAB 2: LOCAL FILE UPLOAD ---
with tab2:
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file:", type=["mp3", "wav"])
    is_beat = st.checkbox("This file is already a vocal-free Beat (Skip Demucs separation)")

    if st.button("📁 Analyze Uploaded File", type="primary"):
        if uploaded_file is not None:
            with st.spinner("📦 Uploading file to API and analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data  = {"skip_demucs": str(is_beat).lower()}

                    res = requests.post(
                        f"{API_URL}/analyze_file/",
                        files=files,
                        data=data,
                        headers=HEADERS
                    )

                    if res.status_code == 200:
                        api_response = res.json()

                    elif res.status_code == 401:
                        if attempt_silent_refresh():
                            st.warning("🔄 Session silently refreshed! Please click Analyze again.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Session completely expired. Please log out and log in again.")

                    elif res.status_code == 429:
                        st.warning(
                            "⏳ **Rate limit reached.**\n\nYou can analyze 3 songs per minute. Please wait.",
                            icon="🛑"
                        )
                    else:
                        st.error(f"API Error: {res.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please upload a file first!")


# ==========================================
# 🎯 ANALYSIS RESULTS DISPLAY
# ==========================================
if api_response and api_response.get("status") in ["success", "completed"]:
    analysis = api_response["analysis"]
    recs     = api_response["recommendations"]

    st.divider()
    st.header("🎯 Analysis Report")

    # 5-column metrics
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

        # Audio preview — use beat_path from response (UUID folder) or fallback for local files
        if api_response.get("source") == "local_file":
            audio_path = "temp_inference/api_uploaded_song.mp3"
        else:
            # beat_path is provided by the worker in api.py response_data
            audio_path = api_response.get("beat_path", "")

        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3")
        else:
            st.info("🎵 Audio preview not available (temp file may have been cleaned up).")

    with col_rec:
        st.subheader("🧬 Vector-Based Recommendations")
        for i, rec in enumerate(recs):
            safe_score = min(100, max(0, int(rec['score'])))
            st.markdown(f"**{i+1}. {rec['title']}**")
            st.caption(f"🔗 [Listen on YouTube](https://youtu.be/{rec['id']})")
            st.progress(safe_score, text=f"Similarity Score: {rec['score']:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)