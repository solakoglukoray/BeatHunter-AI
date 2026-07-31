"""
BeatHunter AI - Frontend (Streamlit) v1.5 Enterprise Edition

Changes in v1.5:
- Added 'My Library' tab to view Search History and Favorite Beats
- Integrated 'Save to Favorites' button into the global analysis results section
- Added pandas for beautiful dataframe rendering
- Injected task_id tracking for Favorite API interactions
"""

import streamlit as st
import requests
import os
import time
import asyncio
import websockets
import json
import nest_asyncio
import pandas as pd  # Added for rendering history & favorites tables

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
if "api_response" not in st.session_state:
    st.session_state.api_response = None

async def wait_for_analysis_ws(task_id):
    import urllib.parse
    encoded_task_id = urllib.parse.quote(task_id, safe="")
    # Replace protocol precisely — "https://" → "wss://", "http://" → "ws://"
    ws_url = API_URL.replace("https://", "wss://").replace("http://", "ws://") + f"/ws/{encoded_task_id}"
    try:
        async with websockets.connect(ws_url) as websocket:
            message = await websocket.recv()
            return json.loads(message)
    except Exception as e:
        print(f"WebSocket failed: {e} — falling back to polling...")
        # WebSocket koparsa cache'de sonuç var mı diye kontrol et
        for _ in range(60):  # 3 dakika bekle
            await asyncio.sleep(3)
            try:
                res = requests.get(
                    f"{API_URL}/task_status/{urllib.parse.quote(task_id, safe='')}",
                    headers=HEADERS
                )
                if res.status_code == 200:
                    data = res.json()
                    if data.get("status") in ["completed", "failed"]:
                        return data
            except Exception:
                pass
        return {"status": "failed", "error": "Timeout: Analysis took too long."}

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

# Updated to include Tab 3 for Library
tab1, tab2, tab3 = st.tabs(["🔗 Analyze from YouTube", "📁 Upload Local File", "📚 My Library"])

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
                            task_id = data["task_id"]

                            # Cache hit — result already ready, no WebSocket needed
                            if current_state.get("status") == "completed":
                                st.success("⚡ Cache Hit! Serving instantly.")
                                st.session_state.api_response = current_state
                                st.session_state.api_response["task_id"] = task_id  # Injected for Favorites tracking
                            else:
                                # New task — connect WebSocket and wait for worker callback
                                st.info("🎟️ Ticket booked! Establishing real-time WebSocket connection...")
                                with st.spinner("🎸 AI is analyzing the beat... Waiting for callback!"):
                                    ws_response = asyncio.run(wait_for_analysis_ws(task_id))

                                if ws_response and ws_response.get("status") == "completed":
                                    st.success("✅ Analysis received via WebSocket!")
                                    st.session_state.api_response = ws_response
                                    st.session_state.api_response["task_id"] = task_id  # Injected for Favorites tracking
                                elif ws_response and ws_response.get("status") == "failed":
                                    st.error(f"❌ Analysis failed: {ws_response.get('error')}")
                                else:
                                    st.error("⚠️ Connection lost or unknown WebSocket error.")
                        else:
                            # Direct response (fallback)
                            st.success("⚡ Instant response received.")
                            st.session_state.api_response = data
                            st.session_state.api_response["task_id"] = f"yt_{song_query}"

                    elif res.status_code == 401:
                        if attempt_silent_refresh():
                            st.warning("🔄 Session silently refreshed! Please click Analyze again.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Session completely expired. Please log out and log in again.")
                    elif res.status_code == 429:
                        st.warning("⏳ **Rate limit reached.**\nYou can analyze 3 songs per minute. Please wait.", icon="🛑")
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
                        st.session_state.api_response = res.json()
                        # Fallback task_id for local files so Favorites button still works
                        st.session_state.api_response["task_id"] = f"file_{uploaded_file.name}"
                    elif res.status_code == 401:
                        if attempt_silent_refresh():
                            st.warning("🔄 Session silently refreshed! Please click Analyze again.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Session completely expired. Please log out and log in again.")
                    elif res.status_code == 429:
                        st.warning("⏳ **Rate limit reached.**\nYou can analyze 3 songs per minute. Please wait.", icon="🛑")
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please upload a file first!")

# --- TAB 3: USER LIBRARY & HISTORY ---
with tab3:
    st.header("📚 My Library")
    st.markdown("View your recent analysis history and favorite beats.")
    
    lib_history, lib_favorites = st.tabs(["🕒 Search History", "⭐ Favorite Beats"])
    
    # 1. History Sub-Tab
    with lib_history:
        if st.button("🔄 Refresh History", key="btn_refresh_hist"):
            st.rerun()
            
        with st.spinner("Loading history..."):
            try:
                res_hist = requests.get(f"{API_URL}/users/me/history", headers=HEADERS)
                if res_hist.status_code == 200:
                    history_data = res_hist.json()
                    if history_data:
                        df_hist = pd.DataFrame(history_data)
                        # Format the date cleanly
                        df_hist['created_at'] = pd.to_datetime(df_hist['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(df_hist[['query', 'source', 'created_at']], use_container_width=True)
                    else:
                        st.info("No search history found. Start analyzing some beats!")
            except Exception as e:
                st.error(f"Connection error: {e}")

    # 2. Favorites Sub-Tab
    with lib_favorites:
        if st.button("🔄 Refresh Favorites", key="btn_refresh_fav"):
            st.rerun()
            
        with st.spinner("Loading favorites..."):
            try:
                res_fav = requests.get(f"{API_URL}/users/me/favorites", headers=HEADERS)
                if res_fav.status_code == 200:
                    fav_data = res_fav.json()
                    if fav_data:
                        df_fav = pd.DataFrame(fav_data)
                        df_fav['created_at'] = pd.to_datetime(df_fav['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                        
                        # --- INTERACTIVE DELETE FEATURE ---
                        # Insert a boolean column for checkboxes
                        df_fav.insert(0, "Select", False)
                        
                        st.markdown("Check the boxes next to the beats you want to remove.")
                        
                        # Render an interactive dataframe
                        edited_df = st.data_editor(
                            df_fav[['Select', 'query', 'created_at', 'task_id']],
                            hide_index=True,
                            column_config={
                                "Select": st.column_config.CheckboxColumn("Select", default=False),
                                "task_id": None, # Hide the technical ID from the user
                                "query": "Beat Name",
                                "created_at": "Added On"
                            },
                            use_container_width=True,
                            key="fav_editor"
                        )
                        
                        # Extract task_ids of selected rows
                        selected_tasks = edited_df[edited_df["Select"]]["task_id"].tolist()
                        
                        # Show delete button only if something is selected
                        if len(selected_tasks) > 0:
                            if st.button("🗑️ Remove Selected", type="primary"):
                                with st.spinner("Removing selected beats..."):
                                    for tid in selected_tasks:
                                        # Call the DELETE API endpoint
                                        requests.delete(f"{API_URL}/users/me/favorites/{tid}", headers=HEADERS)
                                st.success("✅ Removed successfully!")
                                time.sleep(1)
                                st.rerun() # Refresh the page to show updated list
                                
                    else:
                        st.info("You haven't added any beats to your favorites yet.")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ==========================================
# 🎯 ANALYSIS RESULTS DISPLAY
# ==========================================
if st.session_state.api_response and st.session_state.api_response.get("status") in ["success", "completed"]:
    api_response = st.session_state.api_response
    analysis = api_response["analysis"]
    recs     = api_response["recommendations"]

    st.divider()
    st.header("🎯 Analysis Report")

    # Artist/genre come from the fine-tuned PANNs classifier. On held-out tracks: 27.1x
    # chance for artist, 16.8x for genre — against 22.9x/14.4x for the stock AudioSet
    # encoder, 16.7x/9.7x for the classical features before that, and 0.8x (below chance)
    # for the original EfficientNetB3. Top-1 is still only 15.35%, so a single name is not
    # a fact; top-5 hits 37.72%, which is why it leads here.
    #
    # Tempo, Key and Mood are measured from the signal, not predicted, so they are shown
    # without a confidence caveat. The OLD mood label (hand-assigned per artist) was
    # deleted after three measurements found it unrelated to the audio; this one is
    # computed per track against corpus percentiles. Different thing, same word.
    conf     = analysis.get("classification_confidence") or {}
    reliable = conf.get("reliable", False)
    top5     = conf.get("artist_top5") or []

    # Leads the report because it is the question people actually came with: what do I
    # type into YouTube. The old layout answered "which artist" first even when the
    # artist was the weakest thing we knew — on held-out tracks that policy could name
    # an artist for only 10% of tracks, while this one puts a specific label on 97%.
    rec = analysis.get("label_recommendation")
    if rec and rec.get("qualifies"):
        b = rec["best"]
        kind_name = {"artist": "artist", "genre": "genre", "mood": "character"}[b["kind"]]
        st.success(
            f"🔎 **Most reliable label for this track — its {kind_name}:**\n"
            f"## {b['label']}\n"
            f"Search: `{rec['query']}`"
        )
        st.caption(
            f"Chosen because it is right about {b['precision']*100:.0f}% of the time at "
            f"this confidence ({b['evidence']}), measured on {b['n']} held-out tracks. "
            "Artist, genre and character are compared on that one number — how often "
            "each turns out to be correct — not on how confident the model sounds."
        )
    elif rec:
        st.warning(
            "🔎 **No single label is reliable enough for this track.** "
            "The candidates below are the honest answer."
        )

    c1, c2, c3, c4 = st.columns(4)
    if reliable:
        c1.success(f"🎧 **Closest style:**\n### {analysis['artist_type']} Type")
    else:
        c1.info("🎧 **Closest style:**\n### see candidates")
    c2.success(f"🎹 **Genre:**\n### {analysis['genre']}")
    c3.error(f"⏱️ **Tempo:**\n### {analysis['bpm']} BPM")
    c4.info(f"🎵 **Key:**\n### {analysis['key']}")

    # Mood: three measured axes. Absent when models/mood_calibration.json was never built,
    # in which case nothing is shown rather than a fabricated neutral reading.
    mood = analysis.get("mood")
    if mood:
        st.markdown("**Character** — measured from the audio, relative to the catalogue:")
        mc = st.columns(len(mood))
        for col, (axis, val) in zip(mc, mood.items()):
            col.metric(axis.capitalize(), val["label"])
            col.progress(min(1.0, max(0.0, val["score"])))
        st.caption(
            "Each axis is this track's position against the catalogue: 50% is average, "
            "so \"Dark\" means darker than most beats here, not dark in absolute terms."
        )

    if top5:
        st.markdown("**Sounds closest to:**")
        st.dataframe(
            pd.DataFrame([{"Artist": d["label"],
                           "Confidence": f"{d['probability']*100:.1f}%"} for d in top5]),
            hide_index=True, use_container_width=True)

    st.caption(
        "Style and genre are estimates, not identifications. Measured on unseen tracks "
        "the top guess is right about 1 time in 7, and the correct artist is somewhere "
        "in the five above about 3 times in 8 — useful as a starting point, not as an "
        "answer. Tempo and key are measured from the audio and are not estimates."
    )

    if conf and not reliable:
        with st.expander("Why is there no single style shown?"):
            st.markdown(
                "A single name is shown only when confidence passes a "
                f"**{conf.get('min_ratio_required', 150):.0f}x**-chance bar, which happens "
                "for roughly 11% of tracks. That bar is re-measured whenever the model is "
                "retrained: above it the top guess is right about 44% of the time and well "
                "below that under it, so the "
                "candidate list above is the honest answer instead."
            )
            rows = [{"Field": k.capitalize(),
                     "Best guess": conf[k]["label"],
                     "Confidence": f"{conf[k]['probability']*100:.2f}%",
                     "vs. chance": f"{conf[k]['ratio_vs_chance']:.0f}x"}
                    for k in ("artist", "genre") if k in conf]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_img, col_rec = st.columns([1, 1])

    with col_img:
        # Per-request path: a shared filename meant concurrent analyses could show
        # each other's spectrogram. Older cached responses have no path — skip rather
        # than display a stale image belonging to someone else.
        spec = api_response.get("spectrogram_path")
        if spec and os.path.exists(spec):
            st.image(spec, use_container_width=True, caption="Mel-Spectrogram")
        else:
            st.caption("Spectrogram unavailable for this result.")

        # Audio preview — her iki source türü de response'dan path alır
        if st.session_state.api_response.get("source") == "local_file":
            audio_path = st.session_state.api_response.get("file_path", "")
        else:
            audio_path = st.session_state.api_response.get("beat_path", "")

        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3")
        else:
            st.info("🎵 Audio preview not available (temp file may have been cleaned up).")

        # --- ⭐️ ADD TO FAVORITES BUTTON ---
        st.divider()
        col_fav1, col_fav2 = st.columns([1, 2])
        with col_fav1:
            # We safely extract the task_id to use as a unique key
            fav_task_id = st.session_state.api_response.get("task_id", f"unknown_{int(time.time())}")
            fav_query   = st.session_state.api_response.get("query") or st.session_state.api_response.get("filename") or "Unknown Beat"

            if st.button("⭐ Save to Favorites", key=f"fav_btn_{fav_task_id}"):
                fav_payload = {
                    "query": fav_query, 
                    "task_id": fav_task_id
                }
                fav_res = requests.post(
                    f"{API_URL}/users/me/favorites", 
                    json=fav_payload, 
                    headers=HEADERS
                )
                
                if fav_res.status_code == 200:
                    st.toast("✅ Beat added to your favorites!")
                elif fav_res.status_code == 401:
                    st.error("Session expired! Please refresh or log in again.")
                else:
                    st.error("Could not save to favorites. It might already be in your list.")

    with col_rec:
        st.subheader("🧬 Vector-Based Recommendations")
        for i, rec in enumerate(recs):
            safe_score = min(100, max(0, int(rec['score'])))
            st.markdown(f"**{i+1}. {rec['title']}**")
            st.caption(f"🔗 [Listen on YouTube](https://youtu.be/{rec['id']})")
            st.progress(safe_score, text=f"Similarity Score: {rec['score']:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)