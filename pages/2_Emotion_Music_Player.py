import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import pandas as pd
from pathlib import Path
import time

# ====================== 1. CONFIG ======================
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MAX_AUDIO_DURATION = 100
TARGET_FRAMES = 431

EMOTION_CLASSES = ["Calm", "Energetic", "Happy", "Romantic", "Sad"]
EMO_ICONS = {"Calm": "🍃", "Energetic": "🔥", "Happy": "😊", "Romantic": "💖", "Sad": "🥺"}

# ====================== 2. PAGE & PREMIUM CSS ======================
st.set_page_config(page_title="Sinhala Emotion Music Player", page_icon="logo.png", layout="wide")

st.markdown("""
<style>
.stApp { background: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
.main-title { text-align: center; color: #ffffff !important; font-size: 3rem; font-weight: 800; margin-bottom: 0px; text-shadow: 0 4px 10px rgba(0,0,0,0.5); }
.sub-title { text-align: center; color: #bbbbbb; font-size: 1.1rem; margin-bottom: 30px; }

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] { gap: 15px; justify-content: center; padding-top: 20px !important; padding-bottom: 20px !important; }
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    color: #ffffff;
    padding: 12px 30px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255, 215, 0, 0.15) !important;
    border: 2px solid #ffd700 !important;
    color: #ffd700 !important;
    transform: scale(1.05);
}

/* Player Card Styling */
.player-card {
    background: rgba(255, 255, 255, 0.05);
    border-left: 5px solid #ffd700;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* 1. Player Control Buttons (Gold) */
div.stButton > button {
    height: 45px !important; 
    border-radius: 10px !important;
    background: rgba(255, 215, 0, 0.05) !important; 
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    color: #ffd700 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    width: 100% !important;
}

/* 2. Playlist Buttons (White) */
.playlist-container div.stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    text-transform: none !important;
    text-align: left !important;
    padding-left: 20px !important;
}

/* Active Song */
.active-song div.stButton > button {
    border: 1px solid #ffd700 !important;
    background: rgba(255, 215, 0, 0.1) !important;
    color: #ffd700 !important;
}

.footer-text { color: #bbbbbb !important; font-size: 0.9rem !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎧 Sinhala Emotion Music Player</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Experience Music Through the Lens of AI</p>", unsafe_allow_html=True)

# ====================== 3. MODEL ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("<h2 style='color:white;'>🧠 AI Engine</h2>", unsafe_allow_html=True)
    model_path = st.text_input("Model path", "mobileNetV2.keras")
    try:
        model = load_model(model_path)
        st.success("Model Loaded Successfully")
    except:
        st.error("Model Error"); st.stop()
    
    st.markdown("---")
    if os.path.exists("user_feedback.csv"):
        st.markdown("### 📊 Testing Data")
        with open("user_feedback.csv", "rb") as f:
            st.download_button("📥 Download Results CSV", f, "testing_results.csv", "text/csv", use_container_width=True)

# ====================== 4. HELPERS (Corrected Logic) ======================
def extract_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def classify_song(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_AUDIO_DURATION)
    mel = extract_logmel(y)
    
    chunks = []
    hop = TARGET_FRAMES
    for start in range(0, mel.shape[1], hop):
        end = start + TARGET_FRAMES
        segment = mel[:, start:end]
        if segment.shape[1] < TARGET_FRAMES:
            segment = np.pad(segment, ((0,0),(0,TARGET_FRAMES - segment.shape[1])))
        chunks.append(segment)
        
    preds = []
    for seg in chunks:
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)
        x = np.repeat(np.expand_dims(seg, axis=-1), 3, axis=-1)
        x = np.expand_dims(x, axis=0)
        preds.append(model.predict(x, verbose=0)[0])
        
    avg_pred = np.mean(preds, axis=0)
    final_idx = int(np.argmax(avg_pred))
    return EMOTION_CLASSES[final_idx], float(avg_pred[final_idx])

# ====================== 5. LIBRARY LOGIC ======================
if "library" not in st.session_state: st.session_state.library = None

if st.session_state.library is None:
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 20px; border: 2px dashed rgba(255, 215, 0, 0.3); text-align: center; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 1.2rem; color: #ffffff; font-weight: bold;">Upload Your Song</p>
            <h7 style="color: #666; font-size: 0.9rem;">MP3 or WAV (Max 90s Analysis)</h7>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Songs", type=['mp3', 'wav'], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Start AI Analysis", use_container_width=True):
        library = {e: [] for e in EMOTION_CLASSES}
        prog = st.progress(0)
        if not os.path.exists("temp_songs"): os.makedirs("temp_songs")
        for i, f in enumerate(uploaded_files):
            p = os.path.join("temp_songs", f.name)
            with open(p, "wb") as file: file.write(f.getbuffer())
            emo, conf = classify_song(p)
            library[emo].append({"name": Path(f.name).stem, "path": p, "confidence": conf})
            prog.progress((i + 1) / len(uploaded_files))
        st.session_state.library = library
        st.session_state.current_index = {e: 0 for e in EMOTION_CLASSES}
        st.rerun()

# ====================== 6. PLAYER UI ======================
else:
    with st.sidebar:
        if st.button("↺ Reset Library", use_container_width=True):
            st.session_state.library = None; st.rerun()

    tabs = st.tabs([f"{EMO_ICONS[e]} {e}" for e in EMOTION_CLASSES])
    for emo, tab in zip(EMOTION_CLASSES, tabs):
        with tab:
            songs = st.session_state.library.get(emo, [])
            if not songs: st.info(f"No {emo} songs."); continue
            
            idx = st.session_state.current_index.get(emo, 0)
            song = songs[idx]

            # Player Card
            st.markdown(f'<div class="player-card"><div><h5>{song["name"]}</h5>'
                        f'<p style="color:#ffd700;">AI Confidence: {song["confidence"]:.1%}</p></div>'
                        f'<div style="font-size:3.5rem;">{EMO_ICONS[emo]}</div></div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with open(song["path"], "rb") as f: st.audio(f.read())
            
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c2:
                if st.button("⏮ Previous", key=f"p_{emo}"):
                    st.session_state.current_index[emo] = max(0, idx - 1); st.rerun()
            with c4:
                if st.button("Next ⏭", key=f"n_{emo}"):
                    st.session_state.current_index[emo] = (idx + 1) % len(songs); st.rerun()

            # --- Feedback Form ---
            with st.expander("📝 Verify This Prediction (Research)"):
                with st.form(key=f"f_{emo}_{idx}"):
                    u_name = st.text_input("Your Name")
                    u_emo = st.selectbox("Actual Emotion:", ["Select...", "Calm", "Energetic", "Happy", "Romantic", "Sad"])
                    if st.form_submit_button("Submit"):
                        if u_name and u_emo != "Select...":
                            res = {"User": u_name, "Song": song['name'], "System": emo, "Actual": u_emo, "Match": "Yes" if emo == u_emo else "No", "Time": time.strftime("%H:%M:%S")}
                            pd.DataFrame([res]).to_csv("user_feedback.csv", mode='a', header=not os.path.exists("user_feedback.csv"), index=False)
                            st.success("Feedback saved!")

            # Playlist
            st.markdown("#### 📑 Emotion Playlist")
            st.markdown('<div class="playlist-container">', unsafe_allow_html=True)
            for i, s in enumerate(songs):
                active = "active-song" if i == idx else ""
                st.markdown(f'<div class="{active}">', unsafe_allow_html=True)
                if st.button(f"{i+1:02d}. {s['name']}", key=f"l_{emo}_{i}", use_container_width=True):
                    st.session_state.current_index[emo] = i; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><p class='footer-text'>Designed For Sinhala Emotion Recognition | Research 2026</p>", unsafe_allow_html=True)
