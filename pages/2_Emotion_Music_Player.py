import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
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
    transition: all 0.3s ease;
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
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
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
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    background: rgba(255, 215, 0, 0.15) !important;
    border: 1px solid #ffd700 !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
}

/* 2. Playlist Buttons (White) - Scoped to container */
.playlist-container div.stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    text-transform: none !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding-left: 20px !important;
}
.playlist-container div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid #ffffff !important;
}

/* Highlight Active Song in Playlist */
.active-song div.stButton > button {
    border: 1px solid #ffd700 !important;
    background: rgba(255, 215, 0, 0.1) !important;
    color: #ffd700 !important;
}

.footer-text { color: #bbbbbb !important; font-size: 0.9rem !important; text-align: center; letter-spacing: 1px; }
.footer-sub { color: #666666 !important; font-size: 0.8rem !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎧 Sinhala Emotion Music Player</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Experience Music Through the Lens of AI</p>", unsafe_allow_html=True)

# ====================== 3. MODEL ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("<h2 style='color:#ffffff;'>🧠 AI Engine</h2>", unsafe_allow_html=True)
    model_path = st.text_input("Model path", "mobileNetV2.keras")
    try:
        model = load_model(model_path)
        st.success("AI Model Ready")
    except:
        st.error("Model Load Error"); st.stop()
    st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

# ====================== 4. HELPERS ======================
def extract_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def classify_song(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_AUDIO_DURATION)
    if len(y) < TARGET_FRAMES: y = np.pad(y, (0, TARGET_FRAMES - len(y)))
    mel = extract_logmel(y)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if mel.shape[1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0,0),(0,TARGET_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :TARGET_FRAMES]
    x = np.repeat(np.expand_dims(mel, axis=-1), 3, axis=-1)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    return EMOTION_CLASSES[int(np.argmax(pred))], float(np.max(pred))

# ====================== 5. LIBRARY LOGIC ======================
if "library" not in st.session_state:
    st.session_state.library = None

if st.session_state.library is None:
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 20px; border: 2px dashed rgba(255, 215, 0, 0.3); text-align: center; margin: 20px;">
            <h2 style="color: #ffffff;">🎵 Build Your AI Music Library</h2>
            <p style="color: #bbb;">Upload songs to automatically sort them by emotion</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("", type=['mp3', 'wav'], accept_multiple_files=True, key="uploader")

    if uploaded_files:
        if st.button("🚀 Start AI Analysis", use_container_width=True):
            library = {e: [] for e in EMOTION_CLASSES}
            status_container = st.empty()
            progress_bar = st.progress(0)
            if not os.path.exists("temp_songs"): os.makedirs("temp_songs")

            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = os.path.join("temp_songs", uploaded_file.name)
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                emo, conf = classify_song(temp_path)
                library[emo].append({"name": Path(uploaded_file.name).stem, "path": temp_path, "confidence": conf})
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_container.markdown(f"<p style='text-align:center; color:#ffd700;'>Analyzing: {uploaded_file.name}</p>", unsafe_allow_html=True)

            st.session_state.library = library
            st.session_state.current_index = {e: 0 for e in EMOTION_CLASSES}
            st.rerun()

# ====================== 6. BALANCED PLAYER UI ======================
else:
    with st.sidebar:
        if st.button("↺ Reset & Upload New", use_container_width=True):
            st.session_state.library = None
            st.rerun()

    tab_titles = [f"{EMO_ICONS[e]} {e}" for e in EMOTION_CLASSES]
    tabs = st.tabs(tab_titles)

    for emo, tab in zip(EMOTION_CLASSES, tabs):
        with tab:
            songs = st.session_state.library.get(emo, [])
            if not songs:
                st.markdown(f"<div style='text-align:center; padding:50px; color:#666;'>No {emo} songs found in this batch.</div>", unsafe_allow_html=True)
                continue

            idx = st.session_state.current_index.get(emo, 0)
            song = songs[idx]

            # Player Card
            st.markdown(f"""
            <div class="player-card">
                <div>
                    <h5 style="margin:0; color:#ffffff;">{song['name']}</h5>
                    <p style="color:#ffd700; margin:0; font-size:1.1rem;">AI Match: <b>{song['confidence']:.1%}</b></p>
                </div>
                <div style="font-size: 3.5rem;">{EMO_ICONS[emo]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Player Control Section
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p2:
                with open(song["path"], "rb") as f: st.audio(f.read())
            
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c2:
                if st.button("⏮ Previous", key=f"prev_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = max(0, idx - 1); st.rerun()
            with c4:
                if st.button("Next ⏭", key=f"next_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = (idx + 1) % len(songs); st.rerun()

            # Playlist Section (White Buttons)
            st.markdown("<hr style='opacity:0.1;'>", unsafe_allow_html=True)
            st.markdown("#### 📑 Emotion Playlist")
            st.markdown('<div class="playlist-container">', unsafe_allow_html=True)
            for i, s in enumerate(songs):
                active_class = "active-song" if i == idx else ""
                st.markdown(f'<div class="{active_class}">', unsafe_allow_html=True)
                if st.button(f"{i+1:02d}. {s['name']}", key=f"list_{emo}_{i}", use_container_width=True):
                    st.session_state.current_index[emo] = i; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("<br><hr style='opacity:0.2;'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><p class='footer-text'>Designed For Sinhala Emotion Recognition | 2026</p></div>", unsafe_allow_html=True)
