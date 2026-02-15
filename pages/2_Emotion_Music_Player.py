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
}

/* Player Card */
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

/* 1. PLAYER BUTTONS (Gold) */
div.stButton > button {
    height: 45px !important; 
    border-radius: 10px !important;
    background: rgba(255, 215, 0, 0.05) !important; 
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    color: #ffd700 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
div.stButton > button:hover {
    background: rgba(255, 215, 0, 0.15) !important;
    border: 1px solid #ffd700 !important;
}

/* 2. PLAYLIST BUTTONS (White) - Overriding using a custom div class */
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
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid #ffffff !important;
}

/* Highlight Active Song in Playlist */
.active-song div.stButton > button {
    border: 1px solid #ffd700 !important;
    background: rgba(255, 215, 0, 0.1) !important;
    color: #ffd700 !important;
    font-weight: 700 !important;
}

.footer-text { color: #bbbbbb !important; font-size: 0.9rem !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ====================== 3. LOGIC (Model & Helpers) ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

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

# ====================== 4. UI START ======================
st.markdown("<h1 class='main-title'>🎧 Sinhala Emotion Music Player</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Experience Music Through the Lens of AI</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='color:#ffffff;'>🧠 AI Engine</h2>", unsafe_allow_html=True)
    model_path = st.text_input("Model path", "mobileNetV2.keras")
    try:
        model = load_model(model_path)
        st.success("AI Model Ready")
    except:
        st.error("Model Load Error"); st.stop()

if "library" not in st.session_state:
    st.session_state.library = None

if st.session_state.library is None:
    st.markdown('<div style="background: rgba(255,255,255,0.05); padding: 40px; border-radius: 20px; border: 2px dashed rgba(255,215,0,0.3); text-align:center;">'
                '<h2 style="color:white;">🎵 Build Your AI Music Library</h2></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=['mp3', 'wav'], accept_multiple_files=True)
    if uploaded_files:
        if st.button("🚀 Start AI Analysis", use_container_width=True):
            library = {e: [] for e in EMOTION_CLASSES}
            p_bar = st.progress(0)
            if not os.path.exists("temp_songs"): os.makedirs("temp_songs")
            for i, f in enumerate(uploaded_files):
                p = os.path.join("temp_songs", f.name)
                with open(p, "wb") as file: file.write(f.getbuffer())
                emo, conf = classify_song(p)
                library[emo].append({"name": Path(f.name).stem, "path": p, "confidence": conf})
                p_bar.progress((i + 1) / len(uploaded_files))
            st.session_state.library = library
            st.session_state.current_index = {e: 0 for e in EMOTION_CLASSES}
            st.rerun()
else:
    # --- PLAYER UI ---
    with st.sidebar:
        if st.button("↺ Reset Library", use_container_width=True):
            st.session_state.library = None
            st.rerun()

    tabs = st.tabs([f"{EMO_ICONS[e]} {e}" for e in EMOTION_CLASSES])
    for emo, tab in zip(EMOTION_CLASSES, tabs):
        with tab:
            songs = st.session_state.library.get(emo, [])
            if not songs:
                st.info(f"No {emo} songs found."); continue
            
            idx = st.session_state.current_index.get(emo, 0)
            song = songs[idx]

            # Player Card
            st.markdown(f'<div class="player-card"><div><h3 style="margin:0;">{song["name"]}</h3>'
                        f'<p style="color:#ffd700; margin:0;">AI Match: {song["confidence"]:.1%}</p></div>'
                        f'<div style="font-size:3.5rem;">{EMO_ICONS[emo]}</div></div>', unsafe_allow_html=True)

            # Player Controls (Gold Buttons)
            c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
            with c_p2:
                with open(song["path"], "rb") as f: st.audio(f.read())
            
            st.markdown("<br>", unsafe_allow_html=True)
            ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 1, 1])
            with ctrl2:
                if st.button("⏮ Previous", key=f"prev_{emo}"):
                    st.session_state.current_index[emo] = max(0, idx - 1); st.rerun()
            with ctrl4:
                if st.button("Next ⏭", key=f"next_{emo}"):
                    st.session_state.current_index[emo] = (idx + 1) % len(songs); st.rerun()

            # --- PLAYLIST SECTION (White Buttons) ---
            st.markdown("<hr style='opacity:0.1;'>", unsafe_allow_html=True)
            st.markdown("#### 📑 Emotion Playlist")
            
            # Wraps playlist buttons in a div for custom CSS
            st.markdown('<div class="playlist-container">', unsafe_allow_html=True)
            for i, s in enumerate(songs):
                is_active = "active-song" if i == idx else ""
                st.markdown(f'<div class="{is_active}">', unsafe_allow_html=True)
                if st.button(f"{i+1:02d}. {s['name']}", key=f"list_{emo}_{i}", use_container_width=True):
                    st.session_state.current_index[emo] = i; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p class='footer-text'>Powered by <b>MobileNetV2</b> & <b>TensorFlow</b></p>
    <p class='footer-sub'>Designed For Sinhala Emotion Recognition | Research Project 2026</p>
</div>
""", unsafe_allow_html=True)
