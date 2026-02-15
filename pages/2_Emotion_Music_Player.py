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

    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
        justify-content: center;
        padding-bottom: 10px;
        
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    
        color: #ffffff;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 215, 0, 0.15) !important;
        border: 2px solid #ffd700 !important;
        color: #ffd700 !important;
        transform: scale(1.05);
    }
    /* Button & Player Fixes */
    .stButton>button {
        border-radius: 12px !important;
    }
            
            /* Focus Ring Fix - Meka thamai click kalama ena katha border eka ain karanne */
    [data-baseweb="tab"]:focus, 
    [data-baseweb="tab"]:active {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Tab list eke uda line eka kapena eka fix karanna thawa podi thalluwak */
    .stTabs [data-baseweb="tab-list"] {
        padding-top: 70px !important; /* Thawa poddak yata kala */
        border-bottom: 1px solid rgba(255, 255, 255, 0.05); /* Yatin podi separator ekak damma */
    }

/* --- ALL BUTTONS (PURE WHITE) --- */
div.stButton > button {
    height: 48px !important; 
    border-radius: 10px !important;
    background: transparent !important; 
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    width: 100% !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* --- RESET BUTTON ONLY (GOLD) --- */
/* Target the first button in the sidebar */
section[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255, 215, 0, 0.1) !important;
    border: 1px solid #ffd700 !important;
    color: #ffd700 !important;
    font-weight: 700 !important;
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

.footer-text { color: #bbbbbb !important; font-size: 0.9rem !important; }
.footer-sub { color: #666666 !important; font-size: 0.8rem !important; }

/* Audio Center */
.stAudio { width: 100% !important; }
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
    
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True) 

    if os.path.exists("user_feedback.csv"):
        st.markdown("### 📊 Download Responces")
        with open("user_feedback.csv", "rb") as f:
            st.download_button("📥 Download Results CSV", f, "testing_results.csv", "text/csv", use_container_width=True)

    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)     

    # Reset Button (Gold as requested)
    if st.session_state.get('library') is not None:
        if st.button("↺ Reset Library", use_container_width=True):
            st.session_state.library = None
            st.rerun()
            

# ====================== 4. HELPERS ======================
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

# ====================== 6. PLAYER UI ======================
if "library" in st.session_state:
    # Balanced Tabs with Titles and Icons
    tab_titles = [f"{EMO_ICONS[e]} {e}" for e in EMOTION_CLASSES]
    tabs = st.tabs(tab_titles)

    for emo, tab in zip(EMOTION_CLASSES, tabs):
        with tab:
            songs = st.session_state.library.get(emo, [])
            if not songs:
                st.info(f"No {emo} songs detected.")
                continue

            idx = st.session_state.current_index.get(emo, 0)
            song = songs[idx]

            st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)

            # Player Card
            st.markdown(f"""
            <div>
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid #ffd700;
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    display:flex; 
                    justify-content:space-between; 
                    align-items:center;   
                ">        
                    <div>
                        <h5 style="margin:0; color:#ffffff;">{song['name']}</h5>
                        <p style="color:#ddd;">AI Confidence: <b>{song['confidence']:.1%}</b></p>
                    </div>
                    <div style="font-size: 3rem;">{EMO_ICONS[emo]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

         

            # Audio Player
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                with open(song["path"], "rb") as f:
                    st.audio(f.read())

            # Balanced Controls
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c2:
                if st.button("⏮ Previous", key=f"prev_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = max(0, idx - 1)
                    st.rerun()
            with c4:
                if st.button("Next ⏭", key=f"next_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = (idx + 1) % len(songs)
                    st.rerun()

            # Feedback Form
            with st.expander("📝 Verify This Prediction (Research)"):
                with st.form(key=f"f_{emo}_{idx}"):
                    u_name = st.text_input("Your Name")
                    u_emo = st.selectbox("Actual Emotion:", ["Select...", "Calm", "Energetic", "Happy", "Romantic", "Sad"])
                    if st.form_submit_button("Submit"):
                        if u_name and u_emo != "Select...":
                            res = {"User": u_name, "Song": song['name'], "System": emo, "Actual": u_emo, "Match": "Yes" if emo == u_emo else "No", "Time": time.strftime("%H:%M:%S")}
                            pd.DataFrame([res]).to_csv("user_feedback.csv", mode='a', header=not os.path.exists("user_feedback.csv"), index=False)
                            st.success("Feedback saved!")

            st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
            
            # Playlist List
            st.markdown("#### 📑 Songs Playlist")
            for i, s in enumerate(songs):
                btn_label = f"{i+1:02d}. {s['name']}"
                if st.button(btn_label, key=f"list_{emo}_{i}", use_container_width=True):
                    st.session_state.current_index[emo] = i
                    st.rerun()
else:
    st.markdown("<div style='text-align:center; padding:50px; color:#666;'>Add a folder and build library to start.</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p class='footer-text'>Powered by <b>MobileNetV2</b> & <b>TensorFlow</b></p>
    <p class='footer-sub'>Designed For Sinhala Emotion Recognition | Research Project 2026</p>
</div>
""", unsafe_allow_html=True)
