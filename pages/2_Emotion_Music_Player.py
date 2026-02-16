# --------------------------------------------------------------
# Sinhala Song Emotion AI – Emotion-Based Music Player
# UI PREMIUN EDITION - Black & Gold Theme
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os
import pandas as pd
from pathlib import Path
import time
import textwrap

# ====================== 1. CONFIG ======================
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MAX_AUDIO_DURATION = 100
TARGET_FRAMES = 431

EMOTION_CLASSES = ["Calm", "Energetic", "Happy", "Romantic", "Sad"]
EMO_ICONS = {"Calm": "🍃", "Energetic": "🔥", "Happy": "😊", "Romantic": "💖", "Sad": "🥺"}

TEMP_DIR = Path("temp_audio")
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ====================== 2. PAGE & PREMIUM CSS ======================
st.set_page_config(
    page_title="Sinhala Emotion Music Player",
    page_icon="logo.png",
    layout="wide"
)

st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        padding-top: 20px !important;
    }

    /* Titles */
    .main-title {
        text-align: center;
        color: #ffffff !important;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #ccc;
        font-size: 1.2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255, 215, 0, 0.1);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        padding-bottom: 10px;
        padding-top: 10px;    
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #ffffff;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid #ffd700 !important;
        color: #ffd700 !important;
        transform: translateY(-2px);
    }

    /* Buttons - Universal Gold Style */
    div.stButton > button {
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
    }

    /* Target: Playlist & Player Controls */
    div.stButton > button[key*="prev"], 
    div.stButton > button[key*="next"], 
    div.stButton > button[key*="list_"] {
        background-color: transparent !important;
        color: #ffd700 !important;
        border: 1px solid rgba(255, 215, 0, 0.4) !important;
        height: 3rem;
    }
    div.stButton > button[key*="prev"]:hover, 
    div.stButton > button[key*="next"]:hover, 
    div.stButton > button[key*="list_"]:hover {
        background-color: #ffd700 !important;
        color: #000 !important;
        border: 1px solid #ffd700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3) !important;
    }

    /* Sidebar Reset Button (Red) */
    section[data-testid="stSidebar"] div.stButton > button[key="reset_btn"] {
        background: transparent !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        margin-top: 20px;
    }
    section[data-testid="stSidebar"] div.stButton > button[key="reset_btn"]:hover {
        background: #ff4b4b !important;
        color: white !important;
    }

    /* Cards */
    .player-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        text-align: center;
    }
            
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* Divider Color */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Titles
st.markdown("<h1 class='main-title'>🎧 Sinhala Song Emotion AI </h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Emotion-Based Intelligence</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
# ====================== 3. MODEL ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("<h3 style='color:#ffd700; margin-bottom:10px;'>🧠 AI Engine</h3>", unsafe_allow_html=True)
    model_path = st.text_input("Model File", "mobileNetV2.keras")
    try:
        model = load_model(model_path)
        st.success("AI Engine Ready")
    except:
        st.error("Model Not Found")
        st.stop()

# ====================== 4. HELPERS ======================
def extract_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def classify_song(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_AUDIO_DURATION)
    if len(y) < TARGET_FRAMES: y = np.pad(y, (0, TARGET_FRAMES - len(y)))
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

# ====================== 5. RESET BUTTON (Sidebar) ======================
with st.sidebar:
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#ffd700;'>📊 Options</h3>", unsafe_allow_html=True)
    if os.path.exists("responses.csv"):
        with open("responses.csv", "rb") as f:
                st.download_button("📥 Download CSV", f, "song_feedback.csv", "text/csv", use_container_width=True)
    else:
        st.button("📥 No Data Yet", disabled=True, use_container_width=True)

with st.sidebar:
    if st.button("🗑️ Reset Library", key="reset_btn", use_container_width=True):
        if "library" in st.session_state: del st.session_state.library
        if "current_index" in st.session_state: del st.session_state.current_index
        st.rerun()

# ====================== 6. UPLOADER ======================
if "library" not in st.session_state:
    st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.03); padding: 10px; border-radius: 25px; border: 1px dashed #ffd700; text-align: center;">
            <h2 style="color: #ffd700; margin-bottom:10px;">Music Analysis Portal</h2>
            <p style="color: #888;">Drop your Sinhala songs here to let the AI classify them by emotion.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("", type=["mp3", "wav"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 START AI SCAN", use_container_width=True):
            library = {e: [] for e in EMOTION_CLASSES}
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = TEMP_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                emo, conf = classify_song(str(file_path))
                library[emo].append({"name": Path(uploaded_file.name).stem, "path": str(file_path), "confidence": conf})
                progress_bar.progress((i + 1) / len(uploaded_files))

            st.session_state.library = library
            st.session_state.current_index = {e: 0 for e in EMOTION_CLASSES}
            st.rerun()

# ====================== 7. PLAYER UI ======================
if "library" in st.session_state:

    tabs = st.tabs([f"{EMO_ICONS[e]} {e}" for e in EMOTION_CLASSES])

    for emo, tab in zip(EMOTION_CLASSES, tabs):
        with tab:
            songs = st.session_state.library.get(emo, [])
            if not songs:
                st.info(f"The AI hasn't found any {emo} songs yet."); continue

            idx = st.session_state.current_index.get(emo, 0)
            song = songs[idx]

           # Player Card
            # st.markdown(f"""
            # <div class="player-card">
            #     <p style="color:#ffd700; font-size:0.9rem; text-transform:uppercase; margin-bottom:5px;">Currently Playing</p>
            #     <h7 style="margin:0; font-size:1.9rem;">{song['name']}</h7>
            #     <p style="color:#888;">AI Match Confidence: <span style="color:#ffd700;">{song['confidence']:.1%}</span></p>
            #     <div style="font-size: 3.5rem;">{EMO_ICONS[emo]}</div>
            # </div>
            # """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass" style="padding: 20px; margin-bottom: 20px; border-left: 5px solid #ffd700; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; flex: 1; min-width: 0;">
                    <div style="background: linear-gradient(135deg, #ffd700, #ff8c00); padding: 12px; border-radius: 12px; margin-right: 15px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2); flex-shrink: 0;">
                        <span style="font-size: 22px;">🎵</span>
                    </div>
                    <div style="overflow: hidden; line-height: 1.4;">
                        <h4 style="margin: 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 1.1rem; letter-spacing: 0.5px;">
                            {song['name']}
                        </h4>
                        <p style="margin: 2px 0 0 0; color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">
                            AI Confidence: <span style="color:#ffd700; font-weight: bold;">{song['confidence']:.1%}</span>
                        </p>
                    </div>
                </div>
                <div style="font-size: 2.8rem; margin-left: 15px; filter: drop-shadow(0 0 10px rgba(255,215,0,0.3)); flex-shrink: 0;">
                    {EMO_ICONS[emo]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            

            # Audio
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.audio(song["path"])

            # Navigation
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            with c2:
                if st.button("⏮ PREVIOUS", key=f"prev_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = max(0, idx - 1); st.rerun()
            with c4:
                if st.button("NEXT ⏭", key=f"next_{emo}", use_container_width=True):
                    st.session_state.current_index[emo] = (idx + 1) % len(songs); st.rerun()

            # Feedback
            with st.expander("📝 Verify AI Emotion Result"):
                st.markdown("""
                    <p style='font-size: 0.85rem; color: #888; margin-bottom: 15px;'>
                        Help us improve our AI! Tell us if the predicted emotion matches your feel. Please fill in the form for each song.
                    </p>
                """, unsafe_allow_html=True)

                with st.form(key=f"f_{emo}_{idx}"):
                    u_name = st.text_input(
                        "Enter Your Name:", 
                        placeholder="E.g. Kasun Perera",
                        help="Please use the same name for all your entries so we can track your contributions accurately."
                    )
        
                    u_actual = st.selectbox(
                        "What is the actual emotion?", 
                        ["Select...", "Calm", "Energetic", "Happy", "Romantic", "Sad"],
                        help="If you feel the AI is wrong, select the emotion that you think best fits this song segment."
                    )
        
                    submit_btn = st.form_submit_button(
                        "SAVE VERIFICATION",
                        help="Click to securely save your feedback to our research database."
                    )

                    if submit_btn:
                        if u_name and u_actual != "Select...":
                            df = pd.DataFrame([{"Song": song['name'], "AI": emo, "User": u_actual, "Name": u_name, "Date": time.strftime("%Y-%m-%d %H:%M")}])
                            df.to_csv("responses.csv", mode='a', header=not os.path.exists("responses.csv"), index=False)
                
                            #st.balloons() 
                            st.success(f"Thank you {u_name}! Your response has been recorded.")
                            time.sleep(1.5)
                            st.rerun()
                    else:
                        st.warning("⚠️ Please fill in both your name and the actual emotion.")

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Playlist
            st.markdown(f"#### 📑 {emo} Playlist")
            for i, s in enumerate(songs):
                active_style = "border: 1px solid #ffd700 !important; background: rgba(255,215,0,0.1) !important;" if i == idx else ""
                if st.button(f"{i+1:02d}. {s['name']}", key=f"list_{emo}_{i}", use_container_width=True):
                    st.session_state.current_index[emo] = i; st.rerun()

# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p style='color:#888; font-size:0.9rem;'>Powered by MobileNetV2 Architecture & TensorFlow</p>
    <p style='color:#555; font-size:0.8rem;'>Sinhala Emotion Recognition Research Project © 2026</p>
</div>
""", unsafe_allow_html=True)
