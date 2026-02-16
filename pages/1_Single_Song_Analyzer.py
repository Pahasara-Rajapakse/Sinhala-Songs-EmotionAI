# --------------------------------------------------------------
#  Sinhala Song Emotion AI – MobileNetV2 Premium Edition
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa, librosa.display
import io, time, base64
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

# ====================== 1. CONFIG ======================
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MAX_AUDIO_DURATION = 100
NUM_CHUNKS = 10
TARGET_FRAMES = 431

EMOTION_CLASSES = ["Calm", "Energetic", "Happy", "Romantic", "Sad"]
EMO_ICONS = {"Calm": "🍃", "Energetic": "🔥", "Happy": "😊", "Romantic": "💖", "Sad": "🥺"}

# ====================== 2. PAGE & PREMIUM CSS ======================
st.set_page_config(
    page_title="Sinhala AI Analyzer",
    page_icon="logo.png",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        padding-top: 20px !important;  
    }
    
    /* Global Titles */
    .main-title {
        text-align: center;
        color: #ffffff !important;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
            
    .sub-title {
        text-align: center;
        color: #ccc;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }

    /* Glass Cards */
    .glass_card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.1);
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
            
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* Progress Bars Animation */
    .stProgress > div > div > div > div {
        background-color: #ffd700 !important;
    }

    /* Custom Button Style */
    .stButton>button {
        height: 5.8rem !important; 
        border-radius: 20px !important;
        background: rgba(255, 215, 0, 0.05) !important; 
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #ffd700 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
            
    .stButton>button:hover {
        background: rgba(255, 215, 0, 0.15) !important;
        border: 1px solid #ffd700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
        transform: translateY(-2px);
    }

    /* Sidebar Fix */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255, 215, 0, 0.1);
    }

    .footer-text { color: #888 !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ====================== 3. MODEL LOADER ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("<h3 style='color:#ffd700;'>🧠 AI Engine</h3>", unsafe_allow_html=True)
    model_path = st.text_input("Model path", "mobileNetV2.keras")
    try:
        model = load_model(model_path)
        st.success("AI Model Active")
    except Exception as e:
        st.error("Model Not Found")
        st.stop()
    
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#ffd700;'>ℹ️ AI Info </h3>", unsafe_allow_html=True)
    st.write("This engine uses **MobileNetV2** for feature extraction from Log-Mel-Spectrograms.")

# ====================== 4. HELPERS ======================
def extract_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def plot_audio_visuals(y, mel):
    # Waveform
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=SR, ax=ax1, color="#ffd700", alpha=0.6)
    ax1.axis('off')
    fig1.patch.set_alpha(0)
    
    # Mel
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    librosa.display.specshow(mel, sr=SR, cmap='magma', ax=ax2)
    ax2.axis('off')
    fig2.patch.set_alpha(0)
    
    return fig1, fig2

# ====================== 5. MAIN UI LOGIC ======================
st.markdown("<h1 class='main-title'>🎧 Sinhala Song Emotion AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>High-Precision Deep Learning Analysis</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)

if 'active_file' not in st.session_state:
    st.session_state.active_file = None

if st.session_state.active_file is None:
    st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.03); padding: 10px; border-radius: 25px; border: 1px dashed #ffd700; text-align: center;">
            <h2 style="color: #ffd700;">Upload Audio Track</h2>
            <p style="color: #666;">Drag and drop MP3 or WAV files for emotional mapping</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3", "wav"], key="main_uploader")
    if uploaded_file:
        st.session_state.active_file = uploaded_file
        st.rerun()

else:
    uploaded_file = st.session_state.active_file
    
    # Top Control Bar
    col_info, col_reset = st.columns([0.8, 0.2])
    with col_info:
        st.markdown(f"""
        <div class="glass" style="padding: 15px; margin-bottom: 20px; border-left: 5px solid #ffd700;">            
            <div style="display: flex; align-items: center ;">
                <div style="background: linear-gradient(135deg, #ffd700, #ff8c00); padding: 10px; border-radius: 10px; margin-right: 15px; box-shadow: 0 4px 10px rgba(255, 215, 0, 0.3);">
                    <span style="font-size: 20px;">🎵</span>
                </div>
                <div style="overflow: hidden;">
                    <h4 style="margin: 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 1rem; letter-spacing: 0.5px;">{uploaded_file.name}</h4>
                    <p style="margin: 0; color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Acoustic Analysis Active</p>
                </div>
            </div>
        </div>    
        """, unsafe_allow_html=True)

    with col_reset:
        if st.button("↺ NEW SCAN", use_container_width=True):
            st.session_state.active_file = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.audio(uploaded_file)
    
    # Analysis Processing
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    with st.spinner("🧠 AI Deep Scan in Progress..."):
        y, _ = librosa.load(uploaded_file, sr=SR, duration=MAX_AUDIO_DURATION)
        mel_full = extract_logmel(y)
        
        # Visualize
        f1, f2 = plot_audio_visuals(y, mel_full)
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("<p style='color:#888; text-align:center;'>Waveform Signature</p>", unsafe_allow_html=True)
            st.markdown(f"<img src='data:image/png;base64,{fig_to_base64(f1)}' style='width:100%;'>", unsafe_allow_html=True)
        with v_col2:
            st.markdown("<p style='color:#888; text-align:center;'>Spectral Mel-Map</p>", unsafe_allow_html=True)
            st.markdown(f"<img src='data:image/png;base64,{fig_to_base64(f2)}' style='width:100%;'>", unsafe_allow_html=True)

        # Chunk Processing Logic
        chunk_len = len(y) // NUM_CHUNKS
        preds = []
        timeline = []
        for i in range(NUM_CHUNKS):
            chunk = y[i*chunk_len:(i+1)*chunk_len]
            mel = extract_logmel(chunk)
            
            # Pad or Trim to TARGET_FRAMES
            if mel.shape[1] < TARGET_FRAMES:
                mel = np.pad(mel, ((0,0),(0,TARGET_FRAMES - mel.shape[1])), 'constant')
            else:
                mel = mel[:, :TARGET_FRAMES]
            
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            inp = np.repeat(np.expand_dims(mel, axis=[-1]), 3, axis=-1)
            inp = np.expand_dims(inp, axis=0)
            
            p = model.predict(inp, verbose=0)[0]
            preds.append(p)
            timeline.append((i*chunk_len/SR, (i+1)*chunk_len/SR, EMOTION_CLASSES[int(np.argmax(p))]))

        avg_pred = np.mean(preds, axis=0)
        final_idx = int(np.argmax(avg_pred))
        res_emo = EMOTION_CLASSES[final_idx]
        res_conf = avg_pred[final_idx]

    # --- RESULTS DISPLAY ---
    st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(255,215,0,0.05); border: 1px solid #ffd700; border-radius: 25px; padding: 30px; text-align: center; margin: 30px 0;">
        <p style="letter-spacing: 3px; color: #ffd700; font-weight: bold; margin-bottom: 0;">PREDICTED EMOTION</p>
        <h1 style="font-size: 5rem; margin: 10px 0; color: white !important; drop-shadow(0 0 10px rgba(255,215,0,0.3));">{EMO_ICONS[res_emo]} {res_emo}</h1>
        <div style="display: inline-block; padding: 5px 20px; background: #ffd700; color: black; border-radius: 50px; font-weight: 800;">
            {res_conf:.1%} AI CONFIDENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Deep Analytics (The Gold Bars)
    st.markdown("### 📊 Emotional Probability Distribution")
    for emo, prob in zip(EMOTION_CLASSES, avg_pred):
        pct = prob * 100
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>{EMO_ICONS[emo]} {emo}</span>
                <span style="color: #ffd700; font-weight: bold;">{pct:.1f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 10px;">
                <div style="width: {pct}%; background: #ffd700; height: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(255,215,0,0.4);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Segments
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown("### 🎞️ Emotion-Based Segments")
    segments = []
    if timeline:
        cur_emo, cur_start = timeline[0][2], timeline[0][0]
        for s, e, emo in timeline[1:]:
            if emo != cur_emo:
                segments.append((cur_emo, cur_start, s))
                cur_emo, cur_start = emo, s
        segments.append((cur_emo, cur_start, timeline[-1][1]))

        for idx, (emo, start, end) in enumerate(segments, 1):
            icon = EMO_ICONS.get(emo, "🎵")
            duration = end - start
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.03); border-left: 5px solid #ffd700; padding: 15px; border-radius: 12px; margin-bottom: 10px;">
                    <span style="font-size: 1.1rem; font-weight: bold; color: white;">{idx}. {icon} {emo} <small style="color: #888;">({start:.1f}s – {end:.1f}s)</small></span>
                    <span style="float:right; background: rgba(255,215,0,0.1); color: #ffd700; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem;">Duration: {duration:.1f}s</span>
                </div>
                """, unsafe_allow_html=True)
                seg_audio = y[int(start*SR):int(end*SR)]
                buf = io.BytesIO()
                sf.write(buf, seg_audio, SR, format="WAV")
                st.audio(buf.getvalue())

# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p style='color:#888; font-size:0.9rem;'>Powered by MobileNetV2 Architecture & TensorFlow</p>
    <p style='color:#555; font-size:0.8rem;'>Sinhala Emotion Recognition Research Project © 2026</p>
</div>
""", unsafe_allow_html=True)
