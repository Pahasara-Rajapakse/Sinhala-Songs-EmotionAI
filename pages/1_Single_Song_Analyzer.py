# --------------------------------------------------------------
#  Sinhala Song Emotion AI – MobileNetV2 (.keras)
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa, librosa.display
import io, time, base64
import matplotlib.pyplot as plt
import soundfile as sf

# ====================== 1. CONFIG ======================
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MAX_AUDIO_DURATION = 100
NUM_CHUNKS = 10
TARGET_FRAMES = 431

# MUST MATCH TRAINING LABEL ORDER EXACTLY
EMOTION_CLASSES = ["Calm", "Energetic", "Happy", "Romantic", "Sad"]
EMO_ICONS = {"Calm": "🍃", "Energetic": "🔥", "Happy": "😊", "Romantic": "💖", "Sad": "🥺"}

# ====================== 2. PAGE & CSS ======================
st.set_page_config(
    page_title="Sinhala Song Emotion AI",
    page_icon="logo.png",
    #page_icon="🎧",
    layout="wide"
)

# ====================== 6. MAIN UI ======================
st.markdown("<h1 style='text-align:center;'> 🎧 Sinhala Song Emotion AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ccc;'>Emotion Recognition from Sinhala Songs</p>", unsafe_allow_html=True)

st.markdown("""
<style>
         
.stApp {
    background: #000000;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}
h1,h2,h3 {
    color:#ffd700;
    text-shadow:0 2px 8px rgba(0,0,0,.3);
}
.glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
            
/* Footer rules - hama page ekatama eka widiyata gray penna */
.footer-text {
    color: #bbbbbb !important; /* Header eke gray color ekama thamai meka */
    font-size: 0.9rem !important;
    letter-spacing: 1px;
    margin-bottom: 5px !important;
}
.footer-sub {
    color: #666666 !important;
    font-size: 0.8rem !important;
}                                            
                                                                 
</style>
""", unsafe_allow_html=True)

# ====================== 3. MODEL LOADER WITH SIDE BAR ======================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("## 🧠 Model")
    model_path = st.text_input("Model path", "mobileNetV2.keras")

    try:
        model = load_model(model_path)
        st.success("Model Loaded Successfully")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ AI Info")
    st.write("This engine uses **MobileNetV2** for feature extraction from Log-Mel-Spectrograms.")

# ====================== 5. HELPER FUNCTIONS ======================

def extract_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def normalize(mel):
    return (mel - mel.mean()) / (mel.std() + 1e-6)

def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(10, 2.8)) 
    librosa.display.waveshow(y, sr=SR, ax=ax, color="#ffd700", alpha=0.7)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)
    return fig

def plot_mel(mel):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    librosa.display.specshow(mel, sr=SR, cmap='magma', ax=ax)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)
    return fig

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ====================== 6. DYNAMIC UPLOADER LOGIC ======================
if 'active_file' not in st.session_state:
    st.session_state.active_file = None

if st.session_state.active_file is None:
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 20px; border: 2px dashed rgba(255, 215, 0, 0.3); text-align: center; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 1.2rem; color: #ffffff; font-weight: bold;">Upload Your Song</p>
            <h7 style="color: #666; font-size: 0.9rem;">MP3 or WAV (Max 90s Analysis)</h7>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3", "wav"], key="main_uploader")
    if uploaded_file:
        st.session_state.active_file = uploaded_file
        st.rerun()

if st.session_state.active_file is not None:
    uploaded_file = st.session_state.active_file
    y, _ = librosa.load(uploaded_file, sr=SR, duration=MAX_AUDIO_DURATION)
    
    st.markdown("""
        <style>
            [data-testid="column"] { display: flex; align-items: center; }
            .stButton>button {
                height: 60px;
                border-radius: 12px !important;
                background: rgba(0, 255, 127, 0.1) !important;
                border: 1px solid rgba(0, 255, 127, 0.3) !important;
                color: #00ff7f !important;
            }
            .stButton>button:hover {
                background: rgba(0, 255, 127, 0.2) !important;
                border: 1px solid #00ff7f !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass" style="padding: 10px; margin-bottom: 15px; border-left: 5px solid #ffd700;">', unsafe_allow_html=True)
    col_info, col_reset = st.columns([0.75, 0.25])
    with col_info:
        st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <div style="background: #ffd700; padding: 12px; border-radius: 12px; font-size: 20px; margin-right: 15px; color: #000;">🎵</div>
                <div style="overflow: hidden;">
                    <h4 style="margin: 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 1.1rem;">{uploaded_file.name}</h4>
                    <p style="margin: 0; color: #888; font-size: 0.8rem;">Ready for Analysis</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_reset:
        if st.button("✨ Analyze New Song", use_container_width=True):
            st.session_state.active_file = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True) 
    st.audio(uploaded_file)
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 〰️ Waveform")
        st.markdown(f"<div style='background:rgba(255,255,255,0.02); border-radius:15px; padding:10px;'><img src='data:image/png;base64,{fig_to_base64(plot_waveform(y))}' style='width:100%;'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("##### 🌈 Spectral View")
        mel_full = extract_logmel(y)
        st.markdown(f"<div style='background:rgba(255,255,255,0.02); border-radius:15px; padding:10px;'><img src='data:image/png;base64,{fig_to_base64(plot_mel(mel_full))}' style='width:100%;'></div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)   

    # ====================== 7. PREDICTION (FIXED) ======================
    with st.spinner("AI is Analyzing emotion..."):
        chunk_len = len(y) // NUM_CHUNKS
        preds = []
        timeline = []
        progress = st.progress(0)
        status_text = st.empty()

        for i in range(NUM_CHUNKS):
            chunk = y[i*chunk_len:(i+1)*chunk_len]
            mel = extract_logmel(chunk)
            
            if mel.shape[1] < TARGET_FRAMES:
                mel = np.pad(mel, ((0,0),(0,TARGET_FRAMES - mel.shape[1])), 'constant')
            else:
                mel = mel[:, :TARGET_FRAMES]
            
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            inp = np.expand_dims(mel, axis=-1)
            inp = np.repeat(inp, 3, axis=-1)
            inp = np.expand_dims(inp, axis=0)
            
            pred = model.predict(inp, verbose=0)[0]
            preds.append(pred)
            
            emo_idx = int(np.argmax(pred))
            emo = EMOTION_CLASSES[emo_idx]
            timeline.append((i*chunk_len/SR, (i+1)*chunk_len/SR, emo))
            
            progress.progress((i+1)/NUM_CHUNKS)
            status_text.text(f"Processing chunk {i+1}/{NUM_CHUNKS}")

        # Prediction logic loop eken eliyata gaththa (Fixes IndexError)
        avg_pred = np.mean(preds, axis=0)
        final_idx = int(np.argmax(avg_pred))
        emotion = EMOTION_CLASSES[final_idx]
        confidence = float(avg_pred[final_idx])

    # ====================== 8. RESULTS ======================
    st.markdown(f"""
    <div class="glass" style="text-align:center; border-top: 5px solid #ffd700; margin-top:20px; padding:10px;">
        <p style="margin:0; color:#cccccc; letter-spacing:2px; font-weight:bold;">VIBE CHECK RESULT</p>
        <div style="font-size:50px;">{EMO_ICONS[emotion]}</div>
        <h1 style="font-size:50px; color:#ffffff !important;">{emotion}</h1>
        <div style="background:rgba(255,215,0,0.1); display:inline-block; padding:8px 20px; border-radius:50px; border:1px solid rgba(255,215,0,0.3);">
            <b style="color:#ffd700 !important; font-size:1.2rem;">{confidence:.1%} AI Confidence</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.write("")
   
    st.markdown("### 📊 Deep Emotional Analysis")
    for emo, prob in zip(EMOTION_CLASSES, avg_pred):
        icon = EMO_ICONS.get(emo, "🎵")
        percentage = prob * 100
        st.markdown(f"""
        <div style="margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 1rem; font-weight: 600; color: white;">{icon} {emo}</span>
                <span style="font-size: 0.9rem; font-weight: 800; color: #ffd700;">{percentage:.1f}%</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.05); height: 10px; border-radius: 10px; width: 100%; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="width: {percentage}%; background: linear-gradient(90deg, #ffd700, #ff8c00); height: 100%; border-radius: 10px; box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    <p class='footer-text'>Powered by <b>MobileNetV2</b> & <b>TensorFlow</b></p>
    <p class='footer-sub'>Designed For Sinhala Emotion Recognition | Research Project 2026</p>
</div>
""", unsafe_allow_html=True)
