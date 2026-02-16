import streamlit as st
import librosa
import numpy as np
import json
import time

# ==============================
# CONFIG (Must be first)
# ==============================
st.set_page_config(
    page_title="Sinhala Song Personality Profiling",
    page_icon="logo.png",
    layout="wide"
)

SR = 22050 
REFERENCE_PATH = "feature_reference.json"
MAX_AUDIO_DURATION = 90

# CSS for Glassmorphism and Buttons
st.markdown("""
<style>
.stApp {
    background: #000000;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    padding-top: 20px !important;
}
            
/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid rgba(255, 215, 0, 0.1);
}

.sub-title {
    text-align: center;
    color: #ccc;
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 30px;
}                       
                        
.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
            
[data-testid="column"] { display: flex; align-items: center; }

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
</style>
""", unsafe_allow_html=True)

# ==============================
# REFERENCE DATA 
# ==============================
try:
    with open(REFERENCE_PATH, "r") as f:
        FEATURE_REF = json.load(f)
except FileNotFoundError:
    FEATURE_REF = {
        "tempo_bpm": {"min": 51.6796875, "max": 215.33203125},
        "loudness_db": {"min": -21.87506, "max": -3.8104322},
        "timbre_spectral_centroid": {"min": 820.41504892454, "max": 4050.5376104961033}
    }

# ==============================
# LOGIC FUNCTIONS
# ==============================
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=SR, duration=MAX_AUDIO_DURATION)
    tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo_array[0]) if isinstance(tempo_array, (np.ndarray, list)) else float(tempo_array)
    rms = librosa.feature.rms(y=y)[0]
    energy = np.mean(librosa.amplitude_to_db(rms, ref=np.max))
    timbre = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mode = "Major" if (chroma_mean[0] + chroma_mean[4] + chroma_mean[7]) > \
                      (chroma_mean[0] + chroma_mean[3] + chroma_mean[7]) else "Minor"
    return tempo, energy, timbre, mode

def normalize(value, feature):
    min_v, max_v = FEATURE_REF[feature]["min"], FEATURE_REF[feature]["max"]
    return (np.clip(value, min_v, max_v) - min_v) / (max_v - min_v)

def level(norm):
    return "Low" if norm < 0.5 else "High"

def compute_big_five(feature_levels):
    traits = ["Extraversion","Agreeableness","Neuroticism","Conscientiousness","Openness"]
    votes = {k: [] for k in traits}
    rules = {
        ("tempo", "High"): {"Extraversion":2,"Agreeableness":1,"Neuroticism":0,"Conscientiousness":1,"Openness":1},
        ("tempo", "Low"):  {"Extraversion":0,"Agreeableness":1,"Neuroticism":2,"Conscientiousness":1,"Openness":2},
        ("energy", "High"): {"Extraversion":2,"Agreeableness":0,"Neuroticism":1,"Conscientiousness":1,"Openness":1},
        ("energy", "Low"):  {"Extraversion":0,"Agreeableness":2,"Neuroticism":2,"Conscientiousness":2,"Openness":2},
        ("mode", "Major"): {"Extraversion":2,"Agreeableness":2,"Neuroticism":0,"Conscientiousness":1,"Openness":1},
        ("mode", "Minor"): {"Extraversion":0,"Agreeableness":1,"Neuroticism":2,"Conscientiousness":1,"Openness":2},
        ("timbre", "High"): {"Extraversion":1,"Agreeableness":0,"Neuroticism":1,"Conscientiousness":0,"Openness":2},
        ("timbre", "Low"):  {"Extraversion":1,"Agreeableness":2,"Neuroticism":0,"Conscientiousness":2,"Openness":1}
    }
    for feat, lvl in feature_levels.items():
        if (feat, lvl) in rules:
            for t in traits: votes[t].append(rules[(feat, lvl)][t])
    results = {}
    for t, v in votes.items():
        avg = sum(v)/len(v) if v else 0
        lbl = "Low" if avg < 0.67 else "Moderate" if avg < 1.33 else "High"
        results[t] = {"level": lbl, "confidence": avg / 2}
    return results

# ==============================
# UI COMPONENTS
# ==============================
def feature_card(name, value, unit, lvl):
    val_str = f"{value:.2f} {unit}" if isinstance(value, (int, float)) and unit != "" else value
    st.markdown(f"""
    <div style="background:#1f1f1f; padding:18px 25px; border-radius:12px; border: 1px solid #333; margin-bottom:12px; 
                display:flex; justify-content:space-between; align-items:center; height:80px;">
        <div>
            <small style="color:#888; text-transform:uppercase; letter-spacing:1px;">{name}</small><br>
            <span style="font-size:1.1rem; color:#eee; font-weight:500;">{val_str}</span>
        </div>
        <div style="background:rgba(255, 215, 0, 0.1); color:#ffd700; padding:6px 14px; border-radius:8px; 
                    font-weight:bold; border: 1.5px solid #ffd700; min-width:70px; text-align:center;">
            {lvl}
        </div>
    </div>
    """, unsafe_allow_html=True)

def personality_card(trait, data):
    st.markdown(f"""
    <div style="background:#1f1f1f; padding:18px 25px; border-radius:12px; border: 1px solid #333; margin-bottom:12px; height:80px; 
                display:flex; flex-direction:column; justify-content:center;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <b style="color:#eee;">{trait}</b>
            <span style="color:#ffd700; font-weight:bold; font-size:0.9rem; text-transform:uppercase;">{data['level']}</span>
        </div>
        <div style="background:#333; height:8px; border-radius:10px; width:100%; overflow:hidden;">
            <div style="width:{data['confidence']*100:.1f}%; background:#ffd700; height:100%; border-radius:10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# MAIN APP
# ==============================
st.markdown("<h1 style='text-align:center;'>🎧 Sinhala Song Emotion AI </h1>", unsafe_allow_html=True)
st.markdown("<p class ='sub-title'>Analyze the acoustic traits of music to predict listener personality</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)

# --- SESSION STATE FIX ---
# Changed 'active_file' to 'active_file_personality' to prevent cross-page conflict
if 'active_file_personality' not in st.session_state:
    st.session_state.active_file_personality = None

# Show uploader ONLY if no file is active
if st.session_state.active_file_personality is None:
    st.markdown("""
         <div style="background: rgba(255, 215, 0, 0.03); padding: 10px; border-radius: 25px; border: 1px dashed #ffd700; text-align: center;">
            <h2 style="color: #ffd700; margin-bottom:10px;">Drop Your Audio Track</h2>
            <p style="color: #888;">We analyze BPM, Timbre, and Energy to map your vibe.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("", type=["mp3", "wav"], key="personality_uploader")
    if uploaded:
        st.session_state.active_file_personality = uploaded
        st.rerun()

# Processing if file is uploaded
if st.session_state.active_file_personality is not None:
    uploaded_file = st.session_state.active_file_personality
    
    # Premium Player & Reset Button UI (Updated)
    st.markdown('', unsafe_allow_html=True)
    
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
        if st.button("↺ NEW ANALYSIS", use_container_width=True):
            st.session_state.active_file_personality = None
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

    st.audio(uploaded_file)

    # Trigger Analysis
    with st.spinner("🧠 AI is extracting acoustic personality features..."):
        tempo, energy, timbre, mode = extract_features(uploaded_file)
        f_levels = {
            "tempo": level(normalize(tempo, "tempo_bpm")),
            "energy": level(normalize(energy, "loudness_db")),
            "timbre": level(normalize(timbre, "timbre_spectral_centroid")),
            "mode": mode
        }
        personality = compute_big_five(f_levels)

    st.markdown(f"""
        <hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>
                """, unsafe_allow_html=True )
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🎼 Acoustic Features")
        feature_card("Tempo", tempo, "BPM", f_levels["tempo"])
        feature_card("Energy (Loudness)", energy, "dB", f_levels["energy"])
        feature_card("Timbre (Brightness)", timbre, "Hz", f_levels["timbre"])
        feature_card("Musical Mode", mode, "", mode)

    with col2:
        st.subheader("🧠 Personality Profile")
        for trait, data in personality.items():
            personality_card(trait, data)

    # ==============================
    # 5-TRAIT INSIGHT GENERATOR
    # ==============================
    summary_parts = []
    for trait, data in personality.items():
        lvl = data['level'].lower()
        summary_parts.append(f"<b style='color:#ffd700;'>{lvl} {trait}</b>")

    full_traits_str = ", ".join(summary_parts[:-1]) + " and " + summary_parts[-1]

    # FINAL INSIGHT TEXT
    insight_text = f"""
        suggests a profile characterized by {full_traits_str}. 
        This indicates a listener who resonates with this song's specific energy, 
        reflecting a unique psychological alignment with its acoustic structure.
    """

    # DISPLAY INSIGHT BOX
    st.markdown(f"""
    <hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>            
    <div style="
        background: linear-gradient(145deg, rgba(255, 215, 0, 0.05), rgba(0, 0, 0, 0.1));
        padding: 25px; 
        border-radius: 20px; 
        border-left: 5px solid #ffd700; 
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);
        text-align: left; 
        margin: 20px 0;
        backdrop-filter: blur(10px);
    ">
        <h6 style="color: #ffd700; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 10px 0;">
            ✨ AI Personality Insight (Big Five)
        </h6>          
        <p style="margin: 0; font-size: 1.05rem; color: #ffffff; line-height: 1.7; font-style: italic;">
            "Based on the acoustic profiling of this song, your musical preference {insight_text}"
        </p>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p style='color:#888; font-size:0.9rem;'>Powered by <b>Librosa</b> & <b>Streamlit</b></p>
    <p style='color:#555; font-size:0.8rem;'>Designed For Sinhala Emotion Recognition | Research Project © 2026</p>
</div>
""", unsafe_allow_html=True)
