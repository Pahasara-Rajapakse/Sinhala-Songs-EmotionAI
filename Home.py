import streamlit as st
import qrcode
from io import BytesIO

st.set_page_config(
    page_title="Sinhala Song Emotion AI",
    page_icon="logo.png",
    layout="wide"
)

# ====================== STYLES ======================
st.markdown("""
<style>
.stApp {
    background: #000000; 
    color: #ffffff; 
    font-family: 'Inter', sans-serif; 
    padding-top: 20px !important;
}
            
h1, h2, h3 { 
    color: #ffffff !important; 
    text-shadow: 0 2px 10px rgba(0,0,0,0.5); 
    text-align: center; }
            
.sub-title {
    text-align: center;
    color: #ccc;
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 30px;
}
            
/* Sidebar Fix */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid rgba(255, 215, 0, 0.1);
}            

.card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-top: 5px solid #ffd700; 
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    margin: 1rem 0;
    transition: all 0.3s ease;
    height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}            

.card:hover {
    transform: translateY(-8px);
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 215, 0, 0.4);
    box-shadow: 0 10px 30px rgba(255, 215, 0, 0.1);
}

.icon { font-size: 3rem; margin-bottom: 0.5rem; display: block; }
            
p { color: #bbbbbb !important; font-size: 0.95rem; }

.stButton>button {
    width: 100%; border-radius: 50px; height: 45px;
    background: transparent; color: #FFFFFF !important;
    font-weight: bold; border: 1px solid #FFD700; transition: 0.3s ease;
    
}

.stButton>button:hover, .stButton>button:active, .stButton>button:focus {
    background-color: #FFD700 !important;
    color: #000000 !important;
    border: 1px solid #FFD700 !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.4);
}

.stButton>button p { color: inherit !important; }

hr { border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent); }

</style>
""", unsafe_allow_html=True)

# --- QR Code Fix ---
# Sidebar එකේ පෙන්නමු
with st.sidebar:
    st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H, # Error correction එක වැඩි කළා
    box_size=10,
    border=4, # Border එක අනිවාර්යයෙන් 4ක් තියෙන්න ඕනේ
    )

    app_url = "https://sinhala-songs-emotion-ai.streamlit.app"

    qr.add_data(app_url)
    qr.make(fit=True)

    # කළු කොටු සහ සුදු පසුබිම (මේක ඕනම ෆෝන් එකකට රීඩ් වෙනවා)
    img = qr.make_image(fill_color="black", back_color="white") 

    buf = BytesIO()
    img.save(buf, format="PNG")
    st.image(buf, caption="Scan this with your phone", width=250) # Width එක පොඩ්ඩක් අඩු කළා පේන්න ලේසි වෙන්න

# --- PAGE ROUTER SETUP ---
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Function to change page
def nav_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# ====================== PAGE LOGIC ======================

if st.session_state.page == "home":
    # HEADER
    st.markdown("<h1 style='text-align:center;'>🎧 Sinhala Song Emotion AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class = 'sub-title'>Next-generation deep learning system designed for Sinhala Music Emotion Recognition and smart playback.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ====================== FEATURES NAVIGATION ======================

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class='card'>
        <div>
            <span class='icon'>🎯</span>
            <h3 style='margin:0;'>Analyzer</h3>
            <p>Detailed emotional analysis, confidence scores, and audio segmentation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Button eka center karana trick eka
    sub_col1, sub_col2, sub_col3 = st.columns([1.2, 2, 0.8])
    with sub_col2:
        if st.button("Launch Analyzer", key="btn_analyzer"):
            st.switch_page("pages/1_Single_Song_Analyzer.py")

with col2:
    st.markdown("""
    <div class='card'>
        <div>
            <span class='icon'>🎶</span>
            <h3 style='margin:0;'>Smart Player</h3>
            <p>Organize and play your music library based on real-time emotional classification.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sub_col1, sub_col2, sub_col3 = st.columns([1.2, 2, 0.8])
    with sub_col2:
        if st.button("Open Smart Player", key="btn_player"):
            st.switch_page("pages/2_Emotion_Music_Player.py")

with col3:
    st.markdown("""
    <div class='card'>
        <span class='icon'>🧠</span>
        <h3 style='margin:0;'>Personality</h3>
        <p>Predict listener personality traits through acoustic profiling and Big Five metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    sub_col1, sub_col2, sub_col3 = st.columns([1.4, 2, 0.6])
    with sub_col2:
        if st.button("View Profiling", key="btn_personality"):
            st.switch_page("pages/3_Find_The_Personality.py")


# FOOTER
st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding-bottom: 2rem;'>
    <p style='color:#888; font-size:0.9rem;'>Powered by MobileNetV2 Architecture & TensorFlow</p>
    <p style='color:#555; font-size:0.8rem;'>Sinhala Emotion Recognition Research Project © 2026</p>
</div>
""", unsafe_allow_html=True)
