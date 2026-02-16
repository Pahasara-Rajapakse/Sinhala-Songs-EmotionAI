import streamlit as st
import qrcode
from io import BytesIO

st.set_page_config(
    page_title="Sinhala Song Personality Profiling",
    page_icon="logo.png",
    layout="wide"
)

def show_about_us_full():
    
    st.markdown("""
        <style>
        .about-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 215, 0, 0.2);
            margin-bottom: 20px;
        }
        .stat-card {
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 3px solid #ffd700;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .tech-badge {
            background: rgba(255, 215, 0, 0.1);
            color: #ffd700;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            border: 1px solid #ffd700;
            margin-right: 5px;
        }
        section[data-testid="stSidebar"] {
            background-color: #0a0a0a !important;
            border-right: 1px solid rgba(255, 215, 0, 0.1);
        }
        .stApp {
            background: #000000;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            padding-top: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Header Section ---
    st.markdown("""
        <div style="text-align: center; padding-bottom: 10px;">
            <h1 style='color: #ffffff; font-size: 3.2rem; font-weight: 800; margin-bottom: 10px; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);'>
                🎧 Sinhala Song Emotion AI
            </h1>
            <p style='color: #888; font-size: 1.1rem; letter-spacing: 3px; text-transform: uppercase;'>
                Advanced Music Emotion Recognition Engine
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4], gap="large")

    with col1:
        # --- Mission Section ---
        st.markdown("""
            <div class="about-container">
                <h2 style="color: #ffd700; margin-top: 0;">🎯 Project Vision</h2>
                <p style="color: #ddd; line-height: 1.8; font-size: 1.05rem;">
                    This research project aims to bridge the gap between <b>Sinhala Musical Heritage</b> and <b>Computational Psychology</b>. 
                    By utilizing state-of-the-art Deep Learning, we analyze the emotional core of local music to understand how it 
                    correlates with human personality archetypes.
                </p>
                <div style="margin-top: 15px;">
                    <span class="tech-badge">Python</span><span class="tech-badge">Deep Learning</span><span class="tech-badge">Librosa</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- Deep Learning (MobileNetV2) Section ---
        st.markdown("""
            <div class="about-container" style="border-left: 6px solid #ffd700;">
                <h2 style="color: #ffd700; margin-top: 0;">🧠 MobileNetV2 Architecture</h2>
                <p style="color: #ddd; line-height: 1.7;">
                    Our classifier leverages the power of <b>MobileNetV2</b>, a sophisticated Convolutional Neural Network (CNN) 
                    optimized for high-speed feature extraction.
                </p>

            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                <div class="stat-card">
                    <h3 style="color: #ffd700; margin:0;">5</h3>
                    <p style="color: #777; font-size: 0.7rem; margin:0;">EMOTIONS</p>
                </div>
                <div class="stat-card">
                    <h3 style="color: #ffd700; margin:0;">CNN</h3>
                    <p style="color: #777; font-size: 0.7rem; margin:0;">ENGINE</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        
        # QR Code Generation
        app_url = "https://sinhala-songs-emotion-ai.streamlit.app" 
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=8, border=4)
        qr.add_data(app_url.strip())
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.image(buf, use_container_width=True)
        
        st.markdown(f"""
                <p style="color: #888; font-size: 1rem; margin-top: 15px; text-align: center;">
                    Scan to access the research portal directly on your smartphone. Open with your chrome browser.
                </p>
                <div style="padding: 10px; text-align: center; background: rgba(255,215,0,0.05); border: 1px dashed rgba(255,215,0,0.3); border-radius: 5px; margin-top: 10px;">
                    <code style="color: #ffd700; font-size: 1rem;">{app_url}</code>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("<br><hr style='border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,215,0,0.3), transparent);'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center; padding-bottom: 2rem;'>
            <p style='color:#888; font-size:0.9rem;'>Powered by MobileNetV2 Architecture & TensorFlow</p>
            <p style='color:#555; font-size:0.8rem;'>Sinhala Emotion Recognition Research Project © 2026</p>
        </div>
    """, unsafe_allow_html=True)

# පේජ් එක පෙන්නන්න මෙතනින් කෝල් කරන්න
show_about_us_full()