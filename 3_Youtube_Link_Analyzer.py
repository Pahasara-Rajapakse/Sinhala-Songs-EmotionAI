import streamlit as st
from yt_dlp import YoutubeDL
import librosa
import numpy as np
import tensorflow as tf
import os

# ---------------- CONFIG ----------------
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
TARGET_FRAMES = 431
NUM_CHUNKS = 10
EMOTION_CLASSES = ["Calm", "Energetic", "Happy", "Romantic", "Sad"]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model("mobileNetV2.keras")

# ---------------- UI ----------------
st.title("🎧 Sinhala Song Emotion AI – YouTube Video")

yt_link = st.text_input("Paste YouTube Link Here:")

if yt_link:
    try:
        # ---------------- Play Video Directly ----------------
        st.video(yt_link)  # Stream directly from YouTube

        with st.spinner("Downloading audio for emotion analysis..."):
            # ---------------- Download Audio Only ----------------
            audio_file = "temp_audio.mp3"
            ydl_opts_audio = {
                "format": "bestaudio/best",
                "outtmpl": audio_file,
                "quiet": True,
                "noplaylist": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
            }
            with YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([yt_link])

        # ---------------- Process Audio ----------------
        y, _ = librosa.load(audio_file, sr=SR, mono=True, duration=100)
        st.success(f"Audio loaded: {len(y)/SR:.1f} seconds")

        # ---------------- Prediction ----------------
        def extract_logmel(y):
            mel = librosa.feature.melspectrogram(
                y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        preds = []
        chunk_len = len(y) // NUM_CHUNKS
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

        avg_pred = np.mean(preds, axis=0)
        final_idx = int(np.argmax(avg_pred))
        emotion = EMOTION_CLASSES[final_idx]
        confidence = float(avg_pred[final_idx])

        st.markdown(f"### 🎯 Predicted Emotion: **{emotion}**")
        st.markdown(f"Confidence: **{confidence:.2%}**")

        # ---------------- Cleanup ----------------
        os.remove(audio_file)

    except Exception as e:
        st.error(f"Failed to process YouTube link: {e}")
