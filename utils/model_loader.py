import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_emotion_model(path="mobileNetV2.keras"):
    return tf.keras.models.load_model(path)
