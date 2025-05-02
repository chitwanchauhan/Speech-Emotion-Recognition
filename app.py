import streamlit as st
from utils.predict import predict_emotion
import os

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎙 Speech Emotion Recognition App")

st.write("Upload a WAV file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    file_path = os.path.join("temp_audio.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format='audio/wav')

    if st.button("Predict Emotion"):
        with st.spinner("Analyzing..."):
            try:
                emotion = predict_emotion(file_path)
                st.success(f"🎯 Predicted Emotion: *{emotion}*")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")