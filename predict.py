import librosa
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model("model/ser_model.h5")

# Emotion labels used in training — make sure these match your dataset
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_features(audio_path):
    data, sample_rate = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)  # shape (40, 1)
    return np.expand_dims(mfcc, axis=0)   # shape (1, 40, 1)

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]
