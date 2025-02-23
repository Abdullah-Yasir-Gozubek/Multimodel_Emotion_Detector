import librosa
import numpy as np
import sounddevice as sd
import joblib

# Load the pre-trained model and scaler
model = joblib.load('voice_emotion_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to extract features from audio buffer
def extract_features_from_buffer(buffer, sr):
    y = librosa.util.normalize(buffer)  # Normalize
    mfccs = librosa.feature.mfcc(y=buffer, sr=sr, n_mfcc=13)  # Match training: 13 MFCCs
    features = np.mean(mfccs, axis=1)  # Only mean MFCCs, no deltas or extra features
    print("Feature shape:", features.shape)  # Debug: Should be (13,)
    print("Features:", features)  # Debug: Inspect feature values
    return scaler.transform(features.reshape(1, -1))[0]  # Scale features

# Callback function for real-time audio input
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio error: {status}")
    audio_data = indata[:, 0]
    features = extract_features_from_buffer(audio_data, sr=22050)
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    max_prob = max(probabilities)
    emotion = model.classes_[np.argmax(probabilities)]
    if max_prob > 0.3:  # Lower threshold to 0.3
        print(f'Predicted emotion: {emotion}')
    else:
        print(f'Predicted emotion: uncertain (max prob: {max_prob:.2f})')
    print(f'Probabilities: {dict(zip(model.classes_, probabilities))}')
    print(f'Raw audio shape:', audio_data.shape)

# Set up real-time audio stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=22050, blocksize=11025):  # Reduced blocksize
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped by user.")