#test_single_file.py
import librosa
import numpy as np
import joblib

# Paths to your model, scaler, and the test audio file
MODEL_PATH = 'voice_emotion_model.pkl'
SCALER_PATH = 'scaler.pkl'
AUDIO_PATH = r"C:\Projects\Multimodel_Emotion_Detector\Audio_Speech_Actors_01-24\Actor_09\03-01-06-01-01-02-09.wav"

# Load the pre-trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_emotion_from_file(file_path):
    # 1) Load the .wav file at 22050 Hz (matching your training)
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2) Normalize amplitude
    y = librosa.util.normalize(y)
    
    # 3) Extract the same features you used in training (13 MFCCs, then mean across time)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfccs, axis=1).reshape(1, -1)  # shape (1, 13)
    
    # 4) Scale the features using the saved scaler
    features_scaled = scaler.transform(features)
    
    # 5) Predict probabilities for each emotion class
    probabilities = model.predict_proba(features_scaled)[0]
    max_prob = max(probabilities)
    predicted_class_index = np.argmax(probabilities)
    predicted_emotion = model.classes_[predicted_class_index]
    
    # 6) Print results
    print("\n--- Single File Prediction ---")
    print(f"Audio File: {file_path}")
    print(f"Predicted Emotion: {predicted_emotion} (prob={max_prob:.2f})")
    print("Probabilities by class:")
    for emotion_class, prob in zip(model.classes_, probabilities):
        print(f"  {emotion_class}: {prob:.2f}")

if __name__ == "__main__":
    predict_emotion_from_file(AUDIO_PATH)
