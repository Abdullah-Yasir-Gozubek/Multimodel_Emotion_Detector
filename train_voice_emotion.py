import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Function to extract MFCC features from audio files

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    y = librosa.util.normalize(y)  # Normalize audio amplitude
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Reduce to 13 MFCCs for simplicity
    features = np.mean(mfccs, axis=1)  # Only mean MFCCs, no deltas or extra features for now
    return features

# Load your dataset (update with your actual path)
dataset_path = "C:\\Projects\\Multimodel_Emotion_Detector\\Audio_Speech_Actors_01-24"
audio_files = []
labels = []

# Define emotion mapping based on RAVDESS convention for speech
emotion_map = {
    '01': 'neutral',    # 01 = neutral
    '02': 'calm',       # 02 = calm
    '03': 'happy',      # 03 = happy
    '04': 'sad',        # 04 = sad
    '05': 'angry',      # 05 = angry
    '06': 'fearful',    # 06 = fearful
    '07': 'disgust',    # 07 = disgust
    '08': 'surprised'   # 08 = surprised
}

# Walk through the dataset directory
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            audio_files.append(file_path)
            # Extract emotion label from filename (e.g., "03" in "03-01-01-01-01-01-01.wav")
            # Split the filename by '-' and take the third part (index 2, since index 0 is modality, 1 is vocal channel)
            emotion_code = file.split('-')[2]  # e.g., "03" from "03-01-03-..."
            # Remove any extra characters (e.g., '.wav') and get the first two digits
            emotion_code = emotion_code.split('.')[0][:2]  # Ensures we get "03", not "03-01..."
            labels.append(emotion_map[emotion_code])

# Debugging: Print emotion distribution
print("Emotion distribution:", Counter(labels))

# Extract features for all audio files
features = np.array([extract_features(file) for file in audio_files])

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Train SVM model with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 0.1]
}
model = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=3, n_jobs=-1)
model.fit(X_train_res, y_train_res)

print("Best parameters:", model.best_params_)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save the trained model and scaler
joblib.dump(model, 'voice_emotion_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved as 'voice_emotion_model.pkl' and 'scaler.pkl'")