import os
import numpy as np
import librosa
import pywt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_combined_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfcc_mean = np.mean(mfcc, axis=1)

    # DWT
    coeffs = pywt.wavedec(y, 'db4', level=2)
    dwt_features = np.concatenate([np.mean(c).reshape(1) for c in coeffs[:3]])

    # SWT (only 1-level)
    swt_coeffs = pywt.swt(y, 'db4', level=1)
    swt_features = np.array([np.mean(c[0]) for c in swt_coeffs])

    # Cepstrum
    spectrum = np.fft.fft(y)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    cepstrum_features = np.array([np.mean(cepstrum[:3])])

    # Combine
    combined_features = np.concatenate((mfcc_mean, dwt_features, swt_features, cepstrum_features))
    return combined_features

# ------------------------
# Dataset Creation
# ------------------------
X = []
y = []

healthy_path = "data/healthy"
faulty_path = "data/faulty"

for filename in os.listdir(healthy_path):
    if filename.endswith(".wav"):
        path = os.path.join(healthy_path, filename)
        features = extract_combined_features(path)
        X.append(features)
        y.append(0)

for filename in os.listdir(faulty_path):
    if filename.endswith(".wav"):
        path = os.path.join(faulty_path, filename)
        features = extract_combined_features(path)
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

# ------------------------
# Data Preparation
# ------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------
# Model Training
# ------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# Evaluation
# ------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------
# Save Model & Scaler
# ------------------------
joblib.dump(model, "engine_fault_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ------------------------
# Test on New File
# ------------------------
def predict_new_audio(file_path):
    model = joblib.load("engine_fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    features = extract_combined_features(file_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return "Healthy" if prediction == 0 else "Faulty"

result = predict_new_audio("test.wav")
print("Test file prediction:", result)
