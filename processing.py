import librosa
import numpy as np
import os

# File paths
test_file = "test.wav"
eco_file  = "eco_healthy.wav"

# Load Audio files
test, sr_test = librosa.load(test_file, sr=None)
eco, sr_eco = librosa.load(eco_file, sr=None)

# Print basic info
print("Test sample rate:", sr_test)
print("Test shape:", test.shape)
print("First 100 samples of Test:", test[:100])

# Save waveform as .npy for later use
np.save("test.npy", test)

# ---- Save waveform as CSV (time + amplitude) ----
time = np.arange(len(test)) / sr_test
waveform_data = np.column_stack((time, test))

# ---- Save waveform as CSV (time + amplitude) ----
time = np.arange(len(test)) / sr_test
waveform_data = np.column_stack((time, test))

csv_file = "test_waveform.csv"
np.savetxt(csv_file, waveform_data, delimiter=",", header="Time(s),Amplitude", comments="")
print(f"Saved waveform to {csv_file}")

# ---- Feature Extraction ----
mfccs = librosa.feature.mfcc(y=test, sr=sr_test, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=test, sr=sr_test)
spectral_centroid = librosa.feature.spectral_centroid(y=test, sr=sr_test)

print("MFCCs shape:", mfccs.shape)
print("Chroma shape:", chroma.shape)
print("Spectral Centroid shape:", spectral_centroid.shape)


import librosa
import numpy as np
import pandas as pd

# Load audio
file = "test.wav"
data, sr = librosa.load(file, sr=None)   # data = waveform, sr = sample rate

# Create time axis
time_axis = np.arange(len(data)) / sr

# Ensure both are 1D arrays
time_axis = time_axis.flatten()
data = data.flatten()

# Create DataFrame
df = pd.DataFrame({
    "Time (s)": time_axis,
    "Amplitude": data
})

# Save to CSV
df.to_csv("test_waveform.csv", index=False)

print("Saved waveform with time axis to test_waveform.csv")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.stats import entropy
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")

# ---------- Load Audio ----------
file = "test.wav"
try:
    signal, sr = librosa.load(file, sr=48000)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

print(f"Duration of signal is: {len(signal)/sr:.2f} seconds")

# ---------- Save waveform as CSV ----------
time_axis = np.arange(len(signal)) / sr
df = pd.DataFrame({"Time (s)": time_axis, "Amplitude": signal})
df.to_csv("test_waveform.csv", index=False)
print("Saved waveform to test_waveform.csv")

# ---------- Plot Waveform ----------
plt.figure(figsize=(14, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title("Waveform of test.wav")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.ylim((-1, 1))
plt.show()

# ---------- If you want multiple files like in screenshot ----------
# Example: load 3 different audio files
files = {
    "test": "test.wav",
    
}
plt.figure(figsize=(14, 4))
librosa.display.waveshow(signal, sr=sr, color="crimson")   # red line
plt.title("Waveform of test.wav")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.ylim((-1, 1))
plt.grid(True, alpha=0.3)
plt.show()


