# ---------------------------------------------------------
# This script performs WAVELET-BASED DENOISING on an audio file.
# It loads an input WAV file, removes noise using wavelet thresholding,
# plots the original vs denoised signal, and saves the cleaned audio.
# ---------------------------------------------------------


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io.wavfile import write
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# ---------- PARAMETERS ----------
input_file = "label-1.wav"        # Input audio file
output_file = "label-1_denoised.wav"  # Output denoised file
wavelet_type = "db4"              # Wavelet type (Daubechies 4)
decomposition_level = 3           # Number of decomposition levels
threshold_type = "soft"           # Threshold type: 'soft' or 'hard'

# ---------- LOAD AUDIO ----------
print(f"Loading: {input_file}")
signal, sr = librosa.load(input_file, sr=None)
print(f"Sampling rate: {sr} Hz | Signal length: {len(signal)} samples")

# ---------- WAVELET DENOISING ----------
def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply thresholding
    coeffs_denoised = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    
    # Reconstruct the denoised signal
    denoised = pywt.waverec(coeffs_denoised, wavelet)
    return denoised[:len(signal)]

print("Applying wavelet denoising...")
denoised_signal = wavelet_denoise(signal, wavelet_type, decomposition_level, threshold_type)

# ---------- PLOT ORIGINAL vs DENOISED ----------
plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
plt.plot(signal, color='gray')
plt.title("Original Audio Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(denoised_signal, color='blue')
plt.title("Denoised Audio Signal (Wavelet)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# ---------- NORMALIZE AND SAVE ----------
# Normalize to -1 to 1
denoised_signal = denoised_signal / np.max(np.abs(denoised_signal))

# Convert to 16-bit PCM format
scaled = np.int16(denoised_signal * 32767)

# Save denoised .wav file
write(output_file, sr, scaled)
print(f"\n✅ Denoised file saved as: {output_file}")
print(f"Location: {os.path.abspath(output_file)}")
